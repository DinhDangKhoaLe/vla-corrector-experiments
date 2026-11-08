from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .model import LatentJacobianModel


class Mode(str, enum.Enum):
    CONTINUE = "CONTINUE"
    ADJUST = "ADJUST"
    REPLAN = "REPLAN"


@dataclass
class Decision:
    mode: Mode
    action_queue: np.ndarray | None
    prediction_error: float
    filtered_error: float
    controllability: float
    predicted_gain: float
    gain_lcb: float
    uncertainty: float
    reason: str
    correction_success: bool | None = None


class LACoController:
    """Three-mode queue controller; it has no access to simulator state or success."""

    def __init__(self, models: list[LatentJacobianModel], artifact: dict[str, Any], calibration: dict[str, Any], *,
                 device: torch.device, action_low: float = -1., action_high: float = 1., trust_region: float = .15,
                 regularization: float = .1, kappa: float = 1., correction_horizon: int = 2,
                 max_failed_retries: int = 1, condition_limit: float = 1e4, use_uncertainty: bool = True):
        if not models: raise ValueError("Need one or more models")
        self.models, self.device = models, device
        self.lm, self.ls = np.asarray(artifact["latent_mean"]), np.asarray(artifact["latent_std"])
        self.am, self.ass = np.asarray(artifact["action_mean"]), np.asarray(artifact["action_std"])
        self.threshold = float(calibration["threshold"]); self.ewma_alpha = float(calibration.get("ewma_alpha", .35))
        self.action_low, self.action_high, self.trust_region, self.regularization = action_low, action_high, trust_region, regularization
        self.kappa, self.correction_horizon, self.max_failed_retries, self.condition_limit, self.use_uncertainty = kappa, correction_horizon, max_failed_retries, condition_limit, use_uncertainty
        self.reset(np.zeros_like(self.lm))

    def reset(self, observation: np.ndarray) -> None:
        self.queue = np.empty((0, len(self.am)), dtype=np.float32); self.nominal: list[np.ndarray] = []
        self.filtered_error: float | None = None; self.failed_retries = 0; self.pending_error: float | None = None
        self.current_z = np.asarray(observation, dtype=np.float32)

    def _z(self, z: np.ndarray) -> np.ndarray: return (np.asarray(z, dtype=np.float32) - self.lm) / np.maximum(self.ls, 1e-6)
    def _action(self, action: np.ndarray) -> np.ndarray: return (action[:len(self.am)] - self.am) / np.maximum(self.ass, 1e-6)

    @torch.no_grad()
    def _components(self, z: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        zz = torch.as_tensor(z[None], dtype=torch.float32, device=self.device)
        values = [m.components(zz) for m in self.models]
        return [b[0].cpu().numpy() for b, _ in values], [j[0].cpu().numpy() for _, j in values]

    @torch.no_grad()
    def _delta(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        zz = torch.as_tensor(z[None], dtype=torch.float32, device=self.device); aa = torch.as_tensor(self._action(action)[None], dtype=torch.float32, device=self.device)
        return np.mean([m(zz, aa)[0].cpu().numpy() for m in self.models], axis=0)

    def _rollout(self, z: np.ndarray) -> list[np.ndarray]:
        nominal = [z.copy()]
        for action in self.queue:
            nominal.append(nominal[-1] + self._delta(nominal[-1], action))
        return nominal

    def on_new_chunk(self, observation: np.ndarray, actions: np.ndarray) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < len(self.am): raise ValueError("Expected [horizon, action_dim] action chunk")
        self.current_z = self._z(observation); self.queue = actions.copy(); self.nominal = self._rollout(self.current_z)

    def next_action(self) -> np.ndarray:
        if not len(self.queue): raise RuntimeError("No action queued; call on_new_chunk")
        return self.queue[0].copy()

    def mark_executed(self) -> None:
        if not len(self.queue): raise RuntimeError("No action queued")
        self.queue = self.queue[1:]

    def _replan(self, error: float, filtered: float, reason: str, success: bool | None) -> Decision:
        self.queue = np.empty((0, len(self.am)), dtype=np.float32); self.nominal = []
        return Decision(Mode.REPLAN, None, error, filtered, 0., 0., 0., 0., reason, success)

    def step(self, observation: np.ndarray) -> Decision:
        """Call after one executed environment action; return a queue update for the next action."""
        observed = self._z(observation)
        if not self.nominal: return self._replan(float("nan"), float("nan"), "missing_nominal", None)
        expected = self.nominal[1] if len(self.nominal) > 1 else self.nominal[0]
        error_vec = expected - observed; error = float(np.linalg.norm(error_vec))
        filtered = error if self.filtered_error is None else self.ewma_alpha * error + (1-self.ewma_alpha)*self.filtered_error
        self.filtered_error = filtered
        correction_success = None
        if self.pending_error is not None:
            correction_success = error < self.pending_error
            self.failed_retries = 0 if correction_success else self.failed_retries + 1
            self.pending_error = None
            if not correction_success and self.failed_retries > self.max_failed_retries:
                return self._replan(error, filtered, "previous_correction_did_not_reduce_error", correction_success)
        self.current_z = observed
        if filtered <= self.threshold:
            self.nominal = self._rollout(observed)
            return Decision(Mode.CONTINUE, self.queue.copy(), error, filtered, 0., 0., 0., 0., "within_calibrated_normal_range", correction_success)
        if not len(self.queue): return self._replan(error, filtered, "anomaly_with_empty_queue", correction_success)
        # Error points from observed state toward the nominal state.  Solve in standardized action units.
        _bias, jacobians = self._components(observed); j = np.mean(jacobians, axis=0)
        gram = j.T @ j + self.regularization * np.eye(j.shape[1], dtype=np.float32)
        cond = float(np.linalg.cond(gram))
        if not np.isfinite(cond) or cond > self.condition_limit: return self._replan(error, filtered, "ill_conditioned_jacobian", correction_success)
        delta_u = np.linalg.solve(gram, j.T @ error_vec)
        delta_actual = delta_u * self.ass
        norm = float(np.linalg.norm(delta_actual))
        if norm > self.trust_region:
            return self._replan(error, filtered, "correction_exceeds_trust_region", correction_success)
        candidate = self.queue.copy(); span = min(self.correction_horizon, len(candidate))
        for i in range(span): candidate[i, :len(delta_actual)] += delta_actual / span
        if np.any(candidate[:, :len(delta_actual)] < self.action_low) or np.any(candidate[:, :len(delta_actual)] > self.action_high):
            return self._replan(error, filtered, "correction_exceeds_action_limits", correction_success)
        gains = []
        for ji in jacobians:
            gains.append(float(error_vec @ error_vec - np.sum((error_vec - ji @ delta_u) ** 2)))
        gain, uncertainty = float(np.mean(gains)), float(np.std(gains))
        lcb = gain - self.kappa * uncertainty if self.use_uncertainty else gain
        projection = j @ np.linalg.pinv(j) @ error_vec
        rho = float((projection @ projection) / (error_vec @ error_vec + 1e-8))
        if lcb <= 0 or not np.isfinite(lcb):
            return self._replan(error, filtered, "unreliable_or_nonpositive_gain", correction_success)
        self.queue = candidate; self.nominal = self._rollout(observed); self.pending_error = error
        return Decision(Mode.ADJUST, self.queue.copy(), error, filtered, rho, gain, lcb, uncertainty, "abnormal_reliable_bounded_correction", correction_success)
