"""Fixed-K expected-error monitoring and empirical episode calibration.

The scale is supervised expected RMSE, not calibrated Gaussian uncertainty.
Calibration describes successful rollouts under the frozen collection protocol;
it is neither a safety guarantee nor evidence that OGG improves recovery.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import digest, file_digest, read_traces
from .runtime import RuntimeMonitor
from .train import error_statistics


def normalized_error(rmse, expected_rmse, floor):
    if expected_rmse is None or not np.isfinite([rmse, expected_rmse, floor]).all():
        raise ValueError("Expected finite RMSE and predicted RMSE for uncertainty monitoring")
    if rmse < 0 or expected_rmse < 0 or floor <= 0:
        raise ValueError("Invalid RMSE or scale floor")
    return float(rmse / max(expected_rmse, floor))


def settings_signature(cfg):
    values = asdict(cfg.uncertainty)
    # Locations and disabling OGG do not change pre-intervention monitoring.
    for name in ("artifact", "traces", "ogg_enabled"):
        values.pop(name)
    return values


def calibrate_uncertainty(cfg):
    rows = read_traces(cfg.uncertainty.traces)
    if rows[0]["mode"] != "collect-calibration":
        raise ValueError("Use separate collect-calibration rollouts for uncertainty calibration")
    if any(r.get("uncertainty_settings") != settings_signature(cfg) for r in rows):
        raise ValueError("Uncertainty settings differ from the collected traces; recollect")
    good = [r for r in rows if r["success"]]
    u = cfg.uncertainty
    if len(good) < u.min_successful_episodes:
        raise ValueError(f"Need {u.min_successful_episodes} successful calibration episodes, found {len(good)}")
    if any(not r["scores"] for r in good):
        raise ValueError("Successful calibration episode has no matured K-step forecasts")
    for row in rows:
        for s in row["scores"]:
            expected = normalized_error(s["rmse"], s["predicted_rmse"], u.rmse_floor)
            if not np.isclose(s["score"], expected):
                raise ValueError("Calibration requires RMSE/expected-RMSE scores from uncertainty mode")
    # Higher empirical quantile; no formal coverage claim.
    def quantile(values, q):
        return float(np.quantile(values, q, method="higher"))
    maxima = [max(s["score"] for s in r["scores"]) for r in good]
    raw_maxima = [max(s["rmse"] for s in r["scores"]) for r in good]
    records = [s for r in rows for s in r["scores"]]
    stats = error_statistics(records)
    supported = all(str(h) in stats and stats[str(h)]["count"] >= 20 and
                    stats[str(h)]["pearson_correlation"] is not None and
                    stats[str(h)]["pearson_correlation"] >= u.min_error_correlation
                    for h in cfg.model.horizons)
    artifact = {
        "format": "expected_rmse_monitor_v1",
        "model_fingerprint": rows[0]["model_fingerprint"],
        "protocol_fingerprint": rows[0]["protocol_fingerprint"],
        "uncertainty_settings": settings_signature(cfg),
        "threshold": quantile(maxima, u.episode_quantile),
        "raw_threshold": quantile(raw_maxima, u.episode_quantile),
        "ogg_max_expected_rmse": quantile(
            [s["predicted_rmse"] for r in good for s in r["scores"]], u.ogg_uncertainty_quantile),
        "uncertainty_validation": stats,
        "ogg_error_head_validated": supported,
        "successful_episodes": len(good),
        "episode_ids": [[r["task"], r["seed"]] for r in rows],
        "interpretation": "Empirical successful-episode thresholds; no OOD, safety or OGG-benefit guarantee",
    }
    path = Path(u.artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    return artifact


class UncertaintyMonitor(RuntimeMonitor):
    def __init__(self, cfg, predictor, mode, output, policy_signature=None):
        super().__init__(cfg, predictor, mode, output, policy_signature)
        self.protocol_fingerprint = digest({"base": self.protocol_fingerprint,
                                            "score": "rmse-over-expected-rmse-v1",
                                            "implementation": file_digest(Path(__file__)),
                                            "settings": settings_signature(cfg)})
        self.artifact = None
        if not self.shadow:
            with open(cfg.uncertainty.artifact, encoding="utf-8") as f:
                artifact = json.load(f)
            if (artifact.get("format") != "expected_rmse_monitor_v1" or
                    artifact.get("model_fingerprint") != predictor.fingerprint or
                    artifact.get("protocol_fingerprint") != self.protocol_fingerprint):
                raise ValueError("Uncertainty calibration does not match checkpoint/settings/environment; recollect")
            for key in ("threshold", "raw_threshold", "ogg_max_expected_rmse"):
                if not np.isfinite(artifact[key]) or artifact[key] < 0:
                    raise ValueError("Invalid uncertainty calibration threshold")
            self.artifact = artifact
            self.threshold = float(artifact["threshold"])
            self.calibration_ids = {tuple(x) for x in artifact["episode_ids"]}
            logging.info("Uncertainty monitor | K=%d E=%d normalized_threshold=%.6g raw_threshold=%.6g "
                         "OGG_error_cap=%.6g error_head_validated=%s",
                         cfg.model.horizons[0], cfg.evaluation.n_action_steps, self.threshold,
                         artifact["raw_threshold"], artifact["ogg_max_expected_rmse"],
                         artifact["ogg_error_head_validated"])
            if not artifact["ogg_error_head_validated"]:
                logging.warning("Error-head validation is weak; uncertainty mode will use ordinary replans, no OGG")
        self._reset_evidence()

    def _reset_evidence(self):
        self.previous_residual = None
        self.above_count = 0
        self.below_count = 0
        self.latched = False

    def reset(self, task, seed):
        super().reset(task, seed)
        self._reset_evidence()

    def cancel(self):
        super().cancel()
        # Old residuals and pending predictions do not describe the new action plan.
        self.previous_residual = None
        self.above_count = 0

    def observe(self, z, step):
        event = super().observe(z, step)
        if event is None:
            return None
        row = self.scores[-1]  # This mode validates exactly one fixed K.
        row["cosine_error"] = row["score"]
        row["score"] = normalized_error(row["rmse"], row["predicted_rmse"], self.cfg.uncertainty.rmse_floor)
        residual = np.asarray(event["actual"] - event["prediction"]).ravel()
        coherence = None
        if self.previous_residual is not None:
            denom = float(np.linalg.norm(residual) * np.linalg.norm(self.previous_residual))
            if denom > 1e-12:
                coherence = float(np.clip(np.dot(residual, self.previous_residual) / denom, -1, 1))
        self.previous_residual = residual.copy()
        row["direction_coherence"] = coherence
        event.update(row)
        return event

    def recovery_decision(self, event, step):
        """Alarms do not require coherence; only OGG eligibility does."""
        if self.shadow:
            return False, False
        u, a = self.cfg.uncertainty, self.artifact
        normalized_high = event["score"] > a["threshold"]
        raw_high = event["rmse"] > a["raw_threshold"]
        high = normalized_high or raw_high
        low = (event["score"] <= u.release_ratio * a["threshold"] and
               event["rmse"] <= u.release_ratio * a["raw_threshold"])
        self.below_count = self.below_count + 1 if low else 0
        if self.below_count >= u.reset_checks or step >= self.cooldown_until:
            self.latched = False
        if step < self.cooldown_until:
            self.above_count = 0
        else:
            self.above_count = self.above_count + 1 if high else 0
        trigger = high and self.above_count >= u.persistence_checks and not self.latched
        coherent = (event["direction_coherence"] is not None and
                    event["direction_coherence"] >= u.direction_coherence_min)
        ogg = bool(trigger and normalized_high and u.ogg_enabled and
                   self.cfg.evaluation.guidance and self.cfg.evaluation.guidance_eta > 0 and
                   a["ogg_error_head_validated"] and coherent and
                   event["predicted_rmse"] <= a["ogg_max_expected_rmse"])
        if trigger:
            self.latched = True
            self.cooldown_until = step + self.cfg.evaluation.cooldown_steps
            self.above_count = 0
        self.scores[-1].update({"alarm": bool(trigger), "ogg_allowed": ogg,
                               "recovery": "ogg_replan" if ogg else "normal_replan" if trigger else "continue",
                               "raw_alarm_evidence": bool(raw_high),
                               "normalized_alarm_evidence": bool(normalized_high)})
        return bool(trigger), ogg

