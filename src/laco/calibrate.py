from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import load_model_artifact
from .utils import load_config, merge_overrides, save_json


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    """Finite-sample split-conformal quantile (higher score is anomalous)."""
    scores = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        raise ValueError("Cannot calibrate from zero finite scores")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(scores))
    return float(scores[rank - 1]), rank


def filtered_errors(errors: np.ndarray, episodes: np.ndarray, ewma_alpha: float) -> np.ndarray:
    if not 0 < ewma_alpha <= 1:
        raise ValueError("ewma_alpha must be in (0, 1]")
    out = np.empty_like(errors, dtype=np.float64)
    last_by_episode: dict[int, float] = {}
    for i, (err, episode) in enumerate(zip(errors, episodes, strict=True)):
        previous = last_by_episode.get(int(episode), float(err))
        current = ewma_alpha * float(err) + (1.0 - ewma_alpha) * previous
        out[i] = current
        last_by_episode[int(episode)] = current
    return out


@torch.no_grad()
def prediction_errors(cache_dir: Path, model_path: str, split: str, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model, artifact, dev = load_model_artifact(model_path, device)
    z = np.load(cache_dir / "z_t.npy", mmap_mode="r")
    next_z = np.load(cache_dir / "z_next.npy", mmap_mode="r")
    action = np.load(cache_dir / "actions.npy", mmap_mode="r")[:, : model.action_dim]
    labels = np.load(cache_dir / "split.npy", mmap_mode="r")
    episodes = np.load(cache_dir / "episode_id.npy", mmap_mode="r")
    indices = np.flatnonzero(labels.astype(str) == split)
    if len(indices) == 0:
        raise ValueError(f"No transitions with split={split!r}")
    lm, ls = np.asarray(artifact["latent_mean"]), np.asarray(artifact["latent_std"])
    am, ass = np.asarray(artifact["action_mean"]), np.asarray(artifact["action_std"])
    errors: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        ix = indices[start : start + batch_size]
        z0 = (np.asarray(z[ix], dtype=np.float32) - lm) / np.maximum(ls, 1e-6)
        z1 = (np.asarray(next_z[ix], dtype=np.float32) - lm) / np.maximum(ls, 1e-6)
        act = (np.asarray(action[ix], dtype=np.float32) - am) / np.maximum(ass, 1e-6)
        predicted = model(torch.as_tensor(z0, device=dev), torch.as_tensor(act, device=dev)).cpu().numpy()
        errors.append(np.linalg.norm(predicted - (z1 - z0), axis=1))
    return np.concatenate(errors), np.asarray(episodes[indices])


@torch.no_grad()
def select_regularization(cache_dir: Path, model_path: str, device: str, candidates: list[float], trust_region: float) -> tuple[float, dict[str, float]]:
    """Select a conservative Tikhonov value on held-out *prediction* residuals.

    This is a numerical-stability selection, not a claim that held-out demos
    identify counterfactual recovery.  Candidates that routinely need an action
    outside the trust region are penalised.
    """
    model, artifact, dev = load_model_artifact(model_path, device)
    z, action, target = _load_split(cache_dir, "validation", model.action_dim)
    lm, ls = np.asarray(artifact["latent_mean"]), np.asarray(artifact["latent_std"])
    am, ass = np.asarray(artifact["action_mean"]), np.asarray(artifact["action_std"])
    z = (z - lm) / np.maximum(ls, 1e-6); action = (action - am) / np.maximum(ass, 1e-6); target = target / np.maximum(ls, 1e-6)
    zz, aa = torch.as_tensor(z, device=dev), torch.as_tensor(action, device=dev)
    pred = model(zz, aa); _b, jac = model.components(zz)
    residual = (target - pred.cpu().numpy())
    jac_np = jac.cpu().numpy()
    scores: dict[str, float] = {}
    for lam in candidates:
        if lam <= 0: raise ValueError("regularization candidates must be positive")
        gains = []
        for e, j in zip(residual, jac_np, strict=True):
            delta = np.linalg.solve(j.T @ j + lam * np.eye(j.shape[1]), j.T @ e)
            if np.linalg.norm(delta * ass) <= trust_region:
                gains.append(float(e @ e - np.sum((e - j @ delta) ** 2)))
        scores[str(lam)] = float(np.mean(gains)) if gains else float("-inf")
    best = max(candidates, key=lambda value: scores[str(value)])
    return float(best), scores


def _load_split(cache_dir: Path, split: str, action_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(cache_dir / "split.npy", mmap_mode="r").astype(str); ix = np.flatnonzero(labels == split)
    if not len(ix): raise ValueError(f"No transitions with split={split!r}")
    z = np.asarray(np.load(cache_dir / "z_t.npy", mmap_mode="r")[ix], dtype=np.float32)
    nxt = np.asarray(np.load(cache_dir / "z_next.npy", mmap_mode="r")[ix], dtype=np.float32)
    actions = np.asarray(np.load(cache_dir / "actions.npy", mmap_mode="r")[ix, :action_dim], dtype=np.float32)
    return z, actions, nxt - z


def run(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("calibration", config)
    errors, episodes = prediction_errors(Path(c["cache"]), c["model"], c.get("split", "calibration"), c.get("device", "cuda"), int(c.get("batch_size", 2048)))
    filtered = filtered_errors(errors, episodes, float(c.get("ewma_alpha", 0.35)))
    # Conformal unit is an episode maximum, so alpha controls per-episode false alarm under exchangeability.
    unique = np.unique(episodes)
    maxima = np.asarray([filtered[episodes == episode].max() for episode in unique], dtype=np.float64)
    threshold, rank = conformal_quantile(maxima, float(c.get("alpha", 0.05)))
    regularization, lambda_scores = select_regularization(Path(c["cache"]), c["model"], c.get("device", "cuda"),
                                                           [float(x) for x in c.get("lambda_candidates", [.01, .03, .1, .3])],
                                                           float(c.get("trust_region", .15)))
    payload = {"format": "laco_calibration_v1", "alpha": float(c.get("alpha", 0.05)), "threshold": threshold,
               "conformal_rank": rank, "n_calibration_episodes": int(len(unique)), "score": "episode_max_EWMA_l2_residual",
               "regularization": regularization, "regularization_validation_gain": lambda_scores,
               "ewma_alpha": float(c.get("ewma_alpha", 0.35)), "model": str(c["model"]),
               "assumption": "Calibration and deployment episodes are exchangeable nominal episodes; this does not validate counterfactual corrections."}
    save_json(c["output"], payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate LACo anomaly threshold with split conformal episode maxima.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    print(run(merge_overrides(load_config(args.config), args.set)))


if __name__ == "__main__":
    main()
