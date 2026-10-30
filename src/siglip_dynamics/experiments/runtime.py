"""Optional monitor for the existing PI0.5 VLA-Corrector rollout loop."""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .artifacts import digest, file_digest
from .config import from_checkpoint_dict, load_config
from .model import ExperimentalCorrector


class ExperimentPredictor:
    def __init__(self, path, cfg, device):
        path = Path(path)
        self.checkpoint_path = path / "best_model.pt" if path.is_dir() else path
        self.model_dir = self.checkpoint_path.parent
        # Legacy directories use best_cosine_model.pt, not best_model.pt.
        obj = (torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
               if self.checkpoint_path.is_file() else {})
        self.device = torch.device(device)
        if obj.get("format") != "corrector_experiment_v1":
            from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor
            old = SiglipDynamicsPredictor(path, device=str(device))
            if (cfg.features.action_prefix or cfg.features.uncertainty_monitor
                    or cfg.model.horizons != [old.k_step]):
                raise ValueError("Legacy checkpoint supports only single-action, fixed-horizon monitoring; retrain this ablation")
            self.model = old.model
            self.action_dim, self.token_dim = old.action_dim, old.token_dim
            self.k_step, self.h_window = old.k_step, old.h_window
            self.checkpoint_path = old.checkpoint_path
            self.model_dir = old.model_dir
            self.experimental = False
        else:
            trained = from_checkpoint_dict(obj["experiment"])
            if trained.features.uncertainty_monitor != cfg.features.uncertainty_monitor:
                raise ValueError("Checkpoint uncertainty_monitor differs from YAML; use matching mode/checkpoint")
            if trained.features.action_prefix != cfg.features.action_prefix or trained.model.horizons != cfg.model.horizons:
                raise ValueError("YAML prefix/horizons differ from the trained checkpoint; retrain or select matching YAML")
            self.action_dim, self.token_dim = int(obj["action_dim"]), int(obj["token_dim"])
            self.model = ExperimentalCorrector(trained, self.token_dim, self.action_dim)
            self.model.load_state_dict(obj["state_dict"], strict=True)
            self.k_step, self.h_window, self.experimental = max(cfg.model.horizons), 1, True
        self.model.to(device).eval()
        self.model_type = "mlp"
        self.fingerprint = file_digest(self.checkpoint_path)
        self.prediction_calls = 0
        self.prediction_ms = 0.0

    @torch.no_grad()
    def predict(self, z, actions, horizon):
        start = perf_counter()
        zt = torch.as_tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
        at = torch.as_tensor(actions, dtype=torch.float32, device=self.device)[None, :, :self.action_dim]
        if self.experimental:
            delta, log_error = self.model.predict(zt, at, horizon)
            error = float(log_error.exp()[0]) if log_error is not None else None
        else:
            delta, error = self.model(zt, at[:, 0]), None
        delta = delta[0].cpu().numpy()
        if not np.isfinite(delta).all() or (error is not None and not np.isfinite(error)):
            raise FloatingPointError("Nonfinite corrector prediction")
        self.prediction_calls += 1
        self.prediction_ms += (perf_counter() - start) * 1000
        return delta, error

    def predict_delta_z(self, z, action):
        # Original baseline integration. Prefix experiments must call predict with a sequence.
        return self.predict(z, np.asarray(action)[None], self.k_step)[0]


def cosine_error(predicted, actual, epsilon=1e-8):
    p, a = np.asarray(predicted).ravel(), np.asarray(actual).ravel()
    pn, an = np.linalg.norm(p), np.linalg.norm(a)
    if pn <= epsilon and an <= epsilon:
        return 0.0
    if pn <= epsilon or an <= epsilon:
        return 1.0
    return float(1 - np.clip(np.dot(p, a) / (pn * an), -1, 1))


def policy_identity(path):
    root = Path(path)
    if not root.is_dir():
        raise ValueError("Experiments require a local policy snapshot for reproducible identity")
    identity = {}
    # Hash small config/stat files; bind the large weight file to its local file identity.
    for file in sorted(root.iterdir()):
        if file.suffix in {".json", ".safetensors"}:
            stat = file.stat()
            identity[file.name] = (file_digest(file) if stat.st_size < 10_000_000
                                   else {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def implementation_identity():
    versions = {}
    for package in ("torch", "transformers", "gymnasium", "metaworld", "mujoco"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    source = Path(__file__).resolve().parents[2]
    files = ("lerobot/policies/pi05/modeling_pi05.py", "lerobot/envs/metaworld.py",
             "lerobot/scripts/lerobot_eval_modified_detection.py", "siglip_dynamics/experiments/runtime.py")
    return {"versions": versions, "source_hashes": {p: file_digest(source / p) for p in files}}


def terminal_observation(observation, info):
    """Single-vector-env final observation, never a same-step auto-reset observation."""
    for key in ("final_obs", "final_observation"):
        if key not in info:
            continue
        final = info[key]
        mask = info.get("_" + key, [True])
        if not bool(mask[0]):
            continue
        if isinstance(final, dict):
            return final  # Gymnasium already batched the nested dictionary.
        final = final[0]
        def batch(value):
            return {k: batch(v) for k, v in value.items()} if isinstance(value, dict) else np.asarray(value)[None]
        return batch(final)
    return observation


class RuntimeMonitor:
    def __init__(self, cfg, predictor, mode, output, policy_signature=None):
        self.cfg, self.predictor, self.mode = cfg, predictor, mode
        self.shadow = mode != "evaluate"
        self.queue_bounded = cfg.features.action_prefix or cfg.evaluation.queue_bounded_monitor
        ev = cfg.evaluation
        self.environment_fingerprint = digest({"policy": policy_signature if policy_signature is not None else policy_identity(ev.policy),
                                               "tasks": ev.tasks, "episode_length": ev.episode_length,
                                               "chunk_size": ev.chunk_size, "n_action_steps": ev.n_action_steps,
                                               "implementation": implementation_identity()})
        self.protocol_fingerprint = digest({"environment": self.environment_fingerprint,
                                            "horizons": cfg.model.horizons, "prefix": cfg.features.action_prefix,
                                            "queue_bounded": self.queue_bounded,
                                            "score": "cosine-zero-aware-v1", "schedule": "rolling-queue-bounded-prefix-v1"})
        self.calibration_ids = set()
        self.threshold = None
        self.output = Path(output) / "experiment_episodes.jsonl"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(f"Choose a fresh evaluation output; {self.output} exists")
        self.pending, self.scores = [], []
        self.cooldown_until = -1

    def reset(self, task, seed):
        if (task, seed) in self.calibration_ids:
            raise ValueError("Evaluation task-seed overlaps a held-out calibration episode")
        self.task, self.seed = task, seed
        self.pending, self.scores = [], []
        self.cooldown_until = -1
        self.cancellations = 0
        self.start_time = perf_counter()
        self.start_calls = getattr(self.predictor, "prediction_calls", 0)
        self.start_prediction_ms = getattr(self.predictor, "prediction_ms", 0.0)

    def schedule(self, z, actions, step):
        for h in self.cfg.model.horizons:
            if self.queue_bounded and len(actions) < h:
                continue
            prefix = actions[:h] if self.cfg.features.action_prefix else actions[:1]
            pred, error = self.predictor.predict(z, prefix, h)
            self.pending.append({"due": step+h, "start": np.array(z, copy=True),
                                 "prediction": pred, "horizon": h, "uncertainty": error})

    def observe(self, z, step):
        due = [p for p in self.pending if p["due"] == step]
        self.pending = [p for p in self.pending if p["due"] > step]
        event = None
        for p in due:
            actual = z - p["start"]
            score = cosine_error(p["prediction"], actual)
            rmse = float(np.sqrt(np.mean((p["prediction"]-actual)**2)))
            if not np.isfinite(score) or not np.isfinite(rmse):
                raise FloatingPointError("Nonfinite monitor observation")
            self.scores.append({"step": step, "horizon": p["horizon"], "score": score,
                                "rmse": rmse, "predicted_rmse": p["uncertainty"]})
            if event is None or score > event["score"]:
                event = {"score": score, "actual": actual, "prediction": p["prediction"], "horizon": p["horizon"]}
        return event

    def cancel(self):
        self.cancellations += len(self.pending)
        self.pending.clear()

    def finish(self, success, error):
        row = {"task": self.task, "seed": self.seed, "success": bool(success), "error": error,
               "mode": self.mode, "model_fingerprint": self.predictor.fingerprint,
               "protocol_fingerprint": self.protocol_fingerprint,
               "environment_fingerprint": self.environment_fingerprint,
               "scores": self.scores,
               "rollout_wall_time_ms": (perf_counter() - self.start_time) * 1000,
               "monitor_prediction_calls": getattr(self.predictor, "prediction_calls", 0) - self.start_calls,
               "monitor_prediction_ms": getattr(self.predictor, "prediction_ms", 0.0) - self.start_prediction_ms,
               "cancelled_predictions": self.cancellations}
        if self.cfg.features.uncertainty_monitor:
            from .uncertainty import settings_signature
            row["uncertainty_settings"] = settings_signature(self.cfg)
        with open(self.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")


def load_runtime(path, checkpoint, device, mode, output):
    if mode not in ("evaluate", "collect-calibration"):
        raise ValueError(f"Unsupported experiment rollout mode: {mode}")
    cfg = load_config(path)
    if mode == "collect-calibration" and not cfg.features.uncertainty_monitor:
        raise ValueError("Calibration collection requires uncertainty_monitor")
    predictor = ExperimentPredictor(checkpoint, cfg, device)
    # Baseline evaluation retains the original rolling k-step/MAD/OGG code path exactly.
    active = (any(asdict(cfg.features).values()) or cfg.evaluation.queue_bounded_monitor
              or mode != "evaluate" or len(cfg.model.horizons) > 1)
    if cfg.features.uncertainty_monitor:
        from .uncertainty import UncertaintyMonitor
        monitor = UncertaintyMonitor(cfg, predictor, mode, output)
    else:
        monitor = RuntimeMonitor(cfg, predictor, mode, output) if active else None
    return predictor, monitor
