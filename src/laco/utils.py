from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_config(path: str | Path) -> dict[str, Any]:
    """Read JSON or YAML config without silently accepting a missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - requirements install PyYAML
        raise RuntimeError("YAML config requested; install PyYAML or use JSON") from exc
    value = yaml.safe_load(text)
    return value or {}


def merge_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides used by all LACo entry points."""
    result = dict(config)
    for entry in overrides:
        key, sep, raw = entry.partition("=")
        if not sep:
            raise ValueError(f"Expected KEY=VALUE override, got {entry!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = value
    return result


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def episode_split(episode_ids: np.ndarray, seed: int, val_fraction: float, calibration_fraction: float) -> np.ndarray:
    """Return split labels, deterministically constant within each episode."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 <= calibration_fraction < 1.0 and val_fraction + calibration_fraction < 1.0):
        raise ValueError("val_fraction + calibration_fraction must be in [0, 1)")
    labels = np.empty(len(episode_ids), dtype="U11")
    unique = np.unique(episode_ids)
    # A local Generator avoids Python's randomized hash and is reproducible across processes.
    rng = np.random.default_rng(seed)
    permutation = unique.copy()
    rng.shuffle(permutation)
    n_cal = int(round(len(permutation) * calibration_fraction))
    n_val = int(round(len(permutation) * val_fraction))
    mapping = {int(x): "calibration" for x in permutation[:n_cal]}
    mapping.update({int(x): "validation" for x in permutation[n_cal : n_cal + n_val]})
    for x in permutation[n_cal + n_val :]:
        mapping[int(x)] = "train"
    for i, ep in enumerate(episode_ids):
        labels[i] = mapping[int(ep)]
    return labels


def action_normalize(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(action, dtype=np.float32) - mean) / np.maximum(std, 1e-6)


def action_denormalize(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.asarray(action, dtype=np.float32) * np.maximum(std, 1e-6) + mean


def device_from_config(value: str | None) -> torch.device:
    if value and value.startswith("cuda") and torch.cuda.is_available():
        return torch.device(value)
    return torch.device("cpu")
