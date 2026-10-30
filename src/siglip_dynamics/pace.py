"""PACE-style training-free adaptive action-chunk execution.

This module is deliberately independent of the VLA Corrector.  It calibrates
one action-profile prominence threshold per task from demonstrations and, at
test time, executes up to the earliest sufficiently prominent low-motion
valley in a PI0.5 action chunk.  It is an implementation of the method
described in PACE, not the authors' code: the paper does not prescribe an
exact smoothing or prominence routine.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class PaceSettings:
    """Fixed PACE-style action-profile settings.

    ``motion_dims`` intentionally excludes the MetaWorld gripper action.  The
    profile therefore measures arm motion rather than gripper open/close
    transitions.
    """

    max_horizon: int = 50
    motion_dims: tuple[int, ...] = (0, 1, 2)
    smooth_window: int = 5
    prominence_window: int = 10
    min_separation: int = 10
    percentile: float = 5.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PaceSettings":
        allowed = {"max_horizon", "motion_dims", "smooth_window", "prominence_window", "min_separation", "percentile"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown pace settings: {sorted(unknown)}")
        data = dict(values)
        if "motion_dims" in data:
            data["motion_dims"] = tuple(int(v) for v in data["motion_dims"])
        obj = cls(**data)
        if obj.max_horizon < 2 or not obj.motion_dims:
            raise ValueError("pace.max_horizon must be >= 2 and motion_dims must not be empty")
        if obj.smooth_window < 1 or obj.prominence_window < 1 or obj.min_separation < 1:
            raise ValueError("PACE window and separation settings must be positive")
        if not 0.0 <= obj.percentile <= 100.0:
            raise ValueError("pace.percentile must lie in [0, 100]")
        return obj


@dataclass(frozen=True)
class PaceConfig:
    cache: Path
    source: Path
    artifact: Path
    settings: PaceSettings
    max_windows_per_task: int = 0


def load_config(path: str | Path) -> PaceConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("PACE YAML must be a mapping")
    allowed = {"dataset", "calibration", "pace"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown PACE YAML sections: {sorted(unknown)}")
    dataset = raw.get("dataset", {})
    calibration = raw.get("calibration", {})
    if not isinstance(dataset, dict) or not isinstance(calibration, dict) or not isinstance(raw.get("pace", {}), dict):
        raise ValueError("dataset, calibration, and pace must be mappings")
    if set(dataset) - {"cache", "source"}:
        raise ValueError("dataset supports only cache and source")
    if set(calibration) - {"artifact", "max_windows_per_task"}:
        raise ValueError("calibration supports only artifact and max_windows_per_task")
    if "cache" not in dataset or "source" not in dataset or "artifact" not in calibration:
        raise ValueError("PACE YAML requires dataset.cache, dataset.source, and calibration.artifact")
    settings = PaceSettings.from_mapping(raw.get("pace", {}))
    return PaceConfig(
        cache=Path(dataset["cache"]),
        source=Path(dataset["source"]),
        artifact=Path(calibration["artifact"]),
        settings=settings,
        max_windows_per_task=int(calibration.get("max_windows_per_task", 0)),
    )


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or width <= 1:
        return values.copy()
    # Edge padding preserves the profile length and avoids zero-padding valleys.
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width, dtype=np.float64) / float(width), mode="valid")


def speed_profile(actions: np.ndarray, settings: PaceSettings) -> np.ndarray:
    """Smoothed joint-space speed between adjacent actions in a chunk."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions shaped [T,A], got {actions.shape}")
    if actions.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    if max(settings.motion_dims) >= actions.shape[1]:
        raise ValueError(f"motion_dims={settings.motion_dims} incompatible with action dimension {actions.shape[1]}")
    speed = np.linalg.norm(np.diff(actions[:, list(settings.motion_dims)], axis=0), axis=1)
    return _moving_average(speed, settings.smooth_window)


def valley_candidates(profile: np.ndarray, settings: PaceSettings) -> list[dict[str, float | int]]:
    """Return local speed valleys and a bounded local prominence score."""
    profile = np.asarray(profile, dtype=np.float64)
    candidates: list[dict[str, float | int]] = []
    for i in range(1, max(1, profile.size - 1)):
        if i >= profile.size - 1:
            break
        # Tie-break rightward so a flat valley produces at most one candidate.
        if not (profile[i] <= profile[i - 1] and profile[i] < profile[i + 1]):
            continue
        lo = max(0, i - settings.prominence_window)
        hi = min(profile.size, i + settings.prominence_window + 1)
        left_peak = float(np.max(profile[lo:i])) if i > lo else float(profile[i])
        right_peak = float(np.max(profile[i + 1 : hi])) if i + 1 < hi else float(profile[i])
        prominence = max(0.0, min(left_peak, right_peak) - float(profile[i]))
        candidates.append({"index": int(i), "horizon": int(i + 1), "speed": float(profile[i]), "prominence": prominence})
    return candidates


def choose_horizon(actions: np.ndarray, threshold: float, settings: PaceSettings) -> tuple[int, dict[str, Any]]:
    """Choose PACE's earliest qualifying valley, otherwise execute Hmax."""
    n = min(int(np.asarray(actions).shape[0]), settings.max_horizon)
    if n < 2:
        return n, {"reason": "short_chunk", "candidates": []}
    profile = speed_profile(np.asarray(actions)[:n], settings)
    candidates = valley_candidates(profile, settings)
    # Respect d_min while preserving PACE's earliest-boundary preference.
    accepted: list[dict[str, float | int]] = []
    for candidate in candidates:
        horizon = int(candidate["horizon"])
        if accepted and horizon - int(accepted[-1]["horizon"]) < settings.min_separation:
            continue
        accepted.append(candidate)
        if float(candidate["prominence"]) >= threshold:
            return horizon, {
                "reason": "valley",
                "profile": profile.tolist(),
                "candidates": accepted,
                "threshold": float(threshold),
            }
    return n, {"reason": "fallback_max_horizon", "profile": profile.tolist(), "candidates": accepted, "threshold": float(threshold)}


def _episode_task_map(source: Path) -> tuple[dict[int, int], dict[int, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - supplied by the project data stack
        raise RuntimeError("PACE calibration needs pyarrow to read MetaWorld parquet metadata") from exc
    data_dir = source / "data"
    paths = sorted(data_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data found under {data_dir}")
    episode_to_task: dict[int, int] = {}
    for path in paths:
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        episodes = table.column("episode_index").to_numpy(zero_copy_only=False)
        tasks = table.column("task_index").to_numpy(zero_copy_only=False)
        for episode, task in zip(episodes, tasks, strict=True):
            episode_i, task_i = int(episode), int(task)
            previous = episode_to_task.setdefault(episode_i, task_i)
            if previous != task_i:
                raise ValueError(f"Episode {episode_i} has inconsistent task indices: {previous}, {task_i}")
    names: dict[int, str] = {}
    task_path = source / "meta" / "tasks.parquet"
    if task_path.exists():
        table = pq.read_table(task_path)
        columns = set(table.column_names)
        # LeRobot v3 stores the task text as the parquet's pandas index.
        task_text_column = "task" if "task" in columns else "__index_level_0__"
        if {"task_index", task_text_column} <= columns:
            for task, name in zip(
                table.column("task_index").to_numpy(zero_copy_only=False),
                table.column(task_text_column).to_pylist(),
                strict=True,
            ):
                names[int(task)] = str(name)
    return episode_to_task, names


def calibrate(config: PaceConfig) -> dict[str, Any]:
    """Calibrate task thresholds from demonstrations only; no policy rollouts."""
    cache = config.cache
    actions_path, episodes_path = cache / "actions.npy", cache / "episode_index.npy"
    if not actions_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"PACE needs actions.npy and episode_index.npy under {cache}")
    actions = np.load(actions_path, mmap_mode="r")
    episodes = np.load(episodes_path, mmap_mode="r")
    if len(actions) != len(episodes):
        raise ValueError("actions.npy and episode_index.npy have different lengths")
    episode_to_task, task_names = _episode_task_map(config.source)
    per_task: dict[int, list[float]] = {}
    start = 0
    windows_per_task: dict[int, int] = {}
    while start < len(episodes):
        end = start + 1
        episode = int(episodes[start])
        while end < len(episodes) and int(episodes[end]) == episode:
            end += 1
        if episode not in episode_to_task:
            raise KeyError(f"Cache episode {episode} is missing from source parquet metadata")
        task = episode_to_task[episode]
        sequence = np.asarray(actions[start:end], dtype=np.float32)
        max_start = max(0, len(sequence) - config.settings.max_horizon + 1)
        for offset in range(max_start):
            if config.max_windows_per_task and windows_per_task.get(task, 0) >= config.max_windows_per_task:
                break
            profile = speed_profile(sequence[offset : offset + config.settings.max_horizon], config.settings)
            per_task.setdefault(task, []).extend(float(c["prominence"]) for c in valley_candidates(profile, config.settings))
            windows_per_task[task] = windows_per_task.get(task, 0) + 1
        start = end
    all_scores = [score for scores in per_task.values() for score in scores]
    if not all_scores:
        raise ValueError("No PACE valley candidates found; verify action cache and settings")
    global_threshold = float(np.percentile(np.asarray(all_scores), config.settings.percentile))
    thresholds: dict[str, dict[str, Any]] = {}
    for task, scores in sorted(per_task.items()):
        threshold = float(np.percentile(np.asarray(scores), config.settings.percentile)) if scores else global_threshold
        thresholds[str(task)] = {
            "task_name": task_names.get(task, f"task_{task}"),
            "threshold": threshold,
            "candidate_count": len(scores),
            "window_count": windows_per_task.get(task, 0),
        }
    artifact = {
        "format": "pace_style_v1",
        "method_note": "Training-free PACE-style reference implementation; smoothing/prominence are repository-defined.",
        "settings": asdict(config.settings),
        "source": {"cache": str(config.cache), "dataset": str(config.source)},
        "global_threshold": global_threshold,
        "tasks": thresholds,
    }
    config.artifact.parent.mkdir(parents=True, exist_ok=True)
    with config.artifact.open("w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    logging.info("PACE calibrated %d tasks from %d candidate valleys -> %s", len(thresholds), len(all_scores), config.artifact)
    return artifact


class PaceController:
    """Task-threshold lookup plus action-chunk horizon selection for evaluation."""

    def __init__(self, artifact: dict[str, Any]):
        if artifact.get("format") != "pace_style_v1":
            raise ValueError("Not a pace_style_v1 calibration artifact")
        self.settings = PaceSettings.from_mapping(dict(artifact["settings"]))
        self.global_threshold = float(artifact["global_threshold"])
        self.by_name = {
            str(value.get("task_name")): float(value["threshold"])
            for value in dict(artifact["tasks"]).values()
        }

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PaceController":
        config = load_config(config_path)
        with config.artifact.open(encoding="utf-8") as f:
            artifact = json.load(f)
        return cls(artifact)

    def select(self, task_name: str, actions: np.ndarray) -> tuple[int, dict[str, Any]]:
        threshold = self.by_name.get(str(task_name), self.global_threshold)
        horizon, details = choose_horizon(actions, threshold, self.settings)
        details["task_name"] = str(task_name)
        return horizon, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate",))
    parser.add_argument("--config", required=True, help="PACE YAML configuration")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    print(json.dumps(calibrate(config), indent=2))


if __name__ == "__main__":
    main()
