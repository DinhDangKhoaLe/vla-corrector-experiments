"""PACE adapter based on niejunnan25/PACE's MIT-licensed implementation.

Reference: https://github.com/niejunnan25/PACE/tree/main/pace/src/pace
Joint-space selection follows the released speed-index convention. Cartesian
delta policies use translation magnitude. Calibration and online actions use
the same decoded units. Recalibrate old pace_style_v1 artifacts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.signal import find_peaks, savgol_filter


@dataclass(frozen=True)
class PaceSettings:
    max_horizon: int = 50
    motion_dims: tuple[int, ...] = (0, 1, 2)
    arm_groups: tuple[tuple[int, ...], ...] = ()
    signal: str = "delta"
    fps: float = 80.0
    smooth_window: int = 11
    polynomial_order: int = 3
    min_separation: int = 10
    percentile: float = 5.0

    def __post_init__(self):
        if self.max_horizon < 1 or self.signal not in {"joint", "delta"}:
            raise ValueError("Invalid max_horizon or signal")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be positive and finite")
        if self.smooth_window < 1 or self.smooth_window % 2 != 1:
            raise ValueError("smooth_window must be positive and odd")
        if not 0 <= self.polynomial_order < self.smooth_window or self.min_separation < 1:
            raise ValueError("Invalid polynomial_order or min_separation")
        if not 0 <= self.percentile <= 100:
            raise ValueError("percentile must lie in [0, 100]")
        if not self.groups or any(not g or min(g) < 0 for g in self.groups):
            raise ValueError("Specify nonempty, nonnegative arm dimensions")

    @property
    def groups(self):
        return self.arm_groups or (self.motion_dims,)

    @classmethod
    def from_mapping(cls, data):
        data = dict(data)
        if "motion_dims" in data:
            data["motion_dims"] = tuple(data["motion_dims"])
        if "arm_groups" in data:
            data["arm_groups"] = tuple(tuple(g) for g in data["arm_groups"])
        return cls(**data)


@dataclass(frozen=True)
class PaceConfig:
    source: Path
    artifact: Path
    settings: PaceSettings
    dataset_format: str = "lerobot"
    action_key: str = "action"
    episodes_per_task: int = 50
    tasks: tuple[str, ...] = ()


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if set(raw) - {"dataset", "calibration", "pace"}:
        raise ValueError("Unexpected PACE section")
    ds, cal = raw["dataset"], raw["calibration"]
    cfg = PaceConfig(Path(ds["source"]), Path(cal["artifact"]),
                     PaceSettings.from_mapping(raw.get("pace", {})),
                     ds.get("format", "lerobot"), ds.get("action_key", "action"),
                     int(cal.get("episodes_per_task", 50)), tuple(ds.get("tasks", [])))
    if cfg.episodes_per_task < 1 or cfg.dataset_format not in {"lerobot", "robotwin"}:
        raise ValueError("Invalid dataset format or episodes_per_task")
    return cfg


def _profiles(actions, settings):
    actions = np.asarray(actions)
    if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("Actions must be finite, nonempty [time, channels]")
    profiles = []
    for group in settings.groups:
        if max(group) >= actions.shape[1]:
            raise ValueError("Arm dimensions exceed action dimension")
        values = actions[:, list(group)]
        if settings.signal == "joint":
            values = np.diff(values, axis=0) * settings.fps
        profiles.append(np.linalg.norm(values, axis=1))
    return profiles


def _smooth(speed, settings):
    window = min(settings.smooth_window, (len(speed) - 1) // 2 * 2 + 1)
    if window < max(5, settings.polynomial_order + 1):
        return speed.copy()
    return savgol_filter(speed, window, settings.polynomial_order)


def speed_profile(actions, settings):
    """First-arm profile, for diagnostics; selection evaluates every arm."""
    return _smooth(_profiles(actions, settings)[0], settings)


def _valleys(profile, settings):
    if len(profile) < settings.min_separation:
        return np.empty(0, dtype=int)
    return find_peaks(-profile, distance=settings.min_separation)[0]


def _contrast(profile, start, end):
    peak = float(np.max(profile[start:end + 1]))
    return min(peak - float(profile[start]), peak - float(profile[end]))


def choose_horizon(actions, threshold, settings):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("PACE threshold must be positive and finite")
    actions = np.asarray(actions, dtype=np.float64)
    horizon = min(len(actions), settings.max_horizon)
    events = []
    # Official selection uses the full prediction profile even when Hmax < L.
    for arm, speed in enumerate(_profiles(actions, settings)):
        profile = _smooth(speed, settings)
        for valley in _valleys(profile, settings):
            if valley > horizon:
                break
            score = _contrast(profile, 0, int(valley))
            events.append({"arm": arm, "index": int(valley), "contrast": score})
            if score >= threshold:
                horizon = min(horizon, int(valley))
                break
    accepted = horizon < min(len(actions), settings.max_horizon)
    return horizon, {"reason": "valley" if accepted else "fallback_max_horizon",
                     "threshold": float(threshold), "candidates": events}


def calibration_contrasts(episodes, settings):
    """Released rule: active arm, valley-bounded full-demo segments."""
    scores, analyzed, skipped = [], 0, 0
    for actions in episodes:
        if len(actions) < 20:
            skipped += 1
            continue
        speeds = _profiles(actions, settings)
        active = speeds[int(np.argmax([np.mean(s) for s in speeds]))]
        profile = _smooth(active, settings)
        bounds = [0, *_valleys(profile, settings), len(profile) - 1]
        for start, end in zip(bounds[:-1], bounds[1:]):
            if end > start:
                score = _contrast(profile, int(start), int(end))
                if score > 0:
                    scores.append(score)
        analyzed += 1
    return scores, analyzed, skipped


def calibrate(config):
    from .pace_data import read_demonstrations
    task_scores, counts = {}, {}
    for task, actions in read_demonstrations(config):
        if counts.get(task, 0) >= config.episodes_per_task:
            continue
        scores, analyzed, _ = calibration_contrasts([actions], config.settings)
        counts[task] = counts.get(task, 0) + analyzed
        task_scores.setdefault(task, []).extend(scores)
    missing = [task for task, scores in task_scores.items() if not scores]
    if missing or not task_scores:
        raise ValueError(f"No positive PACE calibration contrasts for tasks: {missing or 'all'}")
    tasks = {name: {"task_name": name, "threshold": float(np.percentile(scores, config.settings.percentile)),
                    "candidate_count": len(scores), "episode_count": counts[name]}
             for name, scores in sorted(task_scores.items())}
    artifact = {"format": "pace_v2", "reference": "https://github.com/niejunnan25/PACE",
                "settings": asdict(config.settings), "source": str(config.source), "tasks": tasks,
                "global_threshold": float(np.percentile([s for scores in task_scores.values() for s in scores], config.settings.percentile))}
    if config.artifact.exists():
        raise FileExistsError(f"Refusing to overwrite calibration: {config.artifact}")
    config.artifact.parent.mkdir(parents=True, exist_ok=True)
    config.artifact.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact


class PaceController:
    def __init__(self, artifact: dict[str, Any]):
        if artifact.get("format") != "pace_v2":
            raise ValueError("Recalibrate PACE: this selector requires pace_v2 decoded-action artifacts")
        self.settings = PaceSettings.from_mapping(artifact["settings"])
        self.by_name = {value["task_name"]: float(value["threshold"]) for value in artifact["tasks"].values()}

    @classmethod
    def from_config(cls, path):
        config = load_config(path)
        artifact = json.loads(config.artifact.read_text())
        if PaceSettings.from_mapping(artifact["settings"]) != config.settings:
            raise ValueError("PACE YAML settings differ from calibration; recalibrate")
        return cls(artifact)

    def select(self, task_name, actions, *, max_horizon=None):
        if task_name not in self.by_name:
            raise KeyError(f"No PACE calibration for task {task_name!r}")
        settings = self.settings
        if max_horizon is not None:
            if not isinstance(max_horizon, int) or not 1 <= max_horizon <= settings.max_horizon:
                raise ValueError(f"PACE execution cap must be 1..{settings.max_horizon}")
            settings = replace(settings, max_horizon=max_horizon)
        return choose_horizon(actions, self.by_name[task_name], settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
