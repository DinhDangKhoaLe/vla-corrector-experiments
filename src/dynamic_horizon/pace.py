"""PACE speed valleys and segment contrasts for MetaWorld delta actions.

Selection/calibration follow niejunnan25/PACE 5d0f4a9. MetaWorld commands
are Cartesian deltas, so speed is norm(clipped XYZ * 0.01) * FPS, not the
difference of consecutive delta commands. The same profile is used on demos.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .types import Decision


def speed_profile(actions, fps=80.0, delta_scale=0.01, smoothing_window=11, polynomial_order=3):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 4 or not np.isfinite(actions).all():
        raise ValueError("PACE requires finite MetaWorld [T, 4] actions")
    speed = np.linalg.norm(np.clip(actions[:, :3], -1, 1) * delta_scale, axis=1) * fps
    window = min(smoothing_window, (len(speed) - 1) // 2 * 2 + 1)
    return savgol_filter(speed, window, polynomial_order) if window >= max(5, polynomial_order + 1) else speed


def contrast(speed, start, end):
    peak = float(speed[start:end + 1].max())
    value = min(peak - float(speed[start]), peak - float(speed[end]))
    # Saturated Cartesian commands produce genuinely flat speed segments.
    # Do not calibrate thresholds from Savitzky–Golay floating-point roundoff.
    tolerance = 32 * np.finfo(np.float64).eps * max(1.0, abs(peak))
    return value if value > tolerance else 0.0


def select_horizon(actions, threshold, min_distance=10, **profile_settings):
    speed = speed_profile(actions, **profile_settings)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("PACE threshold must be positive and finite")
    valleys = find_peaks(-speed, distance=min_distance)[0]
    for valley in valleys:
        if contrast(speed, 0, int(valley)) >= threshold:
            return max(1, int(valley))
    return len(actions)


def demo_episodes(source):
    """Read only action/ID columns; embedded images are never decoded."""
    import pyarrow.parquet as pq

    paths = sorted((Path(source) / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No demonstration parquet files in {source}")
    parts = defaultdict(list)
    for path in paths:
        schema = pq.read_schema(path)
        if "task_id" not in schema.names:
            raise ValueError("MetaWorld calibration requires the canonical task_id column")
        table = pq.read_table(path, columns=["episode_index", "frame_index", "task_id", "action"])
        columns = table.to_pydict()
        for ep, frame, task, action in zip(*columns.values(), strict=True):
            parts[int(ep)].append((int(frame), int(task), action))
    for ep, rows in sorted(parts.items()):
        rows.sort(key=lambda row: row[0])
        if len({row[1] for row in rows}) != 1 or len({row[0] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate frames or inconsistent task in episode {ep}")
        yield ep, rows[0][1], np.asarray([row[2] for row in rows], dtype=np.float32)


def calibrate(config):
    settings = dict(config["settings"])
    artifact = Path(settings.pop("artifact"))
    percentile = settings.pop("percentile", 5.0)
    distance = settings.pop("min_distance", 10)
    if artifact.exists():
        raise FileExistsError(f"Refusing to replace calibration: {artifact}")
    values = defaultdict(list)
    counts = defaultdict(int)
    for _, task_id, actions in demo_episodes(config["dataset"]):
        counts[task_id] += 1
        if len(actions) < 20:
            continue
        speed = speed_profile(actions, **settings)
        boundaries = [0, *find_peaks(-speed, distance=distance)[0], len(speed) - 1]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end > start:
                value = contrast(speed, int(start), int(end))
                if value > 0:
                    values[task_id].append(value)
    if set(counts) != set(values):
        raise ValueError(f"No positive speed contrasts for tasks {set(counts) - set(values)}")
    payload = {"method": "pace", "action_representation": "metaworld_delta_xyz",
               "settings": config["settings"], "dataset": str(Path(config["dataset"]).resolve()),
               "thresholds": {str(k): float(np.percentile(v, percentile)) for k, v in values.items()},
               "episode_counts": dict(counts)}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


class PACE:
    def __init__(self, settings):
        self.settings = {k: v for k, v in settings.items() if k not in ("artifact", "percentile")}
        payload = json.loads(Path(settings["artifact"]).read_text())
        # Artifact location is not a calibration parameter: downloaded/copied
        # calibrations remain valid when supplied through --artifact.
        calibrated = {k: v for k, v in payload["settings"].items() if k != "artifact"}
        requested = {k: v for k, v in settings.items() if k != "artifact"}
        if payload["action_representation"] != "metaworld_delta_xyz" or calibrated != requested:
            raise ValueError("PACE calibration settings do not match this run")
        self.thresholds = payload["thresholds"]

    def decide(self, sampler, task, **kwargs):
        prediction = sampler.sample()
        threshold = self.thresholds[str(task)]
        horizon = select_horizon(prediction.actions.cpu().numpy(), threshold, **self.settings)
        return Decision(prediction.actions[:horizon], {"threshold": threshold})
