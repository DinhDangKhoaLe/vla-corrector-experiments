"""Held-out cosine thresholds, with explicit provenance and provisional demo fallback."""

import json
from pathlib import Path

import numpy as np

from siglip_dynamics.experiments.artifacts import file_digest

from .data import TIMING, FrameCache, WindowDataset
from .runtime import Predictor, checkpoint_path, protocol


def read_episodes(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({(row["task"], row["seed"]) for row in rows}) != len(rows):
        raise ValueError("COAP traces must contain unique, nonempty task/seed episodes")
    if any(row.get("error") for row in rows):
        raise ValueError("Resolve crashed COAP episodes before calibration/analysis")
    return rows


def calibrate(cfg, *, source="rollouts", traces=None):
    artifact = Path(cfg.monitor.artifact)
    if artifact.exists():
        raise FileExistsError(f"Choose a fresh calibration artifact: {artifact}")
    predictor = Predictor(checkpoint_path(cfg), cfg.training.device)
    identity = protocol(cfg)
    episodes = []
    if source == "rollouts":
        path = Path(traces or cfg.evaluation.output + "_calibration/coap_episodes.jsonl")
        rows = read_episodes(path)
        if any(
            row.get("model_fingerprint") != predictor.fingerprint
            or row.get("protocol_fingerprint") != identity
            or row.get("mode") != "collect-calibration"
            or row.get("timing") != TIMING
            for row in rows
        ):
            raise ValueError("COAP collection must match the checkpoint and observation/action protocol")
        minima = []
        for row in rows:
            episodes.append({"task": row["task"], "seed": row["seed"]})
            if row["success"] and row["traces"]:
                minima.append(min(event["similarity"] for event in row["traces"]))
        if len(minima) < cfg.monitor.min_successful_episodes:
            raise ValueError("Too few successful held-out COAP collection episodes")
        status = "calibrated on successful held-out policy rollouts"
        scores = minima
        source_info = {"path": str(path), "sha256": file_digest(path)}
    elif source == "demonstrations":
        cache = FrameCache(cfg.training.cache, cfg.training.raw_actions)
        splits = predictor.split_manifest
        dataset = WindowDataset(cache, splits["validation"], cfg.model.window_size, predictor.normalization)
        by_episode = {}
        for index in range(len(dataset)):
            row = dataset[index]
            # Predictor accepts raw commands; the dataset applies the train-fitted normalization.
            start = row["start"]
            score = predictor.score(
                row["observations"].numpy(), cache.actions[start : start + cfg.model.window_size]
            )
            by_episode[row["episode"]] = min(by_episode.get(row["episode"], 1.0), score)
        if len(by_episode) < 2:
            raise ValueError("Need at least two held-out demonstration episodes")
        scores = list(by_episode.values())
        status = "PROVISIONAL demonstration-validation threshold; policy rollout calibration recommended"
        source_info = {"cache": cfg.training.cache, "validation_episodes": sorted(by_episode)}
    else:
        raise ValueError("COAP calibration source must be rollouts or demonstrations")
    if not np.isfinite(scores).all() or min(scores) < -1 or max(scores) > 1:
        raise ValueError("Invalid calibration cosine scores")
    result = {
        "format": "coap_calibration_v1",
        "threshold": float(np.quantile(scores, cfg.monitor.lower_quantile)),
        "statistic": "lower quantile of per-episode minimum cosine; not a guaranteed false-alarm rate",
        "lower_quantile": cfg.monitor.lower_quantile,
        "episodes_used": len(scores),
        "episodes": episodes,
        "model_fingerprint": predictor.fingerprint,
        "protocol_fingerprint": identity,
        "timing": TIMING,
        "status": status,
        "source": source_info,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def summarize(path):
    """Only explicitly labeled windows/intervals support false alarms and detection delay."""
    rows = read_episodes(path)
    labelled = false_alarms = 0
    delays = []
    for row in rows:
        for event in row["traces"]:
            if type(event.get("abnormal")) is bool and not event["abnormal"]:
                labelled += 1
                false_alarms += bool(event["trigger"])
        # Optional ground-truth inclusive intervals, measured in completed-action steps.
        for interval in row.get("anomaly_intervals", []):
            start, end = interval["start_step"], interval["end_step"]
            hits = [
                event["completed_step"]
                for event in row["traces"]
                if event["trigger"] and start <= event["completed_step"] <= end
            ]
            delays.append(min(hits) - start if hits else None)
    successes = sum(row["success"] for row in rows)
    result = {
        "episodes": len(rows),
        "successes": successes,
        "success_rate": successes / len(rows),
        "interventions": sum(row["intervention_count"] for row in rows),
        "vla_calls": sum(row["vla_calls"] for row in rows),
        "vla_seconds": sum(row["vla_seconds"] for row in rows),
        "feature_seconds": sum(row["feature_seconds"] for row in rows),
        "scoring_seconds": sum(row["scoring_seconds"] for row in rows),
        "wall_seconds": sum(row["wall_seconds"] for row in rows),
        "labelled_normal_scored_windows": labelled,
        "false_alarm_rate_per_labelled_normal_window": false_alarms / labelled if labelled else None,
        "detection_delays_steps": delays or None,
        "missed_labelled_intervals": sum(value is None for value in delays),
        "label_note": "Task failure/success is not a per-step anomaly label",
    }
    requests = [request for row in rows for request in row.get("generation_requests", [])]
    if requests:
        temperatures = [request["temperature"] for request in requests]
        result["generation"] = {
            "control_mode": "generation_temperature",
            "chunks": len(requests),
            "warmup_chunks": sum(request["warmup"] for request in requests),
            "temperature_min": min(temperatures),
            "temperature_max": max(temperatures),
            "temperature_mean": sum(temperatures) / len(temperatures),
            "definition": "initial Gaussian action-noise standard-deviation multiplier",
        }
    return result
