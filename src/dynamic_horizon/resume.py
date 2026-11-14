"""Validate completed evaluations and preserve incomplete output on restart."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


def prepare_output(output: Path, request: dict, expected_tasks: int, save_videos: bool) -> bool:
    """Return True to skip a complete run; archive partial output otherwise.

    Older results have no request record; explicitly resuming them requires the
    caller to retain their original horizon/seed, and verifies episodes/videos.
    """
    if not output.exists() or not any(output.iterdir()):
        return False
    recorded = output / "evaluation_request.json"
    if recorded.is_file() and json.loads(recorded.read_text()) != request:
        raise ValueError(f"Resume settings differ from {recorded}; use a fresh output directory")
    summary_path, tasks_path = output / "metrics_summary.json", output / "per_task_metrics.csv"
    complete = False
    if summary_path.is_file() and tasks_path.is_file():
        summary = json.loads(summary_path.read_text())
        with tasks_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        episodes = int(request["episodes"])
        complete = (summary.get("tasks") == expected_tasks and summary.get("episodes") == expected_tasks * episodes
                    and len(rows) == expected_tasks and all(int(row["episodes"]) == episodes for row in rows)
                    and len({(row["task_group"], row["task_id"]) for row in rows}) == expected_tasks)
        if save_videos:
            complete = complete and len(list((output / "videos").rglob("*.mp4"))) == expected_tasks * episodes
    if complete:
        print(f"Resume: skipping completed evaluation {output}", flush=True)
        return True
    archived = output.with_name(output.name + ".incomplete_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.rename(archived)
    print(f"Resume: preserved incomplete results at {archived}; restarting this evaluation", flush=True)
    return False
