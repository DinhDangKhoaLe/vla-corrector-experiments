"""Write consistent task/episode metrics from each evaluator's native output.

Average horizon is total executed environment actions / generated policy chunks,
including chunks shortened by interventions or episode termination.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def read_episodes(output: Path) -> list[dict]:
    if (output / "eval_info.json").is_file():
        info = json.loads((output / "eval_info.json").read_text())
        rows = []
        for task in info["per_task"]:
            metrics = task["metrics"]
            for episode, (success, calls, steps) in enumerate(zip(
                metrics["successes"], metrics["inference_steps"], metrics["execution_steps"], strict=True
            )):
                rows.append(dict(task_group=task["task_group"], task_id=task["task_id"],
                                 episode=episode, success=bool(success), policy_calls=int(calls), steps=int(steps)))
        return rows
    candidates = [output / "episodes.csv", output / "analysis/summary/episode_summary.csv",
                  output / "summary/episode_summary.csv"]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        return [dict(task_group=row["task_group"], task_id=row["task_id"],
                     episode=int(row.get("episode", row.get("episode_idx", 0))),
                     success=_bool(row["success"]),
                     policy_calls=int(row["policy_calls"] if "policy_calls" in row else row["action_chunk_infer_requests"]),
                     steps=int(row["steps"] if "steps" in row else row["env_steps_executed"])) for row in rows]
    raise FileNotFoundError(f"No supported evaluation results in {output}")


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot report zero evaluation episodes")
    calls = sum(row["policy_calls"] for row in rows)
    steps = sum(row["steps"] for row in rows)
    return dict(episodes=len(rows), successes=sum(row["success"] for row in rows),
                success_rate_percent=100 * sum(row["success"] for row in rows) / len(rows),
                policy_calls_total=calls, policy_calls_per_episode=calls / len(rows),
                executed_actions_total=steps, average_horizon=steps / calls if calls else None)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output: Path) -> dict:
    rows = read_episodes(output)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task_group"], row["task_id"])].append(row)
    tasks = [dict(task_group=group, task_id=task_id, **aggregate(episodes))
             for (group, task_id), episodes in groups.items()]
    summary = aggregate(rows)
    summary.update(tasks=len(tasks), videos=len(list((output / "videos").rglob("*.mp4"))),
                   average_horizon_definition="total executed actions / total policy calls")
    _csv(output / "per_episode_metrics.csv", rows)
    _csv(output / "per_task_metrics.csv", tasks)
    (output / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Task metrics: {output / 'per_task_metrics.csv'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_report(args.output)


if __name__ == "__main__":
    main()
