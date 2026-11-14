import csv
import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dynamic_horizon.evaluate import command
from dynamic_horizon.report import write_report


@pytest.mark.parametrize("method", ["baseline", "LACO_replan", "LACO_adjust", "corrector", "pace"])
def test_video_option_saves_all_requested_episodes(method):
    args = Namespace(benchmark="libero", method=method, no_ogg=False, output=None,
                     policy=None, episodes=20, seed=1000, horizon=50, save_videos=True)
    cmd = command(args)
    if method.startswith("LACO_") or method == "corrector":
        assert "evaluation.save_video=true" in cmd
    elif method == "baseline":
        assert "--max_episodes_rendered=20" in cmd
    else:
        assert "--save_raw_video=true" in cmd
        assert "--save_summary_csv=true" in cmd


@pytest.mark.parametrize("format", ["baseline", "laco", "corrector"])
def test_reports_use_actual_executed_actions_and_separate_task_groups(tmp_path, format):
    # 51 actions / 3 chunks = 17; the mean of per-episode horizons would be 13.25.
    if format == "baseline":
        tasks = [{"task_group": group, "task_id": 0, "metrics": {
            "successes": [True, False], "inference_steps": [2, 1], "execution_steps": [50, 1]}}
            for group in ["spatial", "object"]]
        (tmp_path / "eval_info.json").write_text(json.dumps({"per_task": tasks}))
    else:
        rows = [{"task_group": group, "task_id": 0, "success": ok,
                 **({"episode": episode, "policy_calls": calls, "steps": steps} if format == "laco" else
                    {"episode_idx": episode, "action_chunk_infer_requests": calls, "env_steps_executed": steps})}
                for group in ["spatial", "object"]
                for episode, (ok, calls, steps) in enumerate([(True, 2, 50), (False, 1, 1)])]
        path = tmp_path / ("episodes.csv" if format == "laco" else "analysis/summary/episode_summary.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    summary = write_report(tmp_path)
    assert summary["tasks"] == 2
    assert summary["success_rate_percent"] == 50
    assert summary["policy_calls_total"] == 6
    assert summary["policy_calls_per_episode"] == 1.5
    assert summary["average_horizon"] == 17
    with (tmp_path / "per_task_metrics.csv").open() as f:
        tasks = list(csv.DictReader(f))
    assert len(tasks) == 2 and all(float(row["average_horizon"]) == 17 for row in tasks)


def test_laco_saves_one_full_video_per_episode(tmp_path, monkeypatch):
    import laco.evaluate as evaluate
    cfg = SimpleNamespace(device="cpu", output_features={"action": SimpleNamespace(shape=(4,))})
    policy = SimpleNamespace(reset=lambda: None, eval=lambda: policy,
                             predict_action_chunk=lambda batch: torch.zeros(1, 50, 4))

    class Env:
        metadata = {"render_fps": 10}
        unwrapped = property(lambda self: self)
        def reset(self, seed):
            self.steps = 0
            return {}, {}
        def call(self, name):
            return [3 if name == "_max_episode_steps" else np.zeros((16, 16, 3), dtype=np.uint8)]
        def step(self, action):
            self.steps += 1
            return {}, 0., np.array([self.steps == 3]), np.array([False]), {"is_success": self.steps == 2}
        def close(self):
            pass

    monkeypatch.setattr(evaluate.PI05Config, "from_pretrained", lambda path: cfg)
    monkeypatch.setattr(evaluate.PI05Policy, "from_pretrained", lambda *a, **kw: policy)
    monkeypatch.setattr(evaluate, "make_pre_post_processors", lambda **kw: (lambda x: x, lambda x: x))
    monkeypatch.setattr(evaluate, "make_env_pre_post_processors", lambda **kw: (lambda x: x, lambda x: x))
    monkeypatch.setattr(evaluate, "make_env", lambda *a, **kw: {"easy": {0: Env()}})
    monkeypatch.setattr(evaluate, "_input", lambda *a: {})
    videos = []
    monkeypatch.setattr(evaluate, "write_video", lambda path, frames, fps: videos.append((path, frames.shape, fps)))
    result = evaluate.run({"evaluation": {"mode": "baseline", "policy": "unused", "output": str(tmp_path),
                                          "tasks": "easy", "episodes": 2, "save_video": True, "device": "cpu"}})
    assert len(videos) == 2 and videos[0][0] != videos[1][0]
    assert all(shape == (4, 16, 16, 3) for _, shape, _ in videos)
    assert result["success_rate"] == 1  # A later false flag must not erase a successful step.
    summary = write_report(tmp_path)
    assert summary["average_horizon"] == 3


def test_pipeline_finishes_metaworld_before_libero_preparation(tmp_path):
    import os
    import subprocess
    fake = tmp_path / "python"
    trace = tmp_path / "commands.txt"
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$TEST_RUN_TRACE"\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "TEST_RUN_TRACE": str(trace),
           "DYNAMIC_HORIZON_IN_SCOPE": "1"}
    subprocess.run(["bash", "scripts/run_metaworld_then_libero.sh", str(tmp_path / "run")],
                   env=env, check=True, capture_output=True, text=True)
    commands = trace.read_text().splitlines()
    meta_eval = [i for i, c in enumerate(commands) if "dynamic_horizon.evaluate --benchmark metaworld" in c]
    libero_prepare = [i for i, c in enumerate(commands) if "libero" in c and "dynamic_horizon.evaluate" not in c]
    evaluations = [c for c in commands if "dynamic_horizon.evaluate" in c]
    assert len(meta_eval) == 5 and len(evaluations) == 10
    assert all("--no-ogg" not in c for c in evaluations)
    assert max(meta_eval) < min(libero_prepare)
    assert all("--episodes 20 --seed 1000 --save-videos" in c for c in evaluations)
