"""CPU regressions for supported native baseline, PACE and original Corrector."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from siglip_dynamics.experiments import robotwin_eval as rt
from siglip_dynamics.experiments import run
from siglip_dynamics.experiments.config import Experiment, load_config


class FakeEnv:
    step_lim = 5
    plan_success = True

    def __init__(self):
        self.setup_demo()

    def setup_demo(self, **kwargs):
        self.take_action_cnt = 0
        self.eval_success = False

    def get_obs(self):
        return {"observation": {name: {"rgb": np.full((2, 2, 3), self.take_action_cnt, np.uint8)}
                                for name in ("head_camera", "left_camera", "right_camera")},
                "joint_action": {"vector": np.zeros(14, np.float32)}}

    def take_action(self, action, action_type):
        assert action.shape == (14,) and action_type == "qpos"
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == self.step_lim

    def set_instruction(self, **kwargs):
        pass

    def close_env(self):
        pass


class FakePolicy:
    def __init__(self):
        self.calls, self.latent_calls, self.guided = 0, 0, 0

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        self.calls += 1
        return torch.zeros(1, 2, 14)

    def predict_action_chunk_with_aux(self, batch, **kwargs):
        self.guided += int(kwargs["guidance_replan"])
        return {"actions": self.predict_action_chunk(batch)}

    def encode_siglip_only(self, batch):
        self.latent_calls += 1
        return np.zeros((3, 4), np.float32)

    def set_safety_predictor(self, *args, **kwargs):
        pass


class FakePredictor:
    k_step, action_dim, model_type = 2, 14, "mlp"

    def __init__(self):
        self.model = SimpleNamespace(guidance_horizon=2)

    def predict_delta_z(self, z, action):
        return np.zeros_like(z)


def rollout(enabled, pace=None):
    cfg = Experiment()
    cfg.evaluation.robotwin_corrector_enabled = enabled
    cfg.evaluation.n_action_steps = 2
    policy = FakePolicy()
    with patch("lerobot.scripts.lerobot_eval_modified_detection.robust_threshold_step", return_value=(.1, True)):
        row = rt._run_one(env=FakeEnv(), policy=policy, preprocessor=lambda b: b,
                          postprocessor=lambda a: a, monitor=None, cfg=cfg, instruction="test",
                          task="task", seed=1000, video_path=None,
                          predictor=FakePredictor() if enabled else None, pace=pace)
    return row, policy


def test_plain_baseline_never_encodes_corrector_latents():
    row, policy = rollout(False)
    assert row["success"] and row["error"] == ""
    assert policy.latent_calls == policy.guided == row["replans"] == 0
    assert row["policy_calls"] == 3


def test_corrector_uses_cached_latents_and_guided_replan():
    row, policy = rollout(True)
    assert row["success"] and row["error"] == ""
    assert policy.latent_calls == 6
    assert row["ogg_replans"] > 0 and policy.guided > 0


def test_pace_executes_selected_prefix_without_corrector():
    pace = SimpleNamespace(select=lambda task, actions, **kwargs: (1, {}))
    row, policy = rollout(False, pace=pace)
    assert row["success"]
    assert row["policy_calls"] == 5
    assert policy.latent_calls == policy.guided == 0


def test_robotwin_dispatch_works_before_cache_exists():
    cfg = load_config("configs/corrector/robotwin_original_pool64.yaml")
    assert run._is_robotwin_cache(cfg)
    path = "configs/corrector/robotwin_original_pool64.yaml"
    with patch.object(run, "resolve", return_value=cfg), patch.object(rt, "evaluate_robotwin") as evaluate, \
         patch("sys.argv", ["run", "--config", path, "--mode", "evaluate"]):
        run.main()
    evaluate.assert_called_once_with(cfg, mode="evaluate", config_path=Path(path))


@pytest.mark.parametrize("enabled", [False, True])
def test_native_runner_writes_results_and_refuses_overwrite(tmp_path, enabled):
    cfg = Experiment()
    cfg.training.backend = "legacy"
    cfg.evaluation.robotwin_root = str(tmp_path)
    cfg.evaluation.robotwin_tasks = "task"
    cfg.evaluation.robotwin_corrector_enabled = enabled
    cfg.evaluation.robotwin_expert_check = False
    cfg.evaluation.episodes = 2
    cfg.evaluation.n_action_steps = 2
    cfg.evaluation.save_video = False
    cfg.evaluation.output = str(tmp_path / "result")
    official = SimpleNamespace(load_task_args=lambda args: (dict(args), "test"),
                               class_decorator=lambda task: FakeEnv(), build_instruction=lambda *args: "test")
    with patch.object(rt, "_require_robotwin"), patch.object(rt, "_load_task_api", return_value=(official, RuntimeError)), \
         patch.object(rt, "_policy_stack", return_value=(FakePolicy(), lambda b: b, lambda a: a)), \
         patch.object(rt, "ExperimentPredictor", return_value=FakePredictor()):
        rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
        summary = json.loads((tmp_path / "result/summary.json").read_text())
        assert summary["episodes"] == 2 and summary["corrector_enabled"] == enabled
        with pytest.raises(FileExistsError):
            rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
