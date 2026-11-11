"""Native-runner regression tests with a small simulator/policy double."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from siglip_dynamics.experiments.config import Experiment, load_config
from siglip_dynamics.experiments import run, robotwin_eval as rt


class FakeEnv:
    def __init__(self):
        self.reset()
        self.step_lim = 5
        self.plan_success = True

    def reset(self):
        self.take_action_cnt = 0
        self.eval_success = False

    def setup_demo(self, **kwargs):
        self.reset()

    def play_once(self):
        return {"info": {}}

    def check_success(self):
        return True

    def set_instruction(self, **kwargs):
        pass

    def get_obs(self):
        return {"observation": {name: {"rgb": np.full((2, 2, 3), self.take_action_cnt, np.uint8)}
                                for name in ("head_camera", "left_camera", "right_camera")},
                "joint_action": {"vector": np.zeros(14, np.float32)}}

    def take_action(self, action, action_type):
        assert action.shape == (14,) and action_type == "qpos"
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt >= self.step_lim

    def close_env(self):
        pass


class ExpertAssertionOnceEnv(FakeEnv):
    """A RoboTwin expert-filter candidate with an invalid sampled target pose."""

    def __init__(self):
        super().__init__()
        self.expert_attempts = 0

    def play_once(self):
        self.expert_attempts += 1
        if self.expert_attempts == 1:
            raise AssertionError("target_pose cannot be None for move action.")
        return {"info": {}}


class FakePolicy:
    def __init__(self):
        self.guidance = []
        self.latent_calls = 0
        self.plain_calls = 0

    def reset(self):
        pass

    def predict_action_chunk(self, batch):
        self.plain_calls += 1
        return torch.zeros(1, 2, 14)

    def predict_action_chunk_with_aux(self, batch, **kwargs):
        self.guidance.append(kwargs)
        return {"actions": self.predict_action_chunk(batch)}

    def encode_siglip_only(self, batch):
        self.latent_calls += 1
        return np.full((3, 4), float(batch["observation.images.cam_high"].mean()), np.float32)

    def set_safety_predictor(self, *args, **kwargs):
        pass


class FakePredictor:
    k_step = 2
    action_dim = 14
    model_type = "mlp"

    def __init__(self):
        self.calls = 0
        self.model = SimpleNamespace(guidance_horizon=2)

    def predict_delta_z(self, z, action):
        self.calls += 1
        return np.zeros_like(z)


class PrefixMonitor:
    """Minimal ordered-prefix monitor; intentionally has no uncertainty API."""

    def reset(self, task, seed):
        self.scheduled = []

    def schedule(self, z, actions, step):
        self.scheduled.append((step, len(actions)))

    def observe(self, z, step):
        if step % 2:
            return None
        return {"score": 0.9, "actual": np.zeros_like(z), "prediction": np.zeros_like(z), "horizon": 2}

    def cancel(self):
        pass

    def finish(self, success, error):
        pass


class RobotwinRunnerTests(unittest.TestCase):
    def test_policy_stack_selects_plain_class_and_matching_settings(self):
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        cfg = Experiment()
        cfg.training.device = "cpu"
        base = PI05Config(compile_model=True)
        loaded = SimpleNamespace()
        loaded.eval = lambda: loaded
        with tempfile.TemporaryDirectory() as tmp:
            cfg.evaluation.policy = tmp
            with patch("lerobot.configs.policies.PreTrainedConfig.from_pretrained",return_value=base), \
                 patch("lerobot.policies.pi05.modeling_pi05.PI05Policy.from_pretrained",return_value=loaded) as plain, \
                 patch("lerobot.policies.pi05_modified.modeling_pi05_modified.PI05ModifiedPolicy.from_pretrained",return_value=loaded) as guided, \
                 patch("lerobot.policies.factory.make_pre_post_processors",return_value=(None,None)):
                cfg.evaluation.robotwin_corrector_enabled = False
                rt._policy_stack(cfg)
                plain.assert_called_once()
                guided.assert_not_called()
                self.assertFalse(plain.call_args.kwargs["config"].compile_model)
                self.assertEqual(plain.call_args.kwargs["config"].n_action_steps,10)
                cfg.evaluation.robotwin_corrector_enabled = True
                rt._policy_stack(cfg)
                guided.assert_called_once()
                self.assertFalse(guided.call_args.kwargs["config"].compile_model)

    def test_plain_policy_never_encodes_or_uses_corrector(self):
        cfg = Experiment()
        cfg.evaluation.robotwin_corrector_enabled = False
        cfg.evaluation.n_action_steps = 2
        policy = FakePolicy()
        row = rt._run_one(env=FakeEnv(), policy=policy, preprocessor=lambda b:b,
                          postprocessor=lambda a:a, monitor=None, cfg=cfg, instruction="test",
                          task="task", seed=1000, video_path=None)
        self.assertTrue(row["success"])
        self.assertEqual(row["error"], "")
        self.assertEqual(row["steps"], 5)
        self.assertEqual(row["replans"], 0)
        self.assertEqual(policy.latent_calls, 0)
        self.assertEqual(policy.guidance, [])

    def test_original_receives_predictor_and_forces_guided_replan(self):
        cfg = Experiment()
        cfg.evaluation.n_action_steps = 2
        policy, predictor = FakePolicy(), FakePredictor()
        with patch("lerobot.scripts.lerobot_eval_modified_detection.robust_threshold_step", return_value=(0.1,True)):
            row = rt._run_one(env=FakeEnv(), policy=policy, preprocessor=lambda b:b,
                              postprocessor=lambda a:a, monitor=None, cfg=cfg, instruction="test",
                              task="task", seed=1000, video_path=None, predictor=predictor)
        self.assertTrue(row["success"], row["error"])
        self.assertEqual(predictor.calls, 5)
        # Initial state plus one new state per action: the post-action latent is
        # cached instead of being encoded again at the next loop iteration.
        self.assertEqual(policy.latent_calls, 6)
        self.assertGreater(row["replans"], 0)
        self.assertGreater(row["ogg_replans"], 0)
        payloads = [g for g in policy.guidance if g["guidance_replan"]]
        self.assertTrue(payloads)
        self.assertEqual(payloads[0]["guidance_current_z_t"].shape, (3,4))

    def test_prefix_uses_legacy_alarm_without_uncertainty_calibration(self):
        cfg = Experiment()
        cfg.features.action_prefix = True
        cfg.evaluation.prefix_monitor_every_step = True
        cfg.evaluation.n_action_steps = 2
        policy, predictor, monitor = FakePolicy(), FakePredictor(), PrefixMonitor()
        with patch("lerobot.scripts.lerobot_eval_modified_detection.robust_threshold_step", return_value=(0.1, True)):
            row = rt._run_one(env=FakeEnv(), policy=policy, preprocessor=lambda b: b,
                              postprocessor=lambda a: a, monitor=monitor, cfg=cfg, instruction="test",
                              task="task", seed=1000, video_path=None, predictor=predictor)
        self.assertTrue(row["success"], row["error"])
        self.assertGreater(row["replans"], 0)
        self.assertTrue(row["legacy_events"])
        self.assertTrue(any(g["guidance_replan"] for g in policy.guidance))
        self.assertEqual([step for step, _ in monitor.scheduled], list(range(5)))

    def test_prefix_selects_ordered_queue_monitor(self):
        cfg = Experiment()
        cfg.features.action_prefix = True
        predictor = SimpleNamespace(fingerprint="test")
        expected = object()
        with patch("siglip_dynamics.experiments.runtime.RuntimeMonitor", return_value=expected) as monitor:
            result = rt._make_monitor(cfg, predictor, "evaluate", Path("output"))
        self.assertIs(result, expected)
        monitor.assert_called_once()

    def test_prefix_monitor_receives_resume_flag(self):
        cfg = Experiment()
        cfg.features.action_prefix = True
        predictor = SimpleNamespace(fingerprint="test")
        with patch("siglip_dynamics.experiments.runtime.RuntimeMonitor", return_value=object()) as monitor:
            rt._make_monitor(cfg, predictor, "evaluate", Path("output"), resume=True)
        self.assertTrue(monitor.call_args.kwargs["resume"])

    def test_dispatch_passes_requested_mode(self):
        cfg = load_config("configs/corrector/robotwin_original.yaml")
        with patch.object(run,"resolve",return_value=cfg), patch.object(run,"_is_robotwin_cache",return_value=True), \
             patch.object(rt,"evaluate_robotwin") as evaluate, \
             patch("sys.argv",["run","--config","configs/corrector/robotwin_original.yaml","--mode","evaluate"]):
            run.main()
        evaluate.assert_called_once_with(cfg,mode="evaluate",config_path=Path("configs/corrector/robotwin_original.yaml"))

    def test_full_runner_plain_and_original_write_matched_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = SimpleNamespace(
                load_task_args=lambda args:(dict(args), "test"), class_decorator=lambda task:FakeEnv(),
                build_instruction=lambda *args:"test")
            rows = []
            for enabled in (False, True):
                cfg = Experiment()
                cfg.training.backend = "legacy"
                cfg.training.output = str(root / "trained")
                cfg.evaluation.robotwin_root = str(root)
                cfg.evaluation.robotwin_tasks = "task_a,task_b"
                cfg.evaluation.robotwin_corrector_enabled = enabled
                cfg.evaluation.episodes = 2
                cfg.evaluation.n_action_steps = 2
                cfg.evaluation.save_video = False
                cfg.evaluation.output = str(root / str(enabled))
                cfg.evaluation.checkpoint = ""
                policy = FakePolicy()
                with patch.object(rt,"_require_robotwin"), \
                     patch.object(rt,"_load_task_api",return_value=(official,RuntimeError)), \
                     patch.object(rt,"_policy_stack",return_value=(policy,lambda b:b,lambda a:a)), \
                     patch.object(rt,"ExperimentPredictor",return_value=FakePredictor()) as loader:
                    rt.evaluate_robotwin(cfg,mode="evaluate",config_path=Path("test.yaml"))
                    if enabled:
                        self.assertEqual(loader.call_args.args[0], root / "trained/h1_k10")
                    else:
                        loader.assert_not_called()
                summary = json.loads((Path(cfg.evaluation.output)/"summary.json").read_text())
                self.assertEqual(summary["episodes"],4)
                self.assertEqual(summary["corrector_enabled"], enabled)
                self.assertEqual(summary["per_task"]["task_a"]["episodes"],2)
                with (Path(cfg.evaluation.output)/"episode_summary.csv").open() as f:
                    rows.append([(r["task"],r["seed"]) for r in csv.DictReader(f)])
            self.assertEqual(rows[0],rows[1])

    def test_invalid_expert_candidate_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Experiment()
            cfg.training.backend = "legacy"
            cfg.evaluation.robotwin_root = str(root)
            cfg.evaluation.robotwin_tasks = "task_a"
            cfg.evaluation.robotwin_corrector_enabled = False
            cfg.evaluation.episodes = 1
            cfg.evaluation.n_action_steps = 2
            cfg.evaluation.save_video = False
            cfg.evaluation.output = str(root / "result")
            cfg.evaluation.robotwin_expert_check = True
            official = SimpleNamespace(
                load_task_args=lambda args: (dict(args), "test"),
                class_decorator=lambda task: ExpertAssertionOnceEnv(),
                build_instruction=lambda *args: "test",
            )
            with patch.object(rt, "_require_robotwin"), \
                 patch.object(rt, "_load_task_api", return_value=(official, RuntimeError)), \
                 patch.object(rt, "_policy_stack", return_value=(FakePolicy(), lambda b: b, lambda a: a)):
                rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
            with (root / "result" / "episode_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], "100100001")
            self.assertEqual(rows[0]["error"], "")

    def test_shared_scene_manifest_reuses_seed_and_instruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "shared_scenes.json"
            official_first = SimpleNamespace(
                load_task_args=lambda args: (dict(args), "test"),
                class_decorator=lambda task: ExpertAssertionOnceEnv(),
                build_instruction=lambda *args: "validated instruction",
            )
            official_replay = SimpleNamespace(
                load_task_args=lambda args: (dict(args), "test"),
                class_decorator=lambda task: FakeEnv(),
                build_instruction=lambda *args: "must not replace manifest instruction",
            )
            recorded = []
            for index, official in enumerate((official_first, official_replay)):
                cfg = Experiment()
                cfg.training.backend = "legacy"
                cfg.evaluation.robotwin_root = str(root)
                cfg.evaluation.robotwin_tasks = "task_a"
                cfg.evaluation.robotwin_corrector_enabled = False
                cfg.evaluation.episodes = 1
                cfg.evaluation.n_action_steps = 2
                cfg.evaluation.save_video = False
                cfg.evaluation.output = str(root / f"result_{index}")
                cfg.evaluation.robotwin_expert_check = True
                cfg.evaluation.robotwin_scene_manifest = str(manifest)
                with patch.object(rt, "_require_robotwin"), \
                     patch.object(rt, "_load_task_api", return_value=(official, RuntimeError)), \
                     patch.object(rt, "_policy_stack", return_value=(FakePolicy(), lambda b: b, lambda a: a)):
                    rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
                with (root / f"result_{index}" / "episode_summary.csv").open(encoding="utf-8") as handle:
                    recorded.append(list(csv.DictReader(handle))[0]["seed"])
            self.assertEqual(recorded, ["100100001", "100100001"])
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["scenes"]["task_a"], [
                {"seed": 100100001, "instruction": "validated instruction"}
            ])

    def test_existing_episode_records_resume_without_rerunning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Experiment()
            cfg.training.backend = "legacy"
            cfg.evaluation.robotwin_root = str(root)
            cfg.evaluation.robotwin_tasks = "task_a"
            cfg.evaluation.robotwin_corrector_enabled = False
            cfg.evaluation.episodes = 1
            cfg.evaluation.n_action_steps = 2
            cfg.evaluation.save_video = False
            cfg.evaluation.output = str(root / "result")
            cfg.evaluation.robotwin_expert_check = False
            official = SimpleNamespace(
                load_task_args=lambda args: (dict(args), "test"),
                class_decorator=lambda task: FakeEnv(),
                build_instruction=lambda *args: "test",
            )
            policy = FakePolicy()
            with patch.object(rt, "_require_robotwin"), \
                 patch.object(rt, "_load_task_api", return_value=(official, RuntimeError)), \
                 patch.object(rt, "_policy_stack", return_value=(policy, lambda b: b, lambda a: a)):
                rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
                first_calls = policy.plain_calls
                rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
            self.assertEqual(policy.plain_calls, first_calls)
            with (root / "result" / "episode_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)

    def test_fast_step_limit_and_video_cap_are_applied_per_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Experiment()
            cfg.training.backend = "legacy"
            cfg.evaluation.robotwin_root = str(root)
            cfg.evaluation.robotwin_tasks = "task_a"
            cfg.evaluation.robotwin_corrector_enabled = False
            cfg.evaluation.robotwin_expert_check = False
            cfg.evaluation.robotwin_step_limits = {"task_a": 3}
            cfg.evaluation.episodes = 3
            cfg.evaluation.save_video = True
            cfg.evaluation.max_videos_per_task = 1
            cfg.evaluation.output = str(root / "result")
            official = SimpleNamespace(
                load_task_args=lambda args: (dict(args), "test"),
                class_decorator=lambda task: FakeEnv(),
                build_instruction=lambda *args: "test",
            )
            calls = []

            def fake_rollout(**kwargs):
                calls.append((kwargs["env"].step_lim, kwargs["video_path"]))
                return {"task": "task_a", "seed": kwargs["seed"], "success": False,
                        "steps": 3, "replans": 0, "ogg_replans": 0, "error": ""}

            with patch.object(rt, "_require_robotwin"), \
                 patch.object(rt, "_load_task_api", return_value=(official, RuntimeError)), \
                 patch.object(rt, "_policy_stack", return_value=(FakePolicy(), lambda b: b, lambda a: a)), \
                 patch.object(rt, "_run_one", side_effect=fake_rollout):
                rt.evaluate_robotwin(cfg, mode="evaluate", config_path=Path("test.yaml"))
            self.assertEqual([limit for limit, _ in calls], [3, 3, 3])
            self.assertIsNotNone(calls[0][1])
            self.assertIsNone(calls[1][1])
            self.assertIsNone(calls[2][1])
            summary = json.loads((root / "result" / "summary.json").read_text())
            self.assertEqual(summary["robotwin_step_limits"], {"task_a": 3})
            self.assertEqual(summary["max_videos_per_task"], 1)

    def test_baseline_config_has_exact_hard15_suite(self):
        cfg = load_config("configs/corrector/robotwin_pi05_baseline.yaml")
        original = load_config("configs/corrector/robotwin_original.yaml")
        self.assertFalse(cfg.evaluation.robotwin_corrector_enabled)
        self.assertTrue(original.evaluation.robotwin_corrector_enabled)
        self.assertEqual(len(rt._tasks(cfg)),15)
        self.assertEqual(rt._tasks(cfg),rt._tasks(original))
        self.assertEqual(cfg.evaluation.policy,original.evaluation.policy)
        self.assertEqual(cfg.evaluation.episodes,20)


if __name__ == "__main__":
    unittest.main()
