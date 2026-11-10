"""Execute the actual evaluator control flow with a fake policy/environment on CPU.

Load function/class ASTs rather than simulator/robot imports: the local test env
does not have every optional LeRobot robot dependency. No control-flow copy is used.
"""

import ast
import csv
import dataclasses
import datetime as dt
import json
import logging
import sys
import tempfile
import time
import unittest
from collections import deque
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
import yaml
from test_coap import make_cache, tiny_config

from siglip_dynamics.coap.train import train


@dataclasses.dataclass
class FakePolicyConfig:
    pretrained_path: str = "missing-test-policy"
    device: str = "cpu"
    n_action_steps: int = 10
    chunk_size: int = 50
    type: str = "pi05"


class Progress:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeEnv:
    def __init__(self, steps=7, reset_noise_draws=0):
        self.actions = []
        self.resets = 0
        self.steps = steps
        self.reset_noise_draws = reset_noise_draws

    def reset(self, seed):
        torch.rand(self.reset_noise_draws)
        self.step_index = 0
        self.resets += 1
        return {"step": 0}, {}

    def call(self, name):
        return [self.steps] if name == "_max_episode_steps" else ["fake task"]

    def step(self, action):
        self.actions.append(np.array(action, copy=True))
        self.step_index += 1
        return (
            {"step": self.step_index},
            np.array([0.0]),
            np.array([self.step_index == self.steps]),
            np.array([False]),
            {"is_success": np.array([self.step_index == self.steps])},
        )


class FakePolicy(torch.nn.Module):
    latest = None

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inference_observations = []
        self.feature_observations = []
        self.generation_temperatures = []
        self.noise_draws = []
        self.model = SimpleNamespace(paligemma_with_expert=SimpleNamespace(embed_image=lambda image: image))
        FakePolicy.latest = self

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, config):
        return cls(config)

    def reset(self):
        pass

    def encode_siglip_only(self, batch):
        if not getattr(self, "corrector_enabled", False):
            raise AssertionError("COAP must not use a dynamics predictor or cached Corrector feature")
        self.feature_observations.append(batch["step"])
        return np.full((4, 6), batch["step"], dtype=np.float32)

    def set_safety_predictor(self, *args, **kwargs):
        self.corrector_enabled = True

    def _preprocess_images(self, batch):
        self.feature_observations.append(batch["step"])
        return [torch.full((1, 4, 6), float(batch["step"]))], [torch.tensor([True])]

    def predict_action_chunk_with_aux(self, batch, **kwargs):
        if kwargs["guidance_replan"]:
            raise AssertionError("COAP must never request OGG")
        self.inference_observations.append(batch["step"])
        self.noise_draws.append(float(torch.rand(())))
        self.generation_temperatures.append(kwargs.get("generation_temperature"))
        actions = torch.full((1, 50, 2), float(batch["step"]))
        return {"actions": actions, "raw_actions": actions}


def load_evaluator_namespace(env):
    path = Path("src/lerobot/scripts/lerobot_eval_modified_detection.py")
    tree = ast.parse(path.read_text())
    selected = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
            selected.append(node)
        elif (
            isinstance(node, ast.ClassDef)
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "GUIDANCE_EVENT_COLUMNS" for t in node.targets)
        ):
            selected.append(node)
    module = ModuleType("_coap_evaluator_smoke")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        {
            "csv": csv,
            "json": json,
            "logging": logging,
            "Path": Path,
            "np": np,
            "torch": torch,
            "dt": dt,
            "time": time,
            "dataclasses": dataclasses,
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "asdict": dataclasses.asdict,
            "deque": deque,
            "pformat": str,
            "EvalConfig": lambda: SimpleNamespace(batch_size=1, n_episodes=1, use_async_envs=False),
            "PI05Config": FakePolicyConfig,
            "PI05ModifiedConfig": FakePolicyConfig,
            "PI05ModifiedPolicy": FakePolicy,
            "get_safe_torch_device": lambda *a, **k: torch.device("cpu"),
            "set_seed": torch.manual_seed,
            "inside_slurm": lambda: True,
            "make_env": lambda *a, **k: {"easy": {0: env}},
            "close_envs": lambda *a: None,
            "make_pre_post_processors": lambda **k: (lambda x: x, lambda x: x),
            "make_env_pre_post_processors": lambda **k: (lambda x: x, lambda x: x),
            "preprocess_observation": lambda x: x,
            "add_envs_task": lambda env, x: x,
            "tqdm": lambda *a, **k: Progress(),
            "trange": lambda *a, **k: Progress(),
            "ACTION": "action",
        }
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), str(path), "exec"),
        module.__dict__,
    )
    return module


class FakeDynamicsPredictor:
    def __init__(self, path, device):
        self.model = torch.nn.Identity()
        self.action_dim, self.k_step, self.h_window, self.token_dim = 2, 2, 1, 6
        self.model_type = "mlp"
        self.model_dir = Path(path)
        self.checkpoint_path = self.model_dir / "best_cosine_model.pt"

    def predict_delta_z(self, z, action):
        return np.zeros_like(z)


class EvaluatorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_rollout(
        self, root, *, detector, original=False, temperature=False, steps=7, random_settings=None,
        explicit_disable=False, reseed=False, episodes=1, reset_noise_draws=0, prefix=False, action_steps=10,
    ):
        env = FakeEnv(steps, reset_noise_draws)
        module = load_evaluator_namespace(env)
        seed_calls = []
        def set_seed(seed):
            seed_calls.append(seed)
            torch.manual_seed(seed)
        module.set_seed = set_seed
        cls = module.EvalModifiedDetectionConfig
        values = {
            f.name: f.default if f.default is not dataclasses.MISSING else f.default_factory()
            for f in dataclasses.fields(cls)
            if f.name != "env"
        }
        values.update(
            env=SimpleNamespace(type="metaworld", task="easy", episode_length=steps),
            policy=FakePolicyConfig(n_action_steps=action_steps),
            output_dir=root / ("coap" if detector else "policy"),
            seed=1000,
            policy_only=not detector and not original,
            disable_corrector=explicit_disable,
            reseed_policy_each_episode=reseed,
            eval=SimpleNamespace(batch_size=1, n_episodes=episodes, use_async_envs=False),
            save_raw_video=False,
            save_analysis=False,
            compute_baseline_metrics=False,
            save_guidance_dual_video=False,
            save_baseline_sync_video_with_curve=False,
            save_baseline_et_curve_plot=False,
            save_baseline_step_csv=False,
            save_task_plots=False,
            meltdown_use_guidance_replan=False,
            guidance_compare_baseline=False,
            save_summary_csv=True,
            save_summary_json=True,
        )
        if explicit_disable:
            values["policy_only"] = False
        if detector:
            make_cache(root / "cache")
            cfg = tiny_config(root)
            train(cfg)
            cfg.evaluation.policy = "missing-test-policy"
            cfg.evaluation.tasks = "easy"
            cfg.evaluation.episodes = 1
            cfg.evaluation.episode_length = steps
            cfg.evaluation.reseed_policy_each_episode = reseed
            cfg.monitor.threshold = 1.0
            cfg.monitor.allow_provisional_threshold = True
            cfg.monitor.persistence = 1
            cfg.monitor.cooldown_steps = 0
            if temperature:
                cfg.monitor.threshold = None
                cfg.generation.enabled = True
            config = root / "coap.yaml"
            config.write_text(yaml.safe_dump(cfg.to_dict()))
            values["coap_config"] = str(config)
        if original:
            module.SiglipDynamicsPredictor = FakeDynamicsPredictor
            values["safety_model_path"] = "mock-dynamics"
        if random_settings is not None:
            from siglip_dynamics.random_replan import Config, Evaluation

            root.mkdir(parents=True, exist_ok=True)
            cfg = Config(
                random=random_settings,
                evaluation=Evaluation(
                    policy="missing-test-policy",
                    device="cpu",
                    output=str(values["output_dir"]),
                    tasks="easy",
                    episodes=1,
                    episode_length=steps,
                    n_action_steps=action_steps,
                    reseed_policy_each_episode=reseed,
                ),
            )
            config = root / "random.yaml"
            config.write_text(yaml.safe_dump(dataclasses.asdict(cfg)))
            values["random_replan_config"] = str(config)
            values["policy_only"] = False
        # Bypass CLI checkpoint loading while retaining the actual dataclass instance.
        config = object.__new__(cls)
        for key, value in values.items():
            setattr(config, key, value)
        context = nullcontext()
        if prefix:
            from unittest.mock import patch

            from siglip_dynamics.experiments.config import Experiment
            from siglip_dynamics.experiments.runtime import RuntimeMonitor
            cfg = Experiment()
            cfg.features.action_prefix = True
            cfg.model.horizons = [2]
            cfg.evaluation.prefix_monitor_every_step = True
            predictor = FakeDynamicsPredictor("mock-prefix", "cpu")
            predictor.fingerprint = "mock-prefix-checkpoint"
            predictor.prefix_calls = []
            def predict(z, actions, horizon):
                predictor.prefix_calls.append(np.array(actions, copy=True))
                return np.zeros_like(z), None
            predictor.predict = predict
            monitor = RuntimeMonitor(cfg, predictor, "evaluate", values["output_dir"], {})
            config.policy_only = False
            config.safety_model_path = "mock-prefix"
            config.experiment_config = "mock-prefix.yaml"
            module.robust_threshold_step = lambda score, state: (0.1, True)
            context = patch("siglip_dynamics.experiments.runtime.load_runtime", return_value=(predictor, monitor))
        with context:
            module.eval_modified_detection(config)
        FakePolicy.latest.seed_calls = seed_calls
        if prefix:
            FakePolicy.latest.prefix_monitor = monitor
            FakePolicy.latest.prefix_calls = predictor.prefix_calls
        return env, FakePolicy.latest, values["output_dir"]

    def test_explicit_disable_skips_all_controllers_and_preserves_plain_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, policy, folder = self.run_rollout(Path(tmp), detector=False, explicit_disable=True, steps=21)
            self.assertEqual(policy.inference_observations, [0, 10, 20])
            self.assertFalse(policy.feature_observations)
            self.assertFalse(getattr(policy, "corrector_enabled", False))
            self.assertEqual([float(a[0, 0]) for a in env.actions], [0]*10+[10]*10+[20])
            for name in ("coap_episodes.jsonl", "random_replan_episodes.jsonl", "experiment_episodes.jsonl"):
                self.assertFalse((folder / name).exists())

    def test_reseeding_occurs_after_env_reset_and_restarts_each_episode_noise(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, policy, _ = self.run_rollout(
                Path(tmp), detector=False, explicit_disable=True, reseed=True,
                episodes=2, reset_noise_draws=17,
            )
            self.assertEqual(policy.seed_calls, [1000, 1000, 1001])
            expected = [float(torch.rand((), generator=torch.Generator().manual_seed(seed))) for seed in (1000,1001)]
            self.assertEqual(policy.noise_draws, expected)

    def test_rolling_prefix_interrupts_with_actual_action_history_across_replans(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, policy, _ = self.run_rollout(Path(tmp), detector=False, prefix=True)
            self.assertEqual(policy.inference_observations, [0, 2, 3, 4, 5, 6])
            self.assertEqual([float(a[0, 0]) for a in env.actions], [0, 0, 2, 3, 4, 5, 6])
            self.assertEqual([s["step"] for s in policy.prefix_monitor.scores], [2, 3, 4, 5, 6, 7])
            np.testing.assert_array_equal(policy.prefix_calls[1], [[0,0],[2,2]])

    def test_real_evaluator_interrupts_after_action_five_and_replans_from_latest_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, policy, folder = self.run_rollout(Path(tmp), detector=True)
            self.assertEqual(env.resets, 1)
            self.assertEqual(len(env.actions), 7)
            self.assertEqual(policy.feature_observations, list(range(7)))
            self.assertEqual(policy.inference_observations, [0, 5, 6])
            self.assertEqual([float(a[0, 0]) for a in env.actions], [0, 0, 0, 0, 0, 5, 6])
            rows = [json.loads(line) for line in (folder / "coap_episodes.jsonl").read_text().splitlines()]
            self.assertEqual([event["completed_step"] for event in rows[0]["traces"]], [5, 6, 7])
            self.assertEqual(rows[0]["intervention_count"], 2)
            self.assertEqual(rows[0]["vla_calls"], 3)
            self.assertTrue((folder / "coap_summary.json").is_file())

    def test_disabled_detector_preserves_normal_action_schedule_and_skips_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, policy, folder = self.run_rollout(Path(tmp), detector=False)
            self.assertEqual(env.resets, 1)
            self.assertEqual(len(env.actions), 7)
            self.assertEqual(policy.inference_observations, [0])
            self.assertFalse(policy.feature_observations)
            self.assertFalse((folder / "coap_episodes.jsonl").exists())
            self.assertEqual(policy.generation_temperatures, [None])
            self.assertEqual(policy.seed_calls, [1000])

    def test_temperature_changes_sampling_only_at_natural_chunk_boundaries(self):
        from unittest.mock import patch

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("siglip_dynamics.coap.runtime.Predictor.score", return_value=-0.5),
        ):
            env, policy, folder = self.run_rollout(Path(tmp), detector=True, temperature=True, steps=21)
            self.assertEqual(env.resets, 1)
            self.assertEqual(policy.inference_observations, [0, 10, 20])
            self.assertEqual(policy.generation_temperatures, [1.0, 2.0, 2.0])
            self.assertEqual(policy.feature_observations, list(range(21)))
            self.assertEqual([float(a[0, 0]) for a in env.actions], [0] * 10 + [10] * 10 + [20])
            row = json.loads((folder / "coap_episodes.jsonl").read_text())
            self.assertEqual(row["intervention_count"], 0)
            self.assertTrue(all(not event["trigger"] and not event["alarm"] for event in row["traces"]))
            self.assertTrue(all(event["threshold"] is None for event in row["traces"]))
            self.assertEqual([r["completed_step"] for r in row["generation_requests"]], [0, 10, 20])
            summary = json.loads((folder / "coap_summary.json").read_text())
            self.assertEqual(summary["generation"]["chunks"], 3)
            self.assertEqual(summary["generation"]["warmup_chunks"], 1)

    def test_existing_corrector_path_still_uses_dynamics_with_coap_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            env, policy, folder = self.run_rollout(Path(tmp), detector=False, original=True)
            self.assertTrue(policy.corrector_enabled)
            self.assertEqual(len(env.actions), 7)
            self.assertEqual(policy.inference_observations, [0])
            self.assertEqual(policy.feature_observations, list(range(7)))
            self.assertFalse((folder / "coap_episodes.jsonl").exists())

    def test_random_baseline_replans_without_any_learned_features_or_guidance(self):
        from siglip_dynamics.random_replan import RandomSettings

        with tempfile.TemporaryDirectory() as tmp:
            env, policy, folder = self.run_rollout(
                Path(tmp), detector=False, steps=21, random_settings=RandomSettings(probability=1)
            )
            self.assertEqual(env.resets, 1)
            self.assertEqual(policy.inference_observations, [0, 6, 16, 17])
            self.assertEqual([float(a[0, 0]) for a in env.actions], [0] * 6 + [6] * 10 + [16] + [17] * 4)
            self.assertFalse(policy.feature_observations)
            self.assertEqual(policy.generation_temperatures, [None] * 4)
            self.assertFalse((folder / "coap_episodes.jsonl").exists())
            row = json.loads((folder / "random_replan_episodes.jsonl").read_text())
            self.assertEqual(row["intervention_count"], 2)
            self.assertEqual([e["completed_step"] for e in row["traces"] if e["trigger"]], [6, 17])
            self.assertEqual(row["traces"][5]["discarded_actions"], 4)
            summary = json.loads((folder / "random_replan_summary.json").read_text())
            self.assertEqual(summary["interventions"], 2)
            self.assertEqual(summary["vla_calls"], 4)

    def test_zero_probability_matches_plain_policy_execution(self):
        from siglip_dynamics.random_replan import RandomSettings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env, policy, folder = self.run_rollout(
                root / "random", detector=False, steps=21, random_settings=RandomSettings(probability=0)
            )
            plain_env, plain_policy, _ = self.run_rollout(root / "plain", detector=False, steps=21)
            self.assertEqual(policy.inference_observations, plain_policy.inference_observations)
            for action, plain_action in zip(env.actions, plain_env.actions, strict=True):
                np.testing.assert_array_equal(action, plain_action)
            self.assertFalse(policy.feature_observations)
            row = json.loads((folder / "random_replan_episodes.jsonl").read_text())
            self.assertEqual(row["intervention_count"], 0)

    def test_uniform_horizons_execute_selected_prefix_and_requery_latest_observation(self):
        from siglip_dynamics.random_replan import RandomSettings

        for horizon, expected_queries in [(20, [0, 20, 40, 60]), (50, [0, 50])]:
            with self.subTest(horizon=horizon), tempfile.TemporaryDirectory() as tmp:
                env, policy, folder = self.run_rollout(
                    Path(tmp), detector=False, steps=61, action_steps=50,
                    random_settings=RandomSettings(strategy="uniform_horizon", min_horizon=horizon,
                        max_horizon=horizon, warmup_steps=0, cooldown_steps=0),
                )
                self.assertEqual(policy.inference_observations, expected_queries)
                self.assertEqual(env.resets, 1)
                self.assertFalse(policy.feature_observations)
                self.assertEqual(policy.generation_temperatures, [None] * len(expected_queries))
                row = json.loads((folder / "random_replan_episodes.jsonl").read_text())
                self.assertEqual(row["intervention_count"], 3 if horizon == 20 else 0)
                for start in expected_queries:
                    self.assertTrue(all(float(a[0, 0]) == start for a in env.actions[start:start+horizon]))
                self.assertTrue(all(e["planned_horizon"] == horizon for e in row["traces"]))

    def test_cli_mode_validation_keeps_default_corrector_and_excludes_mixed_detectors(self):
        module = load_evaluator_namespace(FakeEnv())
        module.parser = SimpleNamespace(
            get_path_arg=lambda name: "mock-policy", get_cli_overrides=lambda name: {}
        )
        module.PreTrainedConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakePolicyConfig())
        cls = module.EvalModifiedDetectionConfig
        env = SimpleNamespace(type="metaworld")
        original = cls(env=env, safety_model_path="mock-dynamics")
        self.assertFalse(original.coap_config)
        self.assertFalse(original.policy_only)
        self.assertTrue(original.meltdown_use_guidance_replan)
        detector = cls(env=env, coap_config="coap.yaml")
        self.assertFalse(detector.meltdown_use_guidance_replan)
        self.assertEqual(detector.guidance_eta, 0)
        with self.assertRaisesRegex(ValueError, "separately"):
            cls(env=env, coap_config="coap.yaml", safety_model_path="mock-dynamics")
        random_cfg = cls(env=env, random_replan_config="random.yaml")
        self.assertFalse(random_cfg.meltdown_use_guidance_replan)
        self.assertFalse(random_cfg.guidance_compare_baseline)
        self.assertEqual(random_cfg.guidance_eta, 0)
        for other in (
            {"coap_config": "coap.yaml"},
            {"safety_model_path": "mock-dynamics"},
            {"experiment_config": "experiment.yaml"},
            {"pace_config": "pace.yaml"},
            {"policy_only": True},
            {"disable_corrector": True},
        ):
            with self.assertRaisesRegex(ValueError, "separately"):
                cls(env=env, random_replan_config="random.yaml", **other)
        plain = cls(env=env, disable_corrector=True)
        self.assertFalse(plain.meltdown_use_guidance_replan)
        self.assertFalse(plain.guidance_compare_baseline)
        self.assertEqual(plain.guidance_eta, 0)
        for other in ({"safety_model_path":"mock"}, {"experiment_config":"experiment.yaml"},
                      {"pace_config":"pace.yaml"}, {"coap_config":"coap.yaml"}):
            with self.assertRaises(ValueError):
                cls(env=env, disable_corrector=True, **other)


if __name__ == "__main__":
    unittest.main()
