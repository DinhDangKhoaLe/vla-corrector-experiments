"""Temperature control, actual PI0.5 sampling hook, and legacy protocol preservation."""

import ast
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from siglip_dynamics.coap.config import COAPConfig, Generation, from_dict
from siglip_dynamics.coap.runtime import protocol
from siglip_dynamics.coap.temperature import scale_initial_noise, temperature_from_score
from siglip_dynamics.experiments.artifacts import digest


class TemperatureTests(unittest.TestCase):
    def test_mapping_bounds_monotonicity_and_warmup(self):
        settings = Generation(enabled=True)
        scores = [-1, -0.1, 0, 0.125, 0.25, 0.375, 0.5, 1]
        actual = [temperature_from_score(score, settings) for score in scores]
        self.assertEqual(actual, [2, 2, 2, 1.75, 1.5, 1.25, 1, 1])
        self.assertEqual(temperature_from_score(None, settings), 1)
        for value in [math.nan, math.inf, -1.1, 1.1]:
            with self.assertRaises(ValueError):
                temperature_from_score(value, settings)

    def test_noise_scale_preserves_rng_and_default_behavior(self):
        torch.manual_seed(17)
        noise = torch.randn(3, 50, 4)
        original = noise.clone()
        state = torch.random.get_rng_state().clone()
        self.assertIs(scale_initial_noise(noise, 1.0), noise)
        torch.testing.assert_close(scale_initial_noise(noise, 2.0), noise * 2)
        torch.testing.assert_close(noise, original)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        for value in [math.nan, math.inf, 0, -1, True]:
            with self.assertRaises(ValueError):
                scale_initial_noise(noise, value)

    def test_generation_config_validation_and_legacy_fingerprint(self):
        cfg = COAPConfig()
        with tempfile.TemporaryDirectory() as tmp:
            cfg.evaluation.policy = tmp
            evaluation = asdict(cfg.evaluation)
            evaluation.pop("reseed_policy_each_episode")
            for key in ("output", "checkpoint", "episodes", "seed", "collection_seed", "save_video"):
                evaluation.pop(key)
            old = digest(
                {
                    "evaluation": evaluation,
                    "model": asdict(cfg.model),
                    "timing": "observation_before_dispatched_action;score_after_action_returns",
                    "persistence": cfg.monitor.persistence,
                    "cooldown_steps": cfg.monitor.cooldown_steps,
                }
            )
            self.assertEqual(protocol(cfg), old)
            cfg.evaluation.reseed_policy_each_episode = True
            self.assertNotEqual(protocol(cfg), old)
            cfg.evaluation.reseed_policy_each_episode = False
            cfg.generation.enabled = True
            self.assertNotEqual(protocol(cfg), old)
        for fields in (
            {"max_temperature": 0.5},
            {"score_high": 0},
            {"score_low": -2},
            {"warmup_temperature": math.nan},
            {"enabled": "true"},
        ):
            with self.assertRaises(ValueError):
                from_dict({"generation": fields})

    def test_actual_policy_hook_scales_initial_noise_and_never_final_actions(self):
        path = Path("src/lerobot/policies/pi05_modified/modeling_pi05_modified.py")
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PI05ModifiedPolicy")
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "predict_action_chunk_with_aux"
        )
        method.decorator_list = []
        namespace = {
            "Tensor": torch.Tensor,
            "Any": object,
            "torch": torch,
            "OBS_LANGUAGE_TOKENS": "tokens",
            "OBS_LANGUAGE_ATTENTION_MASK": "masks",
            "ACTION": "action",
        }
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[])),
                str(path),
                "exec",
            ),
            namespace,
        )
        noises = []

        class Model:
            def sample_noise(self, shape, device):
                return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)

            def sample_actions(self, *args, **kwargs):
                noises.append(kwargs["noise"].clone())
                return torch.full((1, 50, 4), 7.0)

        policy = SimpleNamespace(
            eval=lambda: None,
            model=Model(),
            config=SimpleNamespace(
                chunk_size=50,
                max_action_dim=4,
                n_action_steps=10,
                output_features={"action": SimpleNamespace(shape=(4,))},
            ),
            _preprocess_images=lambda batch: ([], []),
        )
        batch = {
            "tokens": torch.zeros((1, 2), dtype=torch.long),
            "masks": torch.ones((1, 2), dtype=torch.bool),
        }
        kwargs = {
            "guidance_replan": False,
            "guidance_current_z_t": None,
            "guidance_current_z_hist_t": None,
            "guidance_current_delta_z_t": None,
            "guidance_current_delta_z_right": None,
            "guidance_apply_every": 3,
            "guidance_loss_objective": "attract_delta_z_correction",
            "guidance_compare_baseline": False,
        }
        default = namespace[method.name](policy, batch, **kwargs)
        hotter = namespace[method.name](policy, batch, generation_temperature=2.0, **kwargs)
        torch.testing.assert_close(noises[1], noises[0] * 2)
        torch.testing.assert_close(default["actions"], hotter["actions"])
        self.assertEqual(len(noises), 2)


if __name__ == "__main__":
    unittest.main()
