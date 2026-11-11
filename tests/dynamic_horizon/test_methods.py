"""Algorithm contracts independent of GPU/model downloads.

Run: python -m unittest discover -s tests/dynamic_horizon -v
"""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from dynamic_horizon.a3 import A3, select_draft
from dynamic_horizon.aac import entropy_profile, select_horizon as aac_horizon
from dynamic_horizon.autohorizon import AutoHorizon, select_horizon as attention_horizon
from dynamic_horizon.bcp import ContinuationHead, clipped_objective, continuation_log_probs, efficiency_reward
from dynamic_horizon.pace import PACE, select_horizon as pace_horizon, speed_profile
from dynamic_horizon.train_bcp import optimize_group
from dynamic_horizon.types import Prediction


class MethodTests(unittest.TestCase):
    def test_autohorizon_reads_third_denoising_step(self):
        class Sampler:
            def sample(inner, *, attention_step):
                self.assertEqual(attention_step, 2)
                return Prediction(torch.zeros(50, 32), torch.zeros(50, 4), attention=torch.eye(50))

        self.assertEqual(AutoHorizon({}).decide(Sampler(), 0).horizon, 50)

    def test_attention_anchors_and_full_horizon(self):
        a = torch.zeros(50, 50)
        a[:, 10] = 1
        self.assertEqual(attention_horizon(a), 11)
        self.assertEqual(attention_horizon(torch.eye(50)), 50)
        a.fill_(0)
        a[:, 0] = 1
        self.assertEqual(attention_horizon(a), 1)
        with self.assertRaises(ValueError):
            attention_horizon(torch.full((50, 50), float("nan")))

    def test_aac_entropy_matches_covariance_formula_and_elbow(self):
        rng = np.random.default_rng(23)
        xyz = rng.normal(size=(20, 1, 3))
        xyz = np.repeat(xyz, 50, axis=1)
        xyz[:, 10:] *= 10
        normalized = torch.zeros(20, 50, 32, dtype=torch.float64)
        normalized[..., :3] = torch.from_numpy(xyz)
        decoded = torch.ones(20, 50, 4, dtype=torch.float64)
        decoded[::2, :, 3] = -1
        entropy = entropy_profile(normalized, decoded)
        expected = .5 * np.log((2 * np.pi * np.e) ** 3 * np.linalg.det(np.cov(xyz[:, 0], rowvar=False) + 1e-6 * np.eye(3))) + np.log(2)
        self.assertAlmostEqual(float(entropy[0]), expected, places=8)
        horizon, _ = aac_horizon(normalized, decoded)
        self.assertEqual(horizon, 10)
        # No predicted motion must retain the full horizon, avoiding oscillatory
        # repeated replans around a stationary command.
        horizon, _ = aac_horizon(normalized, torch.zeros_like(decoded))
        self.assertEqual(horizon, 50)

    def test_pace_uses_velocity_not_difference_of_delta_commands(self):
        actions = np.zeros((50, 4))
        actions[:, 0] = 1
        np.testing.assert_allclose(speed_profile(actions), .8, atol=1e-12)
        self.assertEqual(pace_horizon(actions, .01), 50)
        actions[:, 0] = np.interp(np.arange(50), [0, 10, 25, 40, 49], [.1, 1, .05, 1, 1])
        self.assertTrue(23 <= pace_horizon(actions, .1, smoothing_window=5) <= 27)
        self.assertEqual(pace_horizon(actions, 100), 50)

    def test_pace_calibration_can_move_but_settings_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            settings = {"artifact": str(path), "fps": 80, "percentile": 5}
            path.write_text(json.dumps({
                "action_representation": "metaworld_delta_xyz",
                "settings": {**settings, "artifact": "/previous/machine/calibration.json"},
                "thresholds": {"0": .1},
            }))
            self.assertEqual(PACE(settings).thresholds, {"0": .1})
            with self.assertRaisesRegex(ValueError, "settings"):
                PACE({**settings, "fps": 50})

    def test_a3_conditioning_and_replacement(self):
        class Sampler:
            def __init__(self):
                self.masks = []

            def sample(self, fixed_actions=None, fixed_mask=None):
                actions = torch.zeros(5, 4)
                if fixed_mask is not None:
                    self.masks.append(fixed_mask.clone())
                    actions[2, 0] = 9
                return Prediction(torch.zeros(5, 32), actions)

        sampler = Sampler()
        decision = A3({"samples": 2, "delta_thresholds": [1, 1, 1]}).decide(sampler, 0)
        self.assertEqual(decision.horizon, 3)
        self.assertEqual(float(decision.actions[2, 0]), 9)
        self.assertTrue(decision.diagnostics["replacement_appended"])
        self.assertTrue(sampler.masks[0][0])
        self.assertFalse(sampler.masks[0][1])
        self.assertTrue(sampler.masks[1][1])
        self.assertFalse(sampler.masks[1][2])

    def test_a3_outlier_does_not_become_consensus_draft(self):
        actions = np.zeros((8, 50, 4))
        actions[7, :, :3] = 100
        index, confidence = select_draft(actions)
        self.assertNotEqual(index, 7)
        self.assertTrue(np.isfinite(confidence).all())

    def test_bcp_probability_reward_and_clipping(self):
        torch.testing.assert_close(continuation_log_probs(torch.zeros(3)).exp(), torch.tensor([.5, .25, .125, .125]))
        for scale in (-1000, 1000):
            probs = continuation_log_probs(torch.full((7,), float(scale))).exp()
            self.assertAlmostEqual(float(probs.sum()), 1)
        self.assertEqual(efficiency_reward(False, 2, 10), 0)
        self.assertEqual(efficiency_reward(True, 10, 10), 1)
        self.assertGreater(efficiency_reward(True, 2, 10), 1)
        self.assertLess(efficiency_reward(True, 20, 10), 1)
        self.assertAlmostEqual(float(clipped_objective(torch.tensor(2.).log(), 0, 1, 0)), -1.28, places=6)
        self.assertAlmostEqual(float(clipped_objective(torch.tensor(2.).log(), 0, -1, 0)), 2, places=6)

    def test_bcp_group_update_uses_reference_and_changes_head(self):
        torch.manual_seed(8)
        head = ContinuationHead([15, 30, 50], context_dim=4, velocity_dim=3, action_dim=4, width=8, heads=2, layers=1)
        inputs = {"context": torch.randn(5, 4), "actions": torch.randn(50, 4), "velocity_features": torch.randn(50, 3)}
        with torch.no_grad():
            old = head(**inputs)[0]
        trajectories = [[{"inputs": inputs, "index": i, "old_logp": float(old[i])}] for i in range(2)]
        initial = head.output.weight.detach().clone()
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        settings = {"update_epochs": 2, "clip_low": .2, "clip_high": .28, "entropy_bonus": .05, "gradient_clip": 1.}
        result = optimize_group(head, optimizer, trajectories, [1.3, 0, 1.], settings)
        self.assertGreater(result["advantages"][0], 0)
        self.assertLess(result["advantages"][1], 0)
        self.assertEqual(len(result["advantages"]), 3)
        self.assertFalse(torch.equal(initial, head.output.weight))
        self.assertTrue(np.isfinite(result["loss"]))

    def test_bcp_discards_all_failure_group_without_optimizer_step(self):
        head = ContinuationHead([15, 50], context_dim=4, velocity_dim=3,
                                action_dim=4, width=8, heads=2, layers=1)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        initial = {key: value.clone() for key, value in head.state_dict().items()}
        result = optimize_group(head, optimizer, [[{}]], [0, 0], {})
        self.assertTrue(result["skipped"])
        self.assertEqual(optimizer.state, {})
        for key, value in head.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    def test_bcp_evaluation_rejects_head_with_only_failed_groups(self):
        from dynamic_horizon.config import make_method
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "head.pt"
            torch.save({"step": 10, "optimized_groups": 0}, path)
            with self.assertRaisesRegex(ValueError, "resume training"):
                make_method({"method": "bcp", "settings": {"artifact": str(path)}})

    def test_checkpoint_config_accepts_legacy_and_typed_exports(self):
        from dynamic_horizon.policy import load_checkpoint_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            for raw in ({}, {"type": "pi05"}):
                path.write_text(json.dumps(raw))
                self.assertEqual(load_checkpoint_config(directory).chunk_size, 50)
            path.write_text(json.dumps({"type": "pi0"}))
            with self.assertRaisesRegex(ValueError, "π0.5"):
                load_checkpoint_config(directory)
            path.write_text(json.dumps({"unknown_setting": 1}))
            with self.assertRaises(Exception):
                load_checkpoint_config(directory)


if __name__ == "__main__":
    unittest.main()
