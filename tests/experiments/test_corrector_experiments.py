"""CPU tests, runnable without the repository's hardware pytest fixtures.

python -m unittest discover -s tests/experiments -v
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from siglip_dynamics.experiments.config import Experiment, from_checkpoint_dict, from_dict, load_config
from siglip_dynamics.experiments.data import PrefixDataset, episode_split
from siglip_dynamics.experiments.model import ExperimentalCorrector
from siglip_dynamics.experiments.run import eval_command, legacy_train_command, resolve
from siglip_dynamics.experiments.runtime import ExperimentPredictor, RuntimeMonitor, cosine_error, terminal_observation
from siglip_dynamics.experiments.train import train_experiment


def tiny_config():
    c = Experiment()
    c.model.horizons = [1, 2, 4]
    c.model.widths = [8, 8]
    c.model.action_embed_dim = 4
    c.model.uncertainty_hidden = 4
    c.training.device = "cpu"
    c.training.epochs = 1
    c.training.batch_size = 8
    c.evaluation.n_action_steps = 4
    c.features.action_prefix = True
    return c


def make_cache(root):
    root.mkdir()
    n = 8 * 6
    rng = np.random.default_rng(42)
    np.save(root / "z_q.npy", rng.integers(-80, 80, size=(n, 2, 4), dtype=np.int8))
    np.save(root / "z_scale.npy", np.full((n, 2), 0.01, dtype=np.float16))
    np.save(root / "actions.npy", np.arange(n * 2, dtype=np.float32).reshape(n, 2) / 100)
    np.save(root / "episode_index.npy", np.repeat(np.arange(6), 8))
    (root / "metadata.json").write_text(json.dumps({"action_representation": "normalized_action"}))


class ModelAndDataTests(unittest.TestCase):
    def test_episode_split_and_exact_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            make_cache(root)
            ids = np.load(root / "episode_index.npy")
            split = episode_split(ids, 0.2, 0.2, 7)
            self.assertEqual(len(set(sum(split.values(), []))), 6)
            ds = PrefixDataset(root, [1, 2, 4], split["train"])
            for i in range(len(ds)):
                row = ds[i]
                t = ds.starts[i // 3]
                h = row["horizon"]
                self.assertEqual(ids[t], ids[t+h])
                np.testing.assert_allclose(row["actions"][:h], ds.base.a_t_np[t:t+h])
                self.assertTrue(torch.all(row["actions"][h:] == 0))

    def test_baseline_matches_original_mlp(self):
        cfg = tiny_config()
        cfg.features.action_prefix = False
        cfg.model.horizons = [4]
        model = ExperimentalCorrector(cfg, 4, 2).eval()
        z, a = torch.randn(3, 2, 4), torch.randn(3, 2)
        self.assertTrue(torch.equal(model(z, a), model.base(z, a)))

    def test_order_suffix_mask_and_prefix_guidance_gradient(self):
        torch.manual_seed(7)
        cfg = tiny_config()
        model = ExperimentalCorrector(cfg, 4, 2).eval()
        z = torch.randn(2, 2, 4)
        a = torch.randn(2, 4, 2, requires_grad=True)
        pred, _ = model.predict(z, a, 2)
        changed = a.detach().clone()
        changed[:, 2:] += 100
        self.assertTrue(torch.equal(pred, model.predict(z, changed, 2)[0]))
        self.assertFalse(torch.allclose(pred, model.predict(z, a.flip(1), 2)[0]))
        pred.square().sum().backward()
        self.assertTrue((a.grad[:, :2].abs().sum(dim=(0, 2)) > 0).all())
        self.assertTrue((a.grad[:, 2:] == 0).all())
        with self.assertRaisesRegex(ValueError, "Missing future"):
            model.predict(z, a[:, :1], 2)

    def test_uncertainty_loss_does_not_update_predictor(self):
        cfg = tiny_config()
        cfg.features.uncertainty_monitor = True
        cfg.model.horizons = [4]
        m = ExperimentalCorrector(cfg, 4, 2)
        _, log_error = m.predict(torch.randn(2, 2, 4), torch.randn(2, 4, 2), 4)
        log_error.sum().backward()
        self.assertTrue(all(p.grad is None for p in m.base.parameters()))
        self.assertTrue(all(p.grad is None for p in m.prefix_encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in m.uncertainty.parameters()))

    def test_train_save_load_cpu(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_cache(root / "cache")
            cfg = tiny_config()
            cfg.training.cache = str(root / "cache")
            cfg.training.output = str(root / "train")
            train_experiment(cfg)
            predictor = ExperimentPredictor(root / "train", cfg, "cpu")
            ds = PrefixDataset(root / "cache", cfg.model.horizons, [0])
            row = ds[2]
            pred, error = predictor.predict(row["z"].numpy(), row["actions"].numpy(), 4)
            self.assertEqual(pred.shape, (2, 4))
            self.assertIsNone(error)
            self.assertTrue((root / "train/last_model.pt").exists())
            with self.assertRaises(FileExistsError):
                train_experiment(cfg)
            cfg.features.action_prefix = False
            with self.assertRaisesRegex(ValueError, "differ"):
                ExperimentPredictor(root / "train", cfg, "cpu")


class FakePredictor:
    fingerprint = "fake-checkpoint"

    def predict(self, z, actions, horizon):
        # Deterministic test dynamics; future z changes by summed action.
        return np.ones_like(z) * float(np.asarray(actions).sum()), float(horizon) / 10

class RuntimeTests(unittest.TestCase):
    def test_zero_motion_and_magnitude(self):
        z = np.zeros((2, 4))
        self.assertEqual(cosine_error(z, z), 0)
        self.assertEqual(cosine_error(z, np.ones_like(z)), 1)
        self.assertAlmostEqual(cosine_error(np.ones_like(z), 10*np.ones_like(z)), 0)

    def test_final_observation_not_autoreset(self):
        reset = {"pixels": {"image": np.zeros((1, 2, 2, 3))}}
        final = {"pixels": {"image": np.ones((2, 2, 3))}}
        info = {"final_obs": np.array([final], dtype=object), "_final_obs": [True]}
        result = terminal_observation(reset, info)
        self.assertEqual(result["pixels"]["image"].shape, (1, 2, 2, 3))
        self.assertTrue(np.all(result["pixels"]["image"] == 1))
        self.assertIs(terminal_observation(reset, {}), reset)

    def test_queue_bounded_alignment_and_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config()
            m = RuntimeMonitor(cfg, FakePredictor(), "collect-calibration", tmp, {})
            m.reset("task:0", 123)
            z, actions = np.zeros((2, 4)), np.ones((2, 2))
            m.schedule(z, actions, 0)
            self.assertEqual([p["horizon"] for p in m.pending], [1, 2])
            self.assertIsNone(m.observe(z, 0))
            event = m.observe(z+2, 1)
            self.assertEqual(event["horizon"], 1)
            self.assertAlmostEqual(event["score"], 0)
            m.cancel()
            self.assertIsNone(m.observe(z+4, 2))
            self.assertEqual(m.cancellations, 1)
            cfg.features.action_prefix = False
            cfg.evaluation.queue_bounded_monitor = True
            control = RuntimeMonitor(cfg, FakePredictor(), "evaluate", Path(tmp)/"control", {})
            control.reset("task:0", 1)
            control.schedule(z, actions, 0)
            self.assertEqual([p["horizon"] for p in control.pending], [1, 2])

class ConfigurationTests(unittest.TestCase):
    def test_examples_and_cli_switches(self):
        for name in ("baseline", "baseline_matched", "prefix", "uncertainty", "baseline_queue_matched"):
            cfg = load_config(f"configs/corrector/{name}.yaml")
            args = eval_command(cfg, "evaluate", "resolved.yaml")
            self.assertIn("--compute_baseline_metrics=false", args)
            if name == "baseline":
                self.assertFalse(any("experiment_config" in arg or "experiment_mode" in arg for arg in args))
                self.assertIn("siglip_dynamics.train", legacy_train_command(cfg))
                self.assertIn("--train-loss-type=cosine", legacy_train_command(cfg))
            else:
                self.assertIn("--experiment_mode=evaluate", args)
        cfg = resolve("configs/corrector/baseline_matched.yaml", ["features.action_prefix=true"])
        self.assertTrue(cfg.features.action_prefix)
        with self.assertRaisesRegex(ValueError, "Legacy mode"):
            resolve("configs/corrector/baseline.yaml", ["features.action_prefix=true"])
        with self.assertRaises(ValueError):
            from_dict({"features": {"prefix": True}})
        with self.assertRaises(ValueError):
            from_dict({"features": {"action_prefix": "false"}})

    def test_supported_checkpoint_metadata_migration(self):
        raw = tiny_config().to_dict()
        raw["features"]["removed_feature"] = False
        raw["unused_section"] = {"option": 1}
        raw["evaluation"]["unused_seed"] = 3000
        self.assertEqual(from_checkpoint_dict(raw).to_dict(), tiny_config().to_dict())
        with self.assertRaises(ValueError):
            from_dict(raw)
        raw["features"]["removed_feature"] = True
        with self.assertRaisesRegex(ValueError, "unsupported feature"):
            from_checkpoint_dict(raw)

    def test_checkpoint_model_fields_remain_strict(self):
        raw = tiny_config().to_dict()
        raw["model"]["unknown_architecture"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            from_checkpoint_dict(raw)


if __name__ == "__main__":
    unittest.main()
