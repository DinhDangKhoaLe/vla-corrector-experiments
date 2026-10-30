"""CPU integration checks for pooled extraction, training, monitoring and OGG inputs."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from siglip_dynamics.visual_tokens import pool_square_token_grid, validate_cache_grid, validate_checkpoint_grid
from siglip_dynamics.experiments.config import load_config
from siglip_dynamics.experiments.run import eval_command
from siglip_dynamics.experiments.train import train_experiment
from siglip_dynamics.experiments.runtime import ExperimentPredictor
from test_corrector_experiments import tiny_config


def make_pooled_cache(root, shape=None):
    root.mkdir()
    rng = np.random.default_rng(42)
    n = 24 * 6
    tokens = int(np.prod(shape)) if shape else 16
    np.save(root / "z_q.npy", rng.integers(-50, 50, (n, tokens, 4), dtype=np.int8))
    np.save(root / "z_scale.npy", np.full((n, tokens), 0.01, dtype=np.float16))
    np.save(root / "actions.npy", rng.normal(size=(n, 2)).astype(np.float16))
    np.save(root / "episode_index.npy", np.repeat(np.arange(6), 24))
    (root / "metadata.json").write_text(json.dumps({
        "action_representation": "normalized_action", "image_token_grid_size": 0 if shape else 4,
        "image_token_grid_shape": list(shape or []),
        "l_tokens": tokens, "d_dim": 4, "action_dim": 2,
    }))


class SpatialPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_block_means_order_noop_and_gradient(self):
        z = torch.arange(2 * 256 * 3, dtype=torch.float32).reshape(2, 256, 3).requires_grad_()
        self.assertIs(pool_square_token_grid(z, 0), z)
        self.assertIs(pool_square_token_grid(z, 16), z)
        pooled = pool_square_token_grid(z, 4)
        expected = z.reshape(2, 4, 4, 4, 4, 3).mean(dim=(2, 4)).reshape(2, 16, 3)
        torch.testing.assert_close(pooled, expected)
        pooled.sum().backward()
        torch.testing.assert_close(z.grad, torch.full_like(z, 1 / 16))
        for side in (-1, 3, 32):
            with self.assertRaises(ValueError):
                pool_square_token_grid(z, side)

    def test_online_per_camera_pooling_matches_extraction(self):
        from siglip_dynamics.extract import BaseVisualExtractor
        from lerobot.policies.pi05_modified.modeling_pi05_modified import PI05ModifiedPolicy

        cameras = [torch.randn(1, 256, 4), torch.randn(1, 256, 4) + 20]
        extractor = BaseVisualExtractor()
        extractor.encode_single_image = lambda x: x
        offline = extractor.encode_item(
            {"head": cameras[0], "wrist": cameras[1]}, image_key="head",
            wrist_image_key="wrist", use_wrist_image=True, image_token_grid_size=4)
        policy = SimpleNamespace(
            _preprocess_images=lambda batch: (cameras + [torch.zeros_like(cameras[0])],
                                               [torch.tensor([True]), torch.tensor([True]), torch.tensor([False])]),
            model=SimpleNamespace(safety_predictor=SimpleNamespace(image_token_grid_size=4),
                                  paligemma_with_expert=SimpleNamespace(embed_image=lambda x: x)))
        online = PI05ModifiedPolicy.encode_siglip_only(policy, {})
        self.assertEqual(online.shape, (32, 4))
        np.testing.assert_array_equal(offline, online)
        offline_rect = extractor.encode_item(
            {"head": cameras[0], "wrist": cameras[1]}, image_key="head",
            wrist_image_key="wrist", use_wrist_image=True, image_token_grid_shape=[4, 8])
        policy.model.safety_predictor = SimpleNamespace(image_token_grid_size=0, image_token_grid_shape=[4, 8])
        online_rect = PI05ModifiedPolicy.encode_siglip_only(policy, {})
        self.assertEqual(online_rect.shape, (64, 4))
        np.testing.assert_array_equal(offline_rect, online_rect)
        policy.model.safety_predictor = None
        full = PI05ModifiedPolicy.encode_siglip_only(policy, {})
        np.testing.assert_array_equal(full, torch.cat(cameras, 1)[0].numpy())

    def test_cache_and_config_isolation(self):
        for name in ("original", "prefix", "uncertainty"):
            cfg = load_config(f"configs/corrector/metaworld_{name}_pool4.yaml")
            self.assertEqual(cfg.model.image_token_grid_size, 4)
            self.assertIn("pool4", cfg.training.cache)
            self.assertIn("pool4", cfg.training.output)
            self.assertIn("pool4", cfg.evaluation.output)
            self.assertIn("--policy.n_action_steps=10", eval_command(cfg, "evaluate", "resolved.yaml"))
            rect = load_config(f"configs/corrector/metaworld_{name}_pool32.yaml")
            self.assertEqual(rect.model.image_token_grid_size, 0)
            self.assertEqual(rect.model.image_token_grid_shape, [4, 8])
            self.assertIn("pool32", rect.training.cache)
            self.assertIn("pool32", rect.training.output)
            self.assertIn("pool32", rect.evaluation.output)
        for name in ("baseline", "prefix", "uncertainty"):
            self.assertEqual(load_config(f"configs/corrector/{name}.yaml").model.image_token_grid_size, 0)
        with self.assertRaisesRegex(ValueError, "Cache image_token_grid_size"):
            validate_cache_grid({"image_token_grid_size": 0}, 4)
        with self.assertRaisesRegex(ValueError, "expected 16"):
            validate_cache_grid({"image_token_grid_size": 4, "l_tokens": 256}, 4)

    def test_prefix_and_uncertainty_train_reload_and_guidance_gradients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pooled_cache(root / "cache")
            for uncertainty in (False, True):
                cfg = tiny_config()
                cfg.model.horizons = [2]
                cfg.model.image_token_grid_size = 4
                cfg.features.uncertainty_monitor = uncertainty
                cfg.training.cache = str(root / "cache")
                cfg.training.output = str(root / f"train_{uncertainty}")
                train_experiment(cfg)
                predictor = ExperimentPredictor(cfg.training.output, cfg, "cpu")
                self.assertEqual(predictor.model.image_token_grid_size, 4)
                validate_checkpoint_grid(cfg.training.output, 4)
                with self.assertRaisesRegex(ValueError, "Checkpoint image_token_grid_size"):
                    validate_checkpoint_grid(cfg.training.output, 0)
                z = pool_square_token_grid(torch.randn(1, 256, 4), 4)
                actions = torch.randn(1, 2, 2, requires_grad=True)
                predictor.model(z, actions).square().sum().backward()
                self.assertGreater(actions.grad.abs().sum().item(), 0)
                cfg.model.image_token_grid_size = 0
                cfg.training.output += "_wrong"
                with self.assertRaisesRegex(ValueError, "Cache image_token_grid_size"):
                    train_experiment(cfg)

    def test_original_train_checkpoint_roundtrip_and_guidance_gradient(self):
        from siglip_dynamics.config import TrainConfig, SiglipMLPConfig, ModelScale, LossType
        from siglip_dynamics.train import train
        from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pooled_cache(root / "cache")
            cfg = TrainConfig(dataset_path=str(root / "cache"), checkpoint_dir=str(root / "train"),
                              device="cpu", epochs=1, batch_size=8, k_step=2, num_workers=0,
                              train_loss_type=LossType.COSINE,
                              model=SiglipMLPConfig(token_dim=4, action_dim=2, action_embed_dim=4,
                                                   scale=ModelScale.CUSTOM, custom_widths=(8, 8)))
            with patch("siglip_dynamics.train.wandb"):
                train(cfg)
            predictor = SiglipDynamicsPredictor(root / "train", device="cpu")
            self.assertEqual(predictor.model.image_token_grid_size, 4)
            validate_checkpoint_grid(root / "train", 4)
            z = pool_square_token_grid(torch.randn(1, 256, 4), 4)
            action = torch.randn(1, 2, requires_grad=True)
            predictor.model(z, action).square().sum().backward()
            self.assertGreater(action.grad.abs().sum().item(), 0)

            # Old local checkpoints did not serialize a representation setting.
            payload = torch.load(predictor.checkpoint_path, map_location="cpu", weights_only=False)
            payload["train_cfg"].pop("image_token_grid_size")
            torch.save(payload, predictor.checkpoint_path)
            config_file = root / "train/config.json"
            config = json.loads(config_file.read_text())
            config.pop("image_token_grid_size")
            config_file.write_text(json.dumps(config))
            old = SiglipDynamicsPredictor(root / "train", device="cpu")
            self.assertEqual(old.model.image_token_grid_size, 0)
            torch.testing.assert_close(old.model(z, action), predictor.model(z, action))

    def test_rectangular_means_and_invalid_shapes(self):
        z = torch.arange(2 * 256 * 3, dtype=torch.float32).reshape(2, 256, 3).requires_grad_()
        pooled = pool_square_token_grid(z, target_shape=[4, 8])
        expected = z.reshape(2, 4, 4, 8, 2, 3).mean(dim=(2, 4)).reshape(2, 32, 3)
        torch.testing.assert_close(pooled, expected)
        pooled.sum().backward()
        torch.testing.assert_close(z.grad, torch.full_like(z, 1 / 8))
        for shape in ([3, 8], [4], [4, 0], [4, 32]):
            with self.assertRaises(ValueError):
                pool_square_token_grid(z, target_shape=shape)
        with self.assertRaises(ValueError):
            pool_square_token_grid(z, 4, target_shape=[4, 8])

    def test_rectangular_training_and_checkpoint_shapes(self):
        from siglip_dynamics.config import TrainConfig, SiglipMLPConfig, ModelScale
        from siglip_dynamics.train import train
        from lerobot.safety.siglip_dynamics_mlp import SiglipDynamicsPredictor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pooled_cache(root / "cache", shape=[4, 8])
            meta = json.loads((root / "cache/metadata.json").read_text())
            validate_cache_grid(meta, 0, target_shape=[4, 8])
            with self.assertRaises(ValueError):
                validate_cache_grid(meta, 0)
            for uncertainty in (False, True):
                cfg = tiny_config()
                cfg.model.horizons = [2]
                cfg.model.image_token_grid_shape = [4, 8]
                cfg.features.uncertainty_monitor = uncertainty
                cfg.training.cache = str(root / "cache")
                cfg.training.output = str(root / f"exp_{uncertainty}")
                train_experiment(cfg)
                predictor = ExperimentPredictor(cfg.training.output, cfg, "cpu")
                self.assertEqual(predictor.model.image_token_grid_shape, [4, 8])
                validate_checkpoint_grid(cfg.training.output, 0, target_shape=[4, 8])
                with self.assertRaises(ValueError):
                    validate_checkpoint_grid(cfg.training.output, 0)
                z = pool_square_token_grid(torch.randn(1, 256, 4), target_shape=[4, 8])
                actions = torch.randn(1, 2, 2, requires_grad=True)
                delta = predictor.model(z, actions)
                self.assertEqual(delta.shape, (1, 32, 4))
                delta.square().sum().backward()
                self.assertGreater(actions.grad.abs().sum().item(), 0)
            legacy_cfg = TrainConfig(
                dataset_path=str(root / "cache"), checkpoint_dir=str(root / "legacy"),
                device="cpu", epochs=1, batch_size=8, k_step=2, num_workers=0,
                model=SiglipMLPConfig(token_dim=4, action_dim=2, action_embed_dim=4,
                                     scale=ModelScale.CUSTOM, custom_widths=(8, 8)))
            with patch("siglip_dynamics.train.wandb"):
                train(legacy_cfg)
            legacy = SiglipDynamicsPredictor(root / "legacy", device="cpu")
            self.assertEqual(legacy.model.image_token_grid_shape, [4, 8])
            validate_checkpoint_grid(root / "legacy", 0, target_shape=[4, 8])
            action = torch.randn(1, 2, requires_grad=True)
            delta = legacy.model(z, action)
            self.assertEqual(delta.shape, (1, 32, 4))
            delta.square().sum().backward()
            self.assertGreater(action.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
