"""CPU-only COAP tests: python -m unittest discover -s tests/coap -v."""

import json
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from siglip_dynamics.coap.calibration import calibrate, summarize
from siglip_dynamics.coap.config import COAPConfig, from_dict, load_config
from siglip_dynamics.coap.data import EpisodeBatchSampler, FrameCache, WindowDataset
from siglip_dynamics.coap.model import COAP, contrastive_loss, eligible_pairs
from siglip_dynamics.coap.run import eval_command
from siglip_dynamics.coap.runtime import (
    Monitor,
    Predictor,
    after_step,
    fresh_observation_features,
    load_runtime,
)
from siglip_dynamics.coap.train import train


def make_cache(root):
    root.mkdir()
    rng = np.random.default_rng(7)
    n = 12 * 10
    np.save(root / "z_q.npy", rng.integers(-100, 100, (n, 4, 6), dtype=np.int8))
    np.save(root / "z_scale.npy", np.full((n, 4), 0.01, np.float16))
    np.save(root / "episode_index.npy", np.repeat(np.arange(12), 10).astype(np.int32))
    np.save(root / "actions.npy", rng.normal(size=(n, 2)).astype(np.float32))
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": "siglip_frame_cache_v2",
                "num_frames": n,
                "dataset_format": "metaworld",
                "encoder_backend": "pi05",
                "action_representation": "raw_action",
                "l_tokens": 4,
                "d_dim": 6,
                "action_dim": 2,
                "image_token_grid_size": 0,
            }
        )
    )


def tiny_config(root):
    cfg = COAPConfig()
    cfg.model.hidden_dim = 8
    cfg.model.embedding_dim = 4
    cfg.training.cache = str(root / "cache")
    cfg.training.output = str(root / "train")
    cfg.training.device = "cpu"
    cfg.training.epochs = 2
    cfg.training.batch_size = 4
    cfg.training.val_fraction = cfg.training.test_fraction = 0.2
    cfg.monitor.artifact = str(root / "calibration.json")
    cfg.monitor.min_successful_episodes = 2
    cfg.evaluation.output = str(root / "eval")
    return cfg


class DataAndObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_cache(self.root / "cache")
        self.cfg = tiny_config(self.root)

    def test_episode_disjoint_ordered_five_pairs_and_train_only_normalization(self):
        cache = FrameCache(self.cfg.training.cache)
        splits = cache.split(self.cfg.training)
        self.assertEqual(len(set(sum(splits.values(), []))), 12)
        norm = cache.action_normalization(splits["train"], 0.001)
        train_frames = np.isin(cache.episodes, splits["train"])
        np.testing.assert_allclose(norm["mean"], cache.actions[train_frames].mean(0), rtol=1e-5)
        ds = WindowDataset(cache, splits["validation"], 5, norm)
        for i, (episode, start) in enumerate(ds.windows):
            row = ds[i]
            self.assertTrue(np.all(cache.episodes[start : start + 5] == episode))
            np.testing.assert_array_equal(row["observations"], cache.observations(start, start + 5))
            np.testing.assert_allclose(
                row["actions"], (cache.actions[start : start + 5] - norm["mean"]) / norm["std"], rtol=1e-5
            )
        # A five-pair window includes an episode's final row, without reading an o6.
        episode = splits["validation"][0]
        self.assertIn((episode, cache.ranges[episode][1] - 5), ds.windows)
        writable = np.load(self.root / "cache/actions.npy", mmap_mode="r+")
        writable[~train_frames] = 1e6
        writable.flush()
        self.assertEqual(norm, cache.action_normalization(splits["train"], 0.001))

    def test_globally_normalized_cache_requires_verified_raw_sidecar(self):
        path = self.root / "cache/metadata.json"
        metadata = json.loads(path.read_text())
        metadata["action_representation"] = "normalized_action"
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "aligned raw actions"):
            FrameCache(self.cfg.training.cache)

    def test_sampler_and_mask_exclude_overlap_and_duplicate_inputs(self):
        cache = FrameCache(self.cfg.training.cache)
        split = cache.split(self.cfg.training)
        norm = cache.action_normalization(split["train"], 0.001)
        ds = WindowDataset(cache, split["train"], 5, norm)
        sampler = EpisodeBatchSampler(ds, 4, 42)
        for batch in sampler:
            episodes = [ds.windows[i][0] for i in batch]
            self.assertEqual(len(episodes), len(set(episodes)))
        batch = {
            "episode": torch.tensor([0, 0, 0, 1]),
            "start": torch.tensor([0, 1, 8, 0]),
            "observation_hash": ["a", "b", "c", "a"],
            "action_hash": ["w", "x", "y", "z"],
        }
        mask = eligible_pairs(batch, 5)
        self.assertFalse(mask[0, 1])  # heavy overlap
        self.assertFalse(mask[0, 3])  # duplicate observation sequence across episodes
        self.assertTrue(mask[0, 2])
        self.assertTrue(torch.equal(mask, mask.T))
        self.assertTrue(mask.diagonal().all())

    def test_two_towers_unit_norm_order_and_symmetric_contrastive_gradients(self):
        torch.manual_seed(7)
        model = COAP(6, 2, self.cfg.model)
        obs, actions = torch.randn(4, 5, 6), torch.randn(4, 5, 2)
        visual, motor = model(obs, actions)
        torch.testing.assert_close(visual.norm(dim=-1), torch.ones(4))
        torch.testing.assert_close(motor.norm(dim=-1), torch.ones(4))
        self.assertFalse(torch.allclose(visual, model(obs.flip(1), actions)[0]))
        self.assertFalse(torch.allclose(motor, model(obs, actions.flip(1))[1]))
        loss = contrastive_loss(visual, motor, torch.ones(4, 4, dtype=torch.bool), 0.07)
        loss.backward()
        for tower in (model.observation_tower, model.action_tower):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in tower.parameters()), 0)
        identity = torch.eye(3)
        allowed = torch.ones(3, 3, dtype=torch.bool)
        self.assertLess(
            contrastive_loss(identity, identity, allowed, 0.07),
            contrastive_loss(identity, identity.roll(1, 0), allowed, 0.07),
        )
        self.assertIsNone(contrastive_loss(identity, identity, torch.eye(3, dtype=torch.bool), 0.07))
        with self.assertRaises(ValueError):
            contrastive_loss(identity, identity, allowed, 0)

    def test_raw_action_export_does_not_decode_images_and_checks_alignment(self):
        from unittest.mock import patch

        from datasets import Dataset

        from siglip_dynamics.coap.data import prepare_actions

        cache = FrameCache(self.cfg.training.cache)
        source = Dataset.from_dict(
            {
                "action": cache.actions.tolist(),
                "episode_index": cache.episodes.tolist(),
                "frame_index": np.tile(np.arange(10), 12).tolist(),
            }
        )
        with patch("datasets.load_dataset", return_value=source):
            prepare_actions(self.cfg.training.cache, self.root / "dataset")
        raw = self.root / "cache/coap_raw_actions.npy"
        reloaded = FrameCache(self.cfg.training.cache, str(raw))
        np.testing.assert_array_equal(reloaded.actions, cache.actions)
        raw.write_bytes(raw.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            FrameCache(self.cfg.training.cache, str(raw))

    def test_config_and_comparison_commands_are_isolated(self):
        cfg = load_config("configs/coap/metaworld_pool64.yaml")
        self.assertEqual(cfg.model.window_size, 5)
        coap = eval_command(cfg, "evaluate", "resolved.yaml")
        plain = eval_command(cfg, "evaluate", "resolved.yaml", "policy")
        self.assertIn("--coap_config=resolved.yaml", coap)
        self.assertIn("--meltdown_use_guidance_replan=false", coap)
        self.assertFalse(any("safety_model" in arg or "experiment_config" in arg for arg in coap))
        self.assertIn("--disable_corrector=true", plain)
        self.assertFalse(any("--coap_config" in arg for arg in plain))
        with self.assertRaises(ValueError):
            from_dict({"model": {"window_size": 0}})
        with self.assertRaises(ValueError):
            from_dict({"features": {"action_prefix": True}})


class FakePredictor:
    def __init__(self, similarity=-0.5):
        self.settings = SimpleNamespace(window_size=5)
        self.model = SimpleNamespace(observation_dim=2, action_dim=1)
        self.fingerprint = "fake-coap"
        self.similarity = similarity
        self.calls = []

    def score(self, observations, actions):
        self.calls.append((np.array(observations), np.array(actions)))
        return self.similarity


class RuntimeTests(unittest.TestCase):
    def monitor(self, similarity=-0.5, persistence=1, cooldown=0, shadow=False):
        cfg = COAPConfig()
        cfg.monitor.persistence, cfg.monitor.cooldown_steps = persistence, cooldown
        return Monitor(FakePredictor(similarity), cfg.monitor, 0.0, shadow=shadow)

    def execute(self, monitor, step, queue=None, done=False):
        queue = deque([99, 100]) if queue is None else queue
        raw = deque([199, 200])
        event = after_step(monitor, np.array([step, step + 1]), np.array([step]), step, queue, raw, done=done)
        return event, queue, raw

    def test_first_score_after_five_executed_pairs_no_future_actions_or_o6(self):
        monitor = self.monitor()
        monitor.reset("easy:0", 1000)
        for step in range(1, 5):
            event, queue, _ = self.execute(monitor, step)
            self.assertIsNone(event)
            self.assertEqual(list(queue), [99, 100])
            self.assertFalse(monitor.predictor.calls)
        event, queue, raw = self.execute(monitor, 5)
        self.assertTrue(event["trigger"])
        self.assertFalse(queue)
        self.assertFalse(raw)
        obs, actions = monitor.predictor.calls[-1]
        np.testing.assert_array_equal(obs[:, 0], np.arange(1, 6))
        np.testing.assert_array_equal(actions[:, 0], np.arange(1, 6))
        self.assertEqual(event["observation_steps"], [0, 1, 2, 3, 4])
        self.execute(monitor, 6)
        np.testing.assert_array_equal(monitor.predictor.calls[-1][1][:, 0], np.arange(2, 7))

    def test_high_similarity_persistence_cooldown_done_and_episode_reset(self):
        monitor = self.monitor(similarity=0.9)
        for step in range(1, 8):
            event, queue, _ = self.execute(monitor, step)
            self.assertTrue(queue)
            if event:
                self.assertFalse(event["trigger"])
        monitor = self.monitor(persistence=2, cooldown=2)
        for step in range(1, 6):
            self.assertFalse((self.execute(monitor, step)[0] or {}).get("trigger", False))
        self.assertTrue(self.execute(monitor, 6)[0]["trigger"])
        self.assertTrue(self.execute(monitor, 7)[0]["cooldown"])
        self.assertTrue(self.execute(monitor, 8)[0]["cooldown"])
        self.assertFalse(self.execute(monitor, 9)[0]["trigger"])
        self.assertTrue(self.execute(monitor, 10)[0]["trigger"])
        monitor.reset("other", 2)
        self.assertEqual(monitor.steps, 0)
        self.assertFalse(monitor.actions)
        self.assertIsNone(self.execute(monitor, 1)[0])
        terminal = self.monitor()
        for step in range(1, 5):
            self.execute(terminal, step)
        self.assertFalse(self.execute(terminal, 5, done=True)[0]["trigger"])

    def test_disabled_and_shadow_never_discard_actions_and_boundary_not_intervention(self):
        queue, raw = deque([1, 2]), deque([3, 4])
        self.assertIsNone(after_step(None, None, None, 1, queue, raw))
        self.assertEqual(list(queue), [1, 2])
        self.assertEqual(list(raw), [3, 4])
        shadow = self.monitor(shadow=True)
        for step in range(1, 7):
            event, queue, _ = self.execute(shadow, step)
            self.assertEqual(len(queue), 2)
            if event:
                self.assertFalse(event["trigger"])
        monitor = self.monitor()
        for step in range(1, 5):
            self.execute(monitor, step)
        event, _, _ = self.execute(monitor, 5, deque())
        self.assertTrue(event["alarm"])
        self.assertFalse(event["trigger"])

    def test_fresh_feature_pass_and_pooling_without_guidance_predictor(self):
        calls = []

        def encode(image):
            calls.append(image)
            return image

        policy = SimpleNamespace(
            _preprocess_images=lambda batch: ([batch["image"]], [torch.tensor([True])]),
            model=SimpleNamespace(paligemma_with_expert=SimpleNamespace(embed_image=encode)),
        )
        metadata = {"image_token_grid_size": 8, "l_tokens": 64, "d_dim": 3}
        image = torch.randn(1, 256, 3)
        first = fresh_observation_features(policy, {"image": image}, metadata)
        second = fresh_observation_features(policy, {"image": image + 2}, metadata)
        self.assertEqual(len(calls), 2)
        np.testing.assert_allclose(second - first, 2, atol=1e-5)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        make_cache(self.root / "cache")
        self.cfg = tiny_config(self.root)

    def test_train_reload_successful_rollout_calibration_and_provenance(self):
        checkpoint = train(self.cfg)
        predictor = Predictor(checkpoint, "cpu")
        self.assertEqual(predictor.settings.window_size, 5)
        self.assertFalse(any(p.requires_grad for p in predictor.model.parameters()))
        collector = load_runtime(self.cfg, "collect-calibration", self.root / "collect")
        for seed in (10000, 10001):
            collector.reset("easy:0", seed)
            for step in range(1, 8):
                collector.after_execution(
                    np.full(6, step, dtype=np.float32), np.array([step, step / 2]), step
                )
            row = collector.finish(True)
            self.assertEqual(row["intervention_count"], 0)
            self.assertEqual(len(row["traces"]), 3)
        calibration = calibrate(self.cfg, traces=self.root / "collect/coap_episodes.jsonl")
        self.assertEqual(calibration["episodes_used"], 2)
        evaluation = load_runtime(self.cfg, "evaluate", self.root / "eval")
        with self.assertRaisesRegex(ValueError, "overlaps"):
            evaluation.reset("easy:0", 10000)
        evaluation.reset("easy:0", 1000)
        changed = from_dict(self.cfg.to_dict())
        changed.evaluation.n_action_steps = 9
        with self.assertRaisesRegex(ValueError, "protocol"):
            load_runtime(changed, "evaluate", self.root / "other")
        with self.assertRaises(FileExistsError):
            train(self.cfg)
        with self.assertRaises(FileExistsError):
            calibrate(self.cfg, traces=self.root / "collect/coap_episodes.jsonl")

    def test_demonstration_threshold_is_provisional_and_normalization_saved(self):
        checkpoint = train(self.cfg)
        payload = torch.load(checkpoint, weights_only=False)
        self.assertEqual(
            set(payload["normalization"]["fit_episodes"]), set(payload["split_manifest"]["train"])
        )
        result = calibrate(self.cfg, source="demonstrations")
        self.assertTrue(result["status"].startswith("PROVISIONAL"))
        with self.assertRaisesRegex(ValueError, "Provisional"):
            load_runtime(self.cfg, "evaluate", self.root / "eval")
        self.cfg.monitor.allow_provisional_threshold = True
        self.assertIsNotNone(load_runtime(self.cfg, "evaluate", self.root / "eval"))

    def test_labels_are_required_for_false_alarm_and_delay_metrics(self):
        monitor = RuntimeTests().monitor()
        monitor.output = self.root
        monitor.reset("easy:0", 1000)
        for step in range(1, 7):
            monitor.after_execution(np.array([step, step]), np.array([step]), step)
        row = monitor.finish(False, vla_calls=2)
        path = self.root / "coap_episodes.jsonl"
        summary = summarize(path)
        self.assertIsNone(summary["false_alarm_rate_per_labelled_normal_window"])
        self.assertIsNone(summary["detection_delays_steps"])
        row["traces"][0]["abnormal"] = False
        row["anomaly_intervals"] = [{"start_step": 6, "end_step": 7}]
        path.write_text(json.dumps(row) + "\n")
        summary = summarize(path)
        self.assertEqual(summary["false_alarm_rate_per_labelled_normal_window"], 1)
        self.assertEqual(summary["detection_delays_steps"], [0])


if __name__ == "__main__":
    unittest.main()
