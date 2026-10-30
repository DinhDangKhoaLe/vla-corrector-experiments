"""Synthetic end-to-end coverage without loading the VLA or starting real jobs."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from siglip_dynamics.experiments.config import from_dict
from siglip_dynamics.experiments.model import ExperimentalCorrector
from siglip_dynamics.experiments.runtime import ExperimentPredictor
from siglip_dynamics.experiments.train import train_experiment
from siglip_dynamics.experiments.uncertainty import (
    UncertaintyMonitor, calibrate_uncertainty, normalized_error,
)
from test_corrector_experiments import FakePredictor, make_cache, tiny_config


def uncertainty_config(root):
    cfg = tiny_config()
    cfg.features.uncertainty_monitor = True
    cfg.model.horizons = [2]
    cfg.uncertainty.min_successful_episodes = 5
    cfg.uncertainty.traces = str(root / "cal/experiment_episodes.jsonl")
    cfg.uncertainty.artifact = str(root / "artifact.json")
    cfg.evaluation.cooldown_steps = 3
    cfg.validate()
    return cfg


def calibrated_monitor(root, good_head=True):
    cfg = uncertainty_config(root)
    collector = UncertaintyMonitor(cfg, FakePredictor(), "collect-calibration", root / "cal", {})
    for i in range(5):
        collector.reset("task:0", i)
        collector.scores = [
            {"step": j, "horizon": 2, "rmse": j / 100,
             "predicted_rmse": j / 100 if good_head else (11-j) / 100,
             "score": 1.0 if good_head else j / (11-j)}
            for j in range(1, 11)]
        collector.finish(True, "")
    artifact = calibrate_uncertainty(cfg)
    m = UncertaintyMonitor(cfg, FakePredictor(), "evaluate", root / "eval", {})
    m.reset("task:0", 100)
    return cfg, artifact, m


def decision(m, step, score=4.0, rmse=0.2, expected=0.05, coherence=1.0):
    event = {"score": score, "rmse": rmse, "predicted_rmse": expected,
             "direction_coherence": coherence}
    m.scores.append(dict(event))
    return m.recovery_decision(event, step)


class UncertaintyTrainingTests(unittest.TestCase):
    def test_fixed_k_and_mode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = uncertainty_config(Path(tmp))
            for updates in ({"unknown_feature": True}, {"action_prefix": False}):
                raw = cfg.to_dict()
                raw["features"].update(updates)
                with self.assertRaises(ValueError):
                    from_dict(raw)
            cfg.evaluation.n_action_steps = 1
            with self.assertRaisesRegex(ValueError, "Prefix horizons"):
                cfg.validate()

    def test_error_head_detached_and_guidance_uses_exact_k_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = uncertainty_config(Path(tmp))
            model = ExperimentalCorrector(cfg, 4, 2)
            z = torch.randn(2, 2, 4)
            actions = torch.randn(2, 4, 2, requires_grad=True)
            delta, log_error = model.predict(z, actions, 2)
            log_error.sum().backward()
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            self.assertTrue(all(p.grad is None for p in model.prefix_encoder.parameters()))
            self.assertIsNone(actions.grad)
            self.assertTrue(any(p.grad is not None for p in model.uncertainty.parameters()))
            delta.square().sum().backward()
            self.assertGreater(float(actions.grad[:, :2].abs().sum()), 0)
            self.assertEqual(float(actions.grad[:, 2:].abs().sum()), 0)

    def test_training_checkpoint_and_component_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = uncertainty_config(root)
            cache = root / "cache"
            make_cache(cache)
            cfg.training.cache = str(cache)
            cfg.training.output = str(root / "training")
            train_experiment(cfg)
            payload = torch.load(root / "training/best_model.pt", weights_only=False)
            for phase, cosine in (("train", "train"), ("validation", "validation")):
                log = json.loads((root / "training/metrics.jsonl").read_text().splitlines()[0])
                parts = payload["loss_metrics"][phase]
                self.assertAlmostEqual(parts["prediction_loss"], log[cosine] + parts["mse"], places=5)
                self.assertAlmostEqual(parts["total_loss"], parts["prediction_loss"] +
                                       cfg.training.uncertainty_loss_weight * parts["error_head"], places=5)
            predictor = ExperimentPredictor(root / "training", cfg, "cpu")
            pred, error = predictor.predict(np.zeros((2, 4)), np.ones((2, 2)), 2)
            self.assertEqual(pred.shape, (2, 4))
            self.assertGreater(error, 0)
            cfg.features.uncertainty_monitor = False
            with self.assertRaisesRegex(ValueError, "uncertainty_monitor differs"):
                ExperimentPredictor(root / "training", cfg, "cpu")


class UncertaintyRuntimeTests(unittest.TestCase):
    def test_normalization_floor_and_invalid_inputs(self):
        self.assertAlmostEqual(normalized_error(0.2, 0.02, 1e-4), 10)
        self.assertAlmostEqual(normalized_error(0.2, 0.2, 1e-4), 1)
        self.assertEqual(normalized_error(0, 0, 1e-4), 0)
        for value in (None, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                normalized_error(1, value, 1e-4)

    def test_shadow_scoring_and_queue_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = uncertainty_config(root)
            m = UncertaintyMonitor(cfg, FakePredictor(), "collect-calibration", root / "cal", {})
            m.reset("task:0", 1)
            z = np.zeros((2, 4))
            m.schedule(z, np.ones((1, 2)), 0)
            self.assertFalse(m.pending)
            m.schedule(z, np.ones((4, 2)), 0)
            event = m.observe(z + 4.2, 2)  # prediction=4, observed RMSE=.2, expected=.2
            self.assertAlmostEqual(event["score"], 1)
            self.assertAlmostEqual(event["cosine_error"], 0)
            self.assertEqual(m.recovery_decision(event, 2), (False, False))
            m.schedule(z, np.ones((4, 2)), 2)
            m.cancel()
            self.assertEqual(m.cancellations, 1)
            self.assertIsNone(m.observe(z + 9, 4))
            m.finish(True, "")
            row = json.loads(m.output.read_text())
            self.assertIn("uncertainty_settings", row)

    def test_ogg_gate_persistence_cooldown_and_plain_replan(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, artifact, m = calibrated_monitor(Path(tmp))
            self.assertTrue(artifact["ogg_error_head_validated"])
            self.assertEqual(artifact["threshold"], 1.0)
            self.assertEqual(decision(m, 10), (False, False))
            self.assertEqual(decision(m, 11), (True, True))
            m.cancel()
            self.assertEqual(decision(m, 12), (False, False))
            self.assertEqual(decision(m, 14, coherence=-1), (False, False))
            self.assertEqual(decision(m, 15, coherence=-1), (True, False))
            m.cancel()
            # Large expected error masks the normalized score, but raw guard still replans.
            self.assertEqual(decision(m, 18, score=0.2, expected=1.0), (False, False))
            self.assertEqual(decision(m, 19, score=0.2, expected=1.0), (True, False))
            m.cancel()
            cfg.uncertainty.ogg_enabled = False
            decision(m, 22)
            self.assertEqual(decision(m, 23), (True, False))

    def test_unvalidated_head_never_enables_ogg(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, artifact, m = calibrated_monitor(Path(tmp), good_head=False)
            self.assertFalse(artifact["ogg_error_head_validated"])
            decision(m, 10, score=100)
            self.assertEqual(decision(m, 11, score=100), (True, False))

    def test_calibration_binding_overlap_and_allowed_ablation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _, m = calibrated_monitor(root)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                m.reset("task:0", 1)
            cfg.uncertainty.ogg_enabled = False
            control = UncertaintyMonitor(cfg, FakePredictor(), "evaluate", root / "plain", {})
            self.assertIsNotNone(control.threshold)
            cfg.uncertainty.rmse_floor *= 2
            with self.assertRaisesRegex(ValueError, "does not match"):
                UncertaintyMonitor(cfg, FakePredictor(), "evaluate", root / "bad", {})
            cfg.uncertainty.rmse_floor /= 2
            cfg.evaluation.n_action_steps += 1
            with self.assertRaisesRegex(ValueError, "does not match"):
                UncertaintyMonitor(cfg, FakePredictor(), "evaluate", root / "bad2", {})


if __name__ == "__main__":
    unittest.main()
