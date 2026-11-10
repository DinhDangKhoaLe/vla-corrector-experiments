"""Random baseline reproducibility, causal timing, isolation, and CLI validation."""

import contextlib
import io
import math
import random
import tempfile
import unittest
from collections import deque
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from siglip_dynamics.random_replan import (
    Config,
    RandomReplanController,
    RandomSettings,
    after_step,
    eval_command,
    load_config,
    main,
    summarize,
)


class RandomReplanTests(unittest.TestCase):
    def test_warmup_cooldown_boundary_and_terminal_only_count_real_interrupts(self):
        controller = RandomReplanController(RandomSettings(probability=1))
        controller.reset("easy:0", 1000)
        events = [controller.after_execution(step, 3) for step in range(1, 30)]
        self.assertEqual([e["completed_step"] for e in events if e["trigger"]], [6, 17, 28])
        for step in range(1, 6):
            self.assertTrue(events[step - 1]["warmup"])
            self.assertIsNone(events[step - 1]["draw"])
        controller.reset("easy:0", 1000)
        for step in range(1, 6):
            controller.after_execution(step, 1)
        self.assertFalse(controller.after_execution(6, 0)["trigger"])
        self.assertFalse(controller.after_execution(7, 3, done=True)["trigger"])
        self.assertTrue(controller.after_execution(8, 3)["trigger"])
        with self.assertRaises(ValueError):
            controller.after_execution(10, 3)

    def test_only_unexecuted_queues_are_discarded_and_zero_rate_is_noop(self):
        for probability in (0, 1):
            controller = RandomReplanController(RandomSettings(probability=probability, warmup_steps=0))
            actions, raw = deque(["next", "future"]), deque([1, 2])
            event = after_step(controller, 1, actions, raw)
            self.assertEqual(event["discarded_actions"], 2 if probability else 0)
            self.assertEqual(list(actions), [] if probability else ["next", "future"])
            self.assertEqual(list(raw), [] if probability else [1, 2])

    def test_reproducible_episode_stream_without_changing_global_rngs(self):
        settings = RandomSettings(probability=0.5, warmup_steps=0, cooldown_steps=0)
        controller = RandomReplanController(settings)
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()

        def episode(task, seed):
            controller.reset(task, seed)
            return [controller.after_execution(step, 3) for step in range(1, 101)]

        first = episode("easy:0", 1000)
        other = episode("hard:0", 1000)
        self.assertEqual(first, episode("easy:0", 1000))
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, episode("easy:0", 1001))
        second = RandomReplanController(
            RandomSettings(probability=0.5, seed=43, warmup_steps=0, cooldown_steps=0)
        )
        second.reset("easy:0", 1000)
        self.assertNotEqual(first, [second.after_execution(step, 3) for step in range(1, 101)])
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())

    def test_summary_preserves_groups_replans_and_rollout_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            controller = RandomReplanController(RandomSettings(probability=1, warmup_steps=0), tmp)
            for task, success in [("easy:0", True), ("hard:0", False)]:
                controller.reset(task, 1000)
                controller.after_execution(1, 3)
                controller.finish(success, vla_calls=2, wall_seconds=1)
            p = Path(tmp) / "random_replan_episodes.jsonl"
            result = summarize(p)
            self.assertEqual(result["success_rate"], 0.5)
            self.assertEqual(result["interventions"], 2)
            self.assertEqual(result["vla_calls"], 4)
            self.assertEqual(result["groups"]["easy"]["success_rate"], 1)
            self.assertEqual(result["groups"]["hard"]["success_rate"], 0)
            controller.reset("hard:1", 1000)
            controller.finish(False, error="test failure")
            with self.assertRaisesRegex(ValueError, "rollout errors"):
                summarize(p)

    def test_config_dry_run_and_commands_need_no_cache_or_detector(self):
        cfg = load_config("configs/baselines/metaworld_random_replan.yaml")
        command = eval_command(cfg, "resolved.yaml")
        self.assertIn("--random_replan_config=resolved.yaml", command)
        self.assertIn("--meltdown_use_guidance_replan=false", command)
        self.assertIn("--policy.n_action_steps=10", command)
        self.assertIn("--seed=1000", command)
        self.assertFalse(any("coap_config" in v or "safety_model_path" in v for v in command))
        with (
            patch("siglip_dynamics.random_replan.subprocess.run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(["--config", "configs/baselines/metaworld_random_replan.yaml", "--dry-run"])
            run.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cfg.yaml"
            p.write_text(yaml.safe_dump(asdict(Config())))
            for value in (-0.1, 1.1, math.nan, True):
                raw = asdict(Config())
                raw["random"]["probability"] = value
                p.write_text(yaml.safe_dump(raw))
                with self.assertRaises(ValueError):
                    load_config(p)
            p.write_text(yaml.safe_dump(asdict(Config())))
            with self.assertRaises(ValueError):
                load_config(p, ["random.unknown=1"])
            with self.assertRaises(ValueError):
                load_config(p, ["random.seed=-1"])
            with self.assertRaises(ValueError):
                load_config(p, ["evaluation.n_action_steps=51"])
            for probability in (0, 1):
                self.assertEqual(
                    load_config(p, [f"random.probability={probability}"]).random.probability, probability
                )
            raw = asdict(Config())
            raw["evaluation"]["output"] = tmp
            p.write_text(yaml.safe_dump(raw))
            with self.assertRaises(FileExistsError):
                main(["--config", str(p)])


if __name__ == "__main__":
    unittest.main()
