import unittest

import numpy as np

from siglip_dynamics.pace import PaceController, PaceSettings, choose_horizon, speed_profile


class PaceTest(unittest.TestCase):
    def setUp(self):
        self.settings = PaceSettings(
            max_horizon=10,
            motion_dims=(0,),
            smooth_window=1,
            prominence_window=2,
            min_separation=2,
            percentile=5.0,
        )

    def test_earliest_prominent_valley_is_selected(self):
        # Speeds: 3, 1, .1, 3.9, ...; the first qualifying valley is action 3.
        actions = np.array([[0], [3], [4], [4.1], [8], [9], [10], [11], [12], [13]], dtype=np.float32)
        horizon, details = choose_horizon(actions, threshold=0.5, settings=self.settings)
        self.assertEqual(horizon, 3)
        self.assertEqual(details["reason"], "valley")

    def test_no_qualified_valley_uses_max_horizon(self):
        actions = np.arange(10, dtype=np.float32)[:, None]
        horizon, details = choose_horizon(actions, threshold=1.0, settings=self.settings)
        self.assertEqual(horizon, 10)
        self.assertEqual(details["reason"], "fallback_max_horizon")

    def test_controller_uses_task_threshold_then_global_fallback(self):
        controller = PaceController(
            {
                "format": "pace_style_v1",
                "settings": {
                    "max_horizon": 10,
                    "motion_dims": [0],
                    "smooth_window": 1,
                    "prominence_window": 2,
                    "min_separation": 2,
                    "percentile": 5.0,
                },
                "global_threshold": 100.0,
                "tasks": {"6": {"task_name": "Press a button", "threshold": 0.5}},
            }
        )
        actions = np.array([[0], [3], [4], [4.1], [8], [9], [10], [11], [12], [13]], dtype=np.float32)
        selected, _ = controller.select("Press a button", actions)
        fallback, _ = controller.select("unknown", actions)
        self.assertEqual(selected, 3)
        self.assertEqual(fallback, 10)

    def test_speed_profile_has_one_value_per_action_transition(self):
        profile = speed_profile(np.zeros((10, 4), dtype=np.float32), self.settings)
        self.assertEqual(profile.shape, (9,))


if __name__ == "__main__":
    unittest.main()
