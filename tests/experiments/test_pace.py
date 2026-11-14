from dataclasses import asdict

import numpy as np
import pytest

from siglip_dynamics.pace import (
    PaceController, PaceSettings, calibration_contrasts, choose_horizon, speed_profile,
)


def test_joint_index_matches_official_convention():
    # Speeds [0, 2, 4, 2, .1, 2, 4, 2, 0]. Contrast from start = 3.9.
    positions = np.cumsum([0, 0, 2, 4, 2, .1, 2, 4, 2, 0])[:, None]
    settings = PaceSettings(max_horizon=10, motion_dims=(0,), signal="joint", fps=1,
                            smooth_window=1, polynomial_order=0, min_separation=2)
    horizon, _ = choose_horizon(positions, .5, settings)
    assert horizon == 4  # Official index convention: no +1.
    assert speed_profile(positions, settings).shape == (9,)


def test_delta_motion_is_not_differenced_and_gripper_is_excluded():
    actions = np.zeros((50, 4))
    actions[:, 0] = 1
    actions[:, 3] = np.sin(np.arange(50))
    settings = PaceSettings()
    np.testing.assert_allclose(speed_profile(actions, settings), 1)
    assert choose_horizon(actions, .1, settings)[0] == 50


def test_multiarms_use_earliest_boundary():
    speed = np.ones((50, 2))
    speed[:, 0] = 2 + np.cos(np.arange(50) * 2*np.pi/20)
    speed[:7, 1] = [1, 2, 3, 2, 1, .1, 2]
    settings = PaceSettings(signal="delta", arm_groups=((0,), (1,)), smooth_window=1,
                            polynomial_order=0, min_separation=2)
    assert choose_horizon(speed, .5, settings)[0] == 5


def test_calibration_full_episodes_and_short_skip():
    speed = (2 + np.cos(np.arange(100) * 2*np.pi/20))[:, None]
    settings = PaceSettings(motion_dims=(0,), smooth_window=1, polynomial_order=0, min_separation=10)
    scores, analyzed, skipped = calibration_contrasts([speed, speed[:5]], settings)
    assert scores and min(scores) > 0
    assert (analyzed, skipped) == (1, 1)


def test_no_silent_missing_task_or_old_artifact():
    with pytest.raises(ValueError, match="Recalibrate"):
        PaceController({"format": "pace_style_v1"})
    controller = PaceController({"format": "pace_v2", "settings": asdict(PaceSettings()),
                                 "tasks": {"task": {"task_name": "task", "threshold": .1}}})
    with pytest.raises(KeyError):
        controller.select("missing", np.ones((50, 4)))


@pytest.mark.parametrize("cap", [50, 40, 30, 20, 10])
def test_controller_reuses_calibration_with_smaller_execution_cap(cap):
    controller = PaceController({"format": "pace_v2", "settings": asdict(PaceSettings()),
                                 "tasks": {"task": {"task_name": "task", "threshold": .1}}})
    # Constant speed has no valid valley: fallback must obey the runtime cap.
    assert controller.select("task", np.ones((50, 4)), max_horizon=cap)[0] == cap
    assert controller.settings.max_horizon == 50
