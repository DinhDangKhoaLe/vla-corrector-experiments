from __future__ import annotations

import numpy as np
import torch

from laco.calibrate import conformal_quantile, filtered_errors
from laco.controller import LACoController, Mode
from laco.model import LatentJacobianModel
from laco.utils import episode_split


def _controller(threshold: float = 0.01) -> LACoController:
    model = LatentJacobianModel(latent_dim=3, action_dim=3, hidden_dim=2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Flattened [latent, action] identity Jacobian.
        model.jacobian_head.bias.copy_(torch.eye(3).reshape(-1))
    return LACoController([model], {"latent_mean": np.zeros(3), "latent_std": np.ones(3),
                                    "action_mean": np.zeros(3), "action_std": np.ones(3)},
                          {"threshold": threshold, "ewma_alpha": 1.0}, device=torch.device("cpu"),
                          regularization=1e-5, trust_region=1.0)


def test_jacobian_shapes_and_corrective_sign():
    controller = _controller()
    controller.reset(np.zeros(3)); controller.on_new_chunk(np.zeros(3), np.array([[0., 0., 0., .5], [0., 0., 0., .5]], np.float32))
    controller.mark_executed()
    # Observation has moved negative X; positive X correction moves it to nominal.
    decision = controller.step(np.array([-.2, 0., 0.]))
    assert decision.mode is Mode.ADJUST
    assert decision.action_queue is not None and decision.action_queue[0, 0] > 0
    assert decision.action_queue[0, 3] == .5  # gripper is preserved


def test_episode_splits_and_conformal_calibration():
    labels = episode_split(np.array([1, 1, 2, 2, 3, 3]), 42, .2, .2)
    assert labels[0] == labels[1] and labels[2] == labels[3]
    threshold, rank = conformal_quantile(np.array([1., 2., 3.]), .05)
    assert threshold == 3. and rank == 3
    np.testing.assert_allclose(filtered_errors(np.array([1., 3., 10.]), np.array([1, 1, 2]), .5), [1., 2., 10.])


def test_failed_correction_replans():
    controller = _controller(); controller.max_failed_retries = 0
    controller.reset(np.zeros(3)); controller.on_new_chunk(np.zeros(3), np.array([[0., 0., 0., 0.], [0., 0., 0., 0.]], np.float32))
    controller.mark_executed(); first = controller.step(np.array([-.2, 0., 0.]))
    assert first.mode is Mode.ADJUST
    # Simulate a following action that did not lower the residual.
    controller.mark_executed(); second = controller.step(np.array([-.2, 0., 0.]))
    assert second.mode is Mode.REPLAN


def test_replan_mode_triggers_before_adjustment_solver():
    controller = _controller()
    controller.intervention = "replan"
    controller.on_new_chunk(np.zeros(3), np.zeros((3, 4), dtype=np.float32))
    controller.mark_executed()
    # This mode must not require Jacobians or try a correction on a detected alarm.
    controller._components = lambda z: (_ for _ in ()).throw(AssertionError("adjustment attempted"))
    decision = controller.step(np.array([-.2, 0, 0]))
    assert decision.mode is Mode.REPLAN
    assert decision.reason == "error_triggered_replan"
    assert len(controller.queue) == 0


def test_libero_adjustment_preserves_rotation_and_gripper():
    controller = _controller()
    actions = np.tile([0., 0., 0., .2, .3, .4, .5], (3, 1)).astype(np.float32)
    controller.on_new_chunk(np.zeros(3), actions)
    controller.mark_executed()
    decision = controller.step(np.array([-.2, 0., 0.]))
    assert decision.mode is Mode.ADJUST
    assert decision.action_queue.shape == (2, 7)
    np.testing.assert_array_equal(decision.action_queue[:, 3:], actions[1:, 3:])


def test_libero_extractor_configuration_uses_both_cameras():
    from laco.extract import _args_for_dataset
    from laco.utils import load_config
    config = load_config("configs/laco/libero_pi05.yaml")
    args = _args_for_dataset(config["extract"])
    assert args.dataset_format == "libero"
    assert args.image_key == "observation.images.image"
    assert args.wrist_image_key == "observation.images.image2"
    assert args.use_wrist_image
