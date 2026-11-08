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
