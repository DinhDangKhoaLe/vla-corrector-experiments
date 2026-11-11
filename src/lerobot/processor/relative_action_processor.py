#!/usr/bin/env python

"""Compatibility processors for PI0.5 checkpoints saved with action-relativity steps.

Some PI0.5 checkpoints shipped with processor JSON files that include
``relative_actions_processor`` before normalization and
``absolute_actions_processor`` after unnormalization.  The RoboTwin checkpoint
used by this project sets both steps to ``enabled: false``.  Newer local
LeRobot sources no longer register the historical names, which prevents an
otherwise compatible checkpoint from loading at all.

These steps deliberately preserve the saved disabled behaviour.  Enabling a
relative/absolute transform would require the checkpoint's original action
schema, so it fails explicitly rather than silently changing robot actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition

from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("relative_actions_processor")
@dataclass
class RelativeActionsProcessorStep(ProcessorStep):
    """Load a legacy no-op relative-action processor when it was disabled."""

    enabled: bool = False
    exclude_joints: list[str] = field(default_factory=list)
    action_names: list[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.enabled:
            raise NotImplementedError(
                "This checkpoint requests enabled relative action processing, but its action "
                "schema is not available in this VLA-Corrector compatibility layer."
            )
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("absolute_actions_processor")
@dataclass
class AbsoluteActionsProcessorStep(ProcessorStep):
    """Load a legacy no-op absolute-action processor when it was disabled."""

    enabled: bool = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if self.enabled:
            raise NotImplementedError(
                "This checkpoint requests enabled absolute action processing, but its action "
                "schema is not available in this VLA-Corrector compatibility layer."
            )
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
