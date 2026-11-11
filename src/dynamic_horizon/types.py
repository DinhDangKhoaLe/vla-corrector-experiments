from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Prediction:
    normalized: torch.Tensor  # [50, padded_action_dim], for conditional re-decoding
    actions: torch.Tensor  # [50, 4], decoded environment commands (before clipping)
    attention: torch.Tensor | None = None  # [50, 50] action self-attention
    context: torch.Tensor | None = None  # valid visual-language hidden tokens
    velocity_features: torch.Tensor | None = None  # final denoising expert features


@dataclass
class Decision:
    actions: torch.Tensor  # executable prefix, including any A3 replacement action
    diagnostics: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] | None = None

    @property
    def horizon(self):
        return len(self.actions)
