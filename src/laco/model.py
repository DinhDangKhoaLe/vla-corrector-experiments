from __future__ import annotations

import torch
from torch import Tensor, nn


class LatentJacobianModel(nn.Module):
    """Predicts b(z) and a state-dependent local Jacobian J(z)."""

    def __init__(self, latent_dim: int = 32, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim, self.action_dim, self.hidden_dim = latent_dim, action_dim, hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.bias_head = nn.Linear(hidden_dim, latent_dim)
        self.jacobian_head = nn.Linear(hidden_dim, latent_dim * action_dim)

    def components(self, z: Tensor) -> tuple[Tensor, Tensor]:
        h = self.trunk(z)
        b = self.bias_head(h)
        jacobian = self.jacobian_head(h).view(*z.shape[:-1], self.latent_dim, self.action_dim)
        return b, jacobian

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        b, jacobian = self.components(z)
        return b + torch.einsum("...ij,...j->...i", jacobian, action)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
