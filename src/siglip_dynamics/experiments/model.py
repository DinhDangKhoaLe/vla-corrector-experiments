"""A shared residual MLP; ordered prefix and uncertainty heads are opt-in."""

from __future__ import annotations

import torch
from torch import nn

from siglip_dynamics.config import ModelScale, SiglipMLPConfig
from siglip_dynamics.MLP import SiglipResidualMLP

from .config import Experiment


class ExperimentalCorrector(nn.Module):
    def __init__(self, cfg: Experiment, token_dim: int, action_dim: int):
        super().__init__()
        self.uses_action_prefix = cfg.features.action_prefix
        self.horizons = tuple(cfg.model.horizons)
        self.max_horizon = max(self.horizons)
        self.guidance_horizon = self.max_horizon
        self.action_dim = action_dim
        self.token_dim = token_dim
        embed = cfg.model.action_embed_dim
        self.base = SiglipResidualMLP(SiglipMLPConfig(
            token_dim=token_dim, action_dim=action_dim, action_embed_dim=embed,
            dropout=cfg.model.dropout, scale=ModelScale.CUSTOM,
            custom_widths=tuple(cfg.model.widths)))
        self.prefix_encoder = nn.Linear(action_dim * self.max_horizon, embed) if self.uses_action_prefix else None
        self.horizon_embed = nn.Embedding(self.max_horizon + 1, embed) if len(self.horizons) > 1 else None
        self.uncertainty = (nn.Sequential(nn.Linear(token_dim + embed, cfg.model.uncertainty_hidden),
                                        nn.SiLU(), nn.Linear(cfg.model.uncertainty_hidden, 1))
                            if cfg.features.uncertainty_monitor else None)

    def conditioned(self, z, actions, horizon):
        if z.ndim == 2:
            z = z.unsqueeze(0)
            if actions.ndim == 2 and self.uses_action_prefix:
                actions = actions.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(1)
        h = torch.as_tensor(horizon, device=z.device, dtype=torch.long).reshape(-1)
        if h.numel() == 1:
            h = h.expand(z.shape[0])
        if h.shape[0] != z.shape[0] or any(int(v) not in self.horizons for v in h.tolist()):
            raise ValueError("Unsupported horizon or batch mismatch")
        if actions.shape[0] != z.shape[0] or actions.shape[-1] != self.action_dim:
            raise ValueError("Action shape does not match corrector")
        if self.uses_action_prefix:
            if actions.shape[1] < int(h.max()):
                raise ValueError("Missing future actions: cannot pad an unknown executed prefix")
            a = actions[:, :self.max_horizon]
            a = torch.nn.functional.pad(a, (0, 0, 0, self.max_horizon - a.shape[1]))
            mask = torch.arange(self.max_horizon, device=z.device)[None, :] < h[:, None]
            emb = self.prefix_encoder((a * mask[..., None]).flatten(1))
        else:
            emb = self.base.action_proj(actions[:, 0])
        if self.horizon_embed is not None:
            emb = emb + self.horizon_embed(h)
        return z, emb

    def predict(self, z, actions, horizon):
        z, emb = self.conditioned(z, actions, horizon)
        x = self.base.in_proj(torch.cat((z, emb[:, None].expand(-1, z.shape[1], -1)), dim=-1))
        for i, block in enumerate(self.base.blocks):
            x = block(x)
            if i < len(self.base.transitions):
                x = self.base.transitions[i](x)
        delta = self.base.out_proj(self.base.out_norm(x))
        return delta, self.error_head(z, emb)

    def error_head(self, z, emb):
        if self.uncertainty is None:
            return None
        # Error-head loss must not change the predictive representation or OGG gradient.
        features = torch.cat((z.mean(1), emb), dim=-1).detach()
        return self.uncertainty(features).squeeze(-1).clamp(-12, 8)

    def forward(self, z, actions):
        squeeze = z.ndim == 2
        if squeeze and actions.ndim == 1:
            actions = actions.unsqueeze(0)
        pred, _ = self.predict(z, actions, self.guidance_horizon)
        return pred[0] if squeeze else pred
