"""Paper-based BCP head and GRPO math (official code has not been released).

Equations 2-8 and Table A.2 of arXiv:2608.03483v1. Architecture dimensions
not specified in the paper are exposed in configuration.
"""
import math

import torch
from torch import nn
from torch.distributions import Categorical

from .types import Decision


class ContinuationHead(nn.Module):
    def __init__(self, horizons, context_dim=2048, velocity_dim=1024, action_dim=32,
                 width=256, layers=2, heads=8):
        super().__init__()
        if list(horizons) != sorted(set(horizons)) or len(horizons) < 2 or horizons[0] < 1 or horizons[-1] > 50:
            raise ValueError("BCP horizons must increase strictly within [1,50]")
        self.horizons = tuple(horizons)
        self.context_proj = nn.Linear(context_dim, width)
        self.action_proj = nn.Linear(action_dim + velocity_dim, width)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position = nn.Parameter(torch.randn(1, 50, width) * 0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, width * 4, dropout=0.0,
                                       activation="gelu", batch_first=True),
            layers, enable_nested_tensor=False,
        )
        self.output = nn.Linear(width, len(horizons) - 1)

    def forward(self, context, actions, velocity_features):
        if context.ndim == 2:
            context, actions, velocity_features = context[None], actions[None], velocity_features[None]
        context = self.context_proj(context.float())
        action = self.action_proj(torch.cat([velocity_features.float(), actions.float()], -1))
        action = action + self.position[:, :action.shape[1]]
        seq = torch.cat([self.cls.expand(len(context), -1, -1), context, action], dim=1)
        logits = self.output(self.encoder(seq)[:, 0])
        return continuation_log_probs(logits)


def continuation_log_probs(logits):
    """Stable log probability of first stop, plus the all-continue outcome."""
    log_continue = torch.nn.functional.logsigmoid(logits)
    log_stop = torch.nn.functional.logsigmoid(-logits)
    prefixes = torch.cat([logits.new_zeros(*logits.shape[:-1], 1), log_continue.cumsum(-1)], -1)
    return torch.cat([prefixes[..., :-1] + log_stop, prefixes[..., -1:]], -1)


def efficiency_reward(success, calls, reference_calls, delta_plus=0.7, delta_minus=0.3):
    if calls < 1 or reference_calls < 1:
        raise ValueError("BCP rewards require at least one VLA query")
    eta = math.tanh(math.log(reference_calls / calls))
    return float(success) * (1 + delta_plus * max(eta, 0) + delta_minus * min(eta, 0))


def clipped_objective(logp, old_logp, advantage, entropy, clip_low=0.2, clip_high=0.28, entropy_bonus=0.05):
    ratio = (logp - old_logp).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_low, 1 + clip_high) * advantage) - entropy_bonus * entropy


class BCP:
    def __init__(self, head, stochastic=False):
        self.head = head
        self.stochastic = stochastic

    def decide(self, sampler, task, generator=None, **kwargs):
        prediction = sampler.sample(features=True)
        device = next(self.head.parameters()).device
        inputs = {"context": prediction.context.detach().to(device),
                  "actions": prediction.normalized.detach().to(device),
                  "velocity_features": prediction.velocity_features.detach().to(device)}
        with torch.no_grad():
            log_probs = self.head(**inputs)[0]
            index = int(torch.multinomial(log_probs.exp(), 1, generator=generator)) if self.stochastic else int(log_probs.argmax())
        horizon = self.head.horizons[index]
        training = {"inputs": {k: v.cpu() for k, v in inputs.items()},
                    "index": index, "old_logp": float(log_probs[index])} if self.stochastic else None
        return Decision(prediction.actions[:horizon], {"probabilities": log_probs.exp().cpu().tolist()}, training)
