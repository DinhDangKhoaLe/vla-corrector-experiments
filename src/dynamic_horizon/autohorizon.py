"""AutoHorizon bidirectional soft pointer, following the official PyTorch code.

Source: hatchetProject/AutoHorizon, c7504f1, pi0_pytorch.py.
See docs/dynamic_horizon_sources.md for extraction and adaptation details.
"""
import math

import torch
import torch.nn.functional as F

from .types import Decision


def soft_pointer(attention, hold_threshold=0.3, entropy_quantile=0.9, run_length=1):
    n = len(attention)
    a = attention.float() / attention.float().sum(-1, keepdim=True).clamp_min(1e-12)
    mu = (a * torch.arange(n, device=a.device)).sum(-1).cummax(0).values
    entropy = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1) / math.log(n)
    reliable = entropy <= torch.quantile(entropy, min(entropy_quantile, 0.999))
    delta = torch.diff(mu, prepend=mu.new_zeros(1))
    holds = ((delta < hold_threshold) & reliable).float()
    runs = F.conv1d(holds[None, None], holds.new_ones(1, 1, run_length)).flatten()
    indices = torch.where(runs >= run_length)[0]
    stop = int(indices[0]) if len(indices) else n - 1
    return mu, stop


def select_horizon(attention, hold_threshold=0.3, entropy_quantile=0.9, run_length=1):
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError("AutoHorizon needs a square action self-attention matrix")
    if not torch.isfinite(attention).all() or (attention < 0).any():
        raise ValueError("Invalid attention probabilities")
    n = len(attention)
    if n == 1:
        return 1
    if not 1 <= run_length <= n or not 0 < entropy_quantile <= 1:
        raise ValueError("Invalid AutoHorizon run length / entropy quantile")
    forward, sf = soft_pointer(attention, hold_threshold, entropy_quantile, run_length)
    reverse, sb = soft_pointer(attention.flip((0, 1)), hold_threshold, entropy_quantile, run_length)
    backward = (n - 1 - reverse).flip((0,))
    nf = int((forward[sf].floor() + 1).clamp(1, n))
    nb = int((backward[n - 1 - sb].floor() + 1).clamp(1, n))
    return n if nf + nb >= n and ((backward - forward) <= 1).any() else nf


class AutoHorizon:
    def __init__(self, settings):
        self.settings = settings

    def decide(self, sampler, task, **kwargs):
        prediction = sampler.sample(attention_step=self.settings.get("attention_step", 3))
        horizon = select_horizon(prediction.attention, **{
            k: v for k, v in self.settings.items() if k != "attention_step"
        })
        return Decision(prediction.actions[:horizon], {"signal": "bidirectional_self_attention"})
