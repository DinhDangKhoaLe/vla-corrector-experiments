"""AAC entropy elbow and movement floor, adapted to MetaWorld XYZ + gripper.

Based on Adaptive-Action-Chunking/libero-pro action_entropy_pi05.py.
There is no rotation component in MetaWorld's four-dimensional action space.
"""
import math

import torch

from .types import Decision


def entropy_profile(normalized, decoded, covariance_epsilon=1e-6):
    if normalized.ndim != 3 or normalized.shape[0] < 2 or normalized.shape[-1] < 4:
        raise ValueError("AAC requires at least two [sample, time, XYZ+gripper] chunks")
    xyz = normalized[..., :3].double()
    centered = xyz - xyz.mean(0)
    covariance = torch.einsum("nti,ntj->tij", centered, centered) / (len(xyz) - 1)
    covariance += covariance_epsilon * torch.eye(3, device=xyz.device)
    position_entropy = 0.5 * (3 * math.log(2 * math.pi * math.e) + torch.linalg.slogdet(covariance)[1])
    p = (decoded[..., 3].to(xyz.device) >= 0).double().mean(0).clamp(1e-9, 1 - 1e-9)
    return position_entropy - p * p.log() - (1 - p) * (1 - p).log()


def select_horizon(normalized, decoded, movement_threshold=0.01, delta_scale=0.01,
                   gripper_weight=0.2, min_horizon=2, covariance_epsilon=1e-6):
    entropy = entropy_profile(normalized, decoded, covariance_epsilon)
    n = len(entropy)
    if n < 2 or not 1 <= min_horizon <= n:
        raise ValueError("Invalid AAC minimum horizon")
    prefix_mean = entropy.cumsum(0) / torch.arange(1, n + 1, device=entropy.device)
    elbow = max(min_horizon, int(torch.diff(prefix_mean).argmax()) + 1)
    # Official movement floor uses the first sample; delta control is clipped by
    # MetaWorld before multiplying XYZ by its 0.01 action scale.
    xyz = decoded[0, :, :3].clamp(-1, 1) * delta_scale
    displacement = xyz.cumsum(0).norm(dim=-1)
    grip = decoded[0, :, 3] >= 0
    toggled = torch.cat([grip.new_zeros(1), grip[1:] != grip[:-1]]).long().cummax(0).values
    movement = displacement + gripper_weight * toggled
    candidates = torch.where(movement[1:] > movement_threshold)[0]
    floor = int(candidates[0]) + 2 if len(candidates) else n
    return min(n, max(elbow, floor)), entropy


class AAC:
    def __init__(self, settings):
        self.settings = settings

    def decide(self, sampler, task, **kwargs):
        samples = [sampler.sample() for _ in range(self.settings.get("samples", 20))]
        normalized = torch.stack([p.normalized for p in samples])
        decoded = torch.stack([p.actions for p in samples])
        horizon, entropy = select_horizon(normalized, decoded, **{
            k: v for k, v in self.settings.items() if k != "samples"
        })
        # Author code executes the sampled trajectory nearest the sample mean.
        flat = normalized[..., :4].flatten(1)
        index = int((flat - flat.mean(0)).norm(dim=-1).argmin())
        return Decision(decoded[index, :horizon], {
            "sample": index, "entropy": entropy.cpu().tolist(), "samples": len(samples)
        })
