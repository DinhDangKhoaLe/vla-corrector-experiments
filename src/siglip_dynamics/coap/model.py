"""Independent, order-aware observation/action towers and a masked CLIP objective."""

import torch
from torch import nn
from torch.nn import functional


class SequenceTower(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.GELU()
        )
        self.sequence = nn.GRU(
            cfg.hidden_dim,
            cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, cfg.embedding_dim)
        )

    def forward(self, sequence):
        encoded, _ = self.sequence(self.input(sequence))
        return functional.normalize(self.projection(encoded[:, -1]), dim=-1)


class COAP(nn.Module):
    def __init__(self, observation_dim, action_dim, cfg):
        super().__init__()
        self.window_size = cfg.window_size
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.observation_tower = SequenceTower(observation_dim, cfg)
        self.action_tower = SequenceTower(action_dim, cfg)

    def forward(self, observations, actions):
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("COAP expects [batch, window, features] for both towers")
        if observations.shape[:2] != actions.shape[:2] or observations.shape[1] != self.window_size:
            raise ValueError("COAP windows must contain equal, complete ordered pairs")
        if observations.shape[-1] != self.observation_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("COAP feature dimensions differ from training")
        return self.observation_tower(observations), self.action_tower(actions)

    def similarity(self, observations, actions):
        obs, act = self(observations, actions)
        return (obs * act).sum(-1).clamp(-1, 1)


def eligible_pairs(batch, window_size, device=None):
    """Diagonal positives; exclude overlap and exact duplicate inputs in either tower."""
    episode = batch["episode"].to(device)
    start = batch["start"].to(device)
    overlap = (episode[:, None] == episode[None, :]) & ((start[:, None] - start[None, :]).abs() < window_size)
    allowed = ~overlap
    for key in ("observation_hash", "action_hash"):
        fingerprints = batch[key]
        duplicate = torch.tensor([[a == b for b in fingerprints] for a in fingerprints], device=device)
        allowed &= ~duplicate
    allowed.fill_diagonal_(True)
    return allowed


def contrastive_loss(observations, actions, allowed, temperature):
    """Symmetric in-batch cross entropy; anchors with no eligible negatives contribute nothing."""
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("Contrastive temperature must be positive and finite")
    count = observations.shape[0]
    if allowed.shape != (count, count) or not torch.equal(allowed, allowed.T) or not allowed.diagonal().all():
        raise ValueError("Eligible-pair mask must be symmetric and include the positive diagonal")
    usable = allowed.sum(1) > 1
    if not usable.any():
        return None
    logits = (observations @ actions.T / temperature).masked_fill(~allowed, -torch.inf)
    labels = torch.arange(count, device=logits.device)[usable]
    return (
        functional.cross_entropy(logits[usable], labels) + functional.cross_entropy(logits.T[usable], labels)
    ) / 2
