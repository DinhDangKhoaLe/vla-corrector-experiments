"""Episode-disjoint windows with exact temporal alignment of action prefixes."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from siglip_dynamics.dataset import SiglipDynamicsDataset


def episode_split(episode_ids, val_fraction, test_fraction, seed):
    ids = np.unique(episode_ids)
    if len(ids) < 3:
        raise ValueError("At least three episodes are required")
    ids = np.random.default_rng(seed).permutation(ids)
    nv = max(1, int(len(ids) * val_fraction))
    nt = max(1, int(len(ids) * test_fraction))
    if nv + nt >= len(ids):
        raise ValueError("Split leaves no training episodes")
    return {"validation": ids[:nv].tolist(), "test": ids[nv:nv + nt].tolist(),
            "train": ids[nv + nt:].tolist()}


class PrefixDataset(Dataset):
    def __init__(self, cache, horizons, episode_ids):
        self.horizons = tuple(horizons)
        self.base = SiglipDynamicsDataset(str(cache), k_step=max(horizons))
        if self.base.mode != "quantized_frames":
            raise ValueError("Experiments require the frame cache with episode_index.npy")
        allowed = set(episode_ids)
        self.starts = [t for t, _ in self.base.pairs if int(self.base.ep_idx[t]) in allowed]
        if not self.starts:
            raise ValueError("No complete windows in this episode split")

    def __len__(self):
        return len(self.starts) * len(self.horizons)

    def __getitem__(self, i):
        t = self.starts[i // len(self.horizons)]
        h = self.horizons[i % len(self.horizons)]
        b = self.base
        z = b._dequantize(b.z_q[t], b.z_scale[t])
        future = b._dequantize(b.z_q[t+h], b.z_scale[t+h])
        # Padding is masked in the model; no actions after t+h are visible.
        actions = np.zeros((max(self.horizons), b.a_t_np.shape[-1]), dtype=np.float32)
        actions[:h] = b.a_t_np[t:t+h]
        return {"z": torch.from_numpy(z), "delta": torch.from_numpy(future-z),
                "actions": torch.from_numpy(actions), "horizon": h,
                "episode": int(b.ep_idx[t])}
