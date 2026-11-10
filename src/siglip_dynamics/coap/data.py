"""Reuse frozen frame caches; split episodes before windows and fit train-only actions."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from siglip_dynamics.experiments.artifacts import file_digest
from siglip_dynamics.experiments.data import episode_split

TIMING = "observation_before_dispatched_action;score_after_action_returns"


class FrameCache:
    def __init__(self, path, raw_actions=""):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        self.q = np.load(self.path / "z_q.npy", mmap_mode="r", allow_pickle=False)
        self.scale = np.load(self.path / "z_scale.npy", mmap_mode="r", allow_pickle=False)
        self.episodes = np.load(self.path / "episode_index.npy", mmap_mode="r", allow_pickle=False)
        raw = Path(raw_actions) if raw_actions else self.path / "coap_raw_actions.npy"
        if not raw_actions and self.metadata.get("action_representation") == "raw_action":
            raw = self.path / "actions.npy"
        if raw.resolve() != (self.path / "actions.npy").resolve():
            provenance = (
                json.loads(raw.with_suffix(".json").read_text()) if raw.with_suffix(".json").exists() else {}
            )
            if (
                provenance.get("format") != "coap_raw_actions_v1"
                or provenance.get("timing") != TIMING
                or provenance.get("episode_index_sha256") != file_digest(self.path / "episode_index.npy")
            ):
                raise ValueError("COAP requires aligned raw actions; run --mode prepare-actions first")
            if provenance.get("actions_sha256") != file_digest(raw):
                raise ValueError("COAP raw-action sidecar checksum differs")
        elif self.metadata.get("action_representation") != "raw_action":
            raise ValueError(
                "COAP cannot train on globally normalized/clipped actions; export raw actions first"
            )
        self.actions = np.load(raw, mmap_mode="r", allow_pickle=False)
        n = len(self.episodes)
        if (
            self.q.ndim != 3
            or self.scale.shape != self.q.shape[:2]
            or self.q.shape[0] != n
            or self.actions.ndim != 2
            or len(self.actions) != n
            or self.episodes.ndim != 1
            or not np.issubdtype(self.episodes.dtype, np.integer)
        ):
            raise ValueError("COAP cache arrays must align frame by frame")
        self.ranges = {}
        start = 0
        while start < n:
            ep = int(self.episodes[start])
            stop = start + 1
            while stop < n and self.episodes[stop] == ep:
                stop += 1
            if ep in self.ranges:
                raise ValueError("Each episode must occupy one contiguous frame range")
            self.ranges[ep] = (start, stop)
            start = stop

    def observations(self, start, stop):
        # Same per-frame spatial mean is used online. No temporal pooling here.
        frames = self.q[start:stop].astype(np.float32) * self.scale[start:stop].astype(np.float32)[..., None]
        return frames.mean(axis=1)

    def split(self, training):
        return episode_split(
            self.episodes, training.val_fraction, training.test_fraction, training.split_seed
        )

    def action_normalization(self, train_episodes, floor):
        count = 0
        total = np.zeros(self.actions.shape[1], np.float64)
        squares = np.zeros_like(total)
        for episode in train_episodes:
            start, stop = self.ranges[episode]
            values = np.asarray(self.actions[start:stop], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite training actions")
            count += len(values)
            total += values.sum(0)
            squares += (values * values).sum(0)
        if not count:
            raise ValueError("No training actions")
        average = total / count
        scale = np.maximum(np.sqrt(np.maximum(squares / count - average * average, 0)), floor)
        return {
            "mean": average.tolist(),
            "std": scale.tolist(),
            "fit_episodes": list(train_episodes),
            "fit_frames": count,
            "source": "training episodes only; raw dispatched-action coordinates",
        }


class WindowDataset(Dataset):
    def __init__(self, cache, episodes, window_size, normalization):
        self.cache, self.window_size, self.normalization = cache, window_size, normalization
        self.windows = []
        for ep in episodes:
            start, stop = cache.ranges[ep]
            self.windows.extend((ep, t) for t in range(start, stop - window_size + 1))
        if not self.windows:
            raise ValueError("No complete COAP windows in this split")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        stop = start + self.window_size
        obs = np.ascontiguousarray(self.cache.observations(start, stop), dtype=np.float32)
        raw = np.asarray(self.cache.actions[start:stop], dtype=np.float32)
        act = np.ascontiguousarray(
            (raw - np.array(self.normalization["mean"])) / np.array(self.normalization["std"]),
            dtype=np.float32,
        )
        if not np.isfinite(obs).all() or not np.isfinite(act).all():
            raise ValueError("Non-finite COAP input window")
        return {
            "observations": torch.from_numpy(obs),
            "actions": torch.from_numpy(act),
            "episode": episode,
            "start": start,
            "observation_hash": hashlib.sha256(obs.tobytes()).hexdigest(),
            "action_hash": hashlib.sha256(raw.tobytes()).hexdigest(),
        }


class EpisodeBatchSampler(Sampler):
    """At most one window per episode per batch; extend here for task-aware sampling."""

    def __init__(self, dataset, batch_size, seed):
        self.groups = defaultdict(list)
        for i, (episode, _) in enumerate(dataset.windows):
            self.groups[episode].append(i)
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        if len(self.groups) < 2:
            raise ValueError("COAP contrastive batches need at least two episodes per split")

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {ep: rng.permutation(ids).tolist() for ep, ids in self.groups.items()}
        while len(pools) >= 2:
            episodes = rng.permutation(list(pools))[: self.batch_size]
            yield [pools[ep].pop() for ep in episodes]
            pools = {ep: ids for ep, ids in pools.items() if ids}


def prepare_actions(cache_path, dataset_path):
    """Read raw parquet columns in the extractor's order, without decoding images."""
    from datasets import load_dataset

    cache = Path(cache_path)
    target = cache / "coap_raw_actions.npy"
    if target.exists() or target.with_suffix(".json").exists():
        raise FileExistsError(f"Raw action sidecar already exists: {target}")
    metadata = json.loads((cache / "metadata.json").read_text())
    if metadata.get("dataset_format") != "metaworld":
        raise ValueError("Initial raw-action adapter supports the local MetaWorld parquet cache only")
    source = Path(dataset_path)
    source = source / "data" if (source / "data").exists() else source
    dataset = load_dataset("parquet", data_dir=str(source), split="train")
    columns = [metadata.get("action_key", "action"), "episode_index", "frame_index"]
    rows = dataset.select_columns(columns)
    episodes = np.load(cache / "episode_index.npy", mmap_mode="r")
    if len(rows) != len(episodes):
        raise ValueError("Raw dataset and cache frame counts differ; cannot align actions")
    actions = np.asarray(rows[columns[0]], dtype=np.float32)
    ids = np.asarray(rows["episode_index"])
    frames = np.asarray(rows["frame_index"])
    if not np.array_equal(ids, episodes) or not np.isfinite(actions).all():
        raise ValueError("Raw action episode order differs from the frozen feature cache")
    starts = np.r_[True, ids[1:] != ids[:-1]]
    if not np.all(frames[starts] == 0) or not np.all(np.diff(frames)[~starts[1:]] == 1):
        raise ValueError("Raw frame_index must be contiguous within each episode")
    if actions.shape != (len(episodes), int(metadata["action_dim"])):
        raise ValueError("Raw action shape differs from the cache")
    np.save(target, actions)
    provenance = {
        "format": "coap_raw_actions_v1",
        "dataset": str(dataset_path),
        "episode_index_sha256": file_digest(cache / "episode_index.npy"),
        "actions_sha256": file_digest(target),
        "timing": TIMING,
        "alignment": "same parquet loader/order as extraction; checked episode_index and contiguous frame_index",
    }
    target.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
