"""Episode-safe MetaWorld transition extraction and train-only PCA fitting."""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from siglip_dynamics.extract import (
    Pi05Extractor,
    _load_dataset,
    _load_action_quantiles_stats,
    _normalize_action_quantiles,
    _resolve_action_key,
    _resolve_layout,
    _to_float_array_1d,
    _to_numpy,
)

from .utils import episode_split, load_config, merge_overrides, save_json


def _args_for_dataset(c: dict[str, Any]) -> argparse.Namespace:
    """Construct the small compatibility surface expected by the proven loader."""
    return argparse.Namespace(
        dataset_path=c["dataset"], dataset_repo_id=c.get("dataset_repo_id", "lerobot/metaworld_mt50"),
        dataset_loader=c.get("dataset_loader", "auto"), allow_dataset_download=bool(c.get("allow_dataset_download", False)),
        video_tolerance_s=float(c.get("video_tolerance_s", 1e-4)), dataset_format="metaworld",
        image_key=c.get("image_key", "observation.image"), wrist_image_key="", use_wrist_image=False,
        extra_image_keys="", action_key=c.get("action_key", "action"),
    )


def _episode_and_task(item: dict[str, Any], fallback_episode: int) -> tuple[int, int]:
    episode = int(_to_numpy(item.get("episode_index", fallback_episode)).reshape(-1)[0])
    for key in ("task_index", "task_id", "task", "episode_task_index"):
        if key in item:
            raw = _to_numpy(item[key]).reshape(-1)[0]
            try:
                return episode, int(raw)
            except (TypeError, ValueError):
                # Preserve a stable numeric id if a dataset exposes task strings.
                return episode, int(zlib.crc32(str(raw).encode("utf-8")))
    return episode, -1


def _fit_pca(extractor: Pi05Extractor, ds: Any, args: argparse.Namespace, split_by_frame: np.ndarray, dim: int, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA from training frames only, streaming an O(D²) covariance."""
    _fmt, image_key, wrist_key, extra, wrist = _resolve_layout(ds[0], args)
    count, total, total_outer = 0, None, None
    for start in range(0, len(ds), batch):
        ix = [i for i in range(start, min(start + batch, len(ds))) if split_by_frame[i] == "train"]
        if not ix:
            continue
        tokens = extractor.encode_items([ds[i] for i in ix], image_key=image_key, wrist_image_key=wrist_key,
                                        use_wrist_image=wrist, extra_image_keys=extra)
        pooled = tokens.mean(axis=1, dtype=np.float64)
        total = pooled.sum(axis=0) if total is None else total + pooled.sum(axis=0)
        total_outer = pooled.T @ pooled if total_outer is None else total_outer + pooled.T @ pooled
        count += len(pooled)
    if count < dim:
        raise ValueError(f"Need at least pca_dim={dim} training frames; got {count}")
    mean = total / count
    covariance = total_outer / count - np.outer(mean, mean)
    vals, vecs = np.linalg.eigh(covariance)
    components = vecs[:, np.argsort(vals)[::-1][:dim]].T.astype(np.float32)
    return mean.astype(np.float32), components


def run(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("extract", config)
    out = Path(c["output"])
    if out.exists() and any(out.iterdir()) and not bool(c.get("overwrite", False)):
        raise FileExistsError(f"Refusing to overwrite nonempty {out}; use extract.overwrite=true")
    out.mkdir(parents=True, exist_ok=True)
    args = _args_for_dataset(c)
    import logging
    ds, loader_kwargs = _load_dataset(args, logging.getLogger("laco.extract"))
    n = len(ds)
    if n < 2:
        raise ValueError("Dataset needs at least two frames")
    # Assign splits from all episode identifiers before fitting PCA.
    frame_episode = np.asarray([_episode_and_task(ds[i], i)[0] for i in range(n)], dtype=np.int64)
    split_by_frame = episode_split(frame_episode, int(c.get("seed", 42)), float(c.get("val_fraction", .1)), float(c.get("calibration_fraction", .1)))
    extractor = Pi05Extractor(c["policy"], device=c.get("device", "cuda"), local_files_only=bool(c.get("local_files_only", True)))
    pca_mean, components = _fit_pca(extractor, ds, args, split_by_frame, int(c.get("pca_dim", 32)), int(c.get("batch_size", 8)))
    _fmt, image_key, wrist_key, extra, wrist = _resolve_layout(ds[0], args)
    # A transition exists only when adjacent frames belong to the same episode.
    valid = np.flatnonzero(frame_episode[:-1] == frame_episode[1:])
    z_t = np.lib.format.open_memmap(out / "z_t.npy", mode="w+", dtype=np.float32, shape=(len(valid), len(components)))
    z_next = np.lib.format.open_memmap(out / "z_next.npy", mode="w+", dtype=np.float32, shape=z_t.shape)
    first_action = _to_float_array_1d(ds[0][_resolve_action_key(ds[0], args.action_key)])
    normalize_actions = bool(c.get("normalize_actions", True))
    action_stats = None
    action_stats_info: dict[str, Any] = {"mode": "raw"}
    if normalize_actions:
        action_stats, stats_source, stats_key = _load_action_quantiles_stats(
            dataset_path=str(c["dataset"]), dataset_repo_id=str(c.get("dataset_repo_id", "lerobot/metaworld_mt50")),
            action_key=_resolve_action_key(ds[0], args.action_key), action_dim=len(first_action), encoder_policy_path=str(c["policy"]),
        )
        action_stats_info = {"mode": "quantile_q01_q99_to_-1_1", "source": stats_source, "key": stats_key}
    actions = np.lib.format.open_memmap(out / "actions.npy", mode="w+", dtype=np.float32, shape=(len(valid), len(first_action)))
    episodes = np.lib.format.open_memmap(out / "episode_id.npy", mode="w+", dtype=np.int64, shape=(len(valid),))
    tasks = np.lib.format.open_memmap(out / "task_id.npy", mode="w+", dtype=np.int32, shape=(len(valid),))
    timesteps = np.lib.format.open_memmap(out / "timestep.npy", mode="w+", dtype=np.int32, shape=(len(valid),))
    labels = np.empty(len(valid), dtype="U11")
    # Encode every frame once; selecting only same-episode neighbours below prevents
    # cross-boundary transitions while avoiding a second vision pass for each successor.
    reduced = np.lib.format.open_memmap(out / "frame_latents.npy", mode="w+", dtype=np.float32, shape=(n, len(components)))
    extraction_batch = int(c.get("batch_size", 8))
    for start in range(0, n, extraction_batch):
        end = min(start + extraction_batch, n)
        tokens = extractor.encode_items([ds[i] for i in range(start, end)], image_key=image_key, wrist_image_key=wrist_key,
                                        use_wrist_image=wrist, extra_image_keys=extra)
        reduced[start:end] = (tokens.mean(axis=1) - pca_mean) @ components.T
    reduced.flush()
    for offset, frame in enumerate(valid):
        z_t[offset], z_next[offset] = reduced[frame], reduced[frame + 1]
        item = ds[int(frame)]
        action_key = _resolve_action_key(item, args.action_key)
        raw_action = _to_float_array_1d(item[action_key])
        actions[offset] = _normalize_action_quantiles(raw_action, action_stats) if action_stats is not None else raw_action
        ep, task = _episode_and_task(item, int(frame))
        episodes[offset], tasks[offset], timesteps[offset] = ep, task, int(frame)
        labels[offset] = split_by_frame[int(frame)]
    for x in (z_t, z_next, actions, episodes, tasks, timesteps):
        x.flush()
    np.save(out / "split.npy", labels)
    np.savez(out / "pca.npz", mean=pca_mean, components=components)
    metadata = {"format": "laco_transition_cache_v1", "source_dataset": c["dataset"], "dataset_loader": loader_kwargs,
                "policy": c["policy"], "action_representation": "PI05 quantile-normalized action" if action_stats is not None else "raw dataset action",
                "action_normalization": action_stats_info,
                "pca_dim": int(len(components)), "pca_fit_split": "train", "n_frames": n, "n_transitions": len(valid),
                "split_counts": {k: int((labels == k).sum()) for k in ("train", "validation", "calibration")},
                "image_key": image_key, "episode_boundary_safe": True}
    save_json(out / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(run(merge_overrides(load_config(args.config), args.set)), indent=2))


if __name__ == "__main__":
    main()
