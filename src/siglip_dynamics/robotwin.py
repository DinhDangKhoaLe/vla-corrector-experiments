"""Read native RoboTwin/XPolicyLab HDF5 demonstrations for latent extraction.

RoboTwin stores RGB frames as encoded buffers in HDF5.  They must be decoded
with RoboTwin's ``decode_image_bit`` routine: JPEG-style decoding directly
silently swaps channels for part of the public corpus.  This adapter keeps the
source data read-only and exposes the small frame-dataset interface used by
``siglip_dynamics.extract``.
"""

from __future__ import annotations

import importlib.util
import os
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ACTION_PARTS = (
    "left_arm_joint_states",
    "left_ee_joint_states",
    "right_arm_joint_states",
    "right_ee_joint_states",
)


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in user environments
        raise RuntimeError(
            "RoboTwin extraction needs h5py. Install it in the active environment with "
            "`python -m pip install 'h5py>=3.10,<4'`."
        ) from exc
    return h5py


@lru_cache(maxsize=1)
def _decode_image_bit():
    """Load the vendored official decoder without treating RoboTwin as installed."""
    root_hint = os.getenv("ROBOTWIN_ROOT", "").strip()
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path(root_hint)] if root_hint else []
    candidates.append(repo_root / "third_party" / "RoboTwin")
    for root in candidates:
        decoder_path = root / "data" / "decode_image_bit.py"
        if not decoder_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("_siglip_dynamics_robotwin_decoder", decoder_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.decode_image_bit
    searched = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(
        "Cannot find RoboTwin's official image decoder. Set ROBOTWIN_ROOT or place RoboTwin at "
        f"third_party/RoboTwin. Searched: {searched}"
    )


def _read_actions(handle: Any, frame: int | None = None) -> np.ndarray:
    """Read the canonical dual-arm 14-D joint target in RoboTwin ordering."""
    action = handle["action"]
    if hasattr(action, "shape"):
        value = np.asarray(action[:] if frame is None else action[frame], dtype=np.float32)
    elif "vector" in action:
        value = np.asarray(action["vector"][:] if frame is None else action["vector"][frame], dtype=np.float32)
    else:
        missing = [name for name in ACTION_PARTS if name not in action]
        if missing:
            raise KeyError(f"RoboTwin action group missing required parts: {missing}")
        pieces = [np.asarray(action[name][:] if frame is None else action[name][frame], dtype=np.float32)
                  for name in ACTION_PARTS]
        axis = 1 if frame is None else 0
        value = np.concatenate(pieces, axis=axis)
    if value.shape[-1] != 14:
        raise ValueError(f"Expected RoboTwin dual-arm joint action dimension 14, got {value.shape}")
    return value


def _episode_length(handle: Any, path: Path) -> int:
    lengths = []
    if "action" not in handle:
        raise KeyError(f"{path}: missing action group")
    action = handle["action"]
    if hasattr(action, "shape"):
        lengths.append(int(action.shape[0]))
    elif "vector" in action:
        lengths.append(int(action["vector"].shape[0]))
    else:
        lengths.extend(int(action[name].shape[0]) for name in ACTION_PARTS if name in action)
        if len(lengths) != len(ACTION_PARTS):
            raise KeyError(f"{path}: missing one or more joint action fields")
    for camera in CAMERAS:
        key = f"vision/{camera}/colors"
        if key not in handle:
            raise KeyError(f"{path}: missing required three-camera field {key}")
        lengths.append(int(handle[key].shape[0]))
    if len(set(lengths)) != 1:
        raise ValueError(f"{path}: action/camera sequence length mismatch: {lengths}")
    if lengths[0] < 2:
        raise ValueError(f"{path}: episode has fewer than two frames")
    return lengths[0]


class RoboTwinHDF5Dataset:
    """Frame-indexed, three-camera, 14-D action view of native RoboTwin data."""

    def __init__(self, dataset_path: str, task_names: tuple[str, ...] = ()):
        h5py = _h5py()
        self.root = Path(dataset_path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"RoboTwin dataset directory does not exist: {self.root}")
        all_files = sorted({*self.root.rglob("*.hdf5"), *self.root.rglob("*.h5")})
        selected = tuple(dict.fromkeys(name.strip() for name in task_names if name.strip()))
        if selected:
            selected_set = set(selected)
            self.files = [
                path for path in all_files
                if selected_set.intersection(path.relative_to(self.root).parts)
            ]
            found = {part for path in self.files for part in path.relative_to(self.root).parts}
            missing = sorted(selected_set - found)
            if missing:
                raise FileNotFoundError(
                    f"Requested RoboTwin tasks were not found under {self.root}: {missing}"
                )
        else:
            self.files = all_files
        if not self.files:
            raise FileNotFoundError(f"No .hdf5/.h5 episodes below {self.root}")
        self.task_names = selected

        self.lengths: list[int] = []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                length = _episode_length(handle, path)
                # Fail at construction rather than after GPU work has started.
                _read_actions(handle, 0)
            self.lengths.append(length)
        self.ends = np.cumsum(np.asarray(self.lengths, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.ends[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        start = 0 if episode == 0 else int(self.ends[episode - 1])
        return episode, index - start

    def __getitem__(self, index: int) -> dict[str, np.ndarray | int]:
        h5py = _h5py()
        episode, frame = self._locate(index)
        with h5py.File(self.files[episode], "r") as handle:
            item: dict[str, np.ndarray | int] = {
                "actions": _read_actions(handle, frame),
                "episode_index": episode,
            }
            decode = _decode_image_bit()
            for source, destination in zip(CAMERAS, ("head", "left_wrist", "right_wrist"), strict=True):
                encoded = handle[f"vision/{source}/colors"][frame]
                image = np.asarray(decode(encoded))
                if image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError(
                        f"{self.files[episode]} {source} frame={frame}: decoder returned {image.shape}, expected HWC RGB"
                    )
                item[f"observation.images.{destination}"] = image
        return item

    def action_quantiles(self) -> dict[str, np.ndarray]:
        """Exact corpus q01/q99 for the 14-D joint targets, computed episode-wise."""
        h5py = _h5py()
        chunks: list[np.ndarray] = []
        for path in self.files:
            with h5py.File(path, "r") as handle:
                chunks.append(_read_actions(handle).astype(np.float32, copy=False))
        all_actions = np.concatenate(chunks, axis=0)
        return {
            "q01": np.quantile(all_actions, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(all_actions, 0.99, axis=0).astype(np.float32),
        }

