"""Check that a completed cache matches the single-camera MT50 pool64 recipe."""

import json
from pathlib import Path
import sys

import numpy as np


def validate(path: Path) -> None:
    metadata = json.loads((path / "metadata.json").read_text())
    expected = {
        "format": "siglip_frame_cache_v2",
        "dataset_format": "metaworld",
        "encoder_backend": "pi05",
        "image_token_grid_size": 8,
        "l_tokens": 64,
        "d_dim": 2048,
        "action_dim": 4,
        "image_key": "observation.image",
        "action_representation": "normalized_action",
        "use_wrist_image": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Cache {key}: expected {value!r}, got {metadata.get(key)!r}")
    if metadata.get("extra_image_keys") or metadata.get("image_token_grid_shape"):
        raise ValueError("Expected a single camera and the square pool64 grid")
    frames = int(metadata["num_frames"])
    if frames <= 10:
        raise ValueError("Cache has too few frames for K=10 training")
    for name, shape, dtype in (
        ("z_q", (frames, 64, 2048), "int8"),
        ("z_scale", (frames, 64), "float16"),
        ("actions", (frames, 4), "float16"),
        ("episode_index", (frames,), "int32"),
    ):
        array = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid {name}: {array.shape}, {array.dtype}; expected {shape}, {dtype}")
        if not np.isfinite(array[[0, -1]]).all():
            raise ValueError(f"Non-finite boundary values in {name}")
    print(f"Validated pool64 cache: {frames:,} frames, 64×2048 tokens, normalized 4D actions")


if __name__ == "__main__":
    validate(Path(sys.argv[1]))
