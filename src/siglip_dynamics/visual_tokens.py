"""Shared, opt-in spatial averaging for cached and online visual tokens."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F


def pool_square_token_grid(
    z: torch.Tensor, target_side: int = 0, *, target_shape: list[int] | tuple[int, int] | None = None,
) -> torch.Tensor:
    """Pool one camera's [B,L,D] square grid; zero preserves the original tensor.

    Pool each camera separately before concatenation. Feature width and token
    order (row major) are preserved. This does not pool the VLA's own inputs.
    """
    if type(target_side) is not int or target_side < 0:
        raise ValueError("image_token_grid_size must be a nonnegative integer")
    shape = tuple(target_shape or ())
    if shape and (len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape)):
        raise ValueError("image_token_grid_shape must contain two positive integers [rows, columns]")
    if shape and target_side:
        raise ValueError("Set either image_token_grid_size or image_token_grid_shape, not both")
    if target_side == 0 and not shape:
        return z
    if z.ndim != 3:
        raise ValueError(f"Expected visual tokens [B,L,D], got {tuple(z.shape)}")
    side = math.isqrt(int(z.shape[1]))
    if side * side != z.shape[1]:
        raise ValueError(f"Cannot spatially pool non-square token count L={z.shape[1]}")
    if not shape and (target_side > side or side % target_side):
        raise ValueError(f"image-token-grid-size must divide source grid {side}; got {target_side}")
    rows, columns = shape or (target_side, target_side)
    if any(v > side or side % v for v in (rows, columns)):
        raise ValueError(f"image-token-grid-shape must divide source grid {side}; got {(rows, columns)}")
    if rows == columns == side:
        return z
    batch, _, dim = z.shape
    grid = z.reshape(batch, side, side, dim).permute(0, 3, 1, 2)
    kernel = (side // rows, side // columns)
    pooled = F.avg_pool2d(grid, kernel_size=kernel, stride=kernel)
    return pooled.permute(0, 2, 3, 1).reshape(batch, rows * columns, dim)


def validate_cache_grid(metadata: dict, target_side: int, *, target_shape=None) -> None:
    """Reject a training config that names a different cached representation."""
    actual = int(metadata.get("image_token_grid_size", 0))
    actual_shape = tuple(metadata.get("image_token_grid_shape") or ())
    shape = tuple(target_shape or ())
    if actual != target_side or actual_shape != shape:
        raise ValueError(
            f"Cache image_token_grid_size={actual}, shape={actual_shape}, "
            f"but model.image_token_grid_size={target_side}, shape={shape}; "
            "select the matching cache/config or re-extract."
        )
    if (target_side or shape) and "l_tokens" in metadata:
        cameras = 1 + int(bool(metadata.get("use_wrist_image", False))) + len(metadata.get("extra_image_keys", []))
        expected = cameras * (math.prod(shape) if shape else target_side ** 2)
        if int(metadata["l_tokens"]) != expected:
            raise ValueError(f"Pooled cache has L={metadata['l_tokens']}; expected {expected} for {cameras} camera(s)")


def validate_checkpoint_grid(path: str | Path, target_side: int, *, target_shape=None) -> None:
    """Prevent evaluating a pooled checkpoint under an unpooled YAML (or vice versa)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corrector checkpoint does not exist: {path}")
    if path.is_dir():
        candidates = [path / name for name in ("best_cosine_model.pt", "best_mse_model.pt", "best_model.pt")]
        path = next((p for p in candidates if p.is_file()), path)
    # Check the saved payload, rather than relying on the current training cache
    # or an editable sidecar. The existing loaders use the same local checkpoints.
    if path.is_file() and path.suffix == ".pt":
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("format") == "corrector_experiment_v1":
            representation = obj["experiment"].get("model", {})
        else:
            representation = obj.get("train_cfg", {})
        actual = int(representation.get("image_token_grid_size", 0))
        actual_shape = tuple(representation.get("image_token_grid_shape") or ())
    else:
        raise ValueError(f"Expected a Corrector checkpoint file or h1_k10/model directory: {path}")
    shape = tuple(target_shape or ())
    if actual != target_side or actual_shape != shape:
        raise ValueError(
            f"Checkpoint image_token_grid_size={actual}, shape={actual_shape}, "
            f"but model.image_token_grid_size={target_side}, shape={shape}; "
            "select the matching ablation YAML/checkpoint."
        )
