from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import LatentJacobianModel
from .utils import device_from_config


def load_model_artifact(path: str | Path, device: str = "cpu") -> tuple[LatentJacobianModel, dict[str, Any], torch.device]:
    dev = device_from_config(device)
    payload = torch.load(Path(path), map_location=dev, weights_only=False)
    spec = payload["model_spec"]
    model = LatentJacobianModel(**spec).to(dev)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload, dev


def model_ensemble(paths: list[str], device: str) -> tuple[list[LatentJacobianModel], dict[str, Any], torch.device]:
    if not paths:
        raise ValueError("At least one --model is required")
    first, artifact, dev = load_model_artifact(paths[0], device)
    models = [first]
    for path in paths[1:]:
        model, other, _ = load_model_artifact(path, device)
        if other["model_spec"] != artifact["model_spec"]:
            raise ValueError("All ensemble models must use the same model_spec")
        for key in ("latent_mean", "latent_std", "action_mean", "action_std"):
            if not np.allclose(np.asarray(other[key]), np.asarray(artifact[key])):
                raise ValueError(f"Ensemble artifacts disagree on {key}")
        models.append(model)
    return models, artifact, dev
