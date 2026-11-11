"""Configuration and method construction; all file paths are repository-relative."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
METHODS = {"autohorizon", "aac", "pace", "a3", "bcp"}


def resolve_path(value):
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    required = {"method", "policy", "dataset", "evaluation", "settings"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError(f"Config requires {sorted(required)}")
    if config["method"] not in METHODS:
        raise ValueError(f"Unknown method: {config['method']}")
    if config["evaluation"]["prediction_horizon"] != 50 or config["evaluation"]["max_execution_horizon"] != 50:
        raise ValueError("This comparison requires prediction and maximum execution horizons of 50")
    if config["policy"]["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Unsupported π0.5 dtype")
    if config["policy"]["denoising_steps"] < 1:
        raise ValueError("denoising_steps must be positive")
    for key in ("episodes_per_task", "episode_length"):
        if config["evaluation"][key] < 1:
            raise ValueError(f"{key} must be positive")
    config["dataset"] = resolve_path(config["dataset"])
    for key in ("checkpoint", "tokenizer_model"):
        config["policy"][key] = resolve_path(config["policy"][key])
    for section in ("settings", "training", "evaluation"):
        for key in ("artifact", "output"):
            if key in config.get(section, {}):
                config[section][key] = resolve_path(config[section][key])
    return config


def make_method(config, *, head=None, stochastic=False):
    method, settings = config["method"], config["settings"]
    if method == "autohorizon":
        from .autohorizon import AutoHorizon
        return AutoHorizon(settings)
    if method == "aac":
        from .aac import AAC
        return AAC(settings)
    if method == "pace":
        from .pace import PACE
        return PACE(settings)
    if method == "a3":
        from .a3 import A3
        return A3(settings)
    if method == "bcp":
        import torch
        from .bcp import BCP, ContinuationHead
        if head is None:
            checkpoint = torch.load(settings["artifact"], map_location="cpu", weights_only=True)
            if checkpoint["head_config"] != settings["head"]:
                raise ValueError("BCP checkpoint head configuration differs from evaluation config")
            if checkpoint["policy"] != config["policy"]:
                raise ValueError("BCP head was trained with a different policy/tokenizer/inference configuration")
            head = ContinuationHead(**settings["head"])
            head.load_state_dict(checkpoint["head"], strict=True)
            head.to(config["policy"]["device"])
        return BCP(head.eval(), stochastic=stochastic)
    raise ValueError(method)
