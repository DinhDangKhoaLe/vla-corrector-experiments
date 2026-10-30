"""Strict YAML configuration. Paths are relative to the invocation directory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Features:
    action_prefix: bool = False
    uncertainty_monitor: bool = False


@dataclass
class Model:
    horizons: list[int] = field(default_factory=lambda: [10])
    widths: list[int] = field(default_factory=lambda: [2048] * 4)
    action_embed_dim: int = 256
    dropout: float = 0.0
    uncertainty_hidden: int = 128


@dataclass
class Training:
    backend: str = "experimental"  # "legacy" launches the untouched original trainer/evaluator.
    cache: str = "outputs/metaworld_mt50_pi05_latents_tf54_fixed"
    output: str = "outputs/experiments/vla_corrector_baseline"
    device: str = "cuda"
    epochs: int = 30
    batch_size: int = 512
    microbatch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    uncertainty_loss_weight: float = 0.1
    mse_loss_weight: float = 1.0
    cosine_loss_weight: float = 1.0


@dataclass
class Uncertainty:
    # Expected RMSE, not a Gaussian standard deviation or failure probability.
    rmse_floor: float = 1e-4
    persistence_checks: int = 2
    reset_checks: int = 2
    release_ratio: float = 0.8
    direction_coherence_min: float = 0.5
    ogg_enabled: bool = True
    ogg_uncertainty_quantile: float = 0.75
    min_error_correlation: float = 0.1
    episode_quantile: float = 0.95
    min_successful_episodes: int = 50
    traces: str = "outputs/experiments/eval_uncertainty_calibration/experiment_episodes.jsonl"
    artifact: str = "outputs/experiments/uncertainty_calibration.json"


@dataclass
class Evaluation:
    policy: str = "pretrained_weights/pi05_metaworld_compat"
    checkpoint: str = ""  # Empty means training.output/best_model.pt.
    output: str = "outputs/experiments/eval_baseline"
    tasks: str = "button-press-v3"
    episodes: int = 5
    seed: int = 1000
    collection_seed: int = 2000
    episode_length: int = 300
    n_action_steps: int = 10
    chunk_size: int = 50
    guidance: bool = True
    guidance_eta: float = 1.0
    guidance_apply_every: int = 1
    cooldown_steps: int = 10
    save_video: bool = True
    # Schedule-matched single-action control for the prefix ablation.
    queue_bounded_monitor: bool = False
    # Native RoboTwin/SAPIEN rollout settings. They are ignored by MetaWorld.
    robotwin_root: str = "third_party/RoboTwin"
    robotwin_task_config: str = "demo_clean"
    robotwin_tasks: str = ""
    robotwin_expert_check: bool = True
    # Runtime overrides of the original median/MAD state machine.
    thresholds: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    features: Features = field(default_factory=Features)
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    evaluation: Evaluation = field(default_factory=Evaluation)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def validate(self):
        h = self.model.horizons
        if not h or h != sorted(set(h)) or any(type(x) is not int or x < 1 for x in h):
            raise ValueError("model.horizons must be sorted, unique positive integers")
        if not self.model.widths or len(self.model.widths) < 2 or min(self.model.widths) < 1:
            raise ValueError("model.widths needs at least two positive widths")
        tr = self.training
        if tr.backend not in {"legacy", "experimental"}:
            raise ValueError("training.backend must be legacy or experimental")
        if tr.backend == "legacy" and (any(asdict(self.features).values()) or self.evaluation.queue_bounded_monitor
                                       or self.model.horizons != [10] or self.model.widths != [2048] * 4):
            raise ValueError("Legacy mode preserves the original baseline; use training.backend=experimental for ablations")
        if not (0 < tr.val_fraction < 1 and 0 < tr.test_fraction < 1 and
                tr.val_fraction + tr.test_fraction < 1):
            raise ValueError("Need disjoint non-empty train, validation and test fractions")
        if min(tr.epochs, tr.batch_size, tr.microbatch_size, tr.patience) < 1 or tr.lr <= 0:
            raise ValueError("Invalid training schedule")
        ev = self.evaluation
        if not 1 <= ev.n_action_steps <= ev.chunk_size or ev.episode_length < 1 or ev.episodes < 1:
            raise ValueError("Invalid evaluation horizon, episode length or episode count")
        if max(h) > ev.n_action_steps and self.features.action_prefix:
            raise ValueError("Prefix horizons cannot exceed evaluation.n_action_steps")
        if self.features.uncertainty_monitor:
            import math
            u = self.uncertainty
            if len(h) != 1:
                raise ValueError("Uncertainty monitor requires one fixed K")
            if min(tr.mse_loss_weight, tr.cosine_loss_weight, tr.uncertainty_loss_weight) <= 0:
                raise ValueError("Uncertainty training requires positive MSE, cosine and error-head weights")
            if not all(math.isfinite(x) for x in (tr.mse_loss_weight, tr.cosine_loss_weight,
                                                 tr.uncertainty_loss_weight, u.rmse_floor,
                                                 u.direction_coherence_min, u.min_error_correlation)):
                raise ValueError("Uncertainty settings must be finite")
            if u.rmse_floor <= 0 or min(u.persistence_checks, u.reset_checks, u.min_successful_episodes) < 1:
                raise ValueError("Invalid uncertainty floor, persistence or episode count")
            if not (-1 <= u.direction_coherence_min <= 1 and -1 <= u.min_error_correlation <= 1):
                raise ValueError("Invalid uncertainty correlation/coherence setting")
            if not all(0 < x < 1 for x in (u.release_ratio, u.episode_quantile, u.ogg_uncertainty_quantile)):
                raise ValueError("Uncertainty quantiles and release_ratio must lie in (0,1)")

    def to_dict(self):
        return asdict(self)


def from_dict(raw: dict) -> Experiment:
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must be a mapping")
    classes = {"features": Features, "model": Model, "training": Training,
               "evaluation": Evaluation,
               "uncertainty": Uncertainty}
    unknown = set(raw) - set(classes)
    if unknown:
        raise ValueError(f"Unknown YAML sections: {sorted(unknown)}")
    parts = {}
    for name, cls in classes.items():
        values = raw.get(name, {})
        if not isinstance(values, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
        parts[name] = cls(**values)
    cfg = Experiment(**parts)
    for section, names in {"features": ("action_prefix", "uncertainty_monitor"),
                           "uncertainty": ("ogg_enabled",),
                           "evaluation": ("guidance", "save_video", "queue_bounded_monitor",
                                          "robotwin_expert_check")}.items():
        for name in names:
            if type(getattr(getattr(cfg, section), name)) is not bool:
                raise ValueError(f"{section}.{name} must be a YAML boolean")
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> Experiment:
    with open(path, encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f))


def from_checkpoint_dict(raw: dict) -> Experiment:
    """Read supported old checkpoints without exposing removed settings in YAML.

    Old snapshots serialize disabled features and unused sections. Project those
    onto the current schema, but reject any unsupported feature that was enabled.
    Model fields remain strict to avoid silently changing an architecture.
    """
    schema = Experiment().to_dict()
    if not isinstance(raw, dict) or not isinstance(raw.get("features", {}), dict):
        raise ValueError("Invalid checkpoint experiment metadata")
    for name, value in raw.get("features", {}).items():
        if name not in schema["features"] and value is not False:
            raise ValueError(f"Checkpoint requires unsupported feature: {name}")
    projected = {}
    for section, defaults in schema.items():
        if section not in raw:
            continue
        values = raw[section]
        if not isinstance(values, dict):
            raise ValueError(f"Invalid checkpoint section: {section}")
        projected[section] = (values if section == "model" else
                              {key: value for key, value in values.items() if key in defaults})
    return from_dict(projected)
