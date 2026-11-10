"""Configuration for the separate COAP detector; existing Corrector configs are unchanged."""

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Model:
    window_size: int = 5
    hidden_dim: int = 128
    embedding_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.0


@dataclass
class Training:
    cache: str = "outputs/metaworld_mt50_pi05_latents_pool64"
    raw_actions: str = ""  # Empty: raw cache actions, or cache/coap_raw_actions.npy.
    output: str = "outputs/experiments/metaworld_coap_pool64"
    device: str = "cuda"
    seed: int = 42
    split_seed: int = 42
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    epochs: int = 30
    batch_size: int = 128
    lr: float = 0.0003
    weight_decay: float = 0.0001
    temperature: float = 0.07
    patience: int = 5
    normalization_floor: float = 0.001


@dataclass
class Monitor:
    artifact: str = "outputs/experiments/metaworld_coap_pool64/calibration.json"
    threshold: float | None = None
    allow_provisional_threshold: bool = False
    persistence: int = 2
    cooldown_steps: int = 10
    lower_quantile: float = 0.05
    min_successful_episodes: int = 50


@dataclass
class Evaluation:
    policy: str = "pretrained_weights/pi05_metaworld_eval_compat"
    checkpoint: str = ""
    output: str = "outputs/experiments/eval_metaworld_coap_pool64"
    tasks: str = "easy,medium,hard,very_hard"
    episodes: int = 20
    seed: int = 1000
    collection_seed: int = 10000
    episode_length: int = 300
    n_action_steps: int = 10
    chunk_size: int = 50
    save_video: bool = True
    reseed_policy_each_episode: bool = False


@dataclass
class Generation:
    enabled: bool = False
    min_temperature: float = 1.0
    max_temperature: float = 2.0
    score_low: float = 0.0
    score_high: float = 0.5
    warmup_temperature: float = 1.0


@dataclass
class COAPConfig:
    model: Model = field(default_factory=Model)
    training: Training = field(default_factory=Training)
    monitor: Monitor = field(default_factory=Monitor)
    evaluation: Evaluation = field(default_factory=Evaluation)
    generation: Generation = field(default_factory=Generation)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        m, t, r, e = self.model, self.training, self.monitor, self.evaluation
        for value in (
            m.window_size,
            m.hidden_dim,
            m.embedding_dim,
            m.num_layers,
            t.epochs,
            t.batch_size,
            t.patience,
            r.persistence,
            r.min_successful_episodes,
            e.episodes,
            e.episode_length,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("COAP sizes, counts and persistence must be positive integers")
        if m.window_size < 2 or t.batch_size < 2:
            raise ValueError("COAP needs at least two timesteps and two pairs per batch")
        if not 0 <= m.dropout < 1 or r.cooldown_steps < 0:
            raise ValueError("Invalid COAP dropout or cooldown")
        if not (0 < t.val_fraction < 1 and 0 < t.test_fraction < 1 and t.val_fraction + t.test_fraction < 1):
            raise ValueError("COAP needs disjoint train/validation/test fractions")
        if any(not math.isfinite(v) or v <= 0 for v in (t.lr, t.temperature, t.normalization_floor)):
            raise ValueError("COAP learning rate, temperature and normalization floor must be positive")
        if not math.isfinite(t.weight_decay) or t.weight_decay < 0:
            raise ValueError("Invalid weight decay")
        if any(
            type(seed) is not int or seed < 0 for seed in (t.seed, t.split_seed, e.seed, e.collection_seed)
        ):
            raise ValueError("COAP seeds must be nonnegative integers")
        if not 0 < r.lower_quantile < 1:
            raise ValueError("Invalid calibration quantile")
        if r.threshold is not None and (not math.isfinite(r.threshold) or not -1 <= r.threshold <= 1):
            raise ValueError("COAP threshold must be a cosine in [-1,1]")
        if type(r.allow_provisional_threshold) is not bool or type(e.save_video) is not bool:
            raise ValueError("COAP flags must be booleans")
        if type(e.reseed_policy_each_episode) is not bool:
            raise ValueError("COAP reseed_policy_each_episode must be a boolean")
        if not 1 <= e.n_action_steps <= e.chunk_size:
            raise ValueError("Invalid COAP execution cap / action chunk size")
        g = self.generation
        if type(g.enabled) is not bool:
            raise ValueError("COAP generation.enabled must be a boolean")
        values = (g.min_temperature, g.max_temperature, g.warmup_temperature, g.score_low, g.score_high)
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("COAP generation settings must be finite numbers")
        if not 0 < g.min_temperature <= g.warmup_temperature <= g.max_temperature:
            raise ValueError("COAP temperatures require 0 < min <= warmup <= max")
        if not -1 <= g.score_low < g.score_high <= 1:
            raise ValueError("COAP generation score bounds require -1 <= low < high <= 1")


def from_dict(raw):
    if not isinstance(raw, dict):
        raise ValueError("COAP config must be a mapping")
    classes = {
        "model": Model,
        "training": Training,
        "monitor": Monitor,
        "evaluation": Evaluation,
        "generation": Generation,
    }
    if set(raw) - set(classes):
        raise ValueError(f"Unknown COAP sections: {set(raw) - set(classes)}")
    sections = {}
    for name, cls in classes.items():
        values = raw.get(name, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown or invalid COAP {name} fields")
        sections[name] = cls(**values)
    cfg = COAPConfig(**sections)
    cfg.validate()
    return cfg


def load_config(path, overrides=()):
    raw = from_dict(yaml.safe_load(Path(path).read_text())).to_dict()
    for override in overrides:
        key, sep, value = override.partition("=")
        parts = key.split(".")
        if not sep or len(parts) != 2 or parts[0] not in raw or parts[1] not in raw[parts[0]]:
            raise ValueError(f"Unknown COAP override: {override}")
        raw[parts[0]][parts[1]] = yaml.safe_load(value)
    return from_dict(raw)
