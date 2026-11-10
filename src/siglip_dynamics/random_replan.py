"""Training-free random replanning with an RNG independent of policy sampling."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass
class RandomSettings:
    probability: float = 0.02  # Bernoulli probability per eligible completed step.
    seed: int = 42
    warmup_steps: int = 5  # First eligible completed step is 6, as with default COAP persistence.
    cooldown_steps: int = 10
    strategy: str = "bernoulli"
    min_horizon: int = 10
    max_horizon: int = 50

    def validate(self):
        if self.strategy not in {"bernoulli", "uniform_horizon"}:
            raise ValueError("Random strategy must be bernoulli or uniform_horizon")
        if (type(self.min_horizon) is not int or type(self.max_horizon) is not int
                or not 1 <= self.min_horizon <= self.max_horizon):
            raise ValueError("Uniform horizons must be positive integers with min <= max")
        if self.strategy == "uniform_horizon" and (self.warmup_steps or self.cooldown_steps):
            raise ValueError("Uniform horizon scheduling requires warmup_steps=0 and cooldown_steps=0")
        if type(self.probability) not in (int, float) or not math.isfinite(self.probability):
            raise ValueError("Random replan probability must be a finite number")
        if not 0 <= self.probability <= 1:
            raise ValueError("Random replan probability must be in [0,1]")
        for value in (self.seed, self.warmup_steps, self.cooldown_steps):
            if type(value) is not int or value < 0:
                raise ValueError("Random seed, warmup and cooldown must be nonnegative integers")


@dataclass
class Evaluation:
    policy: str = "pretrained_weights/pi05_metaworld_eval_compat"
    device: str = "cuda"
    output: str = "outputs/experiments/eval_metaworld_random_replan_p002_seed42"
    tasks: str = "easy,medium,hard,very_hard"
    episodes: int = 20
    seed: int = 1000
    episode_length: int = 300
    n_action_steps: int = 10
    chunk_size: int = 50
    save_video: bool = True
    reseed_policy_each_episode: bool = False


@dataclass
class Config:
    random: RandomSettings = field(default_factory=RandomSettings)
    evaluation: Evaluation = field(default_factory=Evaluation)

    def validate(self):
        self.random.validate()
        ev = self.evaluation
        for value in (ev.episodes, ev.episode_length, ev.n_action_steps, ev.chunk_size):
            if type(value) is not int or value < 1:
                raise ValueError(
                    "Random baseline episode counts, lengths and chunk sizes must be positive integers"
                )
        if type(ev.seed) is not int or ev.seed < 0 or type(ev.save_video) is not bool:
            raise ValueError("Invalid random baseline evaluation seed or video flag")
        if type(ev.reseed_policy_each_episode) is not bool:
            raise ValueError("reseed_policy_each_episode must be a boolean")
        if ev.n_action_steps > ev.chunk_size:
            raise ValueError("Execution cap cannot exceed predicted chunk size")
        if self.random.strategy == "uniform_horizon" and self.random.max_horizon > ev.n_action_steps:
            raise ValueError("Uniform max_horizon cannot exceed the queued action count")


def load_config(path, overrides=()):
    raw = yaml.safe_load(Path(path).read_text())
    classes = {"random": RandomSettings, "evaluation": Evaluation}
    if not isinstance(raw, dict) or set(raw) - set(classes):
        raise ValueError("Random baseline config accepts only random and evaluation sections")
    sections = {}
    for name, cls in classes.items():
        values = raw.get(name, {})
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError(f"Unknown or invalid random baseline {name} settings")
        sections[name] = cls(**values)
    cfg = Config(**sections)
    for override in overrides:
        key, sep, value = override.partition("=")
        parts = key.split(".")
        if not sep or len(parts) != 2 or parts[0] not in classes:
            raise ValueError(f"Unknown random baseline override: {override}")
        section = getattr(cfg, parts[0])
        if parts[1] not in {f.name for f in fields(section)}:
            raise ValueError(f"Unknown random baseline override: {override}")
        setattr(section, parts[1], yaml.safe_load(value))
    cfg.validate()
    return cfg


class RandomReplanController:
    def __init__(self, settings, output=None):
        settings.validate()
        self.settings = settings
        self.output = Path(output) if output is not None else None
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        self.reset("", 0)

    def reset(self, task, episode_seed):
        self.task, self.episode_seed = str(task), int(episode_seed)
        # A stable per-episode stream; never seed/use global numpy, Python or Torch RNGs.
        identity = json.dumps([self.settings.seed, self.task, self.episode_seed]).encode()
        self.rng_seed = int.from_bytes(hashlib.sha256(identity).digest()[:16], "little")
        self.rng = np.random.default_rng(self.rng_seed)
        self.steps, self.last_trigger = 0, -(10**12)
        self.traces = []
        self.chunk_start_step = 0
        self.planned_horizon = (
            int(self.rng.integers(self.settings.min_horizon, self.settings.max_horizon + 1))
            if self.settings.strategy == "uniform_horizon" else None
        )

    def _after_uniform_execution(self, completed_step, remaining_actions, done):
        # The horizon is drawn before a chunk begins. When it ends, draw the next
        # horizon without touching the RNG used by policy sampling or the environment.
        chunk_steps = completed_step - self.chunk_start_step
        horizon = self.planned_horizon
        eligible = not done and remaining_actions > 0
        trigger = eligible and chunk_steps >= horizon
        event = {
            "completed_step": completed_step,
            "strategy": "uniform_horizon",
            "planned_horizon": horizon,
            "chunk_action_index": chunk_steps,
            "chunk_start_step": self.chunk_start_step,
            "eligible": eligible,
            "draw": None,
            "warmup": False,
            "cooldown": False,
            "terminal": bool(done),
            "remaining_actions": remaining_actions,
            "trigger": trigger,
            "discarded_actions": remaining_actions if trigger else 0,
        }
        if trigger:
            self.last_trigger = completed_step
        if trigger or remaining_actions == 0:
            self.chunk_start_step = completed_step
            if not done:
                self.planned_horizon = int(self.rng.integers(
                    self.settings.min_horizon, self.settings.max_horizon + 1
                ))
        self.traces.append(event)
        return event

    def after_execution(self, completed_step, remaining_actions, *, done=False):
        if completed_step != self.steps + 1:
            raise ValueError("Random replanning needs consecutive completed steps")
        if type(remaining_actions) is not int or remaining_actions < 0:
            raise ValueError("Remaining action count must be a nonnegative integer")
        self.steps = completed_step
        if self.settings.strategy == "uniform_horizon":
            return self._after_uniform_execution(completed_step, remaining_actions, done)
        cooldown = completed_step - self.last_trigger <= self.settings.cooldown_steps
        warmup = completed_step <= self.settings.warmup_steps
        eligible = not done and remaining_actions > 0 and not cooldown and not warmup
        draw = float(self.rng.random()) if eligible and 0 < self.settings.probability < 1 else None
        trigger = eligible and (
            self.settings.probability == 1 or draw is not None and draw < self.settings.probability
        )
        if trigger:
            self.last_trigger = completed_step
        event = {
            "completed_step": completed_step,
            "probability": self.settings.probability,
            "eligible": eligible,
            "draw": draw,
            "warmup": warmup,
            "cooldown": cooldown,
            "terminal": bool(done),
            "remaining_actions": remaining_actions,
            "trigger": trigger,
            "discarded_actions": remaining_actions if trigger else 0,
        }
        self.traces.append(event)
        return event

    def finish(self, success, error="", *, vla_calls=0, vla_seconds=0.0, wall_seconds=0.0):
        row = {
            "task": self.task,
            "seed": self.episode_seed,
            "controller_seed": self.settings.seed,
            "rng_seed": self.rng_seed,
            "settings": asdict(self.settings),
            "success": bool(success),
            "error": error,
            "steps": self.steps,
            "intervention_count": sum(event["trigger"] for event in self.traces),
            "eligible_steps": sum(event["eligible"] for event in self.traces),
            "vla_calls": vla_calls,
            "vla_seconds": vla_seconds,
            "wall_seconds": wall_seconds,
            "traces": self.traces,
        }
        if self.output is not None:
            with (self.output / "random_replan_episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row


def after_step(controller, completed_step, action_queue, raw_action_queue, *, done=False):
    event = controller.after_execution(completed_step, len(action_queue), done=done)
    if event["trigger"]:
        action_queue.clear()
        raw_action_queue.clear()
    return event


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:
        raise ValueError("No random baseline episodes to summarize")
    if len({(row["task"], row["seed"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate random baseline task/seed episodes")
    if any(row["error"] for row in rows):
        raise ValueError("Resolve random baseline rollout errors before summarizing")
    if any(row["settings"] != rows[0]["settings"] for row in rows):
        raise ValueError("Random baseline summaries require one consistent controller configuration")
    groups = {}
    for row in rows:
        group = groups.setdefault(
            row["task"].split(":")[0], {"episodes": 0, "successes": 0, "interventions": 0}
        )
        group["episodes"] += 1
        group["successes"] += row["success"]
        group["interventions"] += row["intervention_count"]
    for group in groups.values():
        group["success_rate"] = group["successes"] / group["episodes"]
    interventions = sum(row["intervention_count"] for row in rows)
    eligible = sum(row["eligible_steps"] for row in rows)
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows),
        "interventions": interventions,
        "interventions_per_episode": interventions / len(rows),
        "eligible_steps": eligible,
        "trigger_fraction_of_eligible_steps": interventions / eligible if eligible else None,
        "vla_calls": sum(row["vla_calls"] for row in rows),
        "vla_seconds": sum(row["vla_seconds"] for row in rows),
        "wall_seconds": sum(row["wall_seconds"] for row in rows),
        "settings": rows[0]["settings"],
        "groups": groups,
    }


def eval_command(cfg, resolved):
    ev = cfg.evaluation
    args = {
        "policy.path": ev.policy,
        "policy.device": ev.device,
        "policy.n_action_steps": ev.n_action_steps,
        "policy.chunk_size": ev.chunk_size,
        "policy.compile_model": False,
        "env.type": "metaworld",
        "env.task": ev.tasks,
        "env.episode_length": ev.episode_length,
        "env.max_parallel_tasks": 1,
        "eval.batch_size": 1,
        "eval.n_episodes": ev.episodes,
        "eval.use_async_envs": False,
        "seed": ev.seed,
        "reseed_policy_each_episode": ev.reseed_policy_each_episode,
        "output_dir": ev.output,
        "rename_map": {"observation.image": "observation.images.base_0_rgb"},
        "random_replan_config": str(resolved),
        "meltdown_use_guidance_replan": False,
        "guidance_eta": 0.0,
        "guidance_compare_baseline": False,
        "compute_baseline_metrics": False,
        "save_analysis": True,
        "save_raw_video": ev.save_video,
        "save_sync_video_with_curve": False,
        "save_baseline_sync_video_with_curve": False,
        "save_baseline_et_curve_plot": False,
        "save_baseline_step_csv": False,
        "save_guidance_dual_video": False,
        "save_et_curve_plot": False,
        "save_task_plots": False,
    }

    def encode(value):
        return json.dumps(value) if isinstance(value, (dict, bool)) else str(value)

    return [sys.executable, "-m", "lerobot.scripts.lerobot_eval_modified_detection"] + [
        f"--{key}={encode(value)}" for key, value in args.items()
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Frozen VLA with Bernoulli interrupts or uniform execution horizons; no detector or OGG"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("evaluate", "summarize"), default="evaluate")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.dry_run:
        print(yaml.safe_dump(asdict(cfg), sort_keys=False))
        print(json.dumps(eval_command(cfg, "<resolved_random_replan.yaml>"), indent=2))
        return
    out = Path(cfg.evaluation.output)
    if args.mode == "summarize":
        print(json.dumps(summarize(out / "random_replan_episodes.jsonl"), indent=2))
        return
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh random baseline output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    resolved = out / "resolved_random_replan.yaml"
    resolved.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    subprocess.run(eval_command(cfg, resolved.resolve()), check=True)


if __name__ == "__main__":
    main()
