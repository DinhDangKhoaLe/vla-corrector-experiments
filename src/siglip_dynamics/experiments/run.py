"""YAML entry point: train, collect calibration, calibrate and evaluate."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

import yaml

from .config import from_dict, load_config


MODES = ("train", "collect-calibration", "calibrate", "evaluate")


def legacy_train_command(cfg):
    """The user's original training recipe; no experimental model/data path is imported."""
    tr, model = cfg.training, cfg.model
    # Keep the legacy architecture/training procedure, but take tensor dimensions
    # from a cache when it supplies them (e.g. RoboTwin has 14-D actions).
    token_dim, action_dim = 2048, 4
    metadata_path = Path(tr.cache) / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        token_dim = int(metadata.get("d_dim", token_dim))
        action_dim = int(metadata.get("action_dim", action_dim))
    args = {"dataset-path": tr.cache, "checkpoint-dir": tr.output, "device": tr.device,
            "model-type": "mlp", "scale": "20m", "token-dim": token_dim, "action-dim": action_dim,
            "action-embed-dim": model.action_embed_dim, "dropout": model.dropout,
            "h-window": 1, "k-step-list": 10, "batch-size": tr.batch_size, "epochs": tr.epochs,
            "lr": tr.lr, "weight-decay": tr.weight_decay, "val-ratio": tr.val_fraction,
            "seed": tr.seed, "grad-clip-norm": 1.0, "patience": tr.patience, "train-loss-type": "cosine"}
    return [sys.executable, "-m", "siglip_dynamics.train"] + [f"--{k}={v}" for k, v in args.items()]


def resolve(path, overrides):
    raw = load_config(path).to_dict()
    for override in overrides:
        key, sep, value = override.partition("=")
        parts = key.split(".")
        if not sep or len(parts) != 2 or parts[0] not in raw or parts[1] not in raw[parts[0]]:
            raise ValueError(f"Override must be a known section.key=value: {override}")
        raw[parts[0]][parts[1]] = yaml.safe_load(value)
    return from_dict(raw)


def eval_command(cfg, mode, resolved_path):
    ev = cfg.evaluation
    seed = ev.collection_seed if mode == "collect-calibration" else ev.seed
    output = ev.output + ("_calibration" if mode == "collect-calibration" else "")
    default_checkpoint = "h1_k10" if cfg.training.backend == "legacy" else "best_model.pt"
    checkpoint = ev.checkpoint or str(Path(cfg.training.output) / default_checkpoint)
    args = {
        "policy.path": ev.policy,
        "policy.device": cfg.training.device, "policy.n_action_steps": ev.n_action_steps,
        "policy.chunk_size": ev.chunk_size, "policy.compile_model": False,
        "env.type": "metaworld", "env.task": ev.tasks, "env.episode_length": ev.episode_length,
        "env.max_parallel_tasks": 1, "eval.batch_size": 1, "eval.n_episodes": ev.episodes,
        "eval.use_async_envs": False, "seed": seed, "output_dir": output,
        "rename_map": {"observation.image": "observation.images.base_0_rgb"},
        "safety_model_path": checkpoint, "safety_k": max(cfg.model.horizons), "safety_history_frames": 1,
        "experiment_config": str(resolved_path), "experiment_mode": mode,
        "guidance_eta": ev.guidance_eta, "guidance_apply_every": ev.guidance_apply_every,
        "guidance_loss_objective": "attract_delta_z_correction", "guidance_compare_baseline": False,
        "meltdown_use_guidance_replan": ev.guidance and mode == "evaluate",
        "meltdown_cooldown_steps": ev.cooldown_steps,
        "save_analysis": True, "compute_baseline_metrics": False,
        "save_raw_video": ev.save_video, "save_sync_video_with_curve": False,
        "save_baseline_sync_video_with_curve": False, "save_guidance_dual_video": False,
        "save_baseline_et_curve_plot": False, "save_baseline_step_csv": False,
        "save_et_curve_plot": False, "save_task_plots": False,
        "save_episode_step_csv": True, "save_guidance_event_csv": True,
        "save_episode_npz": True, "save_summary_csv": True, "save_summary_json": True,
    }
    for name, value in ev.thresholds.items():
        if not name.startswith("threshold_"):
            raise ValueError(f"Only threshold_* keys allowed in evaluation.thresholds: {name}")
        args[name] = value
    if cfg.training.backend == "legacy":
        # Not even the experimental loader/monitor is used for the legacy baseline.
        args.pop("experiment_config")
        args.pop("experiment_mode")
    def encode(value):
        return json.dumps(value) if isinstance(value, (bool, dict, list)) else str(value)
    return [sys.executable, "-m", "lerobot.scripts.lerobot_eval_modified_detection"] + [
        f"--{k}={encode(v)}" for k, v in args.items()]


def _is_robotwin_cache(cfg) -> bool:
    """Whether the selected latent cache was extracted from native RoboTwin data."""
    metadata_path = Path(cfg.training.cache) / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata.get("dataset_format") == "robotwin"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--set", action="append", default=[], help="Override section.key=value (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = resolve(args.config, args.set)
    if cfg.training.backend == "legacy" and args.mode not in ("train", "evaluate"):
        raise ValueError("Legacy baseline supports train/evaluate only; collection is an experimental workflow")
    if args.mode in ("collect-calibration", "calibrate") and not cfg.features.uncertainty_monitor:
        raise ValueError("Calibration is only available for uncertainty_monitor")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        if args.mode in ("evaluate", "collect-calibration"):
            if _is_robotwin_cache(cfg):
                print("RoboTwin native rollout selected (SAPIEN task runner; no MetaWorld command is generated).")
                print(json.dumps({
                    "mode": args.mode,
                    "robotwin_root": cfg.evaluation.robotwin_root,
                    "task_config": cfg.evaluation.robotwin_task_config,
                    "tasks": [x.strip() for x in cfg.evaluation.robotwin_tasks.split(",") if x.strip()],
                    "episodes_per_task": cfg.evaluation.episodes,
                    "policy": cfg.evaluation.policy,
                    "corrector_checkpoint": cfg.evaluation.checkpoint or str(Path(cfg.training.output) / "best_model.pt"),
                }, indent=2))
            else:
                print(json.dumps(eval_command(cfg, args.mode, "<resolved.yaml>"), indent=2))
        elif args.mode == "train" and cfg.training.backend == "legacy":
            print(json.dumps(legacy_train_command(cfg), indent=2))
        return
    if args.mode == "train":
        if cfg.training.backend == "legacy":
            output = Path(cfg.training.output)
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Refusing to overwrite an existing/running baseline in {output}")
            subprocess.run(legacy_train_command(cfg), check=True)
        else:
            from .train import train_experiment
            train_experiment(cfg)
    elif args.mode == "calibrate":
        from .uncertainty import calibrate_uncertainty
        print(json.dumps(calibrate_uncertainty(cfg), indent=2))
    else:
        if _is_robotwin_cache(cfg):
            # RoboTwin is not a Gym/MuJoCo environment. Keep its SAPIEN task lifecycle,
            # three-camera observations and 14-D qpos actions in a dedicated runner.
            from .robotwin_eval import evaluate_robotwin
            evaluate_robotwin(cfg, mode=mode, config_path=Path(args.config))
            return
        suffix = "_calibration" if args.mode == "collect-calibration" else ""
        output = Path(cfg.evaluation.output + suffix)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a fresh evaluation.output; {output} is not empty")
        output.mkdir(parents=True, exist_ok=True)
        resolved = output / "resolved_experiment.yaml"
        resolved.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        env = os.environ.copy()
        for name, value in {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "EGL_PLATFORM": "surfaceless"}.items():
            env.setdefault(name, value)
        subprocess.run(eval_command(cfg, args.mode, resolved.resolve()), check=True, env=env)


if __name__ == "__main__":
    main()
