"""COAP preparation, training, calibration and matched policy/COAP evaluation."""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from .config import load_config


def eval_command(cfg, mode, resolved, comparison="coap"):
    ev = cfg.evaluation
    collection = mode == "collect-calibration"
    if collection and comparison != "coap":
        raise ValueError("Only COAP has a collection mode")
    output = ev.output + ("_calibration" if collection else "_policy" if comparison == "policy" else "")
    args = {
        "policy.path": ev.policy,
        "policy.device": cfg.training.device,
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
        "seed": ev.collection_seed if collection else ev.seed,
        "reseed_policy_each_episode": ev.reseed_policy_each_episode,
        "output_dir": output,
        "rename_map": {"observation.image": "observation.images.base_0_rgb"},
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
    if comparison == "coap":
        args.update(coap_config=str(resolved), coap_mode=mode)
    elif comparison == "policy":
        args.update(disable_corrector=True)
    else:
        raise ValueError("COAP comparison must be coap or policy")

    def encode(value):
        return json.dumps(value) if isinstance(value, (dict, bool)) else str(value)

    return [sys.executable, "-m", "lerobot.scripts.lerobot_eval_modified_detection"] + [
        f"--{key}={encode(value)}" for key, value in args.items()
    ]


def main():
    parser = argparse.ArgumentParser(description="COAP — Contrastive Observation Action Pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("prepare-actions", "train", "collect-calibration", "calibrate", "evaluate", "summarize"),
    )
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dataset-path", default="datasets/metaworld_mt50")
    parser.add_argument("--source", choices=("rollouts", "demonstrations"), default="rollouts")
    parser.add_argument("--traces")
    parser.add_argument("--comparison", choices=("coap", "policy"), default="coap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        if args.mode in ("evaluate", "collect-calibration"):
            print(json.dumps(eval_command(cfg, args.mode, "<resolved_coap.yaml>", args.comparison), indent=2))
        return
    if args.mode == "prepare-actions":
        from .data import prepare_actions

        print(json.dumps(prepare_actions(cfg.training.cache, args.dataset_path), indent=2))
    elif args.mode == "train":
        from .train import train

        print(train(cfg))
    elif args.mode == "calibrate":
        from .calibration import calibrate

        print(json.dumps(calibrate(cfg, source=args.source, traces=args.traces), indent=2))
    elif args.mode == "summarize":
        from .calibration import summarize

        print(
            json.dumps(
                summarize(args.traces or Path(cfg.evaluation.output) / "coap_episodes.jsonl"), indent=2
            )
        )
    else:
        output = cfg.evaluation.output + (
            "_calibration"
            if args.mode == "collect-calibration"
            else "_policy"
            if args.comparison == "policy"
            else ""
        )
        folder = Path(output)
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Use a fresh COAP evaluation output: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        resolved = folder / "resolved_coap.yaml"
        resolved.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        subprocess.run(eval_command(cfg, args.mode, resolved.resolve(), args.comparison), check=True)


if __name__ == "__main__":
    main()
