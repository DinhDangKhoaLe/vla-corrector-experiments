"""CLI: python -m dynamic_horizon.run {calibrate,train,evaluate,check} --config ..."""
import argparse
import json
import os
from pathlib import Path

from .config import load_config, make_method, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "calibrate", "train", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="Override results directory (must be new unless resuming)")
    parser.add_argument("--artifact", help="Override PACE calibration / BCP head checkpoint path")
    parser.add_argument("--tasks", help="Comma-separated difficulty groups or task names")
    parser.add_argument("--episodes", type=int, help="Evaluation episodes per task")
    parser.add_argument("--episode-length", type=int, help="Maximum control steps per rollout")
    parser.add_argument("--steps", type=int, help="Total BCP training updates")
    parser.add_argument("--resume", action="store_true", help="Resume BCP from its saved head/optimizer")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    config = load_config(args.config)
    if args.artifact:
        if config["method"] not in ("pace", "bcp"):
            parser.error("--artifact applies only to PACE and BCP")
        config["settings"]["artifact"] = resolve_path(args.artifact)
    if args.tasks:
        config["evaluation"]["tasks"] = args.tasks
        if "training" in config:
            config["training"]["tasks"] = args.tasks
    for argument, section, key in ((args.episodes, "evaluation", "episodes_per_task"),
                                   (args.episode_length, "evaluation", "episode_length"),
                                   (args.steps, "training", "steps")):
        if argument is not None:
            if argument < 1 or section not in config:
                parser.error(f"Invalid override for {key}")
            config[section][key] = argument
    if args.resume and args.command != "train":
        parser.error("--resume applies only to BCP training")
    if args.command == "check":
        from .rollout import task_names
        required = [Path(config["policy"]["checkpoint"]) / "model.safetensors",
                    Path(config["policy"]["tokenizer_model"]), Path(config["dataset"]) / "meta/info.json"]
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        from .policy import load_checkpoint_config, SentencePieceTokenizer
        checkpoint_config = load_checkpoint_config(config["policy"]["checkpoint"])
        if checkpoint_config.chunk_size != 50 or checkpoint_config.output_features["action"].shape[0] != 4:
            raise ValueError("Expected a MetaWorld π0.5 checkpoint with H=50 and action_dim=4")
        SentencePieceTokenizer(config["policy"]["tokenizer_model"])
        print(json.dumps({"config": config, "tasks": task_names(config["evaluation"]["tasks"]),
                          "note": "Paths/config checked; this does not execute policy inference."}, indent=2))
        return
    if args.command == "calibrate":
        if config["method"] != "pace":
            parser.error("Only PACE uses offline calibration; AutoHorizon/AAC/A³ need none, BCP needs train")
        from .pace import calibrate
        print(json.dumps(calibrate(config), indent=2))
        return
    if args.command == "train" and config["method"] != "bcp":
        parser.error("Only BCP needs training; use calibrate for PACE")
    if args.output:
        config["training" if args.command == "train" else "evaluation"]["output"] = resolve_path(args.output)
    output = Path(config["training" if args.command == "train" else "evaluation"]["output"])
    if output.exists() and not args.resume:
        raise FileExistsError(f"Output already exists; select a new --output: {output}")
    if args.command == "train":
        artifact = Path(config["settings"]["artifact"])
        if artifact.exists() and not args.resume:
            raise FileExistsError(f"BCP head exists; use --resume or a new --artifact: {artifact}")
        if args.resume and not artifact.is_file():
            raise FileNotFoundError(artifact)
    from .policy import Pi05Sampler
    if args.command == "evaluate":
        # Validate required artifacts before paying the model load cost.
        method = make_method(config)
        sampler = Pi05Sampler(config["policy"])
        from .rollout import evaluate
        result = evaluate(config, sampler, method, config["evaluation"]["output"])
    else:
        sampler = Pi05Sampler(config["policy"])
        from .train_bcp import train
        result = train(config, sampler, resume=args.resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
