"""Launch evaluations with shared defaults; --dry-run prints the exact command."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def command(args):
    benchmark, method = args.benchmark, args.method
    save_videos = bool(getattr(args, "save_videos", False))
    label = method + ("_no_ogg" if args.no_ogg else "")
    output = args.output or f"outputs/eval/{label}_{benchmark}_h{args.horizon}_seed{args.seed}"
    from dynamic_horizon.assets import ROBOTWIN_OFFICIAL_POLICY
    policy = args.policy or (ROBOTWIN_OFFICIAL_POLICY if benchmark == "robotwin" else f"pretrained_weights/pi05_{benchmark}_compat")
    prefix = [sys.executable, "-m"]
    if method.startswith("LACO_"):
        if benchmark not in {"metaworld", "libero"}:
            raise ValueError("LACO live evaluation supports MetaWorld and LIBERO")
        config = f"configs/laco/{method}.yaml" if benchmark == "metaworld" else f"configs/laco/libero_{method}.yaml"
        return prefix + ["laco.evaluate", "--config", config,
            "--set", f"evaluation.policy={policy}", "--set", f"evaluation.action_horizon={args.horizon}",
            "--set", f"evaluation.seed={args.seed}", "--set", f"evaluation.episodes={args.episodes}",
            "--set", f"evaluation.output={output}", "--set", f"evaluation.save_video={str(save_videos).lower()}"]
    if method == "corrector":
        return prefix + ["siglip_dynamics.experiments.run", "--config", f"configs/corrector/{benchmark}_original_pool64.yaml",
            "--mode", "evaluate", "--set", f"evaluation.policy={policy}",
            "--set", f"evaluation.n_action_steps={args.horizon}", "--set", f"evaluation.seed={args.seed}",
            "--set", f"evaluation.episodes={args.episodes}", "--set", f"evaluation.output={output}",
            "--set", f"evaluation.guidance={'false' if args.no_ogg else 'true'}",
            "--set", f"evaluation.save_video={str(save_videos).lower()}"]
    if benchmark == "robotwin":
        cmd = prefix + ["siglip_dynamics.experiments.run", "--config", "configs/baseline/robotwin_pi05.yaml",
                        "--mode", "evaluate", "--set", f"evaluation.policy={policy}",
                        "--set", f"evaluation.n_action_steps={args.horizon}", "--set", f"evaluation.seed={args.seed}",
                        "--set", f"evaluation.episodes={args.episodes}", "--set", f"evaluation.output={output}"]
        if method == "pace":
            cmd += ["--set", "evaluation.pace_config=configs/pace/robotwin_pi05.yaml"]
        return cmd
    cfg = yaml.safe_load(Path(f"configs/baseline/{benchmark}_pi05.yaml").read_text())
    params = {"policy.path": policy, "policy.device": cfg["device"], "policy.n_action_steps": args.horizon,
              "policy.chunk_size": cfg["chunk_size"], "policy.compile_model": False,
              "env.type": benchmark, "env.task": cfg["tasks"], "env.max_parallel_tasks": 1,
              "eval.batch_size": 1, "eval.n_episodes": args.episodes, "eval.use_async_envs": False,
              "seed": args.seed, "output_dir": output, "rename_map": cfg["rename_map"]}
    if "episode_length" in cfg:
        params["env.episode_length"] = cfg["episode_length"]
    module = "lerobot.scripts.lerobot_eval"
    if method == "baseline":
        params["max_episodes_rendered"] = args.episodes if save_videos else 0
    if method == "pace":
        module = "lerobot.scripts.lerobot_eval_modified_detection"
        params.update({"pace_config": f"configs/pace/{benchmark}_pi05.yaml", "pace_disable_corrector": True,
                       "meltdown_use_guidance_replan": False, "compute_baseline_metrics": False,
                       "save_raw_video": save_videos, "save_sync_video_with_curve": False,
                       "max_visualizations_per_task": 0,
                       "save_summary_csv": True, "save_summary_json": True,
                       "save_guidance_dual_video": False, "save_baseline_sync_video_with_curve": False,
                       "save_baseline_et_curve_plot": False, "save_baseline_step_csv": False,
                       "save_et_curve_plot": False, "save_task_plots": False})
    return prefix + [module] + [f"--{key}={json.dumps(value) if isinstance(value, (dict, list, bool)) else value}"
                               for key, value in params.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["baseline", "LACO_replan", "LACO_adjust", "corrector", "pace"])
    parser.add_argument("--benchmark", default="metaworld", choices=["metaworld", "libero", "robotwin"])
    parser.add_argument("--horizon", type=int, choices=[50, 40, 30, 20, 10], default=50,
                        help="Maximum executed actions per policy call; prediction chunk remains 50")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy")
    parser.add_argument("--output")
    parser.add_argument("--no-ogg", action="store_true")
    parser.add_argument("--save-videos", action="store_true", help="Save one rollout video per episode per task")
    parser.add_argument("--resume", action="store_true", help="Skip completed evaluations and archive partial output before retrying")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.horizon <= 50 or args.episodes < 1:
        parser.error("horizon must be 1..50; episodes must be positive")
    if args.no_ogg and args.method != "corrector":
        parser.error("--no-ogg applies to Corrector only")
    cmd = command(args)
    print(shlex.join(cmd), flush=True)
    if not args.dry_run:
        if args.benchmark == "robotwin":
            from dynamic_horizon.assets import ROBOTWIN_OFFICIAL_POLICY
            if not (Path(args.policy or ROBOTWIN_OFFICIAL_POLICY) / "model.safetensors").is_file():
                raise FileNotFoundError("RoboTwin requires a converted LeRobot export of the official checkpoint. "
                                        "Downloading/unpacking the OpenPI archive does not create one. See docs/robotwin_official.md.")
        request = {key: getattr(args, key) for key in ("benchmark", "method", "no_ogg", "horizon", "seed", "episodes", "policy")}
        for option in cmd:
            if option.startswith(("evaluation.output=", "--output_dir=")):
                path = Path(option.split("=", 1)[1])
                if args.resume:
                    from dynamic_horizon.resume import prepare_output
                    if args.benchmark not in {"metaworld", "libero"}:
                        raise ValueError("Resume currently supports MetaWorld and LIBERO")
                    if prepare_output(path, request, 50 if args.benchmark == "metaworld" else 40, args.save_videos):
                        return
                if path.exists() and any(path.iterdir()):
                    raise FileExistsError(f"Use a fresh --output directory: {path}")
        subprocess.run(cmd, check=True)
        if args.benchmark in {"metaworld", "libero"}:
            from dynamic_horizon.report import write_report
            output = next(option.split("=", 1)[1] for option in cmd
                          if option.startswith(("evaluation.output=", "--output_dir=")))
            summary = write_report(Path(output))
            if args.save_videos and summary["videos"] != summary["episodes"]:
                raise RuntimeError(f"Expected {summary['episodes']} videos, found {summary['videos']} in {output}/videos")
            (Path(output) / "evaluation_request.json").write_text(json.dumps(request, indent=2) + "\n")


if __name__ == "__main__":
    main()
