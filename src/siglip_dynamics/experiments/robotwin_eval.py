"""Native RoboTwin/SAPIEN evaluator for VLA-Corrector variants.

The module is deliberately dependency-lazy: MetaWorld-only installations can still
import the experiment runner.  A native rollout requires RoboTwin's simulator
requirements (SAPIEN, MPLib, TOPPRA and its assets) in the *same* environment as
the local PI0.5 checkpoint.
"""

from __future__ import annotations

import csv
import importlib
import json
import logging
import os
import random
import sys
from collections import deque
from dataclasses import fields
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
import yaml

from .runtime import ExperimentPredictor


def _require_robotwin(root: Path) -> None:
    missing = [name for name in ("envs", "env_cfg/task_config/demo_clean.yml") if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Invalid evaluation.robotwin_root={root}; missing: {missing}")
    try:
        import sapien  # noqa: F401
        import mplib  # noqa: F401
        import toppra  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Native RoboTwin evaluation needs SAPIEN, MPLib and TOPPRA. The current environment does not "
            "have the official RoboTwin simulator stack installed. Install it in a dedicated compatible "
            "RoboTwin+PI0.5 environment before running collect-calibration/evaluate."
        ) from exc


def _add_robotwin_paths(root: Path) -> None:
    for path in (root, root / "scripts", root / "description" / "utils"):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def _tasks(cfg) -> list[str]:
    tasks = [value.strip() for value in cfg.evaluation.robotwin_tasks.split(",") if value.strip()]
    if not tasks:
        metadata = json.loads((Path(cfg.training.cache) / "metadata.json").read_text(encoding="utf-8"))
        tasks = list(metadata.get("dataset_kwargs", {}).get("robotwin_tasks", []))
    if not tasks:
        raise ValueError("Set evaluation.robotwin_tasks for RoboTwin evaluation")
    return tasks


def _load_task_api(root: Path):
    _add_robotwin_paths(root)
    try:
        official = importlib.import_module("eval_policy_xpolicylab")
        unstable = importlib.import_module("envs.utils.create_actor").UnStableError
    except Exception as exc:  # pragma: no cover - requires simulator installation
        raise RuntimeError("Could not import the installed RoboTwin task API") from exc
    return official, unstable


def _policy_stack(cfg):
    """Load PI0.5 plus the exact checkpoint processor pipeline."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05_modified.configuration_pi05_modified import PI05ModifiedConfig
    from lerobot.policies.pi05_modified.modeling_pi05_modified import PI05ModifiedPolicy

    policy_path = Path(cfg.evaluation.policy)
    if not policy_path.is_dir():
        raise FileNotFoundError(f"RoboTwin PI0.5 policy is not a local directory: {policy_path}")
    base = PreTrainedConfig.from_pretrained(str(policy_path))
    if not isinstance(base, PI05Config):
        raise TypeError(f"RoboTwin evaluator requires PI0.5, got {type(base).__name__}")
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values["n_action_steps"] = int(cfg.evaluation.n_action_steps)
    values["chunk_size"] = int(cfg.evaluation.chunk_size)
    values["device"] = str(cfg.training.device)
    modified = PI05ModifiedConfig(**values)
    modified.pretrained_path = policy_path
    policy = PI05ModifiedPolicy.from_pretrained(str(policy_path), config=modified).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=modified,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": str(cfg.training.device)}},
    )
    return policy, preprocessor, postprocessor


def _model_input(observation: dict[str, Any], instruction: str, preprocessor):
    source = observation["observation"]
    images = {
        "observation.images.cam_high": source["head_camera"]["rgb"],
        "observation.images.cam_left_wrist": source["left_camera"]["rgb"],
        "observation.images.cam_right_wrist": source["right_camera"]["rgb"],
    }
    batch: dict[str, Any] = {
        "observation.state": torch.as_tensor(observation["joint_action"]["vector"], dtype=torch.float32)[None],
        "task": [instruction],
    }
    for key, image in images.items():
        rgb = np.asarray(image, dtype=np.uint8)
        batch[key] = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(torch.float32) / 255.0
    return preprocessor(batch)


def _latent(policy, model_input: dict[str, Any]) -> np.ndarray:
    return np.asarray(policy.encode_siglip_only(model_input), dtype=np.float32)


def _env_actions(postprocessor, actions: torch.Tensor) -> np.ndarray:
    """Convert PI0.5 normalized action samples back to 14-D RoboTwin qpos targets."""
    out = postprocessor(actions)
    result = np.asarray(out.detach().to("cpu"), dtype=np.float32)
    if result.ndim == 3:
        result = result[0]
    if result.ndim != 2 or result.shape[1] != 14:
        raise ValueError(f"PI0.5 postprocessor returned {result.shape}; expected (steps, 14)")
    return result


def _build_instruction(official, task_args: dict[str, Any], episode_info: dict[str, Any], seed: int, episodes: int) -> str:
    try:
        return official.build_instruction(task_args, episode_info, task_args.get("eval_instruction"), episodes, seed)
    except Exception as exc:  # pragma: no cover - external description assets vary
        logging.warning("RoboTwin instruction generation failed for seed %d: %s; using task name", seed, exc)
        return str(task_args["task_name"])


def _close(env) -> None:
    try:
        env.close_env()
    except Exception:
        pass
    try:
        if getattr(env, "viewer", None) is not None:
            env.viewer.close()
    except Exception:
        pass


def _make_monitor(cfg, predictor, mode: str, output: Path):
    if not cfg.features.uncertainty_monitor:
        # The paper-style Corrector has no calibration phase.  Its online MAD/
        # EWMA detector is instantiated per episode in ``_run_one`` below.
        if mode != "evaluate":
            raise ValueError("The legacy original Corrector supports evaluation only")
        return None
    from .uncertainty import UncertaintyMonitor
    return UncertaintyMonitor(cfg, predictor, mode, output, policy_signature={"policy": str(cfg.evaluation.policy)})


def _legacy_threshold_state(cfg):
    """Build the exact legacy MAD/EWMA detector used by the MetaWorld evaluator.

    Numerical thresholds are deliberately *not* copied from a MetaWorld run:
    they operate on the RoboTwin residual stream.  The defaults and state
    transition function are shared with the established evaluator; optional
    ``evaluation.thresholds`` values use the same CLI-style key names.
    """
    from lerobot.scripts.lerobot_eval_modified_detection import RobustThresholdState

    values = dict(cfg.evaluation.thresholds)
    allowed = {
        "threshold_window_size", "threshold_base_noise_floor", "threshold_bootstrap_initial_threshold",
        "threshold_bootstrap_queue_size", "threshold_bootstrap_sigma_scale", "threshold_ewma_alpha",
        "threshold_k_on", "threshold_k_off", "threshold_trigger_margin", "threshold_jump_trigger",
        "threshold_trigger_consecutive_steps", "threshold_reset_consecutive_steps",
        "threshold_hard_retrigger_margin", "threshold_hard_retrigger_consecutive_steps",
        "meltdown_cooldown_steps",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown legacy RoboTwin threshold settings: {sorted(unknown)}")
    defaults = {
        "window_size": 15,
        "base_noise_floor": 0.25,
        "bootstrap_initial_threshold": 10.0,
        "bootstrap_queue_size": 15,
        "bootstrap_sigma_scale": 1.0,
        "ewma_alpha": 0.35,
        "k_on": 3.0,
        "k_off": 2.0,
        "trigger_margin": 0.02,
        "jump_trigger": 0.06,
        "trigger_consecutive_steps": 5,
        "reset_consecutive_steps": 5,
        "hard_retrigger_margin": 0.08,
        "hard_retrigger_consecutive_steps": 3,
        # Keep the experiment YAML's explicit recovery cooldown unless the
        # legacy-style threshold override is supplied.
        "cooldown_steps": int(cfg.evaluation.cooldown_steps),
    }
    for source, target in (
        ("threshold_window_size", "window_size"),
        ("threshold_base_noise_floor", "base_noise_floor"),
        ("threshold_bootstrap_initial_threshold", "bootstrap_initial_threshold"),
        ("threshold_bootstrap_queue_size", "bootstrap_queue_size"),
        ("threshold_bootstrap_sigma_scale", "bootstrap_sigma_scale"),
        ("threshold_ewma_alpha", "ewma_alpha"),
        ("threshold_k_on", "k_on"),
        ("threshold_k_off", "k_off"),
        ("threshold_trigger_margin", "trigger_margin"),
        ("threshold_jump_trigger", "jump_trigger"),
        ("threshold_trigger_consecutive_steps", "trigger_consecutive_steps"),
        ("threshold_reset_consecutive_steps", "reset_consecutive_steps"),
        ("threshold_hard_retrigger_margin", "hard_retrigger_margin"),
        ("threshold_hard_retrigger_consecutive_steps", "hard_retrigger_consecutive_steps"),
        ("meltdown_cooldown_steps", "cooldown_steps"),
    ):
        if source in values:
            defaults[target] = values[source]
    return RobustThresholdState(**defaults)


def _run_one(
    *, env, policy, preprocessor, postprocessor, monitor, cfg, instruction: str, task: str, seed: int,
    video_path: Path | None,
) -> dict[str, Any]:
    policy.reset()
    if monitor is not None:
        monitor.reset(task, seed)
    else:
        # Import the shared cosine residual / threshold transition rather than
        # maintaining a second detector implementation for RoboTwin.
        from .runtime import cosine_error
        from lerobot.scripts.lerobot_eval_modified_detection import robust_threshold_step
        legacy_state = _legacy_threshold_state(cfg)
        legacy_pending: deque[dict[str, Any]] = deque()
        legacy_events: list[dict[str, float | int | bool]] = []
    obs = env.get_obs()
    frames: list[np.ndarray] = []
    queue: deque[tuple[np.ndarray, np.ndarray]] = deque()
    executed = 0
    replans = 0
    ogg_replans = 0
    episode_error = ""
    # Set after a legacy alarm and consumed by the next (forced) PI0.5 call.
    guidance_replan = False
    guidance_current_z_t = None
    guidance_current_delta_z_t = None
    guidance_current_delta_z_right = None
    try:
        while not env.eval_success and env.take_action_cnt < env.step_lim:
            model_input = _model_input(obs, instruction, preprocessor)
            z_now = _latent(policy, model_input)
            if not queue:
                response = policy.predict_action_chunk_with_aux(
                    model_input,
                    guidance_replan=guidance_replan,
                    guidance_current_z_t=guidance_current_z_t,
                    guidance_current_z_hist_t=None,
                    guidance_current_delta_z_t=guidance_current_delta_z_t,
                    guidance_current_delta_z_right=guidance_current_delta_z_right,
                    guidance_apply_every=int(cfg.evaluation.guidance_apply_every),
                    guidance_loss_objective="attract_delta_z_correction",
                    guidance_compare_baseline=False,
                )
                if guidance_replan:
                    ogg_replans += 1
                guidance_replan = False
                guidance_current_z_t = None
                guidance_current_delta_z_t = None
                guidance_current_delta_z_right = None
                normalized = response["actions"][0, : cfg.evaluation.n_action_steps].detach().to("cpu").numpy()
                native = _env_actions(postprocessor, response["actions"][:, : cfg.evaluation.n_action_steps])
                for raw_action, model_action in zip(native, normalized, strict=True):
                    queue.append((raw_action, model_action.astype(np.float32, copy=False)))
                if monitor is not None:
                    monitor.schedule(z_now, [entry[1] for entry in queue], executed)
            raw_action, model_action = queue.popleft()
            if monitor is None:
                # Match the original VLA-Corrector schedule: each executed
                # action receives a K-step forecast from its own current latent.
                prediction = predictor.predict_delta_z(z_now, model_action)
                legacy_pending.append({"due": executed + predictor.k_step, "start": z_now,
                                       "prediction": np.asarray(prediction, dtype=np.float32)})
            if video_path is not None:
                frames.append(np.asarray(obs["observation"]["head_camera"]["rgb"], dtype=np.uint8))
            env.take_action(raw_action, action_type="qpos")
            executed += 1
            obs = env.get_obs()
            z_after = _latent(policy, _model_input(obs, instruction, preprocessor))
            if monitor is not None:
                event = monitor.observe(z_after, executed)
                if event is not None:
                    replan, ogg = monitor.recovery_decision(event, executed)
                    if replan:
                        queue.clear()
                        monitor.cancel()
                        # Match the MetaWorld uncertainty branch: only a
                        # calibrated/validated OGG decision changes the
                        # next replan; every other alarm gets an ordinary
                        # fresh PI0.5 chunk.
                        guidance_replan = bool(ogg)
                        if guidance_replan:
                            guidance_current_z_t = z_after
                            guidance_current_delta_z_t = event["actual"]
                            guidance_current_delta_z_right = event["prediction"]
                            predictor.model.guidance_horizon = int(event["horizon"])
                        replans += 1
            elif legacy_pending and legacy_pending[0]["due"] == executed:
                pending = legacy_pending.popleft()
                actual = z_after - pending["start"]
                e_t = cosine_error(pending["prediction"], actual)
                threshold, replan = robust_threshold_step(e_t, legacy_state)
                legacy_events.append({"step": executed, "residual": float(e_t), "threshold": float(threshold),
                                      "alarm": bool(replan)})
                if replan:
                    # This is the same recovery contract as MetaWorld: discard
                    # remaining actions and immediately request a fresh chunk.
                    queue.clear()
                    # Pending forecasts describe actions from the discarded plan.
                    legacy_pending.clear()
                    guidance_replan = bool(cfg.evaluation.guidance and cfg.evaluation.guidance_eta > 0)
                    guidance_current_z_t = z_after
                    guidance_current_delta_z_t = actual
                    guidance_current_delta_z_right = pending["prediction"]
                    predictor.model.guidance_horizon = int(predictor.k_step)
                    replans += 1
        success = bool(env.eval_success)
    except Exception as exc:  # Preserve a trace row for calibration diagnostics.
        success = False
        episode_error = f"{type(exc).__name__}: {exc}"
        logging.exception("RoboTwin episode crashed | task=%s seed=%d", task, seed)
    if video_path is not None and frames:
        imageio.mimwrite(video_path, frames, fps=10)
    if monitor is not None:
        monitor.finish(success, episode_error)
    return {"task": task, "seed": seed, "success": success, "steps": executed,
            "replans": replans, "ogg_replans": ogg_replans, "error": episode_error,
            "legacy_events": legacy_events if monitor is None else []}


def evaluate_robotwin(cfg, *, mode: str, config_path: Path) -> None:
    """Collect calibration traces or evaluate an uncertainty/original Corrector in RoboTwin."""
    if mode not in {"collect-calibration", "evaluate"}:
        raise ValueError(mode)
    # RoboTwin uses relative asset paths internally, so its simulator must run
    # with CWD at ROBOTWIN_ROOT. Resolve every experiment path first; otherwise
    # a perfectly valid ``outputs/...`` or ``pretrained_weights/...`` reference
    # would accidentally become relative to ``third_party/RoboTwin``.
    workspace = Path.cwd().resolve()

    def absolute(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else workspace / path)

    cfg.training.cache = absolute(cfg.training.cache)
    cfg.training.output = absolute(cfg.training.output)
    cfg.evaluation.policy = absolute(cfg.evaluation.policy)
    cfg.evaluation.output = absolute(cfg.evaluation.output)
    cfg.evaluation.robotwin_root = absolute(cfg.evaluation.robotwin_root)
    if cfg.evaluation.checkpoint:
        cfg.evaluation.checkpoint = absolute(cfg.evaluation.checkpoint)
    cfg.uncertainty.traces = absolute(cfg.uncertainty.traces)
    cfg.uncertainty.artifact = absolute(cfg.uncertainty.artifact)
    root = Path(cfg.evaluation.robotwin_root)
    task_names = _tasks(cfg)
    # Bind calibration to the native task suite rather than the MetaWorld
    # placeholder in Evaluation.tasks.
    cfg.evaluation.tasks = "robotwin:" + cfg.evaluation.robotwin_task_config + ":" + ",".join(task_names)
    _require_robotwin(root)
    original_cwd = Path.cwd()
    os.chdir(root)
    try:
        official, unstable_error = _load_task_api(root)
        output = Path(cfg.evaluation.output + ("_calibration" if mode == "collect-calibration" else ""))
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use a fresh evaluation.output; {output} is not empty")
        output.mkdir(parents=True)
        (output / "resolved_experiment.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        policy, preprocessor, postprocessor = _policy_stack(cfg)
        checkpoint = Path(cfg.evaluation.checkpoint or Path(cfg.training.output) / "best_model.pt")
        predictor = ExperimentPredictor(checkpoint, cfg, cfg.training.device)
        # OGG is a policy-side gradient step.  Register the same Corrector used
        # by the alarm before any rollout; otherwise ``guidance_replan=True``
        # would silently sample an ordinary PI0.5 chunk.
        if cfg.evaluation.guidance and cfg.evaluation.guidance_eta > 0:
            policy.set_safety_predictor(
                predictor.model,
                action_dim=int(predictor.action_dim),
                model_type=str(getattr(predictor, "model_type", "mlp")),
                eta=float(cfg.evaluation.guidance_eta),
            )
        monitor = _make_monitor(cfg, predictor, mode, output)
        records: list[dict[str, Any]] = []
        videos = output / "videos"
        if cfg.evaluation.save_video:
            videos.mkdir()
        base_seed = int(cfg.evaluation.collection_seed if mode == "collect-calibration" else cfg.evaluation.seed)
        for task in task_names:
            task_args, _ = official.load_task_args({"task_name": task, "task_config": cfg.evaluation.robotwin_task_config,
                                                     "ckpt_setting": str(cfg.evaluation.policy), "policy_name": "pi05"})
            task_args["eval_mode"] = True
            task_args["render_freq"] = 0
            task_args["eval_video_log"] = False
            env = official.class_decorator(task)
            accepted = 0
            candidate = 100000 * (1 + base_seed)
            while accepted < cfg.evaluation.episodes:
                episode_info: dict[str, Any] = {"info": {}}
                try:
                    if cfg.evaluation.robotwin_expert_check:
                        env.setup_demo(now_ep_num=accepted, seed=candidate, is_test=True, **task_args)
                        episode_info = env.play_once()
                        viable = bool(env.plan_success and env.check_success())
                        _close(env)
                        if not viable:
                            candidate += 1
                            continue
                    env.setup_demo(now_ep_num=accepted, seed=candidate, is_test=True, **task_args)
                except unstable_error:
                    _close(env)
                    candidate += 1
                    continue
                instruction = _build_instruction(official, task_args, episode_info, candidate, cfg.evaluation.episodes)
                env.set_instruction(instruction=instruction)
                video = videos / f"{task}_ep{accepted:03d}_{candidate}.mp4" if cfg.evaluation.save_video else None
                result = _run_one(env=env, policy=policy, preprocessor=preprocessor, postprocessor=postprocessor,
                                  monitor=monitor, cfg=cfg, instruction=instruction, task=task, seed=candidate,
                                  video_path=video)
                records.append(result)
                logging.info("RoboTwin %s | task=%s ep=%d seed=%d success=%s steps=%d replans=%d", mode, task,
                             accepted, candidate, result["success"], result["steps"], result["replans"])
                _close(env)
                accepted += 1
                candidate += 1
        with open(output / "episode_summary.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["task", "seed", "success", "steps", "replans", "ogg_replans", "error"])
            writer.writeheader()
            writer.writerows(records)
        summary = {"mode": mode, "episodes": len(records), "successes": sum(r["success"] for r in records),
                   "success_rate": float(np.mean([r["success"] for r in records])) if records else 0.0}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logging.info("RoboTwin %s complete: %s", mode, summary)
    finally:
        os.chdir(original_cwd)
