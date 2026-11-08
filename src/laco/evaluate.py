"""Live MetaWorld evaluation of frozen PI0.5, error replanning, and LACo."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.envs.configs import MetaworldEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION

from .artifacts import model_ensemble
from .controller import Decision, LACoController, Mode
from .utils import load_config, merge_overrides, save_json


def _input(obs: Any, env: Any, env_pre: Any, pre: Any) -> dict[str, Any]:
    return pre(env_pre(add_envs_task(env, preprocess_observation(obs))))


@torch.no_grad()
def _latent(policy: PI05Policy, batch: dict[str, Any], pca: dict[str, np.ndarray]) -> np.ndarray:
    images, masks = policy._preprocess_images(batch)  # existing PI0.5 inference preprocessing
    blocks = [policy.model.paligemma_with_expert.embed_image(img) for img, mask in zip(images, masks, strict=False)
              if not isinstance(mask, torch.Tensor) or bool(mask[0].item())]
    if not blocks: blocks = [policy.model.paligemma_with_expert.embed_image(images[0])]
    pooled = torch.cat(blocks, dim=1)[0].float().mean(dim=0).cpu().numpy()
    return (pooled - pca["mean"]) @ pca["components"].T


def _success(info: dict[str, Any]) -> bool:
    value = info.get("is_success", False)
    return bool(np.asarray(value).reshape(-1)[0]) if isinstance(value, (np.ndarray, list, tuple)) else bool(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("evaluation", config)
    mode = str(c.get("mode", "full_laco"))
    if mode not in {"baseline", "replan", "laco", "full_laco"}: raise ValueError("mode must be baseline/replan/laco/full_laco")
    output = Path(c["output"]); output.mkdir(parents=True, exist_ok=True)
    pi_cfg = PI05Config.from_pretrained(c["policy"])
    pi_cfg.device = c.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    pi_cfg.n_action_steps = int(c.get("action_horizon", 10))
    policy = PI05Policy.from_pretrained(c["policy"], config=pi_cfg).eval()
    pre, post = make_pre_post_processors(policy_cfg=pi_cfg, pretrained_path=c["policy"], preprocessor_overrides={
        "device_processor": {"device": str(pi_cfg.device)},
        "rename_observations_processor": {"rename_map": c.get("rename_map", {"observation.image": "observation.images.base_0_rgb"})},
    })
    env_cfg = MetaworldEnv(task=str(c["tasks"]), episode_length=int(c.get("episode_limit", 400)))
    envs = make_env(env_cfg, n_envs=1)
    # Each entry is a distinct task environment; this preserves identical task setup per mode.
    models = artifact = dev = calibration = None
    pca = None
    if mode != "baseline":
        models, artifact, dev = model_ensemble(list(c["models"]), str(pi_cfg.device))
        calibration = json.loads(Path(c["calibration"]).read_text())
        pca_file = c.get("pca") or str(Path(c["cache"]) / "pca.npz")
        pca = dict(np.load(pca_file))
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pi_cfg)
    step_rows: list[dict[str, Any]] = []; episode_rows: list[dict[str, Any]] = []
    for group, task_envs in envs.items():
        for task_id, env in task_envs.items():
            for episode in range(int(c.get("episodes", 1))):
                policy.reset(); obs, _ = env.reset(seed=int(c.get("seed", 1000)) + episode)
                controller = None
                if mode != "baseline":
                    assert models is not None and artifact is not None and calibration is not None and dev is not None
                    controller = LACoController(models, artifact, calibration, device=dev,
                        action_low=float(c.get("action_low", -1.0)), action_high=float(c.get("action_high", 1.0)),
                        trust_region=float(c.get("trust_region", .15)), regularization=float(c.get("regularization", calibration.get("regularization", .1))),
                        kappa=float(c.get("kappa", 1.0)), correction_horizon=int(c.get("correction_horizon", 2)),
                        max_failed_retries=int(c.get("max_failed_retries", 1)), condition_limit=float(c.get("condition_limit", 1e4)),
                        use_uncertainty=(mode == "full_laco"))
                queue = np.empty((0, 4), np.float32); policy_calls = anomalies = adjusts = replans = correction_ok = correction_fail = 0
                t0 = time.perf_counter(); latent_s = corrector_s = 0.; success = False
                for step in range(int(c.get("episode_limit", 400))):
                    model_input = _input(obs, env, env_pre, pre)
                    if not len(queue):
                        start = time.perf_counter(); chunk = policy.predict_action_chunk(model_input)[0, :int(pi_cfg.n_action_steps)].detach().cpu().numpy(); policy_calls += 1
                        if controller is not None:
                            ls = time.perf_counter(); z = _latent(policy, model_input, pca); latent_s += time.perf_counter()-ls; controller.on_new_chunk(z, chunk)
                        queue = chunk.copy()
                    if controller is not None: action = controller.next_action()
                    else: action = queue[0].copy()
                    action_tensor = torch.as_tensor(action[None], dtype=torch.float32, device=pi_cfg.device)
                    env_action = env_post({ACTION: post(action_tensor)})[ACTION].cpu().numpy()
                    obs, _reward, terminated, truncated, info = env.step(env_action)
                    queue = queue[1:]
                    decision = None
                    if controller is not None:
                        controller.mark_executed(); model_input_next = _input(obs, env, env_pre, pre)
                        ls = time.perf_counter(); z = _latent(policy, model_input_next, pca); latent_s += time.perf_counter()-ls
                        cs = time.perf_counter(); decision = controller.step(z); corrector_s += time.perf_counter()-cs
                        queue = controller.queue.copy()
                        if mode == "replan" and decision.mode == Mode.ADJUST:
                            # Same calibrated detector, deliberately no local intervention.
                            controller.queue = np.empty((0, queue.shape[1]), dtype=np.float32)
                            queue = controller.queue
                            decision = Decision(Mode.REPLAN, None, decision.prediction_error, decision.filtered_error,
                                                decision.controllability, decision.predicted_gain, decision.gain_lcb,
                                                decision.uncertainty, "error_triggered_replan", decision.correction_success)
                        anomalies += int(decision.filtered_error > controller.threshold)
                        adjusts += int(decision.mode == Mode.ADJUST); replans += int(decision.mode == Mode.REPLAN)
                        correction_ok += int(decision.correction_success is True); correction_fail += int(decision.correction_success is False)
                    step_rows.append({"task_group": group, "task_id": task_id, "episode": episode, "step": step, "mode": decision.mode.value if decision else "BASELINE",
                                      "prediction_error": decision.prediction_error if decision else "", "filtered_error": decision.filtered_error if decision else "",
                                      "threshold": controller.threshold if controller else "", "rho": decision.controllability if decision else "", "gain": decision.predicted_gain if decision else "",
                                      "gain_lcb": decision.gain_lcb if decision else "", "uncertainty": decision.uncertainty if decision else "",
                                      "correction_success": decision.correction_success if decision else "", "reason": decision.reason if decision else ""})
                    success = _success(info)
                    if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]): break
                episode_rows.append({"task_group": group, "task_id": task_id, "episode": episode, "success": success, "steps": step+1,
                                     "policy_calls": policy_calls, "anomalies": anomalies, "adjustments": adjusts, "replans": replans, "interventions": adjusts + replans,
                                     "correction_successes": correction_ok, "correction_failures": correction_fail, "latent_seconds": latent_s,
                                     "corrector_seconds": corrector_s, "episode_seconds": time.perf_counter()-t0})
    env.close(); _write_csv(output / "steps.csv", step_rows); _write_csv(output / "episodes.csv", episode_rows)
    successful = [r for r in episode_rows if r["success"]]
    summary = {"mode": mode, "episodes": len(episode_rows), "success_rate": float(np.mean([r["success"] for r in episode_rows])),
               "mean_policy_calls": float(np.mean([r["policy_calls"] for r in episode_rows])), "mean_anomalies": float(np.mean([r["anomalies"] for r in episode_rows])),
               "mean_adjustments": float(np.mean([r["adjustments"] for r in episode_rows])), "mean_replans": float(np.mean([r["replans"] for r in episode_rows])),
               "total_latent_seconds": float(sum(r["latent_seconds"] for r in episode_rows)), "total_corrector_seconds": float(sum(r["corrector_seconds"] for r in episode_rows)),
               "total_episode_seconds": float(sum(r["episode_seconds"] for r in episode_rows)),
               # A proxy only: environment success is never exposed to LACo.
               "successful_episode_interventions_per_step_proxy": float(sum(r["interventions"] for r in successful) / max(1, sum(r["steps"] for r in successful)))}
    save_json(output / "summary.json", summary); return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", required=True); parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(); print(json.dumps(run(merge_overrides(load_config(args.config), args.set)), indent=2))


if __name__ == "__main__": main()
