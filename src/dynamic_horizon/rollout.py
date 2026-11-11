"""MetaWorld execution with the repository's camera, task and success conventions."""
import json
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from .types import Decision


def task_names(specification):
    from lerobot.envs.metaworld import DIFFICULTY_TO_TASKS, TASK_NAME_TO_ID
    result = []
    for item in specification.split(","):
        item = item.strip()
        for task in DIFFICULTY_TO_TASKS.get(item, [item]):
            if task not in TASK_NAME_TO_ID:
                raise ValueError(f"Unknown MetaWorld task/group: {task}")
            if task not in result:
                result.append(task)
    if not result:
        raise ValueError("At least one task is required")
    return result


def seed_policy(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(task, settings):
    from lerobot.envs.metaworld import MetaworldEnv
    return MetaworldEnv(task, obs_type="pixels_agent_pos", camera_name=settings["camera"],
                        max_path_length=settings["episode_length"])


class FixedHorizon:
    def decide(self, sampler, task, **kwargs):
        return Decision(sampler.sample().actions)


def reset_env(env, seed):
    import mujoco
    seed_policy(seed)
    # MetaWorld resets the hand through simulation. Clearing residual velocities
    # and solver state first makes repeated task instances physically identical.
    mujoco.mj_resetData(env._env.model, env._env.data)
    observation, _ = env.reset(seed=seed)
    state = np.concatenate([env._env.data.qpos, env._env.data.qvel, env._env._last_rand_vec])
    fingerprint = hashlib.sha256(state.tobytes()).hexdigest()
    return observation, fingerprint


def episode(env, sampler, method, task, settings, env_seed, policy_seed=None, generator=None, log=None):
    from lerobot.envs.metaworld import TASK_DESCRIPTIONS, TASK_NAME_TO_ID
    # This installed MetaWorld release ignores reset(seed) for its global-NumPy
    # reset vectors. Seed before reset as well as after it, so BCP group members
    # and all evaluated methods receive the same initial physical task instance.
    observation, initial_state = reset_env(env, env_seed)
    seed_policy(env_seed if policy_seed is None else policy_seed)
    sampler.policy.reset()
    steps, queries, success, total_reward = 0, 0, False, 0.0
    records, training = [], []
    start = time.perf_counter()
    initial_samples, initial_denoises = sampler.samples, sampler.denoising_calls
    while steps < settings["episode_length"]:
        query_start = time.perf_counter()
        before_samples = sampler.samples
        sampler.prepare(observation, TASK_DESCRIPTIONS[task])
        decision = method.decide(sampler, TASK_NAME_TO_ID[task], generator=generator)
        if not 1 <= decision.horizon <= 50 or decision.actions.shape[1:] != (4,):
            raise ValueError(f"Invalid executable action chunk: {decision.actions.shape}")
        if not torch.isfinite(decision.actions).all():
            raise FloatingPointError("Non-finite action chunk")
        query_seconds = time.perf_counter() - query_start
        queries += 1
        if decision.training is not None:
            training.append(decision.training)
        selected_horizon = decision.horizon
        actions = decision.actions.detach().cpu().numpy()
        record = {"query": queries, "start_step": steps, "selected_horizon": selected_horizon,
                  "samples": sampler.samples - before_samples, "query_seconds": query_seconds,
                  "diagnostics": decision.diagnostics}
        executed = 0
        for action in actions[:settings["episode_length"] - steps]:
            observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            total_reward += float(reward)
            steps += 1
            executed += 1
            success = success or bool(info.get("is_success", False))
            if terminated or truncated or success:
                break
        record["executed_horizon"] = executed
        records.append(record)
        if log is not None:
            log.write(json.dumps(record) + "\n")
            log.flush()
        if terminated or truncated or success:
            break
    result = {"task": task, "env_seed": env_seed, "initial_state_sha256": initial_state,
              "policy_seed": env_seed if policy_seed is None else policy_seed,
              "success": success, "steps": steps, "queries": queries, "reward": total_reward,
              "seconds": time.perf_counter() - start,
              "samples": sampler.samples - initial_samples,
              "denoising_calls": sampler.denoising_calls - initial_denoises,
              "selected_horizons": [r["selected_horizon"] for r in records],
              "executed_horizons": [r["executed_horizon"] for r in records]}
    return result, training


def evaluate(config, sampler, method, output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    status_path = directory / "summary.json"
    status_path.write_text(json.dumps({"status": "running"}))
    settings = config["evaluation"]
    results = []
    try:
        with (directory / "episodes.jsonl").open("w") as episodes:
            for task in task_names(settings["tasks"]):
                env = make_env(task, settings)
                try:
                    for _ in range(settings["episodes_per_task"]):
                        index = len(results)
                        with (directory / f"episode_{index:04d}_queries.jsonl").open("w") as log:
                            result, _ = episode(env, sampler, method, task, settings, settings["seed"] + index, log=log)
                        results.append(result)
                        episodes.write(json.dumps(result) + "\n")
                        episodes.flush()
                        print(json.dumps(result), flush=True)
                finally:
                    env.close()
        summary = {"status": "complete", "episodes": len(results),
                   "success_rate": sum(r["success"] for r in results) / len(results),
                   "queries": sum(r["queries"] for r in results),
                   "samples": sum(r["samples"] for r in results),
                   "by_task": {task: {"episodes": len(subset),
                                      "success_rate": sum(r["success"] for r in subset) / len(subset)}
                               for task in task_names(settings["tasks"])
                               if (subset := [r for r in results if r["task"] == task])}}
        status_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException as exc:
        status_path.write_text(json.dumps({"status": "failed", "completed_episodes": len(results),
                                           "error": str(exc)}, indent=2) + "\n")
        raise
