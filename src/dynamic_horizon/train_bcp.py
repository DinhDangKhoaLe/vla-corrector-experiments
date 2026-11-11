"""Serial single-GPU GRPO training of BCP's continuation head.

One fixed-H=50 reference and G-1 adaptive rollouts share each initial task
instance. The base policy is frozen; only the continuation head is optimized.
"""
import json
import random
from pathlib import Path

import torch

from .bcp import BCP, ContinuationHead, clipped_objective, efficiency_reward
from .rollout import FixedHorizon, episode, make_env, seed_policy, task_names


def optimize_group(head, optimizer, trajectories, rewards, settings):
    """Reference is last in rewards and contributes to advantage normalization."""
    device = next(head.parameters()).device
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
    decisions = sum(map(len, trajectories))
    if not decisions or len(rewards) != len(trajectories) + 1:
        raise ValueError("GRPO needs adaptive decisions and exactly one reference reward")
    head.train()
    metrics = {}
    for _ in range(settings["update_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_entropy = 0.0, 0.0
        for trajectory, advantage in zip(trajectories, advantages[:-1], strict=True):
            for record in trajectory:
                log_probs = head(**{key: value.to(device) for key, value in record["inputs"].items()})[0]
                entropy = -(log_probs.exp() * log_probs).sum()
                loss = clipped_objective(
                    log_probs[record["index"]], record["old_logp"], advantage, entropy,
                    clip_low=settings["clip_low"], clip_high=settings["clip_high"],
                    entropy_bonus=settings["entropy_bonus"],
                ) / decisions
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite BCP objective")
                loss.backward()
                total_loss += float(loss.detach())
                total_entropy += float(entropy.detach()) / decisions
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), settings["gradient_clip"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Non-finite BCP gradient")
        optimizer.step()
        metrics = {"loss": total_loss, "entropy": total_entropy, "gradient_norm": float(gradient_norm)}
    head.eval()
    return {**metrics, "decisions": decisions, "rewards": rewards.cpu().tolist(),
            "advantages": advantages.cpu().tolist()}


def train(config, sampler, resume=False):
    settings = config["training"]
    if settings["group_size"] < 2 or settings["steps"] < 1 or settings["update_epochs"] < 1:
        raise ValueError("BCP requires group_size >= 2 and positive training steps/epochs")
    artifact = Path(config["settings"]["artifact"])
    output = Path(settings["output"])
    seed_policy(settings["seed"])
    head = ContinuationHead(**config["settings"]["head"]).to(config["policy"]["device"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    start_step = 0
    if resume:
        saved = torch.load(artifact, map_location="cpu", weights_only=True)
        if saved["head_config"] != config["settings"]["head"] or saved["policy"] != config["policy"]:
            raise ValueError("Resume model configuration differs from checkpoint")
        previous = saved["config"]["training"]
        if any(previous[k] != v for k, v in settings.items() if k != "steps"):
            raise ValueError("Resume may change only the total number of training steps")
        if saved["config"]["evaluation"]["episode_length"] != config["evaluation"]["episode_length"]:
            raise ValueError("Resume episode length differs from checkpoint")
        head.load_state_dict(saved["head"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        output.mkdir(parents=True, exist_ok=True)
    else:
        if artifact.exists():
            raise FileExistsError(f"BCP artifact exists; use --resume or a new --artifact: {artifact}")
        output.mkdir(parents=True, exist_ok=False)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tasks = task_names(settings["tasks"])
    method = BCP(head.eval(), stochastic=True)
    status_path = output / "summary.json"
    status_path.write_text(json.dumps({"status": "running", "completed_steps": start_step}))
    try:
        with (output / "training.jsonl").open("a" if resume else "w") as log:
            for step in range(start_step, settings["steps"]):
                task = random.Random(settings["seed"] + step).choice(tasks)
                env_seed = settings["seed"] + step
                env = make_env(task, config["evaluation"])
                trajectories, rollouts = [], []
                try:
                    reference, _ = episode(env, sampler, FixedHorizon(), task, config["evaluation"], env_seed,
                                           policy_seed=settings["seed"] + step * settings["group_size"])
                    for index in range(settings["group_size"] - 1):
                        policy_seed = settings["seed"] + step * settings["group_size"] + index + 1
                        generator = torch.Generator(device=config["policy"]["device"]).manual_seed(policy_seed)
                        result, records = episode(env, sampler, method, task, config["evaluation"], env_seed,
                                                  policy_seed=policy_seed, generator=generator)
                        if result["initial_state_sha256"] != reference["initial_state_sha256"]:
                            raise RuntimeError("BCP group rollouts did not share the same initial physical state")
                        rollouts.append(result)
                        trajectories.append(records)
                finally:
                    env.close()
                rewards = [efficiency_reward(r["success"], r["queries"], reference["queries"],
                                              settings["delta_plus"], settings["delta_minus"])
                           for r in rollouts] + [float(reference["success"])]
                metrics = optimize_group(head, optimizer, trajectories, rewards, settings)
                if any(parameter.grad is not None for parameter in sampler.policy.parameters()):
                    raise RuntimeError("Frozen π0.5 received gradients")
                row = {"step": step + 1, "task": task, "env_seed": env_seed,
                       "reference": reference, "adaptive": rollouts, **metrics}
                log.write(json.dumps(row) + "\n")
                log.flush()
                temporary = artifact.with_suffix(artifact.suffix + ".tmp")
                torch.save({"step": step + 1, "head_config": config["settings"]["head"],
                            "head": head.state_dict(), "optimizer": optimizer.state_dict(),
                            "policy": config["policy"], "config": config}, temporary)
                temporary.replace(artifact)
                print(json.dumps({k: v for k, v in row.items() if k not in ("adaptive", "reference")}), flush=True)
                status_path.write_text(json.dumps({"status": "running", "completed_steps": step + 1}))
        result = {"status": "complete", "completed_steps": max(start_step, settings["steps"]), "artifact": str(artifact)}
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        return result
    except BaseException as exc:
        status_path.write_text(json.dumps({"status": "failed", "error": str(exc),
                                           "resume_from": str(artifact)}, indent=2) + "\n")
        raise
