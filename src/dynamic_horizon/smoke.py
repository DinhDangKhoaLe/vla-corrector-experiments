"""Real-model integration check. This is not a benchmark or a trained BCP result."""
import argparse
import json
import os
from pathlib import Path

import torch

from .config import ROOT, load_config, make_method
from .pace import calibrate
from .policy import Pi05Sampler
from .rollout import evaluate, make_env, reset_env, seed_policy
from .train_bcp import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory for smoke artifacts")
    parser.add_argument("--episode-length", type=int, default=55)
    args = parser.parse_args()
    if args.episode_length < 1:
        parser.error("--episode-length must be positive")
    os.environ.setdefault("MUJOCO_GL", "egl")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configurations = {}
    for name in ("autohorizon", "aac", "pace", "a3", "bcp"):
        suffix = "_official" if name == "pace" else ""
        config = load_config(ROOT / f"configs/{name}/metaworld_pi05{suffix}.yaml")
        config["evaluation"].update(tasks="reach-v3", episodes_per_task=1, episode_length=args.episode_length)
        configurations[name] = config
    configurations["pace"]["settings"]["artifact"] = str(output / "pace_calibration.json")
    calibration = calibrate(configurations["pace"])
    assert len(calibration["thresholds"]) == 50
    print("PACE: calibrated all 50 tasks", flush=True)
    sampler = Pi05Sampler(configurations["autohorizon"]["policy"])
    print("π0.5: strict pretrained load complete", flush=True)
    env = make_env("reach-v3", configurations["autohorizon"]["evaluation"])
    try:
        observation, first_state = reset_env(env, 1000)
        _, second_state = reset_env(env, 1000)
        assert first_state == second_state
        seed_policy(1000)
        sampler.prepare(observation, env.task_description)
        noise = sampler.model.sample_noise((1, 50, 32), sampler.device)
        images, masks = sampler.policy._preprocess_images(sampler.batch)
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        native = sampler.model.sample_actions(images, masks, sampler.batch[OBS_LANGUAGE_TOKENS],
                                              sampler.batch[OBS_LANGUAGE_ATTENTION_MASK], noise=noise.clone())
        sampled = sampler.sample(noise=noise.clone(),
                                 attention_step=configurations["autohorizon"]["settings"]["attention_step"],
                                 features=True)
        torch.testing.assert_close(sampled.normalized, native[0], rtol=0, atol=0)
        assert sampled.attention.shape == (50, 50)
        assert sampled.context.shape[-1] == 2048
        assert sampled.velocity_features.shape == (50, 1024)
        fixed_mask = torch.zeros(50, dtype=torch.bool, device=sampler.device)
        fixed_mask[[0, 3, 9]] = True
        conditional = sampler.sample(fixed_actions=sampled.normalized, fixed_mask=fixed_mask)
        torch.testing.assert_close(conditional.normalized[fixed_mask], sampled.normalized[fixed_mask], rtol=0, atol=0)
        assert all(not p.requires_grad for p in sampler.policy.parameters())
        print("Sampler: native parity, attention/features, and exact conditional anchors passed", flush=True)
    finally:
        env.close()
    results = {}
    for name in ("autohorizon", "aac", "pace", "a3"):
        config = configurations[name]
        results[name] = evaluate(config, sampler, make_method(config), output / name)
    config = configurations["bcp"]
    config["settings"]["artifact"] = str(output / "smoke_bcp_head.pt")
    config["training"].update(tasks="reach-v3", steps=1, output=str(output / "bcp_training"))
    # Allow a successful trajectory: all-failure groups must be discarded by
    # BCP and cannot test a genuine optimizer update.
    config["evaluation"]["episode_length"] = max(100, args.episode_length)
    train(config, sampler)
    first = torch.load(config["settings"]["artifact"], map_location="cpu", weights_only=True)
    config["training"]["steps"] = 2
    train(config, sampler, resume=True)
    second = torch.load(config["settings"]["artifact"], map_location="cpu", weights_only=True)
    assert first["step"] == 1 and second["step"] == 2
    training_rows = [json.loads(line) for line in
                     (Path(config["training"]["output"]) / "training.jsonl").read_text().splitlines()]
    updates = sum(not row["skipped"] for row in training_rows)
    assert updates > 0, "BCP smoke needs a successful training group to verify optimization"
    if not training_rows[-1]["skipped"]:
        assert not torch.equal(first["head"]["output.weight"], second["head"]["output.weight"])
    config["evaluation"]["episode_length"] = args.episode_length
    results["bcp"] = evaluate(config, sampler, make_method(config), output / "bcp")
    assert all(p.grad is None for p in sampler.policy.parameters())
    report = {"status": "complete", "native_sampler_exact_parity": True,
              "conditional_anchors_exact": True, "frozen_policy": True,
              "pace_calibration_tasks": len(calibration["thresholds"]),
              "pace_calibration_episodes": sum(calibration["episode_counts"].values()),
              "bcp_train_and_resume_steps": 2, "bcp_optimizer_groups": updates,
              "deterministic_reset": True, "evaluation": results,
              "note": "Smoke BCP head is not benchmark trained; these are integration checks, not performance claims."}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
