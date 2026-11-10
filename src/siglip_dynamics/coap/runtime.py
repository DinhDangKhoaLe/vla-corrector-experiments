"""Score completed, same-timestep pairs; interrupt only the remaining queue."""

import json
import math
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from siglip_dynamics.experiments.artifacts import digest, file_digest
from siglip_dynamics.visual_tokens import pool_square_token_grid

from .config import Generation, Model
from .data import TIMING
from .model import COAP


def checkpoint_path(cfg):
    path = Path(cfg.evaluation.checkpoint or Path(cfg.training.output) / "best_model.pt")
    return path / "best_model.pt" if path.is_dir() else path


def protocol(cfg):
    evaluation = asdict(cfg.evaluation)
    # Preserve the identity of existing continuous-RNG calibration artifacts.
    if not cfg.evaluation.reseed_policy_each_episode:
        evaluation.pop("reseed_policy_each_episode")
    for key in ("output", "checkpoint", "episodes", "seed", "collection_seed", "save_video"):
        evaluation.pop(key)
    # Bind calibration to the policy preprocessing as well as the rollout schedule.
    policy = Path(cfg.evaluation.policy)
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        path = policy / name
        if path.is_file():
            evaluation[name] = file_digest(path)
    identity = {
        "evaluation": evaluation,
        "model": asdict(cfg.model),
        "timing": TIMING,
        "persistence": cfg.monitor.persistence,
        "cooldown_steps": cfg.monitor.cooldown_steps,
    }
    # Keep legacy replan calibration fingerprints exactly unchanged.
    if cfg.generation.enabled:
        identity["generation"] = asdict(cfg.generation)
    return digest(identity)


class Predictor:
    def __init__(self, path, device="cpu"):
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "coap_v1" or payload.get("timing") != TIMING:
            raise ValueError("Expected a COAP checkpoint with the same observation/action timing")
        self.settings = Model(**payload["model"])
        self.model = COAP(payload["observation_dim"], payload["action_dim"], self.settings).to(device).eval()
        self.model.load_state_dict(payload["state_dict"])
        self.model.requires_grad_(False)
        self.device = device
        self.normalization = payload["normalization"]
        self.metadata = payload["cache_metadata"]
        self.split_manifest = payload["split_manifest"]
        self.fingerprint = file_digest(path)

    @torch.inference_mode()
    def score(self, observations, raw_actions):
        obs = torch.tensor(np.asarray(observations), dtype=torch.float32, device=self.device).unsqueeze(0)
        actions = torch.tensor(np.asarray(raw_actions), dtype=torch.float32, device=self.device)
        mean = torch.tensor(self.normalization["mean"], device=self.device)
        std = torch.tensor(self.normalization["std"], device=self.device)
        actions = ((actions - mean) / std).unsqueeze(0)
        return float(self.model.similarity(obs, actions).item())


@torch.inference_mode()
def fresh_observation_features(policy, model_input, metadata):
    """A real encoder pass on the current image, independent of the Corrector/OGG API."""
    images, masks = policy._preprocess_images(model_input)
    blocks = []
    for image, mask in zip(images, masks, strict=True):
        if isinstance(mask, torch.Tensor) and not bool(mask[0].item()):
            continue
        tokens = policy.model.paligemma_with_expert.embed_image(image)
        blocks.append(
            pool_square_token_grid(
                tokens,
                metadata.get("image_token_grid_size", 0),
                target_shape=metadata.get("image_token_grid_shape"),
            )
        )
    if not blocks:
        raise ValueError("COAP received no real camera image")
    tokens = torch.cat(blocks, dim=1)[0]
    if tokens.shape != (metadata["l_tokens"], metadata["d_dim"]):
        raise ValueError("COAP online camera/token layout differs from its feature cache")
    return tokens.float().mean(0).cpu().numpy()


class Monitor:
    def __init__(
        self,
        predictor,
        settings,
        threshold,
        *,
        shadow=False,
        output=None,
        protocol_fingerprint="",
        calibration_status="calibrated",
        generation=None,
    ):
        self.predictor, self.settings, self.threshold = predictor, settings, threshold
        self.shadow, self.output = shadow, Path(output) if output else None
        self.protocol_fingerprint = protocol_fingerprint
        self.calibration_status = calibration_status
        self.generation = generation if generation is not None else Generation()
        if not -1 <= threshold <= 1:
            raise ValueError("Invalid COAP cosine threshold")
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
        self.reset("", 0)

    def reset(self, task, seed):
        if (task, int(seed)) in getattr(self, "calibration_episodes", set()):
            raise ValueError("COAP evaluation episode overlaps its calibration collection")
        self.task, self.seed = task, int(seed)
        self.observations = deque(maxlen=self.predictor.settings.window_size)
        self.actions = deque(maxlen=self.predictor.settings.window_size)
        self.steps, self.low_count, self.last_trigger = 0, 0, -(10**12)
        self.traces = []
        self.feature_seconds = self.score_seconds = 0.0
        self.latest_similarity = None
        self.generation_requests = []

    def temperature_for_next_chunk(self, completed_step):
        """Called only at a natural chunk request; uses completed past pairs only."""
        from .temperature import temperature_from_score

        if not self.generation.enabled or self.shadow:
            raise ValueError("COAP temperature control is not enabled")
        if completed_step != self.steps:
            raise ValueError("Generation temperature must use the most recent completed step")
        temperature = temperature_from_score(self.latest_similarity, self.generation)
        self.generation_requests.append(
            {
                "completed_step": completed_step,
                "similarity": self.latest_similarity,
                "temperature": temperature,
                "warmup": self.latest_similarity is None,
            }
        )
        return temperature

    def after_execution(self, features, dispatched_action, completed_step, *, done=False, can_interrupt=True):
        if completed_step != self.steps + 1:
            raise ValueError("COAP needs one completed action per consecutive observation/action pair")
        features = np.asarray(features, dtype=np.float32)
        action = np.asarray(dispatched_action, dtype=np.float32)
        if (
            features.shape != (self.predictor.model.observation_dim,)
            or action.shape != (self.predictor.model.action_dim,)
            or not np.isfinite(features).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError("Invalid fresh observation / dispatched action dimensions or values")
        self.steps = completed_step
        self.observations.append(features.copy())
        self.actions.append(action.copy())
        if len(self.actions) < self.predictor.settings.window_size:
            return None
        started = time.perf_counter()
        similarity = self.predictor.score(self.observations, self.actions)
        elapsed = time.perf_counter() - started
        self.score_seconds += elapsed
        if not math.isfinite(similarity) or not -1 <= similarity <= 1:
            raise ValueError("COAP must return a finite direct cosine similarity")
        self.latest_similarity = similarity
        cooldown = completed_step - self.last_trigger <= self.settings.cooldown_steps
        self.low_count = self.low_count + 1 if similarity < self.threshold and not cooldown else 0
        alarm = self.low_count >= self.settings.persistence
        trigger = alarm and not self.generation.enabled and not self.shadow and not done and can_interrupt
        if trigger:
            self.last_trigger, self.low_count = completed_step, 0
        event = {
            "completed_step": completed_step,
            "observation_steps": list(range(completed_step - len(self.actions), completed_step)),
            "similarity": similarity,
            "mismatch": 1 - similarity,
            "threshold": self.threshold,
            "alarm": alarm,
            "trigger": trigger,
            "cooldown": cooldown,
            "score_seconds": elapsed,
            "timing": TIMING,
        }
        if self.generation.enabled:
            from .temperature import temperature_from_score

            event.update(
                control_mode="generation_temperature",
                threshold=None,
                alarm=False,
                cooldown=False,
                next_chunk_temperature=temperature_from_score(similarity, self.generation),
            )
        self.traces.append(event)
        return event

    def finish(self, success, error="", *, vla_calls=0, vla_seconds=0.0, wall_seconds=0.0):
        row = {
            "task": self.task,
            "seed": self.seed,
            "success": bool(success),
            "error": error,
            "model_fingerprint": self.predictor.fingerprint,
            "protocol_fingerprint": self.protocol_fingerprint,
            "mode": "collect-calibration" if self.shadow else "evaluate",
            "timing": TIMING,
            "calibration_status": self.calibration_status,
            "steps": self.steps,
            "traces": self.traces,
            "intervention_count": sum(event["trigger"] for event in self.traces),
            "vla_calls": vla_calls,
            "vla_seconds": vla_seconds,
            "wall_seconds": wall_seconds,
            "feature_seconds": self.feature_seconds,
            "scoring_seconds": self.score_seconds,
        }
        if self.generation.enabled:
            row.update(
                control_mode="generation_temperature",
                generation_settings=asdict(self.generation),
                generation_requests=self.generation_requests,
            )
        if self.output:
            with (self.output / "coap_episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row


def after_step(
    monitor, features, dispatched_action, completed_step, action_queue, raw_action_queue, *, done=False
):
    """Shared evaluator hook; disabled COAP is a no-op, and future actions never enter scoring."""
    if monitor is None:
        return None
    event = monitor.after_execution(
        features, dispatched_action, completed_step, done=done, can_interrupt=bool(action_queue)
    )
    if event and event["trigger"]:
        event["discarded_actions"] = len(action_queue)
        action_queue.clear()
        raw_action_queue.clear()
    return event


def load_runtime(cfg, mode, output):
    predictor = Predictor(checkpoint_path(cfg), cfg.training.device)
    if asdict(predictor.settings) != asdict(cfg.model):
        raise ValueError("COAP runtime architecture differs from checkpoint")
    shadow = mode == "collect-calibration"
    if mode not in {"collect-calibration", "evaluate"}:
        raise ValueError("Unsupported COAP runtime mode")
    identity = protocol(cfg)
    artifact = Path(cfg.monitor.artifact)
    if cfg.generation.enabled:
        if shadow:
            raise ValueError("Collect calibration with generation control disabled")
        if cfg.monitor.threshold is not None:
            raise ValueError(
                "Generation temperature replaces COAP threshold replanning; unset monitor.threshold"
            )
        threshold, status = -1.0, "score-controlled generation temperature; no replan threshold"
    elif shadow:
        threshold, status = -1.0, "collection without interventions"
    elif cfg.monitor.threshold is not None:
        if not cfg.monitor.allow_provisional_threshold:
            raise ValueError("Explicit COAP threshold requires allow_provisional_threshold=true")
        threshold, status = cfg.monitor.threshold, "PROVISIONAL explicit threshold"
    else:
        if not artifact.is_file():
            raise FileNotFoundError("Calibrate COAP first, or explicitly allow a provisional threshold")
        calibration = json.loads(artifact.read_text())
        if (
            calibration.get("model_fingerprint") != predictor.fingerprint
            or calibration.get("protocol_fingerprint") != identity
        ):
            raise ValueError("COAP calibration does not match checkpoint or rollout protocol")
        threshold, status = calibration["threshold"], calibration["status"]
        if status.startswith("PROVISIONAL") and not cfg.monitor.allow_provisional_threshold:
            raise ValueError("Provisional demonstration threshold requires allow_provisional_threshold=true")
    monitor = Monitor(
        predictor,
        cfg.monitor,
        threshold,
        shadow=shadow,
        output=output,
        protocol_fingerprint=identity,
        calibration_status=status,
        generation=cfg.generation,
    )
    if not shadow and not cfg.generation.enabled and cfg.monitor.threshold is None:
        monitor.calibration_episodes = {(row["task"], row["seed"]) for row in calibration.get("episodes", [])}
    else:
        monitor.calibration_episodes = set()
    return monitor
