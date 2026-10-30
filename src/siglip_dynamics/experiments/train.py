"""Train ablations with episode-disjoint validation and an untouched test split."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Experiment
from siglip_dynamics.visual_tokens import validate_cache_grid
from .data import PrefixDataset, episode_split
from .model import ExperimentalCorrector


def error_statistics(records):
    """Diagnostics, not a calibrated failure probability."""
    result = {}
    for h in sorted({int(r["horizon"]) for r in records}):
        rows = [r for r in records if int(r["horizon"]) == h]
        predicted = np.asarray([r["predicted_rmse"] for r in rows], dtype=float)
        observed = np.asarray([r["rmse"] for r in rows], dtype=float)
        good = np.isfinite(predicted) & np.isfinite(observed)
        predicted, observed = predicted[good], observed[good]
        corr = None
        if len(predicted) >= 3 and predicted.std() > 1e-12 and observed.std() > 1e-12:
            corr = float(np.corrcoef(predicted, observed)[0, 1])
        bins = []
        if len(predicted):
            for ids in np.array_split(np.argsort(predicted), min(5, len(predicted))):
                bins.append({"count": len(ids), "predicted": float(predicted[ids].mean()),
                             "observed": float(observed[ids].mean())})
        result[str(h)] = {"count": len(predicted), "pearson_correlation": corr, "bins": bins}
    return result


def train_experiment(cfg: Experiment):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Experimental trainer is single-process; launch with python, not multi-process torchrun")
    tr = cfg.training
    out = Path(tr.output)
    if (out / "best_model.pt").exists() or (out / "last_model.pt").exists():
        raise FileExistsError(f"Choose a fresh training.output; checkpoints already exist in {out}")
    torch.manual_seed(tr.seed)
    np.random.seed(tr.seed)
    root = Path(tr.cache)
    with open(root / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    validate_cache_grid(metadata, cfg.model.image_token_grid_size, target_shape=cfg.model.image_token_grid_shape)
    if metadata.get("action_representation") != "normalized_action":
        raise ValueError("Re-extract with --use-normalized-delta-action; experiments require policy-normalized actions")
    ids = np.load(root / "episode_index.npy", mmap_mode="r")
    splits = episode_split(ids, tr.val_fraction, tr.test_fraction, tr.seed)
    train_ds = PrefixDataset(root, cfg.model.horizons, splits["train"])
    val_ds = PrefixDataset(root, cfg.model.horizons, splits["validation"])
    token_dim = train_ds.base.z_q.shape[-1]
    action_dim = train_ds.base.a_t_np.shape[-1]
    model = ExperimentalCorrector(cfg, token_dim, action_dim).to(tr.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=tr.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tr.epochs, eta_min=tr.lr * 0.01)
    loaders = {"train": DataLoader(train_ds, batch_size=tr.batch_size, shuffle=True, num_workers=0),
               "validation": DataLoader(val_ds, batch_size=tr.batch_size, shuffle=False, num_workers=0)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_manifest.json").write_text(json.dumps(splits, indent=2))
    (out / "experiment.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    logging.info("Corrector parameters: %s | train=%s validation=%s", sum(p.numel() for p in model.parameters()),
                 len(train_ds), len(val_ds))
    best, stale = float("inf"), 0
    for epoch in range(1, tr.epochs + 1):
        metrics, records = {}, []
        loss_metrics = {}
        for phase, loader in loaders.items():
            model.train(phase == "train")
            total, count = 0.0, 0
            sums = {"mse": 0.0, "error_head": 0.0, "prediction_loss": 0.0, "total_loss": 0.0}
            for batch in loader:
                if phase == "train":
                    optimizer.zero_grad(set_to_none=True)
                batch_count = len(batch["z"])
                # Accumulate to the configured effective batch without a 512-frame GPU allocation.
                for start in range(0, batch_count, tr.microbatch_size):
                    sl = slice(start, start + tr.microbatch_size)
                    z, target, actions, h = (batch[k][sl].to(tr.device) for k in ("z", "delta", "actions", "horizon"))
                    with torch.set_grad_enabled(phase == "train"):
                        pred, log_error = model.predict(z, actions, h)
                        cosine = 1 - F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=-1)
                        loss = cosine.mean()
                        mse = F.mse_loss(pred, target)
                        if cfg.features.uncertainty_monitor:
                            loss = tr.cosine_loss_weight * loss + tr.mse_loss_weight * mse
                        prediction_loss = loss
                        error_loss = pred.new_zeros(())
                        rmse = (pred.detach() - target).square().mean(dim=(1, 2)).sqrt()
                        if log_error is not None:
                            error_loss = F.smooth_l1_loss(log_error, rmse.clamp_min(1e-6).log())
                            loss = loss + tr.uncertainty_loss_weight * error_loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite loss in {phase}, epoch {epoch}")
                        if phase == "train":
                            (loss * (len(z) / batch_count)).backward()
                    total += float(cosine.detach().sum())
                    for key, value in (("mse", mse), ("error_head", error_loss),
                                       ("prediction_loss", prediction_loss), ("total_loss", loss)):
                        sums[key] += float(value.detach()) * len(z)
                    count += len(z)
                    if phase == "validation" and log_error is not None:
                        records.extend({"horizon": hi, "predicted_rmse": pi, "rmse": ri}
                                       for hi, pi, ri in zip(h.tolist(), log_error.detach().exp().tolist(), rmse.tolist()))
                if phase == "train":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            metrics[phase] = total / count
            loss_metrics[phase] = {key: value / count for key, value in sums.items()}
        scheduler.step()
        payload = {"format": "corrector_experiment_v1", "experiment": cfg.to_dict(),
                   "token_dim": token_dim, "action_dim": action_dim, "state_dict": model.state_dict(),
                   "epoch": epoch, "validation_cosine_loss": metrics["validation"],
                   "loss_metrics": loss_metrics,
                   "uncertainty_validation": error_statistics(records)}
        # Local trusted checkpoints, written atomically so evaluation never sees half a file.
        tmp = out / "checkpoint.tmp"
        torch.save(payload, tmp)
        tmp.replace(out / "last_model.pt")
        selection = (loss_metrics["validation"]["prediction_loss"] if cfg.features.uncertainty_monitor
                     else metrics["validation"])
        if selection < best:
            best, stale = selection, 0
            torch.save(payload, tmp)
            tmp.replace(out / "best_model.pt")
        else:
            stale += 1
        with open(out / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics, "losses": loss_metrics}) + "\n")
        logging.info("Epoch %d | train cosine=%.6f validation cosine=%.6f stale=%d", epoch,
                     metrics["train"], metrics["validation"], stale)
        if cfg.features.uncertainty_monitor:
            logging.info("Loss components | train=%s validation=%s", loss_metrics["train"], loss_metrics["validation"])
        if stale >= tr.patience:
            break
