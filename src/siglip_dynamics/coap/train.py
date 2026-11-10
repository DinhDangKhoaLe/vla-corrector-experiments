"""Train only the two small towers on frozen cached observations."""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import TIMING, EpisodeBatchSampler, FrameCache, WindowDataset
from .model import COAP, contrastive_loss, eligible_pairs


def train(cfg):
    cfg.validate()
    tr, settings = cfg.training, cfg.model
    out = Path(tr.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh COAP training output: {out}")
    torch.manual_seed(tr.seed)
    np.random.seed(tr.seed)
    cache = FrameCache(tr.cache, tr.raw_actions)
    splits = cache.split(tr)
    normalization = cache.action_normalization(splits["train"], tr.normalization_floor)
    datasets = {
        name: WindowDataset(cache, splits[name], settings.window_size, normalization)
        for name in ("train", "validation")
    }
    samplers = {name: EpisodeBatchSampler(ds, tr.batch_size, tr.seed) for name, ds in datasets.items()}
    loaders = {
        name: DataLoader(ds, batch_sampler=samplers[name], num_workers=0) for name, ds in datasets.items()
    }
    model = COAP(cache.q.shape[-1], cache.actions.shape[-1], settings).to(tr.device)
    logging.info(
        "COAP: %d trainable parameters; device=%s; train windows=%d validation windows=%d",
        sum(p.numel() for p in model.parameters()),
        tr.device,
        len(datasets["train"]),
        len(datasets["validation"]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr.lr, weight_decay=tr.weight_decay)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
    (out / "action_normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    best, stale, started = float("inf"), 0, time.perf_counter()
    for epoch in range(1, tr.epochs + 1):
        metrics = {"epoch": epoch}
        for phase, loader in loaders.items():
            model.train(phase == "train")
            samplers[phase].epoch = epoch if phase == "train" else 0
            total, count, skipped, batches = 0.0, 0, 0, 0
            for batch in loader:
                observations = batch["observations"].to(tr.device)
                actions = batch["actions"].to(tr.device)
                allowed = eligible_pairs(batch, settings.window_size, tr.device)
                with torch.set_grad_enabled(phase == "train"):
                    obs, act = model(observations, actions)
                    loss = contrastive_loss(obs, act, allowed, tr.temperature)
                    if loss is None:
                        skipped += 1
                        continue
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite COAP contrastive loss")
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                anchors = int((allowed.sum(1) > 1).sum().item())
                total += float(loss.detach()) * anchors
                count += anchors
                batches += 1
            if not count:
                raise ValueError(
                    f"No eligible COAP negatives in {phase}; need distinct windows from multiple episodes"
                )
            metrics[phase] = {
                "contrastive_loss": total / count,
                "anchors": count,
                "batches": batches,
                "skipped_batches": skipped,
            }
        validation = metrics["validation"]["contrastive_loss"]
        payload = {
            "format": "coap_v1",
            "model": asdict(settings),
            "observation_dim": cache.q.shape[-1],
            "action_dim": cache.actions.shape[-1],
            "state_dict": model.state_dict(),
            "normalization": normalization,
            "cache_metadata": cache.metadata,
            "timing": TIMING,
            "split_manifest": splits,
            "epoch": epoch,
            "validation_loss": validation,
            "training_config": cfg.to_dict(),
        }
        for name in ("last_model.pt", "best_model.pt") if validation < best else ("last_model.pt",):
            temp = out / (name + ".tmp")
            torch.save(payload, temp)
            temp.replace(out / name)
        if validation < best:
            best, stale = validation, 0
        else:
            stale += 1
        metrics["elapsed_seconds"] = time.perf_counter() - started
        with (out / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics) + "\n")
        logging.info(
            "COAP epoch %d train=%.5f validation=%.5f stale=%d",
            epoch,
            metrics["train"]["contrastive_loss"],
            validation,
            stale,
        )
        if stale >= tr.patience:
            break
    return out / "best_model.pt"
