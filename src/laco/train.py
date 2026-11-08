from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import LatentJacobianModel
from .utils import device_from_config, load_config, merge_overrides, save_json


def _data(cache: Path, split: str, action_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.load(cache / "split.npy", mmap_mode="r").astype(str)
    ix = np.flatnonzero(labels == split)
    if not len(ix):
        raise ValueError(f"No {split} samples in {cache}")
    z = np.asarray(np.load(cache / "z_t.npy", mmap_mode="r")[ix], dtype=np.float32)
    target = np.asarray(np.load(cache / "z_next.npy", mmap_mode="r")[ix], dtype=np.float32) - z
    actions = np.asarray(np.load(cache / "actions.npy", mmap_mode="r")[ix, :action_dim], dtype=np.float32)
    return z, actions, target


def _normalizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(0).astype(np.float32), np.maximum(x.std(0), 1e-6).astype(np.float32)


def run(config: dict[str, Any]) -> dict[str, Any]:
    c = config.get("train", config)
    cache, output = Path(c["cache"]), Path(c["output"])
    output.mkdir(parents=True, exist_ok=True)
    action_dim = int(c.get("action_dim", 3))
    z_train, a_train, y_train = _data(cache, "train", action_dim)
    z_val, a_val, y_val = _data(cache, "validation", action_dim)
    lm, ls = _normalizer(z_train)
    am, ass = _normalizer(a_train)
    def standardize(z, a, y): return (z-lm)/ls, (a-am)/ass, y/ls
    z_train, a_train, y_train = standardize(z_train, a_train, y_train)
    z_val, a_val, y_val = standardize(z_val, a_val, y_val)
    dev = device_from_config(c.get("device", "cuda"))
    spec = {"latent_dim": z_train.shape[1], "action_dim": action_dim, "hidden_dim": int(c.get("hidden_dim", 128))}
    model = LatentJacobianModel(**spec).to(dev)
    if model.parameter_count >= 1_000_000:
        raise ValueError(f"Corrector has {model.parameter_count:,} parameters; must remain under 1M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c.get("lr", 3e-4)), weight_decay=float(c.get("weight_decay", 1e-4)))
    batch = int(c.get("batch_size", 512)); epochs = int(c.get("epochs", 100)); patience = int(c.get("patience", 10)); cosine_w = float(c.get("cosine_weight", 0.0))
    loader = DataLoader(TensorDataset(torch.from_numpy(z_train), torch.from_numpy(a_train), torch.from_numpy(y_train)), batch_size=batch, shuffle=True)
    val = (torch.from_numpy(z_val).to(dev), torch.from_numpy(a_val).to(dev), torch.from_numpy(y_val).to(dev))
    best, best_state, stale = float("inf"), None, 0
    history = []
    torch.manual_seed(int(c.get("seed", 42)))
    for epoch in range(epochs):
        model.train(); total = 0.0
        for z, a, y in loader:
            z, a, y = z.to(dev), a.to(dev), y.to(dev)
            pred = model(z, a); loss = torch.nn.functional.mse_loss(pred, y)
            if cosine_w:
                loss = loss + cosine_w * (1 - torch.nn.functional.cosine_similarity(pred, y, dim=-1).mean())
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(c.get("grad_clip", 1.0))); optimizer.step()
            total += loss.item() * len(z)
        model.eval()
        with torch.no_grad(): val_loss = torch.nn.functional.mse_loss(model(val[0], val[1]), val[2]).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(z_train), "validation_mse": val_loss})
        if val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience: break
    assert best_state is not None
    artifact = {"format": "laco_jacobian_v1", "model": best_state, "model_spec": spec, "latent_mean": lm, "latent_std": ls,
                "action_mean": am, "action_std": ass, "action_representation": "first three PI05-normalized MetaWorld action dimensions",
                "best_validation_mse": best, "parameter_count": model.parameter_count, "config": c}
    torch.save(artifact, output / "best.pt")
    save_json(output / "metrics.json", {"best_validation_mse": best, "epochs_ran": len(history), "parameter_count": model.parameter_count, "history": history})
    return {"checkpoint": str(output / "best.pt"), "best_validation_mse": best, "parameter_count": model.parameter_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight LACo latent Jacobian.")
    parser.add_argument("--config", required=True); parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args(); print(json.dumps(run(merge_overrides(load_config(args.config), args.set)), indent=2))


if __name__ == "__main__": main()
