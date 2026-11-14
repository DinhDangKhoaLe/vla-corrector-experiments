"""Read decoded demonstration actions without loading images or VLA weights."""
from __future__ import annotations

import json

import numpy as np


def read_demonstrations(config):
    if config.dataset_format == "robotwin":
        import h5py
        from .robotwin import _read_actions
        paths = sorted({*config.source.rglob("*.h5"), *config.source.rglob("*.hdf5")})
        if not paths:
            raise FileNotFoundError(f"No native HDF5 demonstrations below {config.source}")
        for path in paths:
            parts = path.relative_to(config.source).parts
            task = next((t for t in config.tasks if t in parts), parts[0])
            if config.tasks and task not in config.tasks:
                continue
            with h5py.File(path, "r") as handle:
                if config.action_key not in handle:
                    raise KeyError(f"{path}: missing calibration field {config.action_key!r}")
                # The released joint-space calibration uses observed joint
                # states, not policy-normalized actions or overlapping chunks.
                yield task, _read_actions({"action": handle[config.action_key]})
        return

    import pyarrow.parquet as pq
    names = {}
    tasks_path = config.source / "meta/tasks.parquet"
    if tasks_path.exists():
        for r in pq.read_table(tasks_path).to_pylist():
            names[int(r["task_index"])] = str(r.get("task", r.get("__index_level_0__")))
    else:
        for line in (config.source / "meta/tasks.jsonl").read_text().splitlines():
            r = json.loads(line)
            names[int(r["task_index"])] = r["task"]
    paths = sorted((config.source / "data").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No LeRobot parquet data below {config.source}")
    # V3 files may split an episode. Retain only one unfinished trajectory.
    current, task, actions = None, None, []
    emitted = set()
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["episode_index", "task_index", config.action_key]):
            for row in batch.to_pylist():
                episode = int(row["episode_index"])
                task_name = names[int(row["task_index"])]
                if current is not None and episode != current:
                    emitted.add(current)
                    if not config.tasks or task in config.tasks:
                        yield task, np.asarray(actions, dtype=np.float32)
                    actions = []
                if episode in emitted:
                    raise ValueError("PACE requires episode-contiguous parquet files")
                if episode == current and task != task_name:
                    raise ValueError("Multiple tasks in one calibration episode")
                current, task = episode, task_name
                actions.append(row[config.action_key])
    if actions and (not config.tasks or task in config.tasks):
        yield task, np.asarray(actions, dtype=np.float32)
