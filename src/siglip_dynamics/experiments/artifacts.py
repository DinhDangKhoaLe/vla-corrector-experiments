"""Shared provenance and trace validation for experiment artifacts."""

from __future__ import annotations

import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def read_traces(path):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("Trace file is empty")
    ids = [(r["task"], r["seed"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate task/seed episodes in trace file")
    for key in ("model_fingerprint", "protocol_fingerprint", "environment_fingerprint", "mode"):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f"Trace file mixes {key}")
    if any(r.get("error") for r in rows):
        raise ValueError("Resolve crashed episodes before calibration")
    return rows
