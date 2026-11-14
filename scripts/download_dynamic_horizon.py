"""Print download plan by default; pass --execute to fetch model/data assets.

Does not fetch anything on import or in dry-run mode. RoboTwin data is selected
by task to avoid accidentally downloading the entire 1.5 TB repository.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from dynamic_horizon.assets import (
    ROBOTWIN_OFFICIAL_ARCHIVE, ROBOTWIN_OFFICIAL_DIRECTORY,
    ROBOTWIN_OFFICIAL_REPO, ROBOTWIN_OFFICIAL_REVISION, unpack_official_robotwin,
)

def plan(benchmark, tasks):
    jobs = []
    if benchmark in {"all", "metaworld"}:
        jobs += [("dataset", "lerobot/metaworld_mt50", "datasets/metaworld_mt50", None),
                 ("model", "tiantianx/pi05_metaworld", "pretrained_weights/pi05_metaworld", None)]
    if benchmark in {"all", "libero"}:
        jobs += [("dataset", "lerobot/libero", "datasets/libero", None),
                 ("model", "lerobot/pi05_libero_base", "pretrained_weights/pi05_libero", None)]
    if benchmark in {"all", "robotwin"}:
        # The official weights are hosted in a dataset repository as an OpenPI
        # archive. Do not pass them through the LeRobot compatibility-view tool.
        jobs += [("dataset", ROBOTWIN_OFFICIAL_REPO, ROBOTWIN_OFFICIAL_DIRECTORY,
                  [ROBOTWIN_OFFICIAL_ARCHIVE]),
                 ("dataset", "TianxingChen/RoboTwin2.0", "datasets/robotwin_archives",
                  [f"dataset/{task}/aloha-agilex_clean_50.zip" for task in tasks])]
    return jobs


def unpack_robotwin(root):
    for archive in sorted(root.glob("dataset/*/aloha-agilex_clean_50.zip")):
        target = Path("datasets/robotwin") / archive.parent.name
        provenance = dict(archive=archive.name, task=archive.parent.name, bytes=archive.stat().st_size)
        marker = target / "source_archive.json"
        if target.exists() and any(target.iterdir()):
            if marker.is_file() and json.loads(marker.read_text()) == provenance:
                print(f"Reusing unpacked clean demonstrations: {target}")
                continue
            raise FileExistsError(f"Refusing to overwrite unpacked episodes: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".demos-unpack-", dir=target.parent) as temporary:
            staging = Path(temporary) / "episodes"
            staging.mkdir()
            with zipfile.ZipFile(archive) as zf:
                for member in zf.infolist():
                    destination = (staging / member.filename).resolve()
                    if not destination.is_relative_to(staging.resolve()):
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                    if member.external_attr >> 16 & 0o170000 == 0o120000:
                        raise ValueError("Symlink in dataset archive")
                zf.extractall(staging)
            (staging / marker.name).write_text(json.dumps(provenance, indent=2) + "\n")
            staging.rename(target)
        print(f"Extracted {archive} -> {target}; archive retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["all", "metaworld", "libero", "robotwin"], default="all")
    parser.add_argument("--robotwin-tasks", default="adjust_bottle", help="Comma-separated task directories, or all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--unpack", action="store_true", help="Unpack official RoboTwin checkpoint and selected demo ZIPs; retain archives")
    args = parser.parse_args()
    tasks = args.robotwin_tasks.split(",")
    if tasks == ["all"]:
        tasks = ["*"]
    if any(not t or (t != "*" and not t.replace("_", "").isalnum()) for t in tasks):
        parser.error("Invalid task directory")
    jobs = plan(args.benchmark, tasks)
    print(json.dumps([dict(type=t, repo=r, destination=d, include=p,
                          revision=ROBOTWIN_OFFICIAL_REVISION if d == ROBOTWIN_OFFICIAL_DIRECTORY else "main")
                      for t,r,d,p in jobs], indent=2))
    if not args.execute:
        print("Plan only. Use --execute to download; --unpack to extract RoboTwin episodes.")
        return
    from huggingface_hub import snapshot_download
    for kind, repo, directory, patterns in jobs:
        revision = ROBOTWIN_OFFICIAL_REVISION if directory == ROBOTWIN_OFFICIAL_DIRECTORY else None
        snapshot_download(repo_id=repo, repo_type=kind, local_dir=directory, allow_patterns=patterns, revision=revision)
        if kind == "model":
            target = Path(directory + "_compat")
            refresh = ["--refresh-config"] if target.exists() else []
            subprocess.run([sys.executable, "scripts/make_pi05_compat.py", "--source", directory,
                            "--output", str(target), *refresh], check=True)
    if args.unpack and args.benchmark in {"all", "robotwin"}:
        unpack_official_robotwin(Path(ROBOTWIN_OFFICIAL_DIRECTORY) / ROBOTWIN_OFFICIAL_ARCHIVE,
                                Path(ROBOTWIN_OFFICIAL_DIRECTORY) / "extracted")
        unpack_robotwin(Path("datasets/robotwin_archives"))
    if args.benchmark in {"all", "robotwin"}:
        print("Official RoboTwin OpenPI archive downloaded. LeRobot conversion is a separate required step; see docs/robotwin_official.md.")


if __name__ == "__main__":
    main()
