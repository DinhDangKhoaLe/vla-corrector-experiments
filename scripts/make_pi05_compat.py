#!/usr/bin/env python3
"""Create a zero-copy local PI0.5 compatibility view for this repository.

The published MetaWorld PI0.5 snapshot retains three OpenPI configuration
fields which older local LeRobot PI0.5 config parsers do not accept.  This
script writes a patched config and symlinks every other checkpoint asset, so
the large model weights are never duplicated or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


UNSUPPORTED = ("use_relative_actions", "relative_exclude_joints", "action_feature_names")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Downloaded PI0.5 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New, empty compatibility-view directory")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise FileNotFoundError(f"--source must be a PI0.5 checkpoint directory: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing --output: {output}")

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    removed = [key for key in UNSUPPORTED if config.pop(key, None) is not None]
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for source_file in source.iterdir():
        if source_file.name == "config.json":
            continue
        target = output / source_file.name
        os.symlink(os.path.relpath(source_file, start=output), target)
    print(f"Created {output} (patched: {', '.join(removed) or 'none'}; checkpoint files are symlinked)")


if __name__ == "__main__":
    main()
