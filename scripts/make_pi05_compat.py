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


# These are OpenPI/checkpoint metadata fields, not fields of this repository's
# PI05Config dataclass.  They must be removed from the local compatibility view
# before either the extractor or a live evaluator calls PI05Config.from_pretrained.
UNSUPPORTED = ("use_relative_actions", "relative_exclude_joints", "action_feature_names")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Downloaded PI0.5 checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="New, empty compatibility-view directory")
    parser.add_argument("--refresh-config", action="store_true", help="Refresh only config.json in an existing symlink view")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir() or not (source / "config.json").is_file():
        raise FileNotFoundError(f"--source must be a PI0.5 checkpoint directory: {source}")
    if output.exists() and not args.refresh_config:
        raise FileExistsError(f"Refusing to overwrite existing --output: {output}")
    if output == source:
        raise ValueError("Source and compatibility output must differ")
    if output.exists():
        for asset in source.iterdir():
            if asset.name == "config.json":
                continue
            link = output / asset.name
            if not link.exists() and not link.is_symlink():
                continue
            if not link.is_symlink() or link.resolve() != asset.resolve():
                raise ValueError(f"Existing output is not this source's compatibility view: {link}")

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    removed = [key for key in UNSUPPORTED if config.pop(key, None) is not None]
    config["type"] = "pi05"
    output.mkdir(parents=True, exist_ok=args.refresh_config)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for source_file in source.iterdir():
        if source_file.name == "config.json":
            continue
        target = output / source_file.name
        if not target.is_symlink():
            os.symlink(os.path.relpath(source_file, start=output), target)
    print(f"Created {output} (patched: {', '.join(removed) or 'none'}; checkpoint files are symlinked)")


if __name__ == "__main__":
    main()
