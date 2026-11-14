"""Published asset identifiers and safe extraction of the official RoboTwin release."""
from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path

ROBOTWIN_OFFICIAL_REPO = "TianxingChen/RoboTwin2.0"
ROBOTWIN_OFFICIAL_REVISION = "a967b852afa21a9cbf19a198f7e653109042e87c"
ROBOTWIN_OFFICIAL_ARCHIVE = "cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz"
ROBOTWIN_OFFICIAL_DIRECTORY = "pretrained_weights/robotwin_official"
ROBOTWIN_OFFICIAL_EXPORT = "pretrained_weights/pi05_robotwin_official"
ROBOTWIN_OFFICIAL_POLICY = ROBOTWIN_OFFICIAL_EXPORT + "_compat"


def unpack_official_robotwin(archive: Path, output: Path) -> Path:
    """Extract the published archive without converting or relabeling its weights."""
    provenance = dict(repo=ROBOTWIN_OFFICIAL_REPO, revision=ROBOTWIN_OFFICIAL_REVISION,
                      archive=ROBOTWIN_OFFICIAL_ARCHIVE, format="OpenPI; not a LeRobot export")
    marker = output / "source_archive.json"
    if output.exists():
        if marker.is_file() and json.loads(marker.read_text()) == provenance:
            print(f"Reusing extracted official RoboTwin archive: {output}")
            return output
        raise FileExistsError(f"Refusing to overwrite existing checkpoint directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".robotwin-unpack-", dir=output.parent) as temporary:
        staging = Path(temporary) / "checkpoint"
        staging.mkdir()
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(staging, filter="data")
        (staging / marker.name).write_text(json.dumps(provenance, indent=2) + "\n")
        staging.rename(output)
    print(f"Extracted official RoboTwin archive -> {output}; original retained")
    return output
