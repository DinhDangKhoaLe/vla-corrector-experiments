"""Apply the two dependency patches required by the official RoboTwin installer.

Files in the active environment are backed up before patching. Never edits a
checkpoint, dataset, or installed package outside the selected conda env.
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path


def patch(module, relative, old, new):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Missing dependency: {module}")
    path = Path(spec.origin).parent / relative
    if not path.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f"Dependency is outside the active environment: {path}")
    text = path.read_text()
    if old not in text:
        if new in text:
            return
        raise RuntimeError(f"Dependency source changed; review patch manually: {path}")
    backup = path.with_suffix(path.suffix + ".dynamic_horizon.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new))
    print(f"Patched {path}; backup: {backup}")


if __name__ == "__main__":
    if os.environ.get("CONDA_DEFAULT_ENV") != "dynamic_horizon":
        raise RuntimeError("Activate dynamic_horizon before applying simulator patches")
    patch("sapien", "wrapper/urdf_loader.py", 'urdf_file[:-4] + "srdf"', 'urdf_file[:-4] + ".srdf"')
    patch("mplib", "planner.py", "< 1e-4 or collide or not within_joint_limit:", "< 1e-4 or not within_joint_limit:")
