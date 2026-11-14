#!/usr/bin/env bash
# Simulator source/build only. No datasets, assets, or policy checkpoints.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
test "${CONDA_DEFAULT_ENV:-}" = dynamic_horizon || { echo 'Activate dynamic_horizon first.' >&2; exit 1; }
if ! python -c 'import torch, numpy, scipy, sapien, mplib, setuptools_scm' >/dev/null; then
  echo 'Core dependencies are incomplete. Run bash scripts/setup_dynamic_horizon.sh successfully before RoboTwin setup.' >&2
  exit 1
fi
command -v nvcc >/dev/null || { echo 'CUDA toolkit/NVCC is required to build CuRobo.' >&2; exit 1; }
ROBOTWIN_DIR="$PROJECT_ROOT/third_party/RoboTwin"
if test ! -e "$ROBOTWIN_DIR"; then
  git clone --no-checkout https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_DIR"
  git -C "$ROBOTWIN_DIR" checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
fi
test "$(git -C "$ROBOTWIN_DIR" rev-parse HEAD)" = ea8b21121ebb3cd201ff5b3fe361944ac94eda3f || {
  echo 'Existing RoboTwin checkout differs from the documented pin; use a separate checkout.' >&2; exit 1;
}
git -C "$ROBOTWIN_DIR" submodule update --init --recursive
python -m pip install -c environments/requirements-dynamic-horizon.txt -e "$ROBOTWIN_DIR/XPolicyLab"
CUROBO_DIR="$ROBOTWIN_DIR/envs/curobo"
if test ! -e "$CUROBO_DIR"; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi
python -m pip install -c environments/requirements-dynamic-horizon.txt -e "$CUROBO_DIR" --no-build-isolation
# Official RoboTwin utilities import pytorch3d even when rendering RGB only.
MAX_JOBS="${MAX_JOBS:-2}" python -m pip install -c environments/requirements-dynamic-horizon.txt \
  'git+https://github.com/facebookresearch/pytorch3d.git@33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba' --no-build-isolation
python scripts/patch_robotwin_dependencies.py
python -m pip check
python scripts/check_dynamic_horizon.py --robotwin
echo 'Next: download the official clean-only checkpoint/demos explicitly; see docs/robotwin_official.md.'
echo 'The official OpenPI archive requires a validated LeRobot conversion before this repo can evaluate it.'
