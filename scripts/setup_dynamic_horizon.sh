#!/usr/bin/env bash
# Run explicitly to install packages. This script never downloads model/data assets.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
command -v conda >/dev/null || { echo 'Install Miniconda/Miniforge first.' >&2; exit 1; }
if conda run -n dynamic_horizon python -c 'import sys' >/dev/null 2>&1; then
  echo 'Resuming installation in the existing dynamic_horizon environment.'
else
  conda env create -f environments/dynamic_horizon.yml
fi
conda run --no-capture-output -n dynamic_horizon python -c \
  'import sys; assert sys.version_info[:2] == (3, 12), "dynamic_horizon must use Python 3.12"'
# Bootstrap before pip builds the dependency set. Otherwise egl_probe fails
# under CMake 4 and none of the main dependencies (including torch) are installed.
conda run --no-capture-output -n dynamic_horizon python -m pip install \
  'cmake>=3.29,<4' 'setuptools>=71,<81' wheel
# egl_probe does not declare cmake as a build dependency. PEP 517 isolation
# hides the Python module used by pip's cmake executable, so build it once
# against the bootstrapped tools before the main (isolated) dependency install.
conda run --no-capture-output -n dynamic_horizon python -m pip install \
  --no-build-isolation 'egl_probe==1.0.2'
conda run --no-capture-output -n dynamic_horizon python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r environments/requirements-dynamic-horizon.txt -e '.[pi,metaworld,libero]'
conda run --no-capture-output -n dynamic_horizon python -m pip check
conda run --no-capture-output -n dynamic_horizon python scripts/check_dynamic_horizon.py
echo 'Next: conda activate dynamic_horizon; source scripts/activate_dynamic_horizon.sh'
echo 'Then: bash scripts/setup_robotwin.sh (simulator source + CUDA extensions only)'
