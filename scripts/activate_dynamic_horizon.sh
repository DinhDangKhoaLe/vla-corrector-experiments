#!/usr/bin/env bash
# Source this after `conda activate dynamic_horizon`.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export TOKENIZERS_PARALLELISM=false
export ROBOTWIN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/third_party/RoboTwin"
