#!/usr/bin/env bash
set -euo pipefail
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
CONFIG_PATH="${1:-configs/laco/metaworld_pi05.yaml}"
python -m laco.evaluate --config "$CONFIG_PATH" "${@:2}"
