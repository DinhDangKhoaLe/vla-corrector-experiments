#!/usr/bin/env bash
# Sourced by the remote setup and training entry points.
BREV_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
BREV_ENV_PREFIX=${BREV_ENV_PREFIX:-"$BREV_REPO_ROOT/.venv"}

brev_activate() {
    if [[ ! -x "$BREV_ENV_PREFIX/bin/python" ]]; then
        echo "Environment missing: run scripts/brev/setup.sh first." >&2
        return 1
    fi
    export PATH="$BREV_ENV_PREFIX/bin:$PATH"
    # FFmpeg shared libraries are needed by TorchCodec.
    export LD_LIBRARY_PATH="$BREV_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export HF_HOME="$BREV_REPO_ROOT/.cache/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless
    export WANDB_MODE=${WANDB_MODE:-disabled} WANDB_SILENT=true
    export PYTHONUNBUFFERED=1
    cd -- "$BREV_REPO_ROOT"
}

brev_quote() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
