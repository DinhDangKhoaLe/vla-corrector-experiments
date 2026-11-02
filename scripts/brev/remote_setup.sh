#!/usr/bin/env bash
# Runs on the GPU instance. Normally launched by setup.sh.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
cd -- "$BREV_REPO_ROOT"
mkdir -p outputs/brev
exec 9>outputs/brev/gpu.lock
flock -n 9 || { echo 'Setup or training is already running.' >&2; exit 1; }
status_file=outputs/brev/setup.status
echo "running pid=$$ started=$(date -Is)" > "$status_file"
trap 'rc=$?; if (( rc != 0 )); then echo "failed exit=$rc at=$(date -Is)" > "$status_file"; fi' EXIT
echo "Repository: $BREV_REPO_ROOT"
nvidia-smi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
    echo 'This environment requires Linux x86_64.' >&2; exit 1;
}
# Includes room for the datasets library's intermediate Arrow copy.
python3 - "$BREV_REPO_ROOT" "${BREV_MIN_FREE_GB:-160}" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free / 1e9
minimum = float(sys.argv[2])
print(f"Free space: {free:.1f} GB; required headroom: {minimum:.1f} GB")
if free < minimum:
    raise SystemExit("Insufficient space; choose a larger --remote-dir mount.")
PY

miniforge="$BREV_REPO_ROOT/.cache/miniforge3"
if [[ ! -x "$miniforge/bin/conda" ]]; then
    mkdir -p .cache
    version=${BREV_MINIFORGE_VERSION:-26.9.4-1}
    installer="$BREV_REPO_ROOT/.cache/Miniforge3-$version-Linux-x86_64.sh"
    url="https://github.com/conda-forge/miniforge/releases/download/$version/$(basename "$installer")"
    curl -fL --retry 5 "$url" -o "$installer"
    curl -fL --retry 5 "$url.sha256" -o "$installer.sha256"
    (cd .cache && sha256sum --check "$(basename "$installer").sha256")
    bash "$installer" -b -p "$miniforge"
    rm -- "$installer" "$installer.sha256"
fi
if [[ ! -x "$BREV_ENV_PREFIX/bin/python" ]]; then
    "$miniforge/bin/conda" create -y --override-channels -c conda-forge \
        -p "$BREV_ENV_PREFIX" python=3.12 'ffmpeg>=7,<8' pip 'setuptools>=71,<81' wheel
fi
brev_activate
python -m pip install --no-cache-dir -c scripts/brev/constraints.txt \
    -e '.[metaworld,pi]' --no-build-isolation pyyaml pillow matplotlib
python -m pip check
python - <<'PY'
import torch
from torchcodec.decoders import VideoDecoder
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
import metaworld
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the installed environment")
print(f"CUDA OK: {torch.cuda.get_device_name(0)}; torch={torch.__version__}")
PY
python -m pip freeze > outputs/brev/installed-requirements.txt

hf download lerobot/metaworld_mt50 --repo-type dataset --local-dir datasets/metaworld_mt50
hf download tiantianx/pi05_metaworld --local-dir pretrained_weights/pi05_metaworld
if [[ ! -e pretrained_weights/pi05_metaworld_compat ]]; then
    python scripts/make_pi05_compat.py --source pretrained_weights/pi05_metaworld \
        --output pretrained_weights/pi05_metaworld_compat
fi

cache=outputs/metaworld_mt50_pi05_latents_pool64
if [[ -e "$cache/metadata.json" ]]; then
    python scripts/brev/validate_cache.py "$cache"
    echo 'Completed cache already exists; skipping extraction.'
else
    # The extractor opens arrays with w+ and cannot resume. Preserve partial work.
    if [[ -d "$cache" ]]; then
        mv -- "$cache" "$cache.incomplete.$(date +%Y%m%dT%H%M%S).$$"
    fi
    python -m siglip_dynamics.extract \
        --dataset-path ./datasets/metaworld_mt50 \
        --dataset-repo-id lerobot/metaworld_mt50 \
        --dataset-loader parquet --dataset-format metaworld \
        --image-key observation.image --no-use-wrist-image --action-key action \
        --output-path "./$cache" --encoder-backend pi05 \
        --encoder-policy-path ./pretrained_weights/pi05_metaworld_compat \
        --encoder-local-files-only --use-normalized-delta-action \
        --image-token-grid-size 8 --max-output-gb 32 --device cuda
    python scripts/brev/validate_cache.py "$cache"
fi
echo "complete at=$(date -Is)" > "$status_file"
echo 'Setup and pool64 extraction complete. Start training with: bash scripts/brev/train.sh --local original'
