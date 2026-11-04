#!/usr/bin/env bash
# One-H100 BBBB pipeline: extract torso-camera PI0.5 8x8 latents, verify the
# cache, then train original, action-prefix, and uncertainty Correctors.
#
# Submit from the repository root:
#   mkdir -p /mnt/data/khoaldd/jobs/logs
#   sbatch scripts/train_bbbb_all_1h100.sbatch
#
# This script uses paths ending in _1h100, so it can coexist with the 3-H100
# original-only job without sharing a cache or checkpoints.

#SBATCH --job-name=train_all_corrector
#SBATCH --nodes=1
#SBATCH --nodelist=worker-1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH --output=/mnt/data/khoaldd/jobs/logs/bbbb_all1_%j.out
#SBATCH --error=/mnt/data/khoaldd/jobs/logs/bbbb_all1_%j.err
#SBATCH --container-image=/mnt/data/khoaldd/containers/ubuntu22-cuda128-lerobot.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data

set -euo pipefail

# All paths can be overridden with `sbatch --export=ALL,NAME=value ...`.
REPO_ROOT="${REPO_ROOT:-/mnt/data/khoaldd/vla-corrector-experiments}"
CONDA_BASE="${CONDA_BASE:-/mnt/data/khoaldd/miniforge3}"
ENV_NAME="${ENV_NAME:-corrector}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/bbbb}"
POLICY_DIR="${POLICY_DIR:-$REPO_ROOT/pretrained_weights/folding_best_lerobot}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/outputs/bbbb_pi05_latents_torso_grid8_1h100}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
# BBBB has sub-millisecond video timestamp rounding drift. 1 ms is still far
# below a video frame interval and selects the nearest correctly decoded frame.
VIDEO_TOLERANCE_S="${VIDEO_TOLERANCE_S:-0.001}"

ORIGINAL_CONFIG="$REPO_ROOT/configs/corrector/bbbb_original.yaml"
PREFIX_CONFIG="$REPO_ROOT/configs/corrector/bbbb_prefix.yaml"
UNCERTAINTY_CONFIG="$REPO_ROOT/configs/corrector/bbbb_uncertainty.yaml"

ORIGINAL_TRAIN="${ORIGINAL_TRAIN:-$REPO_ROOT/outputs/bbbb_original_corrector_torso_grid8_1h100}"
PREFIX_TRAIN="${PREFIX_TRAIN:-$REPO_ROOT/outputs/bbbb_prefix_corrector_torso_grid8_1h100}"
UNCERTAINTY_TRAIN="${UNCERTAINTY_TRAIN:-$REPO_ROOT/outputs/bbbb_uncertainty_corrector_torso_grid8_1h100}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_empty() {
    local path="$1"
    if [[ -e "$path" ]] && [[ -n "$(find "$path" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
        die "Refusing to overwrite non-empty path: $path"
    fi
}

run_phase() {
    echo
    echo "===== $1 ====="
    shift
    "$@"
}

trap 'echo "Pipeline failed at $(date -Is)" >&2' ERR

[[ -d "$REPO_ROOT" ]] || die "Repository not found: $REPO_ROOT"
[[ -f "$DATASET_DIR/meta/info.json" && -d "$DATASET_DIR/videos" ]] || \
    die "Expected complete LeRobot BBBB data (meta/, data/, videos/) at: $DATASET_DIR"
[[ -f "$POLICY_DIR/config.json" && -f "$POLICY_DIR/model.safetensors" ]] || \
    die "Converted PI0.5 policy is incomplete: $POLICY_DIR"
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || die "Conda setup script not found: $CONDA_BASE"
[[ "$EXTRACT_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "EXTRACT_BATCH_SIZE must be a positive integer"
for config in "$ORIGINAL_CONFIG" "$PREFIX_CONFIG" "$UNCERTAINTY_CONFIG"; do
    [[ -f "$config" ]] || die "Corrector config not found: $config"
done

cd "$REPO_ROOT"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python - "$VIDEO_TOLERANCE_S" <<'PY' || die "VIDEO_TOLERANCE_S must be a finite positive number of seconds"
import math
import sys
value = float(sys.argv[1])
assert math.isfinite(value) and value > 0
PY

# Use this checkout's PI0.5/LeRobot implementation.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=disabled
export WANDB_SILENT=true

echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Started: $(date -Is)"
echo "Cache: $CACHE_DIR"
echo "Video timestamp tolerance: $VIDEO_TOLERANCE_S s"
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print("PyTorch", torch.__version__, "on", torch.cuda.get_device_name(0))
PY

# metadata.json is created only once extraction completes. A directory without
# it is partial and is never reused.
if [[ -f "$CACHE_DIR/metadata.json" ]]; then
    run_phase "Validate existing torso-only 8x8 cache" python - "$CACHE_DIR/metadata.json" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
assert metadata["image_token_grid_size"] == 8, metadata
assert metadata.get("image_token_grid_shape", []) == [], metadata
assert metadata["action_dim"] == 16, metadata
assert metadata.get("image_key") == "observation.images.cam_torso", metadata
assert metadata.get("use_wrist_image") is False, metadata
print("Using existing completed BBBB torso 8x8 cache:", sys.argv[1])
PY
elif [[ -e "$CACHE_DIR" ]]; then
    die "Partial cache exists at $CACHE_DIR. Delete it manually, or set CACHE_DIR to a new path."
else
    run_phase "Extract torso-camera PI0.5 8x8 spatial-mean latents" \
        python -m siglip_dynamics.extract \
        --dataset-path "$DATASET_DIR" \
        --dataset-repo-id Kronze157/bbbb \
        --dataset-loader lerobot \
        --video-tolerance-s "$VIDEO_TOLERANCE_S" \
        --dataset-format libero \
        --image-key observation.images.cam_torso \
        --no-use-wrist-image \
        --action-key action \
        --output-path "$CACHE_DIR" \
        --max-output-gb 800 \
        --extract-batch-size "$EXTRACT_BATCH_SIZE" \
        --image-token-grid-size 8 \
        --encoder-backend pi05 \
        --encoder-policy-path "$POLICY_DIR" \
        --encoder-local-files-only \
        --use-normalized-delta-action \
        --device cuda
fi

# Some older deployed checkouts do not include the optional recompute verifier.
# The completed-cache validation above still checks the required metadata; use
# the stronger five-frame recompute check when this module is available.
if python -c 'import siglip_dynamics.verify_extraction' >/dev/null 2>&1; then
    run_phase "Verify extracted BBBB cache" \
        python -m siglip_dynamics.verify_extraction \
        --dataset-path "$DATASET_DIR" \
        --dataset-repo-id Kronze157/bbbb \
        --dataset-loader lerobot \
        --dataset-format libero \
        --extracted-path "$CACHE_DIR" \
        --encoder-policy-path "$POLICY_DIR" \
        --encoder-local-files-only \
        --device cuda \
        --num-samples 5
else
    echo "WARNING: siglip_dynamics.verify_extraction is unavailable; skipping optional embedding recompute check."
fi

require_empty "$ORIGINAL_TRAIN"
run_phase "Train Original Corrector on 1 H100" \
    python -m siglip_dynamics.experiments.run \
    --config "$ORIGINAL_CONFIG" --mode train \
    --set "training.cache=$CACHE_DIR" \
    --set "training.output=$ORIGINAL_TRAIN"

require_empty "$PREFIX_TRAIN"
run_phase "Train Action-Prefix Corrector on 1 H100" \
    python -m siglip_dynamics.experiments.run \
    --config "$PREFIX_CONFIG" --mode train \
    --set "training.cache=$CACHE_DIR" \
    --set "training.output=$PREFIX_TRAIN"

require_empty "$UNCERTAINTY_TRAIN"
run_phase "Train Uncertainty Corrector on 1 H100" \
    python -m siglip_dynamics.experiments.run \
    --config "$UNCERTAINTY_CONFIG" --mode train \
    --set "training.cache=$CACHE_DIR" \
    --set "training.output=$UNCERTAINTY_TRAIN"

echo "Finished successfully: $(date -Is)"
