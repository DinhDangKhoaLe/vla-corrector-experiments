#!/usr/bin/env bash
# Extract (or reuse) the BBBB torso-camera 8x8 PI0.5 cache and train only
# the original VLA-Corrector with 3 H100 GPUs via single-node DDP.
#
# Submit from the repository root:
#   mkdir -p /mnt/data/khoaldd/jobs/logs
#   sbatch scripts/train_bbbb_original_3h100.sbatch
#
# This job deliberately has its own cache and checkpoint paths, so it can run
# alongside the one-GPU pipeline without sharing output files.

#SBATCH --job-name=folding_corrector_3gpus
#SBATCH --nodes=1
#SBATCH --nodelist=worker-1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH --output=/mnt/data/khoaldd/jobs/logs/bbbb_orig3_%j.out
#SBATCH --error=/mnt/data/khoaldd/jobs/logs/bbbb_orig3_%j.err
#SBATCH --container-image=/mnt/data/khoaldd/containers/ubuntu22-cuda128-lerobot.sqsh
#SBATCH --container-mounts=/mnt/data:/mnt/data

set -euo pipefail

# All paths can be overridden with `sbatch --export=ALL,NAME=value ...`.
REPO_ROOT="${REPO_ROOT:-/mnt/data/khoaldd/vla-corrector-experiments}"
CONDA_BASE="${CONDA_BASE:-/mnt/data/khoaldd/miniforge3}"
ENV_NAME="${ENV_NAME:-corrector}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/bbbb}"
POLICY_DIR="${POLICY_DIR:-$REPO_ROOT/pretrained_weights/folding_best_lerobot}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/outputs/bbbb_pi05_latents_torso_grid8_3h100}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
# BBBB has sub-millisecond video timestamp rounding drift. 1 ms is still far
# below a video frame interval and accepts the nearest correctly decoded frame.
VIDEO_TOLERANCE_S="${VIDEO_TOLERANCE_S:-0.001}"
ORIGINAL_TRAIN="${ORIGINAL_TRAIN:-$REPO_ROOT/outputs/bbbb_original_corrector_torso_grid8_3h100}"

# This is per GPU. 170 x 3 = an effective global batch size of 510, the
# closest practical value to the prior single-GPU batch size of 512.
ORIGINAL_PER_GPU_BATCH_SIZE="${ORIGINAL_PER_GPU_BATCH_SIZE:-170}"

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
[[ "$ORIGINAL_PER_GPU_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || die "ORIGINAL_PER_GPU_BATCH_SIZE must be a positive integer"

cd "$REPO_ROOT"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python - "$VIDEO_TOLERANCE_S" <<'PY' || die "VIDEO_TOLERANCE_S must be a finite positive number of seconds"
import math
import sys
value = float(sys.argv[1])
assert math.isfinite(value) and value > 0
PY

# Ensure the local PI0.5/LeRobot implementation is used.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=disabled
export WANDB_SILENT=true

# Prevent 3 DDP ranks from oversubscribing the 32 allocated CPU cores. TF32
# uses H100 Tensor Cores for FP32 matmuls; validate once if bitwise identity
# with old one-GPU checkpoints is required.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE="${TORCH_ALLOW_TF32_CUBLAS_OVERRIDE:-1}"

echo "Job: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "Started: $(date -Is)"
echo "Cache: $CACHE_DIR"
echo "Video timestamp tolerance: $VIDEO_TOLERANCE_S s"
echo "Original output: $ORIGINAL_TRAIN"
echo "DDP: 3 GPUs x $ORIGINAL_PER_GPU_BATCH_SIZE samples = $((3 * ORIGINAL_PER_GPU_BATCH_SIZE)) global batch"
nvidia-smi
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() >= 3, f"Expected 3 visible GPUs, found {torch.cuda.device_count()}"
print("PyTorch", torch.__version__, "with", torch.cuda.device_count(), "visible GPUs")
for i in range(3):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
PY

# Extraction is all-or-nothing: metadata.json is created only after a complete
# cache. Never reuse a directory that exists without that file.
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

# The legacy original trainer supports DDP. Invoke it directly rather than
# wrapping `experiments.run`: this avoids multiple wrapper processes racing to
# create/check the same output directory.
require_empty "$ORIGINAL_TRAIN"
run_phase "Train Original Corrector on 3 H100 GPUs" \
    torchrun --standalone --nproc_per_node=3 \
    -m siglip_dynamics.train \
    --dataset-path "$CACHE_DIR" \
    --checkpoint-dir "$ORIGINAL_TRAIN" \
    --device cuda \
    --model-type mlp \
    --scale 20m \
    --token-dim 2048 \
    --action-dim 16 \
    --action-embed-dim 256 \
    --dropout 0.0 \
    --h-window 1 \
    --k-step-list 10 \
    --batch-size "$ORIGINAL_PER_GPU_BATCH_SIZE" \
    --epochs 30 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --val-ratio 0.1 \
    --seed 42 \
    --grad-clip-norm 1.0 \
    --patience 5 \
    --train-loss-type cosine

echo "Finished successfully: $(date -Is)"
