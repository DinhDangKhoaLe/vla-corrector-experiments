#!/usr/bin/env bash
set -euo pipefail

# Full clean MetaWorld MT50 8x8 experiment:
#   extract -> train Original/Prefix/Uncertainty -> calibrate Uncertainty
#   -> evaluate standalone PI0.5/Original/Prefix/Uncertainty.
# Existing non-empty output directories are intentionally never overwritten.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-lerobot-dynamicVLA}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
fi

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_PLATFORM=surfaceless
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=disabled
export WANDB_SILENT=true

EVAL_SEED="${EVAL_SEED:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
CALIBRATION_EPISODES="${CALIBRATION_EPISODES:-10}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-16}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"

test -d datasets/metaworld_mt50
test -f pretrained_weights/pi05_metaworld_compat/config.json
test -f pretrained_weights/pi05_metaworld_compat/model.safetensors

echo '[1/9] Extract 8x8 (64-token) MetaWorld latent cache'
python -m siglip_dynamics.extract \
  --dataset-path ./datasets/metaworld_mt50 \
  --dataset-repo-id lerobot/metaworld_mt50 \
  --dataset-loader parquet \
  --dataset-format metaworld \
  --image-key observation.image \
  --no-use-wrist-image \
  --action-key action \
  --output-path ./outputs/metaworld_mt50_pi05_latents_pool64 \
  --encoder-backend pi05 \
  --encoder-policy-path ./pretrained_weights/pi05_metaworld_compat \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --image-token-grid-size 8 \
  --extract-batch-size "$EXTRACT_BATCH_SIZE" \
  --max-output-gb 32 \
  --device cuda

echo '[2/9] Train released Original VLA-Corrector recipe on pooled latents'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml \
  --mode train

echo '[3/9] Train every-step ordered-prefix Corrector'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool64.yaml \
  --mode train

echo '[4/9] Train every-step ordered-prefix Uncertainty Corrector'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode train

echo '[5/9] Collect held-out Uncertainty calibration episodes'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode collect-calibration \
  --set evaluation.episodes="$CALIBRATION_EPISODES" \
  --set evaluation.save_video=false

echo '[6/9] Calibrate Uncertainty thresholds'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode calibrate

echo '[7/9] Evaluate standalone fixed-H=50 PI0.5 baseline'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_pi05_baseline_pool64.yaml \
  --mode evaluate \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.save_video="$SAVE_VIDEO"

echo '[8/9] Evaluate Original VLA-Corrector'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_original_pool64.yaml \
  --mode evaluate \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.save_video="$SAVE_VIDEO"

echo '[9/9] Evaluate Prefix and Uncertainty Correctors'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_prefix_pool64.yaml \
  --mode evaluate \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.save_video="$SAVE_VIDEO"

python -m siglip_dynamics.experiments.run \
  --config configs/corrector/metaworld_uncertainty_pool64.yaml \
  --mode evaluate \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.save_video="$SAVE_VIDEO"

echo 'Finished. Compare macro_task_group_success_rate in each analysis/summary/global_summary.json.'
