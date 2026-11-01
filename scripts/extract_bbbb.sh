#!/usr/bin/env bash
# Extract frozen PI0.5 visual latents from the local Kronze157/bbbb snapshot.
# Extracts only the torso camera. Its native 16x16 PI0.5 token grid is
# mean-pooled to 8x8: 64 tokens per frame.
# Usage: scripts/extract_bbbb.sh [DATASET_DIR] [PI05_CHECKPOINT_DIR] [OUTPUT_DIR]
set -euo pipefail

dataset_dir=${1:-datasets/bbbb}
policy_dir=${2:-pretrained_weights/pi05_bbbb}
output_dir=${3:-outputs/bbbb_pi05_latents_torso_grid8}

if [[ ! -f "$dataset_dir/meta/info.json" || ! -d "$dataset_dir/videos" ]]; then
  echo "Expected a complete LeRobot v3 snapshot at $dataset_dir (meta/, data/, and videos/)." >&2
  exit 2
fi
if [[ ! -f "$policy_dir/config.json" ]]; then
  echo "Expected a loadable PI0.5/PI05 checkpoint directory at $policy_dir." >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite $output_dir. Choose a new, empty output directory." >&2
  exit 2
fi

# Do not use --dataset-loader=parquet: it reads metadata rows but does not
# decode this v3 dataset's MP4-backed camera features.
# The full cache is about 682 GB for 5.2M frames and 64 pooled visual tokens.
python -m siglip_dynamics.extract \
  --dataset-path "$dataset_dir" \
  --dataset-repo-id Kronze157/bbbb \
  --dataset-loader lerobot \
  --dataset-format libero \
  --image-key observation.images.cam_torso \
  --no-use-wrist-image \
  --action-key action \
  --output-path "$output_dir" \
  --max-output-gb 800 \
  --extract-batch-size 16 \
  --image-token-grid-size 8 \
  --encoder-backend pi05 \
  --encoder-policy-path "$policy_dir" \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --device cuda

python -m siglip_dynamics.verify_extraction \
  --dataset-path "$dataset_dir" \
  --dataset-repo-id Kronze157/bbbb \
  --dataset-loader lerobot \
  --dataset-format libero \
  --extracted-path "$output_dir" \
  --encoder-policy-path "$policy_dir" \
  --encoder-local-files-only \
  --device cuda \
  --num-samples 5

