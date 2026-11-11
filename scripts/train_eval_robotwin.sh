#!/usr/bin/env bash
set -euo pipefail

# RoboTwin Clean demonstrations:
#   extract 8x8 latents -> train Original/Prefix/Uncertainty
#   -> collect/calibrate Uncertainty -> evaluate PI0.5/Original/Prefix/Uncertainty.
#
# Defaults are tuned for one Quadro RTX 8000 (48 GiB):
#   ROBOTWIN_CAMERA_MODE=head   (head-camera Corrector latent)
#   ROBOTWIN_TASK_SET=hard15    (use full50 for all Clean tasks)
# The PI0.5 policy still receives all three cameras.  CAMERA_MODE=head is a
# one-camera Corrector-latent setup, matching the requested lightweight run.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-robotwin-dynamicVLA}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV_NAME"
fi

export ROBOTWIN_ROOT="$PWD/third_party/RoboTwin"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export WANDB_MODE=disabled
export WANDB_SILENT=true
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CAMERA_MODE="${ROBOTWIN_CAMERA_MODE:-head}"
TASK_SET="${ROBOTWIN_TASK_SET:-hard15}"
EVAL_SEED="${EVAL_SEED:-1000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
CALIBRATION_EPISODES="${CALIBRATION_EPISODES:-10}"
SAVE_VIDEO="${SAVE_VIDEO:-true}"
MAX_VIDEOS_PER_TASK="${MAX_VIDEOS_PER_TASK:-2}"
ROBOTWIN_FAST_STEP_LIMITS="${ROBOTWIN_FAST_STEP_LIMITS:-true}"
ORIGINAL_BATCH_SIZE="${ORIGINAL_BATCH_SIZE:-512}"
EXPERIMENT_BATCH_SIZE="${EXPERIMENT_BATCH_SIZE:-512}"

HARD15="hanging_mug,move_stapler_pad,handover_block,click_bell,place_cans_plasticbox,place_mouse_pad,put_object_cabinet,blocks_ranking_size,move_pillbottle_pad,scan_object,turn_switch,stamp_seal,place_can_basket,move_can_pot,place_bread_basket"
FULL50="adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,hanging_mug,lift_pot,move_can_pot,move_pillbottle_pad,move_playingcard_away,move_stapler_pad,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_burger_fries,place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_mouse_pad,place_object_basket,place_object_scale,place_object_stand,place_phone_stand,place_shoe,press_stapler,put_bottles_dustbin,put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"

case "$TASK_SET" in
  hard15) TASKS="$HARD15"; TASK_LABEL=hard15 ;;
  full50) TASKS="$FULL50"; TASK_LABEL=full50 ;;
  *) echo "ROBOTWIN_TASK_SET must be hard15 or full50" >&2; exit 2 ;;
esac

if [[ "$ROBOTWIN_FAST_STEP_LIMITS" == true ]]; then
  if [[ "$TASK_SET" != hard15 ]]; then
    echo "Fast mean+100 step limits are defined for hard15 only; set ROBOTWIN_FAST_STEP_LIMITS=false for full50." >&2
    exit 2
  fi
  STEP_LIMITS_FILE="configs/corrector/robotwin_hard15_mean_plus100_step_limits.yaml"
  STEP_LIMITS_JSON="$(python -c 'import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))' "$STEP_LIMITS_FILE")"
  LIMIT_LABEL=meanplus100
else
  STEP_LIMITS_JSON='{}'
  LIMIT_LABEL=official_limits
fi

case "$CAMERA_MODE" in
  three)
    CAMERA_LABEL=3cam
    EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-16}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-32}"
    if [[ "$TASK_SET" == hard15 ]]; then MAX_OUTPUT_GB=70; else MAX_OUTPUT_GB=230; fi
    ;;
  head)
    CAMERA_LABEL=1cam
    EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-32}"
    MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-64}"
    if [[ "$TASK_SET" == hard15 ]]; then MAX_OUTPUT_GB=24; else MAX_OUTPUT_GB=80; fi
    ;;
  *) echo "ROBOTWIN_CAMERA_MODE must be three or head" >&2; exit 2 ;;
esac

CACHE="outputs/robotwin_pi05_${TASK_LABEL}_latents_${CAMERA_LABEL}_grid8"
ORIGINAL_TRAIN="outputs/experiments/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_original_grid8"
PREFIX_TRAIN="outputs/experiments/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_prefix_grid8"
UNCERTAINTY_TRAIN="outputs/experiments/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_uncertainty_grid8"
BASELINE_EVAL="outputs/eval/robotwin_${TASK_LABEL}_pi05_${LIMIT_LABEL}_seed${EVAL_SEED}"
ORIGINAL_EVAL="outputs/eval/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_original_grid8_${LIMIT_LABEL}_seed${EVAL_SEED}"
PREFIX_EVAL="outputs/eval/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_prefix_grid8_${LIMIT_LABEL}_seed${EVAL_SEED}"
UNCERTAINTY_EVAL="outputs/eval/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_uncertainty_grid8_${LIMIT_LABEL}_seed${EVAL_SEED}"
SCENE_MANIFEST="outputs/eval/robotwin_${TASK_LABEL}_seed${EVAL_SEED}_scene_manifest.json"
UNCERTAINTY_TRACES="${UNCERTAINTY_EVAL}_calibration/experiment_episodes.jsonl"
UNCERTAINTY_ARTIFACT="outputs/experiments/robotwin_${TASK_LABEL}_${CAMERA_LABEL}_uncertainty_grid8_${LIMIT_LABEL}_calibration.json"

test -d third_party/RoboTwin/data/demo_clean
test -f pretrained_weights/robotwin_pi05_vanilla_compat/config.json
test -f pretrained_weights/robotwin_pi05_vanilla_compat/model.safetensors

echo "RoboTwin protocol: tasks=${TASK_LABEL} cameras=${CAMERA_LABEL} limits=${LIMIT_LABEL} videos_per_task=${MAX_VIDEOS_PER_TASK}"

echo '[1/10] Extract RoboTwin 8x8 latent cache'
python -m siglip_dynamics.extract \
  --dataset-path ./third_party/RoboTwin/data/demo_clean \
  --dataset-format robotwin \
  --robotwin-tasks "$TASKS" \
  --robotwin-camera-mode "$CAMERA_MODE" \
  --image-key observation.images.head \
  --wrist-image-key observation.images.left_wrist \
  --action-key actions \
  --output-path "$CACHE" \
  --encoder-backend pi05 \
  --encoder-policy-path ./pretrained_weights/robotwin_pi05_vanilla_compat \
  --encoder-local-files-only \
  --use-normalized-delta-action \
  --robotwin-action-normalization policy_quantile \
  --image-token-grid-size 8 \
  --extract-batch-size "$EXTRACT_BATCH_SIZE" \
  --max-output-gb "$MAX_OUTPUT_GB" \
  --device cuda

echo '[2/10] Train paper-code Original Corrector on pooled RoboTwin latents'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_original_grid8.yaml \
  --mode train \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$ORIGINAL_TRAIN" \
  --set training.batch_size="$ORIGINAL_BATCH_SIZE"

echo '[3/10] Train every-step ordered-prefix Corrector'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_prefix.yaml \
  --mode train \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$PREFIX_TRAIN" \
  --set training.batch_size="$EXPERIMENT_BATCH_SIZE" \
  --set training.microbatch_size="$MICRO_BATCH_SIZE"

echo '[4/10] Train every-step ordered-prefix Uncertainty Corrector'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode train \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$UNCERTAINTY_TRAIN" \
  --set training.batch_size="$EXPERIMENT_BATCH_SIZE" \
  --set training.microbatch_size="$MICRO_BATCH_SIZE" \
  --set uncertainty.traces="$UNCERTAINTY_TRACES" \
  --set uncertainty.artifact="$UNCERTAINTY_ARTIFACT"

echo '[5/10] Collect held-out Uncertainty calibration episodes'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode collect-calibration \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$UNCERTAINTY_TRAIN" \
  --set evaluation.checkpoint="$UNCERTAINTY_TRAIN/best_model.pt" \
  --set evaluation.output="$UNCERTAINTY_EVAL" \
  --set evaluation.robotwin_tasks="$TASKS" \
  --set evaluation.robotwin_step_limits="$STEP_LIMITS_JSON" \
  --set evaluation.episodes="$CALIBRATION_EPISODES" \
  --set evaluation.collection_seed=2000 \
  --set evaluation.save_video=false \
  --set uncertainty.traces="$UNCERTAINTY_TRACES" \
  --set uncertainty.artifact="$UNCERTAINTY_ARTIFACT"

echo '[6/10] Calibrate Uncertainty thresholds'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode calibrate \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$UNCERTAINTY_TRAIN" \
  --set evaluation.output="$UNCERTAINTY_EVAL" \
  --set evaluation.robotwin_tasks="$TASKS" \
  --set evaluation.robotwin_step_limits="$STEP_LIMITS_JSON" \
  --set evaluation.episodes="$CALIBRATION_EPISODES" \
  --set uncertainty.traces="$UNCERTAINTY_TRACES" \
  --set uncertainty.artifact="$UNCERTAINTY_ARTIFACT"

echo '[7/10] Evaluate standalone PI0.5 and create the shared scene manifest'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_pi05_baseline.yaml \
  --mode evaluate \
  --set training.cache="$CACHE" \
  --set evaluation.output="$BASELINE_EVAL" \
  --set evaluation.robotwin_tasks="$TASKS" \
  --set evaluation.robotwin_scene_manifest="$SCENE_MANIFEST" \
  --set evaluation.robotwin_step_limits="$STEP_LIMITS_JSON" \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.max_videos_per_task="$MAX_VIDEOS_PER_TASK" \
  --set evaluation.save_video="$SAVE_VIDEO"

echo '[8/10] Evaluate Original Corrector on the identical scenes'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_original_grid8.yaml \
  --mode evaluate \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$ORIGINAL_TRAIN" \
  --set evaluation.checkpoint="$ORIGINAL_TRAIN/h1_k10" \
  --set evaluation.output="$ORIGINAL_EVAL" \
  --set evaluation.robotwin_tasks="$TASKS" \
  --set evaluation.robotwin_scene_manifest="$SCENE_MANIFEST" \
  --set evaluation.robotwin_step_limits="$STEP_LIMITS_JSON" \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.max_videos_per_task="$MAX_VIDEOS_PER_TASK" \
  --set evaluation.save_video="$SAVE_VIDEO"

echo '[9/10] Evaluate Prefix Corrector on the identical scenes'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_prefix.yaml \
  --mode evaluate \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$PREFIX_TRAIN" \
  --set evaluation.checkpoint="$PREFIX_TRAIN/best_model.pt" \
  --set evaluation.output="$PREFIX_EVAL" \
  --set evaluation.robotwin_tasks="$TASKS" \
  --set evaluation.robotwin_scene_manifest="$SCENE_MANIFEST" \
  --set evaluation.robotwin_step_limits="$STEP_LIMITS_JSON" \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.max_videos_per_task="$MAX_VIDEOS_PER_TASK" \
  --set evaluation.save_video="$SAVE_VIDEO"

echo '[10/10] Evaluate Uncertainty Corrector on the identical scenes'
python -m siglip_dynamics.experiments.run \
  --config configs/corrector/robotwin_uncertainty.yaml \
  --mode evaluate \
  --set model.robotwin_camera_mode="$CAMERA_MODE" \
  --set training.cache="$CACHE" \
  --set training.output="$UNCERTAINTY_TRAIN" \
  --set evaluation.checkpoint="$UNCERTAINTY_TRAIN/best_model.pt" \
  --set evaluation.output="$UNCERTAINTY_EVAL" \
  --set evaluation.robotwin_tasks="$TASKS" \
  --set evaluation.robotwin_scene_manifest="$SCENE_MANIFEST" \
  --set evaluation.robotwin_step_limits="$STEP_LIMITS_JSON" \
  --set evaluation.seed="$EVAL_SEED" \
  --set evaluation.episodes="$EVAL_EPISODES" \
  --set evaluation.max_videos_per_task="$MAX_VIDEOS_PER_TASK" \
  --set evaluation.save_video="$SAVE_VIDEO" \
  --set uncertainty.traces="$UNCERTAINTY_TRACES" \
  --set uncertainty.artifact="$UNCERTAINTY_ARTIFACT"

echo "Finished RoboTwin ${TASK_LABEL}, camera_mode=${CAMERA_MODE}, seed=${EVAL_SEED}."
