#!/usr/bin/env bash
# Prepare missing artifacts, evaluate MetaWorld, then prepare/evaluate LIBERO.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Keep the batch job outside the terminal's cgroup. A memory limit or oomd
# kill then stops the job rather than the interactive shell and terminal.
if [[ "${DYNAMIC_HORIZON_IN_SCOPE:-0}" != 1 ]] && command -v systemd-run >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  exec systemd-run --user --scope --quiet --unit="dynamic-horizon-$$" \
    -p MemoryHigh=36G -p MemoryMax=44G -p MemorySwapMax=512M \
    env DYNAMIC_HORIZON_IN_SCOPE=1 bash "${BASH_SOURCE[0]}" "$@"
fi
source scripts/activate_dynamic_horizon.sh
RUN_ROOT="${1:-outputs/eval/metaworld_then_libero_$(date +%Y%m%d_%H%M%S)}"
HORIZON="${HORIZON:-50}"
EPISODES="${EPISODES:-20}"
SEED="${SEED:-1000}"
case "$HORIZON" in 10|20|30|40|50) ;; *) echo 'HORIZON must be 10, 20, 30, 40, or 50' >&2; exit 2 ;; esac
if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ || ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo 'EPISODES must be positive; SEED must be a nonnegative integer' >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs"
printf 'Outputs and console logs: %s\n' "$RUN_ROOT"
CURRENT_STAGE=initialization
trap 'code=$?; printf "Stage %s stopped (exit %s). Logs: %s/logs\n" "$CURRENT_STAGE" "$code" "$RUN_ROOT" >&2; exit "$code"' ERR

run_logged() {
  local label="$1"
  shift
  CURRENT_STAGE="$label"
  "$@" 2>&1 | python -m dynamic_horizon.log_stream --log "$RUN_ROOT/logs/${label}.log"
}

prepare_corrector() {
  local benchmark="$1" cache="outputs/${1}_pi05_latents_pool64"
  if [[ "$benchmark" == metaworld ]]; then cache=outputs/metaworld_mt50_pi05_latents_pool64; fi
  if [[ ! -f "$cache/metadata.json" ]]; then
    run_logged "${benchmark}_corrector_extract" bash scripts/extract_corrector_pool64.sh "$benchmark"
  fi
  if [[ ! -f "outputs/experiments/${benchmark}_original_pool64/h1_k10/best_cosine_model.pt" ]]; then
    run_logged "${benchmark}_corrector_train" python -m siglip_dynamics.experiments.run \
      --config "configs/corrector/${benchmark}_original_pool64.yaml" --mode train
  fi
}

prepare_laco() {
  local benchmark="$1" model="outputs/laco/${1}_jacobian_seed42/best.pt"
  local calibration="outputs/laco/${1}_calibration.json"
  if [[ "$benchmark" == metaworld ]]; then
    model=outputs/laco/jacobian_seed42/best.pt
    calibration=outputs/laco/calibration.json
  fi
  if [[ ! -f "outputs/laco/${benchmark}_transitions/metadata.json" ]]; then
    run_logged "${benchmark}_laco_extract" python -m laco.extract --config "configs/laco/${benchmark}_pi05.yaml"
  fi
  if [[ ! -f "$model" ]]; then
    run_logged "${benchmark}_laco_train" python -m laco.train --config "configs/laco/${benchmark}_pi05.yaml"
  fi
  if [[ ! -f "$calibration" ]]; then
    run_logged "${benchmark}_laco_calibrate" python -m laco.calibrate --config "configs/laco/${benchmark}_pi05.yaml"
  fi
}

prepare_pace() {
  local benchmark="$1"
  if [[ ! -f "outputs/pace/${benchmark}_decoded_v2.json" ]]; then
    run_logged "${benchmark}_pace_calibrate" python -m siglip_dynamics.pace calibrate \
      --config "configs/pace/${benchmark}_pi05.yaml"
  fi
}

evaluate_all() {
  local benchmark="$1" method
  for method in baseline LACO_replan LACO_adjust corrector pace; do
    run_logged "${benchmark}_${method}_eval" python -m dynamic_horizon.evaluate \
      --benchmark "$benchmark" --method "$method" \
      --horizon "$HORIZON" --episodes "$EPISODES" --seed "$SEED" --save-videos --resume \
      --output "$RUN_ROOT/$benchmark/$method"
  done
}

# Existing MetaWorld pool64 cache/Corrector are reused. Build only missing artifacts.
prepare_corrector metaworld
prepare_laco metaworld
prepare_pace metaworld
evaluate_all metaworld

# LIBERO extraction/training/calibration starts after all MetaWorld evaluations finish.
prepare_corrector libero
prepare_laco libero
prepare_pace libero
evaluate_all libero

printf 'Finished. Per-task results: %s/{metaworld,libero}/*/per_task_metrics.csv\n' "$RUN_ROOT"
