#!/usr/bin/env bash
# Train -> held-out collection -> calibration -> evaluation, optionally after Prefix.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
instance=primitive-pink-bovid
remote_dir=/data/vla-corrector-experiments
local_run=false
after_prefix=false
eval_prefix_first=false
through=evaluate
forward=()
while (($#)); do
    case "$1" in
        --local) local_run=true; shift ;;
        --instance) instance=${2:?}; shift 2 ;;
        --remote-dir) remote_dir=${2:?}; shift 2 ;;
        --after-prefix) after_prefix=true; forward+=(--after-prefix); shift ;;
        --eval-prefix-first) after_prefix=true; eval_prefix_first=true; forward+=(--eval-prefix-first); shift ;;
        --through) through=${2:?}; forward+=(--through "$through"); shift 2 ;;
        -h|--help)
            echo 'Usage: bash scripts/brev/uncertainty.sh [--after-prefix | --eval-prefix-first] [--through train|calibrate|evaluate] [--instance NAME] [--remote-dir PATH] [--local]'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$through" in train|calibrate|evaluate) ;; *) echo 'Invalid --through stage.' >&2; exit 2 ;; esac
if ! $local_run; then
    [[ "$remote_dir" == /* && "$remote_dir" != / ]] || { echo '--remote-dir must be an absolute project path.' >&2; exit 2; }
    command='python3 scripts/brev/launch.py uncertainty-workflow'
    for arg in "${forward[@]}"; do command+=" $(brev_quote "$arg")"; done
    brev exec "$instance" "cd $(brev_quote "$remote_dir") && $command"
    exit 0
fi
brev_activate
mkdir -p outputs/brev
exec 8>outputs/brev/uncertainty-workflow.lock
flock -n 8 || { echo 'An uncertainty workflow is already queued or running.' >&2; exit 1; }
workflow_status=outputs/brev/uncertainty-workflow.status
stage_status=''
echo "starting pid=$$ at=$(date -Is)" > "$workflow_status"
trap 'rc=$?; if (( rc != 0 )); then echo "failed exit=$rc stage=${stage_status:-preflight} at=$(date -Is)" > "$workflow_status"; if [[ -n "$stage_status" ]]; then echo "failed exit=$rc at=$(date -Is)" > "$stage_status"; fi; fi' EXIT

config=configs/corrector/metaworld_uncertainty_pool64.yaml
policy=pretrained_weights/pi05_metaworld_eval_compat
common=(--config "$config" --set "evaluation.policy=$policy")
# Preserve existing training, calibration and evaluation outputs. Check before queuing.
python - "$eval_prefix_first" <<'PY'
from pathlib import Path
import sys
from siglip_dynamics.experiments.config import load_config
cfg = load_config('configs/corrector/metaworld_uncertainty_pool64.yaml')
assert cfg.features.action_prefix and cfg.features.uncertainty_monitor
assert cfg.model.image_token_grid_size == 8
assert cfg.training.cache == 'outputs/metaworld_mt50_pi05_latents_pool64'
for name in (cfg.training.output, cfg.evaluation.output, cfg.evaluation.output + '_calibration'):
    p = Path(name)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f'Existing uncertainty output: {p}; inspect it before rerunning')
if Path(cfg.uncertainty.artifact).exists():
    raise FileExistsError('A calibration artifact already exists; inspect it before rerunning')
if not Path('pretrained_weights/pi05_metaworld_eval_compat/policy_preprocessor.json').is_file():
    raise FileNotFoundError('Prepare the working evaluation policy/tokenizer compatibility view first')
if sys.argv[1] == 'true':
    prefix = load_config('configs/corrector/metaworld_prefix_pool64.yaml')
    p = Path(prefix.evaluation.output)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f'Existing Prefix evaluation output: {p}; inspect it before rerunning')
print('Uncertainty config and fresh output paths verified', flush=True)
PY
if $after_prefix; then
    prerequisite=outputs/brev/train-prefix.status
    [[ -f "$prerequisite" ]] || { echo 'Prefix status file is missing.' >&2; exit 1; }
    echo "waiting prerequisite=prefix pid=$$ at=$(date -Is)" > "$workflow_status"
    echo 'Queued: waiting for Prefix training to finish successfully.'
    while true; do
        prefix_status=$(cat "$prerequisite")
        case "$prefix_status" in
            complete\ *) break ;;
            failed\ *) echo "Prefix failed: $prefix_status" >&2; exit 1 ;;
            running\ *)
                if [[ "$prefix_status" =~ pid=([0-9]+) ]]; then
                    kill -0 "${BASH_REMATCH[1]}" 2>/dev/null || { echo 'Prefix process exited without a completion status.' >&2; exit 1; }
                else
                    echo 'Unrecognized Prefix running status.' >&2; exit 1
                fi ;;
            *) echo "Unrecognized Prefix status: $prefix_status" >&2; exit 1 ;;
        esac
        sleep 15
    done
fi
exec 9>outputs/brev/gpu.lock
flock -w 30 9 || { echo 'GPU is busy after the prerequisite finished.' >&2; exit 1; }
grep -q '^complete ' outputs/brev/setup.status
python scripts/brev/validate_cache.py outputs/metaworld_mt50_pi05_latents_pool64

run_stage() {
    local name=$1
    shift
    stage_status="outputs/brev/$name.status"
    echo "running stage=$name pid=$$ at=$(date -Is)" > "$workflow_status"
    echo "running pid=$$ at=$(date -Is)" > "$stage_status"
    echo "Starting uncertainty stage: $name"
    "$@"
    echo "complete at=$(date -Is)" > "$stage_status"
    stage_status=''
}
if $eval_prefix_first; then
    run_stage eval-prefix python -m siglip_dynamics.experiments.run \
        --config configs/corrector/metaworld_prefix_pool64.yaml \
        --set "evaluation.policy=$policy" --mode evaluate
fi
run_stage train-uncertainty python -m siglip_dynamics.experiments.run "${common[@]}" --mode train
if [[ "$through" != train ]]; then
    run_stage collect-calibration-uncertainty python -m siglip_dynamics.experiments.run "${common[@]}" \
        --mode collect-calibration --set evaluation.episodes=10 --set evaluation.save_video=false
    run_stage calibrate-uncertainty python -m siglip_dynamics.experiments.run "${common[@]}" --mode calibrate
fi
if [[ "$through" == evaluate ]]; then
    run_stage eval-uncertainty python -m siglip_dynamics.experiments.run "${common[@]}" --mode evaluate
fi
echo "complete through=$through at=$(date -Is)" > "$workflow_status"
