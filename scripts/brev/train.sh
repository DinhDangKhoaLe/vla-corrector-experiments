#!/usr/bin/env bash
# Local launcher, or use --local inside the instance to run in the foreground.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"
instance=primitive-pink-bovid
remote_dir=/data/vla-corrector-experiments
local_run=false
variant=original
overrides=()
while (($#)); do
    case "$1" in
        --local) local_run=true; shift ;;
        --instance) instance=${2:?}; shift 2 ;;
        --remote-dir) remote_dir=${2:?}; shift 2 ;;
        --set) overrides+=(--set "${2:?}"); shift 2 ;;
        original|prefix|uncertainty|all) variant=$1; shift ;;
        -h|--help)
            echo 'Usage: bash scripts/brev/train.sh [original|prefix|uncertainty|all] [--instance NAME] [--remote-dir PATH] [--set training.epochs=30] [--local]'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
if ! $local_run; then
    [[ "$remote_dir" == /* && "$remote_dir" != / ]] || { echo '--remote-dir must be an absolute project path.' >&2; exit 2; }
    quoted_dir=$(brev_quote "$remote_dir")
    # Fail immediately while setup/extraction/training holds the GPU lock.
    brev exec "$instance" "test -f $quoted_dir/outputs/brev/setup.status && grep -q '^complete ' $quoted_dir/outputs/brev/setup.status && flock -n $quoted_dir/outputs/brev/gpu.lock true"
    command="python3 scripts/brev/launch.py train $variant"
    for arg in "${overrides[@]}"; do command+=" $(brev_quote "$arg")"; done
    brev exec "$instance" "cd $quoted_dir && $command"
    exit 0
fi
brev_activate
mkdir -p outputs/brev
exec 9>outputs/brev/gpu.lock
flock -n 9 || { echo 'Setup or training is already running.' >&2; exit 1; }
status_file="outputs/brev/train-$variant.status"
echo "running pid=$$ started=$(date -Is)" > "$status_file"
trap 'rc=$?; if (( rc != 0 )); then echo "failed exit=$rc at=$(date -Is)" > "$status_file"; fi' EXIT
grep -q '^complete ' outputs/brev/setup.status
python scripts/brev/validate_cache.py outputs/metaworld_mt50_pi05_latents_pool64
variants=("$variant")
[[ "$variant" != all ]] || variants=(original prefix uncertainty)
for mode in "${variants[@]}"; do
    python -m siglip_dynamics.experiments.run \
        --config "configs/corrector/metaworld_${mode}_pool64.yaml" \
        --mode train "${overrides[@]}"
done
echo "complete at=$(date -Is)" > "$status_file"
