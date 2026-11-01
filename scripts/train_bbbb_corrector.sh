#!/usr/bin/env bash
# Train one offline Corrector variant for Kronze157/bbbb.
# Usage: scripts/train_bbbb_corrector.sh {original|prefix|uncertainty}
set -euo pipefail

variant=${1:?"usage: $0 {original|prefix|uncertainty}"}
case "$variant" in
  original|prefix|uncertainty) ;;
  *) echo "Unknown variant '$variant'; use original, prefix, or uncertainty." >&2; exit 2 ;;
esac

export WANDB_MODE=${WANDB_MODE:-disabled}
export WANDB_SILENT=${WANDB_SILENT:-true}

# The legacy original trainer supports distributed torchrun; the experimental
# prefix/uncertainty trainer is intentionally single-process.
if [[ "$variant" == original ]]; then
  torchrun --nproc_per_node=1 -m siglip_dynamics.experiments.run \
    --config "configs/corrector/bbbb_${variant}.yaml" --mode train
else
  python -m siglip_dynamics.experiments.run \
    --config "configs/corrector/bbbb_${variant}.yaml" --mode train
fi
