#!/usr/bin/env bash
set -euo pipefail
BENCHMARK="${1:-metaworld}"
shift "$(( $# > 0 ? 1 : 0 ))"
NORMALIZATION=(--use-normalized-delta-action)
POLICY="pretrained_weights/pi05_${BENCHMARK}_compat"
case "$BENCHMARK" in
  metaworld) DATASET=datasets/metaworld_mt50; REPO=lerobot/metaworld_mt50; WRIST=(--no-use-wrist-image) ;;
  # The downloaded pi05_libero_base processors have empty feature/stat maps:
  # they apply identity normalization. Match that exact policy action space.
  libero) DATASET=datasets/libero; REPO=lerobot/libero; WRIST=(--use-wrist-image); NORMALIZATION=() ;;
  robotwin) DATASET=datasets/robotwin; REPO=; WRIST=(--robotwin-action-normalization=policy_quantile)
    POLICY=pretrained_weights/pi05_robotwin_official_compat
    if [[ ! -f "$POLICY/model.safetensors" ]]; then
      echo 'Official RoboTwin weights require a validated LeRobot export first; see docs/robotwin_official.md.' >&2
      exit 2
    fi ;;
  *) echo 'Expected metaworld, libero, or robotwin.' >&2; exit 2 ;;
esac
CACHE="outputs/${BENCHMARK}_pi05_latents_pool64"
if test "$BENCHMARK" = metaworld; then CACHE=outputs/metaworld_mt50_pi05_latents_pool64; fi
python -m siglip_dynamics.extract \
  --dataset-path "$DATASET" --dataset-repo-id "$REPO" --dataset-format "$BENCHMARK" \
  --output-path "$CACHE" \
  --local-pi0-checkpoint-dir "$POLICY" \
  --image-token-grid-size 8 "${NORMALIZATION[@]}" \
  --action-key action --extract-batch-size 8 --device cuda "${WRIST[@]}" "$@"
