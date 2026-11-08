#!/usr/bin/env bash
set -euo pipefail
CONFIG_PATH="${1:-configs/laco/metaworld_pi05.yaml}"
python -m laco.train --config "$CONFIG_PATH" "${@:2}"
