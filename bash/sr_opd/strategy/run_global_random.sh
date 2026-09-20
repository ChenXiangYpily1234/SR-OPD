#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Strategy ablations write every artifact under <output-root>/<model-pair>/strategy/.
export BOUNDARY_METHOD_DIR="${BOUNDARY_METHOD_DIR:-strategy}"
source "$SCRIPT_DIR/../common/base_config.sh"
SEED="${1:-$SEED}"
exec bash "$OPD_ROOT/bash/train/sr-opd.sh" --selector-mode global_random --seed "$SEED" \
  --method-dir "$BOUNDARY_METHOD_DIR" --run-name "strategy-global-random-seed${SEED}-${BOUNDARY_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
