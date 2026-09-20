#!/usr/bin/env bash
# Shortest-Wrong baseline: mixed prompts only, exactly one query, targeting the
# shortest incorrect sibling. Length-only control for the SR-OPD ranking rule.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# Strategy ablations write every artifact under <output-root>/<model-pair>/strategy/.
export BOUNDARY_METHOD_DIR="${BOUNDARY_METHOD_DIR:-strategy}"
source "$SCRIPT_DIR/../common/base_config.sh"
SEED="${1:-$SEED}"
exec bash "$OPD_ROOT/bash/train/sr-opd.sh" --selector-mode shortest_wrong --seed "$SEED" \
  --method-dir "$BOUNDARY_METHOD_DIR" --run-name "strategy-shortest-wrong-seed${SEED}-${BOUNDARY_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"