#!/bin/bash
# Frontier-TLR: matched-support baseline for SR-OPD.
# Uses Student response length and entropy to rank failed siblings of mixed prompts.
# Usage: bash bash/sr_opd/main/run_frontier_tlr.sh [seed]
{
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEED="${1:-${SEED:-42}}"
# Point MODELS_DIR at a directory holding both checkpoints, or set
# TEACHER_MODEL_PATH / STUDENT_MODEL_PATH explicitly.
MODELS_DIR="${MODELS_DIR:-}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODELS_DIR:+$MODELS_DIR/Qwen3-4B}}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODELS_DIR:+$MODELS_DIR/Qwen3-1.7B}}"
if [[ -z "$TEACHER_MODEL_PATH" || -z "$STUDENT_MODEL_PATH" ]]; then
    echo "ERROR: set MODELS_DIR (holding both checkpoints) or TEACHER_MODEL_PATH/STUDENT_MODEL_PATH" >&2
    exit 2
fi
for model_path in "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH"; do
    [[ -d "$model_path" ]] || { echo "ERROR: model dir not found: $model_path" >&2; exit 2; }
done

export ACTOR_MODEL_PATH="$STUDENT_MODEL_PATH"
export REWARD_MODEL_PATH="$TEACHER_MODEL_PATH"
export MAX_RESP_LENGTH=8192
# K=4 rollouts — identical to SR-OPD / Frontier-Shortest / Frontier-Random.
export FF_K_ROLLOUTS=4
export N_RESPONSES=4
export FF_MAX_NO_SUCCESS_RETRIES=0
export FF_TOTAL_EPOCHS=1
# Frontier-TLR selector: mixed-only gating + wrong-only candidates + TLR score.
# No Boundary hidden-state capture is needed; the selector uses only response
# length and Student token entropy already available from the rollout forward.
export FF_SELECTOR_MODE=frontier_tlr
# BOUNDARY_SELECTOR_SCORE_MODE is only meaningful for boundary_opd modes;
# keep the default "legacy" so BoundaryOPDSettings validation is not triggered.
export BOUNDARY_OPD_FALLBACK_TO_FF_COST=false
export BOUNDARY_METHOD_DIR="${BOUNDARY_METHOD_DIR:-sr_opd}"
# Student token entropies are required by the TLR score.  They are computed
# during the pre-Teacher log-prob forward when LOG_PROB_TOP_K > 0.
# FF-OPD sets LOG_PROB_TOP_K=16 by default; keep that value.
export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"

if [[ -n "${FRONTIER_TLR_SMOKE_STEPS:-}" ]]; then
    [[ "$FRONTIER_TLR_SMOKE_STEPS" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: FRONTIER_TLR_SMOKE_STEPS must be a positive integer" >&2
        exit 2
    }
    export FF_FRESH_STEP_LIMIT="$FRONTIER_TLR_SMOKE_STEPS"
fi
RUN_SUFFIX="${FRONTIER_TLR_SMOKE_STEPS:+-smoke${FRONTIER_TLR_SMOKE_STEPS}}"
export BOUNDARY_RUN_NAME="${BOUNDARY_RUN_NAME:-frontier-tlr-k4-seed${SEED}${RUN_SUFFIX}}"

echo "Frontier-TLR: mixed-only gating, wrong-only candidates, TLR score selector, K=4, seed=$SEED"
echo "  Selector:  Score_LH = (1 - L̂) * (1 - Ĥ), normalised over wrong candidates per prompt"
echo "  Teacher:   $TEACHER_MODEL_PATH"
echo "  Student:   $STUDENT_MODEL_PATH"
echo "  Run:       $BOUNDARY_RUN_NAME"
echo "  Smoke:     ${FRONTIER_TLR_SMOKE_STEPS:-off}"
echo ""
echo "Comparison table (all methods share identical gating and query budget):"
echo "  Frontier-Random   mixed-only  wrong-only  random        1/mixed"
echo "  Frontier-Shortest mixed-only  wrong-only  shortest      1/mixed"
echo "  Frontier-TLR      mixed-only  wrong-only  TLR score     1/mixed  <-- this run"
echo "  SR-OPD           mixed-only  wrong-only  SR score     1/mixed"

# Frontier-TLR does not use Boundary hidden-state capture, so no calibration
# artifact is needed.  Route directly through sr-opd.sh (bypassing the calibration
# check in _run_sr_opd.sh) using the same environment already set above.
OPD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
exec bash "$OPD_ROOT/bash/train/sr-opd.sh" \
    --selector-mode frontier_tlr \
    --seed "$SEED" \
    --run-name "$BOUNDARY_RUN_NAME" \
    --method-dir "$BOUNDARY_METHOD_DIR"
}
