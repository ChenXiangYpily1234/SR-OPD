#!/bin/bash
# SR-OPD: success-referenced rollout routing, M=16, K=4.
# Set TEACHER_MODEL_PATH and STUDENT_MODEL_PATH for the model pair.
# Usage: bash bash/sr_opd/main/run_sr_opd_m16.sh [seed]
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
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
export FF_K_ROLLOUTS=4
export N_RESPONSES=4
export FF_MAX_NO_SUCCESS_RETRIES=0
export FF_TOTAL_EPOCHS=1
export BOUNDARY_SELECTOR_SCORE_MODE=persistent_departure_area
export BOUNDARY_OPD_NUM_BOUNDARIES=16
export BOUNDARY_OPD_SIMILARITY_METRIC=centered_hidden_state_cosine
export BOUNDARY_OPD_FALLBACK_TO_FF_COST=false
export BOUNDARY_METHOD_DIR="${BOUNDARY_METHOD_DIR:-sr_opd}"
if [[ -n "${SR_OPD_SMOKE_STEPS:-}" ]]; then
    [[ "$SR_OPD_SMOKE_STEPS" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: SR_OPD_SMOKE_STEPS must be a positive integer" >&2
        exit 2
    }
    export FF_FRESH_STEP_LIMIT="$SR_OPD_SMOKE_STEPS"
fi
RUN_SUFFIX="${SR_OPD_SMOKE_STEPS:+-smoke${SR_OPD_SMOKE_STEPS}}"
export BOUNDARY_RUN_NAME="${BOUNDARY_RUN_NAME:-sr-opd-m16-seed${SEED}${RUN_SUFFIX}}"

echo "SR-OPD [M ablation]: direct hidden states, uses_hidden_difference=false, M=16, K=4, seed=$SEED"
echo "SR-OPD invariant: extra_student_forward=0 (hidden states reuse the existing Student log-prob forward)"
echo "  Teacher: $TEACHER_MODEL_PATH"
echo "  Student: $STUDENT_MODEL_PATH"
echo "  Run:     $BOUNDARY_RUN_NAME"
echo "  Smoke:   ${SR_OPD_SMOKE_STEPS:-off}"
exec bash "$SCRIPT_DIR/../common/_run_sr_opd.sh" boundary_opd 16 "$SEED"
}
