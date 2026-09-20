#!/bin/bash
set -euo pipefail

MODE="${1:?selector mode is required}"
NUM_BOUNDARIES="${2:?num boundaries is required}"
SEED="${3:-${SEED:-42}}"

case "$MODE" in
    boundary_opd|shortest_wrong|all_wrong|frontier_tlr) ;;
    *) echo "ERROR: invalid Boundary-OPD selector mode '$MODE'" >&2; exit 2 ;;
esac
case "$NUM_BOUNDARIES" in
    2|4|8|16|32|64) ;;
    *) echo "ERROR: num boundaries must be 2, 4, 8, 16, 32, or 64" >&2; exit 2 ;;
esac
[[ "$SEED" =~ ^[0-9]+$ ]] || {
    echo "ERROR: seed must be non-negative" >&2
    exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
source "$SCRIPT_DIR/base_config.sh"
# Environment model paths take precedence over bash/models.env.
if [[ -z "${ACTOR_MODEL_PATH:-}" || -z "${REWARD_MODEL_PATH:-}" ]]; then
    [[ -f "$OPD_ROOT/bash/models.env" ]] || {
        echo "ERROR: ACTOR_MODEL_PATH/REWARD_MODEL_PATH are unset and bash/models.env does not exist." >&2
        echo "       Copy bash/models.env.example to bash/models.env, or export both variables." >&2
        exit 2
    }
    source "$OPD_ROOT/bash/models.env"
fi
: "${ACTOR_MODEL_PATH:?ACTOR_MODEL_PATH is not set}"
: "${REWARD_MODEL_PATH:?REWARD_MODEL_PATH is not set}"
source "$OPD_ROOT/bash/lib/layout.sh"
TRAIN_DATASET="${TRAIN_DATASET:-$OPD_ROOT/datasets/train/dapo-math-17k.parquet}"
export BOUNDARY_OPD_NUM_BOUNDARIES="$NUM_BOUNDARIES"
export BOUNDARY_OPD_FALLBACK_TO_FF_COST=false

# SR-OPD compares centered direct hidden states; it is the only similarity
# metric the released routing rule uses.
SIMILARITY_METRIC="${BOUNDARY_OPD_SIMILARITY_METRIC:-centered_hidden_state_cosine}"
case "$SIMILARITY_METRIC" in
    centered_hidden_state_cosine) ;;
    *) echo "ERROR: unsupported similarity metric '$SIMILARITY_METRIC' (expected centered_hidden_state_cosine)" >&2; exit 2 ;;
esac
export BOUNDARY_OPD_SIMILARITY_METRIC="$SIMILARITY_METRIC"

# Calibration collection itself must not re-enter the calibration gate.
if [[ "$MODE" != "shortest_wrong" && "${BOUNDARY_OPD_CALIBRATION_COLLECT:-false}" != "true" ]]; then
    boundary_ensure_calibration "$NUM_BOUNDARIES"
    : "${BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH:?calibrated hidden metric requires BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH}"
    : "${BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256:?calibrated hidden metric requires BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256}"
    ACTUAL_MODEL_HASH="$(PYTHONPATH="$OPD_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}" python3 -m verl.utils.boundary_calibration hash "$ACTOR_MODEL_PATH")"
    ACTUAL_DATA_HASH="$(PYTHONPATH="$OPD_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}" python3 -m verl.utils.boundary_calibration hash "$TRAIN_DATASET")"
    if [[ -n "${BOUNDARY_OPD_CALIBRATION_MODEL_HASH:-}" && "$BOUNDARY_OPD_CALIBRATION_MODEL_HASH" != "$ACTUAL_MODEL_HASH" ]]; then
        echo "ERROR: configured calibration model hash does not match ACTOR_MODEL_PATH" >&2
        exit 2
    fi
    if [[ -n "${BOUNDARY_OPD_CALIBRATION_DATA_HASH:-}" && "$BOUNDARY_OPD_CALIBRATION_DATA_HASH" != "$ACTUAL_DATA_HASH" ]]; then
        echo "ERROR: configured calibration data hash does not match TRAIN_DATASET" >&2
        exit 2
    fi
    export BOUNDARY_OPD_CALIBRATION_MODEL_HASH="$ACTUAL_MODEL_HASH"
    export BOUNDARY_OPD_CALIBRATION_DATA_HASH="$ACTUAL_DATA_HASH"
    VALIDATE_ARGS=(
        validate
        --manifest "$BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH"
        --manifest-sha256 "$BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256"
        --model-hash "$BOUNDARY_OPD_CALIBRATION_MODEL_HASH"
        --data-hash "$BOUNDARY_OPD_CALIBRATION_DATA_HASH"
        --num-boundaries "$NUM_BOUNDARIES"
        --whitening-regularization "${BOUNDARY_OPD_WHITENING_REGULARIZATION:-1.0e-4}"
    )
    [[ "$SIMILARITY_METRIC" != "whitened_hidden_cosine" && "$SIMILARITY_METRIC" != "whitened_lm_head_cosine" ]] || VALIDATE_ARGS+=(--require-cholesky)
    [[ "$SIMILARITY_METRIC" != "centered_hidden_state_cosine" ]] || VALIDATE_ARGS+=(--representation-domain direct_hidden_state)
    PYTHONPATH="$OPD_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}" \
        python3 -m verl.utils.boundary_calibration "${VALIDATE_ARGS[@]}"
fi

TIMESTAMP="${BOUNDARY_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_NAME="${BOUNDARY_RUN_NAME:-boundary-contrast-full-m${NUM_BOUNDARIES}-seed${SEED}-${TIMESTAMP}}"

MODEL_PAIR="$(opd_model_pair "$(basename "$REWARD_MODEL_PATH")" "$(basename "$ACTOR_MODEL_PATH")")"
OUTPUT_ROOT="${BOUNDARY_OUTPUT_ROOT:-$OPD_ROOT/runs}"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$OPD_ROOT/$OUTPUT_ROOT"
METHOD_DIR="${BOUNDARY_METHOD_DIR:-ff_opd}"
RUN_DIR="$(opd_run_dir "$OUTPUT_ROOT" "$MODEL_PAIR" "$METHOD_DIR" "$RUN_NAME")"
if [[ -e "$RUN_DIR" && "${BOUNDARY_ALLOW_EXISTING_RUN_DIR:-1}" != "1" ]]; then
    echo "ERROR: refusing to overwrite existing Boundary run: $RUN_DIR" >&2
    echo "       set BOUNDARY_ALLOW_EXISTING_RUN_DIR=1 to reuse the directory (resume training)" >&2
    exit 2
fi

echo "==> Launching Boundary training: run_name=$RUN_NAME selector_mode=${BOUNDARY_SELECTOR_SCORE_MODE:-legacy}"
echo "    representation_domain=direct_hidden_state uses_hidden_difference=false M=$NUM_BOUNDARIES K=${FF_K_ROLLOUTS:-4}"

exec bash "$OPD_ROOT/bash/train/sr-opd.sh" \
    --selector-mode "$MODE" \
    --seed "$SEED" \
    --run-name "$RUN_NAME" \
    --method-dir "$METHOD_DIR" \
    --output-root "$OUTPUT_ROOT"
