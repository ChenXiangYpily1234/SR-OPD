#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NUM_BOUNDARIES="${1:-${BOUNDARY_OPD_NUM_BOUNDARIES:-16}}"
SEED="${BOUNDARY_OPD_CALIBRATION_SEED:-42}"
case "$NUM_BOUNDARIES" in
    4|8|16|32) ;;
    *) echo "ERROR: num boundaries must be 4, 8, 16, or 32" >&2; exit 2 ;;
esac
[[ "$SEED" == "42" ]] || {
    echo "ERROR: Boundary calibration seed is frozen at 42" >&2
    exit 2
}
TIMESTAMP="${BOUNDARY_RUN_TIMESTAMP:-$(date +%Y%m%d-%H%M%S)}"
if [[ -z "${ACTOR_MODEL_PATH:-}" || -z "${REWARD_MODEL_PATH:-}" ]]; then
    [[ -f "$OPD_ROOT/bash/models.env" ]] || {
        echo "ERROR: ACTOR_MODEL_PATH/REWARD_MODEL_PATH are unset and bash/models.env does not exist." >&2
        echo "       Copy bash/models.env.example to bash/models.env, or export both variables." >&2
        exit 2
    }
    source "$OPD_ROOT/bash/models.env"
fi
TRAIN_DATASET="${TRAIN_DATASET:-$OPD_ROOT/datasets/train/dapo-math-17k.parquet}"
PYTHONPATH="$OPD_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH

# SR-OPD always calibrates in the direct-hidden-state domain.
export BOUNDARY_OPD_SIMILARITY_METRIC="${BOUNDARY_OPD_SIMILARITY_METRIC:-centered_hidden_state_cosine}"
export BOUNDARY_OPD_CALIBRATION_COLLECT=true
# An SR-OPD smoke limit applies only to the subsequent training run. Calibration
# must still consume its complete frozen prompt subset.
export FF_FRESH_STEP_LIMIT=0
REPRESENTATION_DOMAIN="${BOUNDARY_OPD_CALIBRATION_REPRESENTATION_DOMAIN:-direct_hidden_state}"
[[ "$REPRESENTATION_DOMAIN" == "direct_hidden_state" ]] || {
    echo "ERROR: SR-OPD calibrates direct hidden states only, got: $REPRESENTATION_DOMAIN" >&2
    exit 2
}
export BOUNDARY_SELECTOR_SCORE_MODE=persistent_departure_area
export BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS="${BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS:-64}"
export BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH="${BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH:-$OPD_ROOT/calibration/artifacts/$REPRESENTATION_DOMAIN/$(basename "$ACTOR_MODEL_PATH")-m${NUM_BOUNDARIES}-seed42-p${BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS}/manifest.json}"
export BOUNDARY_OPD_CALIBRATION_MODEL_HASH="${BOUNDARY_OPD_CALIBRATION_MODEL_HASH:-$(python3 -m verl.utils.boundary_calibration hash "$ACTOR_MODEL_PATH")}"
export BOUNDARY_OPD_CALIBRATION_DATA_HASH="${BOUNDARY_OPD_CALIBRATION_DATA_HASH:-$(python3 -m verl.utils.boundary_calibration hash "$TRAIN_DATASET")}"
export BOUNDARY_RUN_NAME="sr-opd-calibration-m${NUM_BOUNDARIES}-seed${SEED}-${TIMESTAMP}"
export BOUNDARY_METHOD_DIR=boundary_calibration
export BOUNDARY_OUTPUT_ROOT="${BOUNDARY_CALIBRATION_RUN_ROOT:-$OPD_ROOT/calibration/runs}"

bash "$SCRIPT_DIR/../common/_run_sr_opd.sh" boundary_opd "$NUM_BOUNDARIES" "$SEED"
[[ -f "$BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH" ]] || {
    echo "ERROR: calibration finished without creating $BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH" >&2
    exit 2
}
echo "Calibration artifact generated: domain=$REPRESENTATION_DOMAIN path=$BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH"
