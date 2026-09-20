#!/bin/bash
#SBATCH --job-name=ff-opd
#SBATCH --output=slurm-train-ff-opd-%j.out
#SBATCH --error=slurm-train-ff-opd-%j.err
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# SR-OPD and matched rollout selectors: shared training entry
# point. Selector modes share the full training path and query exactly the
# current step's Frontier prompts.

# Read the complete script before starting a long training job.
{
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash bash/train/sr-opd.sh [options]

  --seed N                Routing and training seed (default: 42)
  --run-name NAME         Run identifier
  --selector-mode MODE    FF selector mode (default: boundary_opd)
  --method-dir NAME       Output method directory (default: ff_opd)
  --model-pair NAME       Output model-pair directory
  --output-root PATH      Run root (default: runs)
  --mini-batch-size N     Rollout prompt batch size
  --ppo-max-token-len N   Max tokens per GPU per Actor update micro-batch
  --max-no-success-retries N
                          NO_SUCCESS retry rounds: 0, 1, or 2 (default: 0)
  --profile-audit-rate R  Share of Frontier attempts whose full confidence
                          profiles are stored in metrics/ff_profiles.jsonl
                          (default: 0.02)
  --cost-aware on|off     Backward-compatible FF selector switch. If
                          --selector-mode is omitted, off selects nearest_only;
                          an explicit --selector-mode always takes precedence.

FF-OPD defaults to K=4 (override with the FF_K_ROLLOUTS environment variable,
e.g. FF_K_ROLLOUTS=8) and to one fresh dataset epoch (override the number of
passes over the RL prompt set with FF_TOTAL_EPOCHS, e.g. FF_TOTAL_EPOCHS=2; the
fixed FF-OPD base_epochs=1 recipe knob is intentionally left untouched) with no
NO_SUCCESS retries. Retries remain available only as an explicit ablation.
Select modes share the full training path and query exactly the current step's
Frontier prompts.
EOF
}

BASE_EPOCHS=1
MAX_NO_SUCCESS_RETRIES=0
LEGACY_TEACHER_QUERY_RATIO=0.25
K_ROLLOUTS="${FF_K_ROLLOUTS:-4}"
SEED=42
COST_AWARE=on
SELECTOR_MODE="${FF_SELECTOR_MODE:-boundary_opd}"
SELECTOR_MODE_EXPLICIT=0
METHOD_DIR="ff_opd"
RUN_NAME=""
MODEL_PAIR=""
OUTPUT_ROOT=""
MINI_BATCH_SIZE=""
PPO_MAX_TOKEN_LEN=""
PROFILE_AUDIT_SAMPLE_RATE=0.02

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed) SEED="${2:?missing --seed}"; shift 2 ;;
        --selector-mode)
            SELECTOR_MODE="${2:?missing --selector-mode}"
            SELECTOR_MODE_EXPLICIT=1
            shift 2
            ;;
        --method-dir) METHOD_DIR="${2:?missing --method-dir}"; shift 2 ;;
        --cost-aware) COST_AWARE="${2:?missing --cost-aware}"; shift 2 ;;
        --profile-audit-rate)
            PROFILE_AUDIT_SAMPLE_RATE="${2:?missing --profile-audit-rate}"; shift 2 ;;
        --max-no-success-retries)
            MAX_NO_SUCCESS_RETRIES="${2:?missing --max-no-success-retries}"; shift 2 ;;
        --run-name) RUN_NAME="${2:?missing --run-name}"; shift 2 ;;
        --model-pair) MODEL_PAIR="${2:?missing --model-pair}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing --output-root}"; shift 2 ;;
        --mini-batch-size) MINI_BATCH_SIZE="${2:?missing --mini-batch-size}"; shift 2 ;;
        --ppo-max-token-len) PPO_MAX_TOKEN_LEN="${2:?missing --ppo-max-token-len}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "$SEED" =~ ^[0-9]+$ ]] || {
    echo "ERROR: seed must be non-negative" >&2
    exit 2
}

[[ "$K_ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: FF_K_ROLLOUTS must be a positive integer" >&2
    exit 2
}

[[ "$PROFILE_AUDIT_SAMPLE_RATE" =~ ^(0|1)(\.[0-9]+)?$ ]] || {
    echo "ERROR: --profile-audit-rate must be within [0, 1]" >&2
    exit 2
}

[[ "$MAX_NO_SUCCESS_RETRIES" =~ ^[012]$ ]] || {
    echo "ERROR: --max-no-success-retries must be 0, 1, or 2" >&2
    exit 2
}

case "$COST_AWARE" in
    on|off) ;;
    *) echo "ERROR: --cost-aware must be 'on' or 'off'" >&2; exit 2 ;;
esac
# Preserve the historical nearest-only entry point while making the new
# selector mode authoritative whenever it is explicitly provided.
if [[ "$COST_AWARE" == "off" && "$SELECTOR_MODE_EXPLICIT" -eq 0 ]]; then
    SELECTOR_MODE=nearest_only
fi
case "$SELECTOR_MODE" in
    global_random|random_wrong|all_wrong|random_correct|nearest_only|cost_only|farthest_only|boundary_opd|persistent_drop|interior_nearest_failure|most_divergent_failure|shortest_wrong|bc_0p4n_random|frontier_tlr) ;;
    *) echo "ERROR: invalid --selector-mode '$SELECTOR_MODE'" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib/layout.sh"
if [[ -z "$RUN_NAME" ]]; then
    COST_TAG="cost"
    [[ "$COST_AWARE" == "off" ]] && COST_TAG="nc"
    RUN_NAME="ff-${COST_TAG}-k${K_ROLLOUTS}-q${LEGACY_TEACHER_QUERY_RATIO}-retry${MAX_NO_SUCCESS_RETRIES}-s${SEED}-$(opd_timestamp)"
fi

export FF_OPD_ENABLE=True
export FF_BASE_EPOCHS="$BASE_EPOCHS"
export FF_K_ROLLOUTS="$K_ROLLOUTS"
export FF_FRONTIER_ONLY=True
export FF_LEGACY_TEACHER_QUERY_RATIO="$LEGACY_TEACHER_QUERY_RATIO"
export FF_MAX_NO_SUCCESS_RETRIES="$MAX_NO_SUCCESS_RETRIES"
export FF_SEED="$SEED"
export FF_SELECTOR_MODE="$SELECTOR_MODE"
export BOUNDARY_OPD_NUM_BOUNDARIES="${BOUNDARY_OPD_NUM_BOUNDARIES:-16}"
export BOUNDARY_OPD_HIDDEN_LAYER="${BOUNDARY_OPD_HIDDEN_LAYER:--1}"
export BOUNDARY_OPD_CAPTURE_LOCATION="${BOUNDARY_OPD_CAPTURE_LOCATION:-pre_lm_head}"
export BOUNDARY_OPD_CAPTURE_METHOD="${BOUNDARY_OPD_CAPTURE_METHOD:-auto}"
export BOUNDARY_OPD_HIDDEN_STORAGE_DTYPE="${BOUNDARY_OPD_HIDDEN_STORAGE_DTYPE:-float16}"
export BOUNDARY_OPD_DISTANCE_COMPUTE_DTYPE="${BOUNDARY_OPD_DISTANCE_COMPUTE_DTYPE:-float32}"
export BOUNDARY_OPD_HIDDEN_NORM_EPSILON="${BOUNDARY_OPD_HIDDEN_NORM_EPSILON:-1.0e-6}"
export BOUNDARY_OPD_SIMILARITY_METRIC="${BOUNDARY_OPD_SIMILARITY_METRIC:-raw_hidden_cosine}"
export BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH="${BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH:-}"
export BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256="${BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256:-}"
export BOUNDARY_OPD_CALIBRATION_MODEL_HASH="${BOUNDARY_OPD_CALIBRATION_MODEL_HASH:-}"
export BOUNDARY_OPD_CALIBRATION_DATA_HASH="${BOUNDARY_OPD_CALIBRATION_DATA_HASH:-}"
export BOUNDARY_OPD_CALIBRATION_COLLECT="${BOUNDARY_OPD_CALIBRATION_COLLECT:-false}"
export BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS="${BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS:-64}"
export BOUNDARY_OPD_WHITENING_REGULARIZATION="${BOUNDARY_OPD_WHITENING_REGULARIZATION:-1.0e-4}"
export BOUNDARY_OPD_LM_HEAD_VOCAB_CHUNK_SIZE="${BOUNDARY_OPD_LM_HEAD_VOCAB_CHUNK_SIZE:-2048}"
export BOUNDARY_OPD_SINKHORN_EPSILON="${BOUNDARY_OPD_SINKHORN_EPSILON:-0.05}"
export BOUNDARY_OPD_SINKHORN_MAX_ITERATIONS="${BOUNDARY_OPD_SINKHORN_MAX_ITERATIONS:-2000}"
export BOUNDARY_OPD_SINKHORN_TOLERANCE="${BOUNDARY_OPD_SINKHORN_TOLERANCE:-1.0e-5}"
export BOUNDARY_OPD_DTW_GAP_PENALTY="${BOUNDARY_OPD_DTW_GAP_PENALTY:-0.0}"
export BOUNDARY_SELECTOR_SCORE_MODE="${BOUNDARY_SELECTOR_SCORE_MODE:-legacy}"
export BOUNDARY_SELECTOR_WINDOW_RATIO="${BOUNDARY_SELECTOR_WINDOW_RATIO:-0.125}"
export BOUNDARY_OPD_DETACH_HIDDEN="${BOUNDARY_OPD_DETACH_HIDDEN:-true}"
export BOUNDARY_OPD_FALLBACK_TO_FF_COST="${BOUNDARY_OPD_FALLBACK_TO_FF_COST:-false}"
export BOUNDARY_OPD_DEBUG_STORE_PAIRWISE="${BOUNDARY_OPD_DEBUG_STORE_PAIRWISE:-false}"
export BOUNDARY_OPD_SM_PDA_COMPUTE_ORIGINAL_PDA_DIAGNOSTICS="${BOUNDARY_OPD_SM_PDA_COMPUTE_ORIGINAL_PDA_DIAGNOSTICS:-true}"
export BOUNDARY_OPD_REPRESENTATION_DYNAMICS_M64="${BOUNDARY_OPD_REPRESENTATION_DYNAMICS_M64:-false}"
export BOUNDARY_OPD_L31_DISTANCE_MIN="${BOUNDARY_OPD_L31_DISTANCE_MIN:-0.05}"
export BOUNDARY_OPD_L31_DISTANCE_MAX="${BOUNDARY_OPD_L31_DISTANCE_MAX:-0.60}"
export BOUNDARY_OPD_L31_REACHABILITY_THRESHOLD="${BOUNDARY_OPD_L31_REACHABILITY_THRESHOLD:-0.50}"
export BOUNDARY_OPD_L31_FORK_THRESHOLD="${BOUNDARY_OPD_L31_FORK_THRESHOLD:-0.35}"
export BOUNDARY_OPD_L31_FORK_PERSISTENCE="${BOUNDARY_OPD_L31_FORK_PERSISTENCE:-2}"
export BOUNDARY_OPD_L31_BRANCH_TOKENS="${BOUNDARY_OPD_L31_BRANCH_TOKENS:-32}"
export BOUNDARY_OPD_L31_TOP_K="${BOUNDARY_OPD_L31_TOP_K:-32}"
export BOUNDARY_OPD_L31_DTW_BAND_RATIO="${BOUNDARY_OPD_L31_DTW_BAND_RATIO:-0.20}"
if [[ "$COST_AWARE" == "on" ]]; then
    export FF_SELECTOR_COST_AWARE=True
else
    export FF_SELECTOR_COST_AWARE=False
fi
export FF_CSV_FLUSH_INTERVAL=1
export FF_PROFILE_AUDIT_SAMPLE_RATE="$PROFILE_AUDIT_SAMPLE_RATE"
export FF_DEBUG_ASSERTIONS=True
export FF_SAVE_QUEUE_STATE=True
# base_epochs is a fixed FF-OPD recipe knob (must stay 1). The number of passes
# over the RL prompt dataset is trainer.total_epochs, decoupled here so it can be
# raised (e.g. FF_TOTAL_EPOCHS=2) without touching the fixed base_epochs=1.
export TRAIN_TOTAL_EPOCHS="${FF_TOTAL_EPOCHS:-$BASE_EPOCHS}"
export OPD_TARGET_MODE=sampled_token
[[ -z "$PPO_MAX_TOKEN_LEN" ]] || export PPO_MAX_TOKEN_LEN_PER_GPU="$PPO_MAX_TOKEN_LEN"

# Fixed Full-OPD Teacher query path parameters for the shared training body.
QUERY_METHOD=full
QUERY_RATIO=1.0
EXPLORATION_RATE=0.0
POST_METHOD=full
TOKEN_RETAIN_RATIO=1.0
export N_RESPONSES="$K_ROLLOUTS"
export TRAINING_SEED="$SEED"

echo "==> Launching Boundary training: run_name=$RUN_NAME selector_mode=${BOUNDARY_SELECTOR_SCORE_MODE:-legacy}"
if [[ "$QUERY_METHOD" != "full" ]]; then
    echo "ERROR: --query-method must be full" >&2; exit 2
fi
: "${QUERY_RATIO:=0.20}"
[[ "$POST_METHOD" == "full" || "$POST_METHOD" == "ta" ]] || {
    echo "ERROR: --post-method must be full or ta" >&2; exit 2;
}
if [[ "$POST_METHOD" == "ta" ]]; then
    USE_TA_MASK=true
else
    USE_TA_MASK=false
    TOKEN_RETAIN_RATIO=1.0
fi
if [[ -n "$MODEL_PAIR" && ( "$MODEL_PAIR" == */* || "$MODEL_PAIR" == "." || "$MODEL_PAIR" == ".." ) ]]; then
    echo "ERROR: --model-pair must be a single directory name" >&2
    exit 2
fi
if [[ -n "$METHOD_DIR" && ( "$METHOD_DIR" == */* || "$METHOD_DIR" == "." || "$METHOD_DIR" == ".." ) ]]; then
    echo "ERROR: --method-dir must be a single directory name" >&2
    exit 2
fi
QUERY_RATIO=1.0
EXPLORATION_RATE=0.0
export QUERY_METHOD QUERY_RATIO EXPLORATION_RATE POST_METHOD USE_TA_MASK TOKEN_RETAIN_RATIO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../verl" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$OPD_ROOT/bash/lib/layout.sh"
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
export ACTOR_MODEL_NAME="$(basename "$ACTOR_MODEL_PATH")"
export REWARD_MODEL_NAME="$(basename "$REWARD_MODEL_PATH")"
if [[ -z "$MODEL_PAIR" ]]; then
    MODEL_PAIR="$(opd_model_pair "$REWARD_MODEL_NAME" "$ACTOR_MODEL_NAME")"
fi
if [[ -z "$METHOD_DIR" ]]; then
    METHOD_DIR="$(opd_method_dir "$QUERY_METHOD" "$POST_METHOD" "${EXPERIMENT_METHOD:-$QUERY_METHOD}")"
fi

# Top-k values are auxiliary inputs for TA selection. TLR keeps its published
# per-trajectory OPSD objective; the remaining methods share the sampled-token
# Actor objective.
if [[ "$USE_TA_MASK" == "true" ]]; then
    export TA_OPD_ENABLE=True
else
    export TA_OPD_ENABLE=False
fi
export TA_OPD_MODE=${TA_OPD_MODE:-teachability}
export TA_OPD_TOPK=${TA_OPD_TOPK:-16}
export TA_SCORE_TOP_K_STRATEGY=${TA_SCORE_TOP_K_STRATEGY:-union}
export TA_OPD_RETAIN_RATIO="$TOKEN_RETAIN_RATIO"
export TA_OPD_RETAIN_TOKEN_COUNT=${TA_OPD_RETAIN_TOKEN_COUNT:-""}
export TA_OPD_NORMALIZE_SCOPE=${TA_OPD_NORMALIZE_SCOPE:-batch}
export TA_OPD_NORMALIZE_BY=${TA_OPD_NORMALIZE_BY:-selected_tokens}
export TA_OPD_DETACH_SCORE=${TA_OPD_DETACH_SCORE:-True}
export TA_OPD_LOG_STATS=${TA_OPD_LOG_STATS:-True}
# Teacher rewards use sampled-token log probabilities.
export OPD_TARGET_MODE=sampled_token
export STUDENT_ENABLE_THINKING=False


# Run outputs
export EXPERIMENT_METHOD="${EXPERIMENT_METHOD:-$QUERY_METHOD}"
export MODE_TAG="${MODE_TAG_OVERRIDE:-$(opd_mode_tag "$QUERY_RATIO" "$EXPLORATION_RATE" "$POST_METHOD" "$TOKEN_RETAIN_RATIO")}"

if [ -n "${RUN_NAME:-}" ]; then
    export RUN_ID="$RUN_NAME"
else
    export RUN_ID="$(opd_training_run_name "$METHOD_DIR" "$QUERY_RATIO" \
        "$EXPLORATION_RATE" "$TOKEN_RETAIN_RATIO" "$TRAINING_SEED" "$(opd_timestamp)")"
fi
if [[ "$RUN_ID" == */* || "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
    echo "ERROR: --run-name must be a single directory name" >&2
    exit 2
fi

export OUTPUT_ROOT_CONFIG="${OUTPUT_ROOT:-runs}"
if [[ "$OUTPUT_ROOT_CONFIG" = /* ]]; then
    export OUTPUT_ROOT="$OUTPUT_ROOT_CONFIG"
else
    export OUTPUT_ROOT="$OPD_ROOT/$OUTPUT_ROOT_CONFIG"
fi
export METHOD_DIR
export RUN_DIR="$(opd_run_dir "$OUTPUT_ROOT" "$MODEL_PAIR" "$METHOD_DIR" "$RUN_ID")"
export RUN_LOG_DIR="$RUN_DIR/logs"
export RUN_METADATA_DIR="$RUN_DIR/metadata"
export RUN_EVAL_MODELS_DIR="$RUN_DIR/eval_models"
export RUN_EVAL_RESULTS_DIR="$RUN_DIR/eval_results"
export RUN_COST_ANALYSIS_DIR="$RUN_DIR/cost_analysis"
export RUN_ARTIFACTS_DIR="$RUN_DIR/artifacts"
export RUN_METRICS_DIR="$RUN_DIR/metrics"
mkdir -p "$RUN_LOG_DIR" "$RUN_METADATA_DIR" "$RUN_EVAL_MODELS_DIR" \
    "$RUN_EVAL_RESULTS_DIR" "$RUN_COST_ANALYSIS_DIR" "$RUN_METRICS_DIR"
export LOG_FILE="$RUN_LOG_DIR/train.log"
exec > >(tee -a "$LOG_FILE") 2>&1

set -x
echo "OPD mode: $EXPERIMENT_METHOD (Teacher query path: $QUERY_METHOD)"
echo "Run directory: $RUN_DIR"
echo "Log file: $LOG_FILE"
echo "Start time: $(date)"

ray stop --force
export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1
export PROJECT_NAME='OnPolicyDistillation'
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTORCH_ALLOC_CONF=expandable_segments:True

export ADV_ESTIMATOR=token_reward_direct
export GRPO_OUTCOME_WEIGHT=1.0

export MAX_PROMPT_LENGTH=2048
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-8192}
export MAX_MODEL_LEN=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH ))

export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-64}
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-2}
export REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
export REWARD_MICRO_BATCH_SIZE_PER_GPU=${REWARD_MICRO_BATCH_SIZE_PER_GPU:-8}
# Per-GPU token budget for log-probability computation.
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
# FSDP parameter, optimizer, and activation offloading.
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
export ACTOR_ACTIVATION_OFFLOAD=${ACTOR_ACTIVATION_OFFLOAD:-True}
export REF_PARAM_OFFLOAD=${REF_PARAM_OFFLOAD:-True}
# Activation checkpointing and optional entropy/log-probability collection.
export ACTOR_GRADIENT_CHECKPOINTING=${ACTOR_GRADIENT_CHECKPOINTING:-True}
export TEACHER_COMPUTE_ENTROPY=${TEACHER_COMPUTE_ENTROPY:-True}
export ROLLOUT_CALCULATE_LOG_PROBS=${ROLLOUT_CALCULATE_LOG_PROBS:-True}
export TEACHER_MAX_TOKEN_LEN_PER_GPU=${TEACHER_MAX_TOKEN_LEN_PER_GPU:-32768}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.70}
export TEMPERATURE=${TEMPERATURE:-1.0}
export TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
export REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
export N_RESPONSES=${N_RESPONSES:-4}
export TRAINING_SEED
# Checkpoint frequency and per-epoch saving.
readonly FINAL_ONLY_SAVE_FREQ=1000000000

if [[ "${FF_OPD_ENABLE:-False}" == "True" ]]; then
    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
else
    export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
fi
# Selector support is configured independently below for TA. TLR uses K=0
# with its per-trajectory reduction.
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-"none"}

export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
export IS_PLOT=${IS_PLOT:-False}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-"token-mean"}

export COST_METRICS_ENABLE=${COST_METRICS_ENABLE:-True}
export COST_METRICS_TRACK_GPU_MEMORY=${COST_METRICS_TRACK_GPU_MEMORY:-True}
export COST_METRICS_TRACK_TEACHER_CALL=${COST_METRICS_TRACK_TEACHER_CALL:-True}

if [[ ! "$LOG_PROB_TOP_K" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LOG_PROB_TOP_K must be a non-negative integer" >&2
    exit 2
fi
if [[ "${FF_OPD_ENABLE:-False}" != "True" ]] && (( LOG_PROB_TOP_K != 0 )); then
    echo "ERROR: unified sampled-token OPD backbone requires LOG_PROB_TOP_K=0" >&2
    exit 2
fi
if [[ "${FF_OPD_ENABLE:-False}" != "True" && "$TOP_K_STRATEGY" != "only_stu" ]]; then
    echo "ERROR: unified sampled-token OPD loss requires TOP_K_STRATEGY=only_stu" >&2
    exit 2
fi

if [[ "$TA_OPD_ENABLE" == "True" ]]; then
    [[ "$TA_OPD_NORMALIZE_SCOPE" == "batch" ]] || {
        echo "ERROR: TA-OPD paper mode requires TA_OPD_NORMALIZE_SCOPE=batch" >&2
        exit 2
    }
    [[ "$TA_OPD_NORMALIZE_BY" == "selected_tokens" ]] || {
        echo "ERROR: TA-OPD paper mode requires TA_OPD_NORMALIZE_BY=selected_tokens" >&2
        exit 2
    }
    [[ "$TA_OPD_MODE" != "random" ]] || {
        echo "ERROR: TA-OPD paper mode requires a deterministic selector" >&2
        exit 2
    }
    (( TA_OPD_TOPK > 0 )) || {
        echo "ERROR: TA-OPD selector requires TA_OPD_TOPK > 0" >&2
        exit 2
    }
    [[ "$TA_SCORE_TOP_K_STRATEGY" == "union" ]] || {
        echo "ERROR: TA-OPD teachability selector requires TA_SCORE_TOP_K_STRATEGY=union" >&2
        exit 2
    }
fi

# Model / dataset paths
export TRAIN_DATASET=${TRAIN_DATASET:-$OPD_ROOT/datasets/train/dapo-math-17k.parquet}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k}

export PROJECT_PATH="$RUN_DIR"
export PARALLEL_SIZE=${PARALLEL_SIZE:-1}
export TRAIN_NNODES=${TRAIN_NNODES:-1}
[[ "$TRAIN_NNODES" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: TRAIN_NNODES must be a positive integer" >&2
    exit 2
}
# Auto-detect GPU count when not explicitly set.
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
    _DETECTED_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$_DETECTED_GPUS" =~ ^[1-9][0-9]*$ ]] || _DETECTED_GPUS=1
    export GPUS_PER_NODE="$_DETECTED_GPUS"
    unset _DETECTED_GPUS
fi
export CKPT_PATH="$RUN_DIR/checkpoints"

export OUTLINES_CACHE_DIR=~/.cache/outlines/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)_$$}
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=true
export SWANLAB_LOG_DIR="$RUN_DIR/swanlab"
export HYDRA_FULL_ERROR=1
export EXPERIMENT_NAME="$RUN_ID"
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"

# Resolve token budgets before recording the configuration.
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( MAX_MODEL_LEN > 16384 ? MAX_MODEL_LEN : 16384 ))}
(( PPO_MAX_TOKEN_LEN_PER_GPU >= MAX_MODEL_LEN )) || {
    echo "ERROR: PPO_MAX_TOKEN_LEN_PER_GPU=$PPO_MAX_TOKEN_LEN_PER_GPU must be >= MAX_MODEL_LEN=$MAX_MODEL_LEN when chunked prefill is enabled" >&2
    exit 2
}
# vLLM prefill batching budget; decoupled from the training-side token budget so
# rollout prefill can batch more aggressively without growing training logits.
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$PPO_MAX_TOKEN_LEN_PER_GPU}
(( ROLLOUT_MAX_NUM_BATCHED_TOKENS >= MAX_MODEL_LEN )) || {
    echo "ERROR: ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS must be >= MAX_MODEL_LEN=$MAX_MODEL_LEN when chunked prefill is enabled" >&2
    exit 2
}

cat > "$RUN_METADATA_DIR/run.env" <<EOF
QUERY_METHOD=$QUERY_METHOD
METHOD=$EXPERIMENT_METHOD
MODE_TAG=$MODE_TAG
RUN_ID=$RUN_ID
MODEL_PAIR=$MODEL_PAIR
METHOD_DIR=$METHOD_DIR
RUN_DIR=$RUN_DIR
OUTPUT_ROOT=$OUTPUT_ROOT_CONFIG
ACTOR_MODEL_PATH=$ACTOR_MODEL_PATH
REWARD_MODEL_PATH=$REWARD_MODEL_PATH
TRAIN_DATASET=$TRAIN_DATASET
MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH
MAX_RESP_LENGTH=$MAX_RESP_LENGTH
MINI_BATCH_SIZE=$MINI_BATCH_SIZE
N_RESPONSES=$N_RESPONSES
ACTOR_MICRO_BATCH_SIZE_PER_GPU=$ACTOR_MICRO_BATCH_SIZE_PER_GPU
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU
REWARD_MICRO_BATCH_SIZE_PER_GPU=$REWARD_MICRO_BATCH_SIZE_PER_GPU
ROLLOUT_GPU_MEM_UTIL=$ROLLOUT_GPU_MEM_UTIL
ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU
ACTOR_PARAM_OFFLOAD=$ACTOR_PARAM_OFFLOAD
ACTOR_OPTIMIZER_OFFLOAD=$ACTOR_OPTIMIZER_OFFLOAD
ACTOR_ACTIVATION_OFFLOAD=$ACTOR_ACTIVATION_OFFLOAD
REF_PARAM_OFFLOAD=$REF_PARAM_OFFLOAD
ACTOR_GRADIENT_CHECKPOINTING=$ACTOR_GRADIENT_CHECKPOINTING
TEACHER_COMPUTE_ENTROPY=$TEACHER_COMPUTE_ENTROPY
ROLLOUT_CALCULATE_LOG_PROBS=$ROLLOUT_CALCULATE_LOG_PROBS
TEACHER_MAX_TOKEN_LEN_PER_GPU=$TEACHER_MAX_TOKEN_LEN_PER_GPU
TRAINING_SEED=$TRAINING_SEED
CHECKPOINT_POLICY=last_step_only
VALIDATION=disabled
LOG_PROB_TOP_K=$LOG_PROB_TOP_K
TOP_K_STRATEGY=$TOP_K_STRATEGY
TA_SCORE_TOP_K_STRATEGY=$TA_SCORE_TOP_K_STRATEGY
QUERY_RATIO=$QUERY_RATIO
EXPLORATION_RATE=$EXPLORATION_RATE
USE_TA_MASK=$USE_TA_MASK
POST_METHOD=$POST_METHOD
TOKEN_RETAIN_RATIO=$TOKEN_RETAIN_RATIO
TA_OPD_ENABLE=$TA_OPD_ENABLE
TA_OPD_MODE=$TA_OPD_MODE
TA_OPD_RETAIN_RATIO=$TA_OPD_RETAIN_RATIO
TA_OPD_NORMALIZE_SCOPE=$TA_OPD_NORMALIZE_SCOPE
TA_OPD_NORMALIZE_BY=$TA_OPD_NORMALIZE_BY
OPD_TARGET_MODE=$OPD_TARGET_MODE
STUDENT_ENABLE_THINKING=$STUDENT_ENABLE_THINKING
TLR_OPD_ENABLE=${TLR_OPD_ENABLE:-False}
TLR_K_ROLLOUTS=${TLR_K_ROLLOUTS:-4}
FF_MAX_NO_SUCCESS_RETRIES=${FF_MAX_NO_SUCCESS_RETRIES:-0}
BOUNDARY_OPD_SIMILARITY_METRIC=${BOUNDARY_OPD_SIMILARITY_METRIC:-raw_hidden_cosine}
GIT_COMMIT=$(git -C "$OPD_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
SLURM_JOB_ID=${SLURM_JOB_ID:-local}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}
GPUS_PER_NODE=$GPUS_PER_NODE
TRAIN_NNODES=$TRAIN_NNODES
PARALLEL_SIZE=$PARALLEL_SIZE
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd ',' - || echo unavailable)
EOF

echo "PPO_MAX_TOKEN_LEN_PER_GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "effective_max_response_length=$MAX_RESP_LENGTH"
echo "effective_token_retain_ratio=$TA_OPD_RETAIN_RATIO"
echo "effective_log_prob_top_k=$LOG_PROB_TOP_K"
echo "effective_student_enable_thinking=$STUDENT_ENABLE_THINKING"
echo "effective_ff_max_no_success_retries=${FF_MAX_NO_SUCCESS_RETRIES:-0}"

if [[ "$TA_OPD_ENABLE" == "True" ]]; then
    TA_OPD_ARGS="+actor_rollout_ref.rollout.ta_opd_enable=True \
    +actor_rollout_ref.rollout.ta_opd_mode=teachability \
    +actor_rollout_ref.rollout.ta_opd_topk=$TA_OPD_TOPK \
    +actor_rollout_ref.rollout.ta_opd_score_top_k_strategy=$TA_SCORE_TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.ta_opd_retain_ratio=$TA_OPD_RETAIN_RATIO \
    +actor_rollout_ref.rollout.ta_opd_normalize_scope=$TA_OPD_NORMALIZE_SCOPE \
    +actor_rollout_ref.rollout.ta_opd_normalize_by=selected_tokens \
    +actor_rollout_ref.rollout.ta_opd_detach_score=True \
    +actor_rollout_ref.rollout.ta_opd_log_stats=$TA_OPD_LOG_STATS"
else
    TA_OPD_ARGS="+actor_rollout_ref.rollout.ta_opd_enable=False"
fi

COST_METRICS_ARGS="+actor_rollout_ref.rollout.cost_metrics_enable=$COST_METRICS_ENABLE \
+actor_rollout_ref.rollout.cost_metrics_track_timing=True \
+actor_rollout_ref.rollout.cost_metrics_track_gpu_memory=$COST_METRICS_TRACK_GPU_MEMORY \
+actor_rollout_ref.rollout.cost_metrics_track_teacher_call=$COST_METRICS_TRACK_TEACHER_CALL \
+actor_rollout_ref.rollout.cost_metrics_warn_once=True"

# FF-OPD is isolated behind its root config switch. All selector modes share
# this same rollout, retry, Teacher, loss, optimizer, and evaluation pipeline.
FF_OPD_ARGS="ff_opd.enable=${FF_OPD_ENABLE:-False} \
algorithm.ff_selector_mode=${FF_SELECTOR_MODE:-boundary_opd} \
algorithm.boundary_opd.num_boundaries=${BOUNDARY_OPD_NUM_BOUNDARIES:-16} \
algorithm.boundary_opd.hidden_layer=${BOUNDARY_OPD_HIDDEN_LAYER:--1} \
algorithm.boundary_opd.capture_location=${BOUNDARY_OPD_CAPTURE_LOCATION:-pre_lm_head} \
algorithm.boundary_opd.capture_method=${BOUNDARY_OPD_CAPTURE_METHOD:-auto} \
algorithm.boundary_opd.hidden_storage_dtype=${BOUNDARY_OPD_HIDDEN_STORAGE_DTYPE:-float16} \
algorithm.boundary_opd.distance_compute_dtype=${BOUNDARY_OPD_DISTANCE_COMPUTE_DTYPE:-float32} \
algorithm.boundary_opd.hidden_norm_epsilon=${BOUNDARY_OPD_HIDDEN_NORM_EPSILON:-1.0e-6} \
algorithm.boundary_opd.similarity_metric=${BOUNDARY_OPD_SIMILARITY_METRIC:-raw_hidden_cosine} \
algorithm.boundary_opd.calibration_artifact_path=${BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH:-\"\"} \
algorithm.boundary_opd.calibration_artifact_sha256=${BOUNDARY_OPD_CALIBRATION_ARTIFACT_SHA256:-\"\"} \
algorithm.boundary_opd.calibration_model_hash=${BOUNDARY_OPD_CALIBRATION_MODEL_HASH:-\"\"} \
algorithm.boundary_opd.calibration_data_hash=${BOUNDARY_OPD_CALIBRATION_DATA_HASH:-\"\"} \
algorithm.boundary_opd.calibration_collect=${BOUNDARY_OPD_CALIBRATION_COLLECT:-false} \
algorithm.boundary_opd.calibration_num_prompts=${BOUNDARY_OPD_CALIBRATION_NUM_PROMPTS:-64} \
algorithm.boundary_opd.whitening_regularization=${BOUNDARY_OPD_WHITENING_REGULARIZATION:-1.0e-4} \
algorithm.boundary_opd.lm_head_vocab_chunk_size=${BOUNDARY_OPD_LM_HEAD_VOCAB_CHUNK_SIZE:-2048} \
algorithm.boundary_opd.sinkhorn_epsilon=${BOUNDARY_OPD_SINKHORN_EPSILON:-0.05} \
algorithm.boundary_opd.sinkhorn_max_iterations=${BOUNDARY_OPD_SINKHORN_MAX_ITERATIONS:-2000} \
algorithm.boundary_opd.sinkhorn_tolerance=${BOUNDARY_OPD_SINKHORN_TOLERANCE:-1.0e-5} \
algorithm.boundary_opd.dtw_gap_penalty=${BOUNDARY_OPD_DTW_GAP_PENALTY:-0.0} \
algorithm.boundary_opd.score_mode=${BOUNDARY_SELECTOR_SCORE_MODE:-legacy} \
algorithm.boundary_opd.window_ratio=${BOUNDARY_SELECTOR_WINDOW_RATIO:-0.125} \
algorithm.boundary_opd.detach_hidden=${BOUNDARY_OPD_DETACH_HIDDEN:-true} \
algorithm.boundary_opd.fallback_to_ff_cost=${BOUNDARY_OPD_FALLBACK_TO_FF_COST:-false} \
algorithm.boundary_opd.debug_store_pairwise=${BOUNDARY_OPD_DEBUG_STORE_PAIRWISE:-false} \
algorithm.boundary_opd.sm_pda_compute_original_pda_diagnostics=${BOUNDARY_OPD_SM_PDA_COMPUTE_ORIGINAL_PDA_DIAGNOSTICS:-true} \
algorithm.boundary_opd.representation_dynamics_m64=${BOUNDARY_OPD_REPRESENTATION_DYNAMICS_M64:-false} \
ff_opd.base_epochs=${FF_BASE_EPOCHS:-1} \
ff_opd.k_rollouts=${FF_K_ROLLOUTS:-4} \
ff_opd.fresh_step_limit=${FF_FRESH_STEP_LIMIT:-0} \
ff_opd.frontier_only=${FF_FRONTIER_ONLY:-true} \
ff_opd.legacy_teacher_query_ratio=${FF_LEGACY_TEACHER_QUERY_RATIO:-0.25} \
ff_opd.max_no_success_retries=${FF_MAX_NO_SUCCESS_RETRIES:-0} \
ff_opd.seed=${FF_SEED:-42} \
ff_opd.selector_cost_aware=${FF_SELECTOR_COST_AWARE:-True} \
ff_opd.selector_cost_alpha=${FF_SELECTOR_COST_ALPHA:-0.5} \
ff_opd.selector_score_eps=${FF_SELECTOR_SCORE_EPS:-1.0e-8} \
ff_opd.csv_path=$RUN_METRICS_DIR/ff.csv \
ff_opd.csv_flush_interval=${FF_CSV_FLUSH_INTERVAL:-1} \
ff_opd.profile_jsonl_path=$RUN_METRICS_DIR/ff_profiles.jsonl \
ff_opd.profile_audit_sample_rate=${FF_PROFILE_AUDIT_SAMPLE_RATE:-0.02} \
ff_opd.log_kl_profiles=${FF_LOG_KL_PROFILES:-False} \
ff_opd.kl_profile_jsonl_path=$RUN_METRICS_DIR/ff_kl_profiles.jsonl \
ff_opd.debug_assertions=${FF_DEBUG_ASSERTIONS:-True} \
ff_opd.save_queue_state=${FF_SAVE_QUEUE_STATE:-True} \
actor_rollout_ref.rollout.opd_target_mode=sampled_token"

TLR_OPD_ARGS="tlr_opd.enabled=${TLR_OPD_ENABLE:-False} \
tlr_opd.rollouts_per_prompt=${TLR_K_ROLLOUTS:-4} \
tlr_opd.csv_path=$RUN_METRICS_DIR/tlr_selection_records.csv \
tlr_opd.seed=$TRAINING_SEED"

# Start Ray on a dedicated node using fixed ports.
purge_stale_ray_state() {
    set +e
    ray stop --force >/dev/null 2>&1
    local port
    for port in "${RAY_HEAD_PORT:-6379}" \
                "${RAY_METRICS_EXPORT_PORT:-18081}" \
                "${RAY_DASHBOARD_AGENT_GRPC_PORT:-18082}" \
                "${RAY_DASHBOARD_AGENT_LISTEN_PORT:-18083}" \
                "${RAY_RUNTIME_ENV_AGENT_PORT:-18084}"; do
        if command -v fuser >/dev/null 2>&1; then
            fuser -k "${port}/tcp" >/dev/null 2>&1
        fi
    done
    # Clean up remaining Ray processes on this dedicated node.
    pkill -9 -f 'raylet|gcs_server|runtime_env_agent|dashboard_agent|plasma_store' >/dev/null 2>&1
    sleep 2
    set -e
}
purge_stale_ray_state

RAY_WORKER_STEP_PIDS=()
cleanup_ray_cluster() {
    local pid
    set +e
    for pid in "${RAY_WORKER_STEP_PIDS[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done
    for pid in "${RAY_WORKER_STEP_PIDS[@]}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray_cluster EXIT

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_HEAD_NODE="$(hostname -s)"
RAY_HEAD_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -n "$RAY_HEAD_IP" ]] || RAY_HEAD_IP="$(hostname -i | awk '{print $1}')"

if (( TRAIN_NNODES > 1 )); then
    [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || {
        echo "ERROR: TRAIN_NNODES=$TRAIN_NNODES requires a multi-node Slurm allocation" >&2
        exit 2
    }
    command -v scontrol >/dev/null 2>&1 || {
        echo "ERROR: scontrol is required for multi-node Ray startup" >&2
        exit 2
    }
    command -v srun >/dev/null 2>&1 || {
        echo "ERROR: srun is required for multi-node Ray startup" >&2
        exit 2
    }
    mapfile -t RAY_SLURM_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    if (( ${#RAY_SLURM_NODES[@]} != TRAIN_NNODES )); then
        echo "ERROR: requested TRAIN_NNODES=$TRAIN_NNODES but Slurm allocated ${#RAY_SLURM_NODES[@]} nodes" >&2
        exit 2
    fi
    EXPECTED_HEAD_NODE="${RAY_SLURM_NODES[0]%%.*}"
    if [[ "${RAY_HEAD_NODE%%.*}" != "$EXPECTED_HEAD_NODE" ]]; then
        echo "ERROR: batch script must run on first allocated node $EXPECTED_HEAD_NODE, got $RAY_HEAD_NODE" >&2
        exit 2
    fi
fi

RAY_CLUSTER_ADDRESS="$RAY_HEAD_IP:$RAY_HEAD_PORT"
ray start --head \
    --node-ip-address="$RAY_HEAD_IP" \
    --port="$RAY_HEAD_PORT" \
    --num-gpus="$GPUS_PER_NODE" \
    --min-worker-port=20000 \
    --max-worker-port=29999 \
    --metrics-export-port="${RAY_METRICS_EXPORT_PORT:-18081}" \
    --dashboard-agent-grpc-port="${RAY_DASHBOARD_AGENT_GRPC_PORT:-18082}" \
    --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-18083}" \
    --runtime-env-agent-port="${RAY_RUNTIME_ENV_AGENT_PORT:-18084}"

if (( TRAIN_NNODES > 1 )); then
    for worker_node in "${RAY_SLURM_NODES[@]:1}"; do
        srun \
            --nodes=1 \
            --ntasks=1 \
            --nodelist="$worker_node" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
            --gres="gpu:$GPUS_PER_NODE" \
            --exclusive \
            env \
                RAY_CLUSTER_ADDRESS="$RAY_CLUSTER_ADDRESS" \
                RAY_WORKER_GPUS="$GPUS_PER_NODE" \
                RAY_MIN_WORKER_PORT=20000 \
                RAY_MAX_WORKER_PORT=29999 \
                RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-18081}" \
                RAY_DASHBOARD_AGENT_GRPC_PORT="${RAY_DASHBOARD_AGENT_GRPC_PORT:-18082}" \
                RAY_DASHBOARD_AGENT_LISTEN_PORT="${RAY_DASHBOARD_AGENT_LISTEN_PORT:-18083}" \
                RAY_RUNTIME_ENV_AGENT_PORT="${RAY_RUNTIME_ENV_AGENT_PORT:-18084}" \
                bash -c '
                    ray stop --force >/dev/null 2>&1 || true
                    exec ray start \
                        --address="$RAY_CLUSTER_ADDRESS" \
                        --num-gpus="$RAY_WORKER_GPUS" \
                        --min-worker-port="$RAY_MIN_WORKER_PORT" \
                        --max-worker-port="$RAY_MAX_WORKER_PORT" \
                        --metrics-export-port="$RAY_METRICS_EXPORT_PORT" \
                        --dashboard-agent-grpc-port="$RAY_DASHBOARD_AGENT_GRPC_PORT" \
                        --dashboard-agent-listen-port="$RAY_DASHBOARD_AGENT_LISTEN_PORT" \
                        --runtime-env-agent-port="$RAY_RUNTIME_ENV_AGENT_PORT" \
                        --block
                ' &
        RAY_WORKER_STEP_PIDS+=("$!")
    done
fi

RAY_READY=0
for _ in $(seq 1 60); do
    RAY_ALIVE_NODES="$(
        RAY_ADDRESS="$RAY_CLUSTER_ADDRESS" python3 -c \
            'import os, ray; ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR"); print(sum(n["Alive"] for n in ray.nodes()))' \
            2>/dev/null || echo 0
    )"
    if [[ "$RAY_ALIVE_NODES" == "$TRAIN_NNODES" ]]; then
        RAY_READY=1
        break
    fi
    sleep 2
done
if (( ! RAY_READY )); then
    echo "ERROR: Ray cluster did not reach $TRAIN_NNODES alive nodes; last count=$RAY_ALIVE_NODES" >&2
    exit 2
fi
export RAY_ADDRESS="$RAY_CLUSTER_ADDRESS"
echo "Ray cluster ready: nodes=$RAY_ALIVE_NODES, GPUs=$((TRAIN_NNODES * GPUS_PER_NODE))"

# Launch training
cd "$VERL_ROOT"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    algorithm.grpo_outcome_weight=$GRPO_OUTCOME_WEIGHT \
    data.shuffle=True \
    data.seed=$TRAINING_SEED \
    data.train_files="$TRAIN_DATASET" \
    data.val_files=null \
    data.train_batch_size=$((${MINI_BATCH_SIZE}*${PARALLEL_SIZE})) \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESP_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=$STUDENT_ENABLE_THINKING \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=$ACTOR_ACTIVATION_OFFLOAD \
    actor_rollout_ref.model.enable_gradient_checkpointing=$ACTOR_GRADIENT_CHECKPOINTING \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.actor.loss_agg_mode=$LOSS_AGG_MODE \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$MODEL_DTYPE \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    +actor_rollout_ref.rollout.seed=$TRAINING_SEED \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$LOG_PROB_MAX_TOKEN_LEN_PER_GPU \
    +actor_rollout_ref.rollout.log_prob_top_k=$LOG_PROB_TOP_K \
    +actor_rollout_ref.rollout.top_k_strategy=$TOP_K_STRATEGY \
    +actor_rollout_ref.rollout.reward_weight_mode=$REWARD_WEIGHT_MODE \
    +actor_rollout_ref.rollout.teacher_temperature=$TEACHER_TEMPERATURE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.n=$N_RESPONSES \
    actor_rollout_ref.rollout.repetition_penalty=$REPETITION_PENALTY \
    actor_rollout_ref.rollout.calculate_log_probs=$ROLLOUT_CALCULATE_LOG_PROBS \
    actor_rollout_ref.rollout.opd_metrics_dir="$RUN_METRICS_DIR" \
    actor_rollout_ref.rollout.opd_metrics_dump_interval=${OPD_METRICS_DUMP_INTERVAL:-10} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype=$MODEL_DTYPE \
    reward_model.micro_batch_size_per_gpu=$REWARD_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.forward_max_token_len_per_gpu=$TEACHER_MAX_TOKEN_LEN_PER_GPU \
    +reward_model.compute_teacher_entropy=$TEACHER_COMPUTE_ENTROPY \
    custom_reward_function.path="$VERL_ROOT/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.log_val_generations=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.seed=$TRAINING_SEED \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$TRAIN_NNODES \
    trainer.save_freq=${TRAIN_SAVE_FREQ:-$FINAL_ONLY_SAVE_FREQ} \
    +trainer.save_only_final=${TRAIN_SAVE_ONLY_FINAL:-True} \
    +trainer.save_every_epoch=${TRAIN_SAVE_EVERY_EPOCH:-False} \
    trainer.test_freq=-1 \
    trainer.total_epochs=${TRAIN_TOTAL_EPOCHS:-1} \
    trainer.default_local_dir="$CKPT_PATH" \
    trainer.is_plot=$IS_PLOT \
    $TA_OPD_ARGS \
    $FF_OPD_ARGS \
    $TLR_OPD_ARGS \
    $COST_METRICS_ARGS

# Keep the framework's global_step_* directory for resume compatibility, while
# exposing the public zero-padded checkpoint name used by paper artifacts.
if [[ "${BOUNDARY_OPD_CALIBRATION_COLLECT:-false}" == "true" ]]; then
    [[ -f "${BOUNDARY_OPD_CALIBRATION_ARTIFACT_PATH:-}" ]] || {
        echo "ERROR: calibration trainer returned without an artifact" >&2
        exit 2
    }
    echo "Calibration trainer returned successfully; skipping checkpoint post-processing"
else
    FINAL_CHECKPOINT_PATH="$(find "$CKPT_PATH" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
    if [[ -n "$FINAL_CHECKPOINT_PATH" ]]; then
        FINAL_STEP="${FINAL_CHECKPOINT_PATH##*_}"
        ln -sfn "global_step_${FINAL_STEP}" "$CKPT_PATH/step_$(printf '%04d' "$FINAL_STEP")"
    fi
fi

# Done
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "=========================================="
    echo "End time: $(date)"
    echo "=========================================="
fi

exit 0
}  # end of read-ahead brace group (see top of file)
