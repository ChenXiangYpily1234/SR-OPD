#!/bin/bash
#SBATCH --job-name=opd-pairA-qwen3
#SBATCH --output=slurm-pairA-qwen3-%j.out
#SBATCH --error=slurm-pairA-qwen3-%j.err
#SBATCH --account=test
#SBATCH --partition=TEST1
#SBATCH --exclude=g[81-82]
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Qwen3-4B -> Qwen3-1.7B: Full OPD, TA-OPD, and TLR-OPD.
# Set MODELS_DIR or TEACHER_MODEL_PATH / STUDENT_MODEL_PATH.
# Usage: bash bash/run_group_a_qwen3_4b_to_qwen3_1.7b.sh --help

{
set -Eeuo pipefail

PHASE=all
METHODS="full,ta,tlr"
OUTPUT_ROOT_ARG=""
EVAL_BASE_TEACHER=true

usage() {
    cat <<'EOF'
Usage: bash bash/run_group_a_qwen3_4b_to_qwen3_1.7b.sh [options]
  --phase all|train|eval  Phase selection (default: all)
  --methods full,ta,tlr  Method subset (default: full,ta,tlr)
  --output-root PATH    Output directory (default: runs/)
  --skip-teacher-eval   Skip the base Teacher evaluation
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="${2:?missing --phase}"; shift 2 ;;
        --methods) METHODS="${2:?missing --methods}"; shift 2 ;;
        --output-root) OUTPUT_ROOT_ARG="${2:?missing --output-root}"; shift 2 ;;
        --skip-teacher-eval) EVAL_BASE_TEACHER=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done
[[ "$PHASE" == "all" || "$PHASE" == "train" || "$PHASE" == "eval" ]] || {
    echo "ERROR: --phase must be all, train, or eval" >&2; exit 2;
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$OPD_ROOT/bash/lib/layout.sh"

MODELS_DIR="${MODELS_DIR:-}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODELS_DIR:+$MODELS_DIR/Qwen3-4B}}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODELS_DIR:+$MODELS_DIR/Qwen3-1.7B}}"
if [[ -z "$TEACHER_MODEL_PATH" || -z "$STUDENT_MODEL_PATH" ]]; then
    echo "ERROR: set MODELS_DIR (holding both checkpoints) or TEACHER_MODEL_PATH/STUDENT_MODEL_PATH" >&2
    exit 2
fi

TRAIN_SEED=42
TA_TOKEN_RETAIN_RATIO="${TA_TOKEN_RETAIN_RATIO:-0.10}"
OUTPUT_ROOT="${OUTPUT_ROOT_ARG:-${OUTPUT_ROOT:-$OPD_ROOT/runs}}"

export MAX_RESP_LENGTH=8192
export ROLLOUT_GPU_MEM_UTIL=0.8
export ACTOR_MICRO_BATCH_SIZE_PER_GPU=2
export MINI_BATCH_SIZE=64
export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIMIZER_OFFLOAD=False
export ACTOR_ACTIVATION_OFFLOAD=False
export REF_PARAM_OFFLOAD=False
export PPO_MAX_TOKEN_LEN_PER_GPU=65536
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=49152
export TEACHER_COMPUTE_ENTROPY=False
export TEACHER_MAX_TOKEN_LEN_PER_GPU=49152
export ROLLOUT_CALCULATE_LOG_PROBS=False
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=36864

for p in "$TEACHER_MODEL_PATH" "$STUDENT_MODEL_PATH"; do
    [[ -d "$p" ]] || { echo "ERROR: model dir not found: $p" >&2; exit 2; }
done

MODEL_PAIR="$(opd_model_pair "$(basename "$TEACHER_MODEL_PATH")" "$(basename "$STUDENT_MODEL_PATH")")"

FULL_RUN_DIR="$OUTPUT_ROOT/$MODEL_PAIR/full_opd/full-opd-trainseed${TRAIN_SEED}"
TA_RUN_DIR="$OUTPUT_ROOT/$MODEL_PAIR/ta_opd/ta-opd-r${TA_TOKEN_RETAIN_RATIO}-trainseed${TRAIN_SEED}"
TLR_RUN_DIR="$OUTPUT_ROOT/$MODEL_PAIR/tlr_opd/tlr-opd-trainseed${TRAIN_SEED}"
BASE_STUDENT_EVAL_DIR="$OUTPUT_ROOT/$MODEL_PAIR/base/base-student"
BASE_TEACHER_EVAL_DIR="$OUTPUT_ROOT/$MODEL_PAIR/teacher/base-teacher"

export ACTOR_MODEL_PATH="$STUDENT_MODEL_PATH"
export REWARD_MODEL_PATH="$TEACHER_MODEL_PATH"

echo "=============================================="
echo "  组别 A: $MODEL_PAIR"
echo "  Teacher: $TEACHER_MODEL_PATH"
echo "  Student: $STUDENT_MODEL_PATH"
echo "  Output:  $OUTPUT_ROOT"
echo "  Phase:   $PHASE   Methods: $METHODS"
echo "  Train seed: $TRAIN_SEED   Eval: base seed 42 · 每题 16 采样 (eval.sh 内置)"
echo "=============================================="

has_checkpoint() {
    find "$1/checkpoints" -maxdepth 1 -type d -name 'global_step_*' \
        -print -quit 2>/dev/null | grep -q .
}

has_eval_results() {
    find "$1/eval_results" -name 'grading_results.json' \
        -print -quit 2>/dev/null | grep -q .
}

method_enabled() {
    [[ ",$METHODS," == *",$1,"* ]]
}

run_eval() {
    local run_dir="$1"
    if ! has_checkpoint "$run_dir"; then
        echo "[WARN] 无 checkpoint, 跳过评测: $run_dir" >&2
        return 0
    fi
    if has_eval_results "$run_dir"; then
        echo "[SKIP] 已有评测结果: $run_dir"
        return 0
    fi
    bash "$OPD_ROOT/bash/eval/eval.sh" --run-dir "$run_dir"
}

if [[ "$PHASE" == "all" || "$PHASE" == "train" ]]; then

    if method_enabled full; then
        echo ""; echo "########## [1/3] Full OPD ##########"
        if has_checkpoint "$FULL_RUN_DIR"; then
            echo "[SKIP] 已有训练 checkpoint: $FULL_RUN_DIR"
        else
            bash "$OPD_ROOT/bash/train/full-opd.sh" --epochs 1 \
                --model-pair "$MODEL_PAIR" \
                --seed "$TRAIN_SEED" \
                --run-name "$(basename "$FULL_RUN_DIR")" \
                --output-root "$OUTPUT_ROOT"
        fi
    fi

    if method_enabled ta; then
        echo ""; echo "########## [2/3] TA-OPD (retain=$TA_TOKEN_RETAIN_RATIO) ##########"
        if has_checkpoint "$TA_RUN_DIR"; then
            echo "[SKIP] 已有训练 checkpoint: $TA_RUN_DIR"
        else
            bash "$OPD_ROOT/bash/train/ta-opd.sh" \
                --token-retain-ratio "$TA_TOKEN_RETAIN_RATIO" \
                --model-pair "$MODEL_PAIR" \
                --seed "$TRAIN_SEED" \
                --run-name "$(basename "$TA_RUN_DIR")" \
                --output-root "$OUTPUT_ROOT"
        fi
    fi

    if method_enabled tlr; then
        echo ""; echo "########## [3/3] TLR-OPD ##########"
        if has_checkpoint "$TLR_RUN_DIR"; then
            echo "[SKIP] 已有训练 checkpoint: $TLR_RUN_DIR"
        else
            bash "$OPD_ROOT/bash/train/tlr-opd.sh" \
                --model-pair "$MODEL_PAIR" \
                --seed "$TRAIN_SEED" \
                --run-name "$(basename "$TLR_RUN_DIR")" \
                --output-root "$OUTPUT_ROOT"
        fi
    fi

    if [[ "$EVAL_BASE_TEACHER" == "true" ]]; then
        echo ""; echo "########## [EVAL] base Teacher ##########"
        if has_eval_results "$BASE_TEACHER_EVAL_DIR"; then
            echo "[SKIP] 已有评测结果: $BASE_TEACHER_EVAL_DIR"
        else
            bash "$OPD_ROOT/bash/eval/eval.sh" \
                --run-dir "$BASE_TEACHER_EVAL_DIR" \
                --model-path "$TEACHER_MODEL_PATH"
        fi
    fi

    method_enabled full && { echo ""; echo "########## [EVAL] Full OPD ##########"; run_eval "$FULL_RUN_DIR"; }
    method_enabled ta   && { echo ""; echo "########## [EVAL] TA-OPD ##########";   run_eval "$TA_RUN_DIR"; }
    method_enabled tlr  && { echo ""; echo "########## [EVAL] TLR-OPD ##########";  run_eval "$TLR_RUN_DIR"; }
fi

echo ""
echo "=============================================="
echo "  组别 A 完成: $MODEL_PAIR"
echo "=============================================="
echo "  训练 run:"
echo "    Full OPD : $FULL_RUN_DIR"
echo "    TA-OPD   : $TA_RUN_DIR"
echo "    TLR-OPD  : $TLR_RUN_DIR"
echo "  base 推理:"
echo "    Student  : $BASE_STUDENT_EVAL_DIR"
echo "    Teacher  : $BASE_TEACHER_EVAL_DIR"
echo "  各 run 评测汇总见 <run_dir>/eval_results/seeds-42-n16/eval_summary.txt"
echo "=============================================="
}
