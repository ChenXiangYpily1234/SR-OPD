#!/bin/bash
# Unified OPD checkpoint evaluation entry point.
#
# 用法:
#   bash bash/eval/eval.sh --run-dir RUN_DIR [--step N]
#   EVAL_GPU_IDS="0 2" bash bash/eval/eval.sh --run-dir RUN_DIR [--step N]
#
# 参数:
#   --run-dir PATH  Training run directory containing checkpoints/
#   --step N        Evaluate only one checkpoint step
#   --model-path P  Evaluate a raw Hugging Face model (uses step 0)
#
# 示例:
#   bash bash/eval/eval.sh --run-dir runs/full-opd/my-run
#   bash bash/eval/eval.sh --run-dir runs/full-q1.0-explore0.10/my-run --step 279
#
# 流程:
#   1. 扫描所有 checkpoint 运行目录
#   2. 对每个 checkpoint step:
#      a. 判断是否已有 huggingface 导出，若无则合并 FSDP 分片
#      b. 自动检测 GPU；每张 GPU 常驻一个 vLLM，按样本分片
#      c. 每道题随机采样 16 条 response（Avg@16/Pass@16 协议：base seed=42，
#         第 j 条 rollout 的采样 seed 为 42+j，即 42..57；16 条采样相互独立，
#         无需 16 次独立评测运行）
#      d. 每个 benchmark 写出一份 jsonl 文件（每题 16 行, rollout_id=0..15），
#         用规则匹配打分，输出 Avg@16 和 Pass@16 结果

set -Eeuo pipefail

ACTIVE_EVAL_PIDS=()
cleanup_eval_children() {
    local pid
    for pid in "${ACTIVE_EVAL_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${ACTIVE_EVAL_PIDS[@]:-}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
    # Clean up remaining evaluation workers on this dedicated node.
    local pattern
    for pattern in \
        'import gen_vllm' 'gen_vllm\.py' \
        'vllm\.v1\.engine' 'vllm\.engine' 'VLLM::EngineCore' \
        'from multiprocessing\.spawn import spawn_main'; do
        pkill -TERM -f "$pattern" 2>/dev/null || true
    done
}
trap cleanup_eval_children EXIT
trap 'exit 130' INT TERM

# ──────────────────────────── 参数解析 ────────────────────────────
EVAL_STEP_FILTER=""
RUN_DIR_ARG=""
RAW_MODEL_PATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR_ARG="${2:?--run-dir requires a path}"
            shift 2
            ;;
        --step)
            EVAL_STEP_FILTER="${2:?--step requires a value}"
            shift 2
            ;;
        --model-path)
            RAW_MODEL_PATH="${2:?--model-path requires a path}"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash bash/eval/eval.sh --run-dir RUN_DIR [--step N] [--model-path MODEL]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument '$1'" >&2
            exit 2
            ;;
    esac
done
[[ -n "$RUN_DIR_ARG" ]] || { echo "ERROR: --run-dir is required" >&2; exit 2; }
if [[ -n "$RAW_MODEL_PATH" ]]; then
    [[ -d "$RAW_MODEL_PATH" ]] || { echo "ERROR: model directory not found: $RAW_MODEL_PATH" >&2; exit 2; }
    mkdir -p "$RUN_DIR_ARG"
else
    [[ -d "$RUN_DIR_ARG" ]] || { echo "ERROR: run directory not found: $RUN_DIR_ARG" >&2; exit 2; }
fi
RUN_TOP_DIR="$(cd "$RUN_DIR_ARG" && pwd)"

# ──────────────────────────── 路径配置 ────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$OPD_ROOT/bash/lib/layout.sh" ]; then
    source "$OPD_ROOT/bash/lib/layout.sh"
else
    opd_eval_seed_tag() {
        local joined="" seed
        for seed in "$@"; do
            [[ -z "$joined" ]] || joined+="_"
            joined+="$seed"
        done
        printf 'seeds-%s\n' "$joined"
    }
    opd_eval_checkpoint_name() {
        printf 'step_%04d\n' "$1"
    }
fi

eval_run_short() {
    local run_dir="$1" run_name run_base raw_name
    run_name="$(basename "$run_dir")"
    if [ "$run_name" = "checkpoints" ]; then
        run_base="$(basename "$(dirname "$run_dir")")"
    else
        run_base="$run_name"
    fi
    if [[ -n "$PIPELINE_TAG" && "$PIPELINE_TAG" != "$run_base" ]]; then
        raw_name="${PIPELINE_TAG}-${run_base}"
    else
        raw_name="${PIPELINE_TAG:-$run_base}"
    fi
    printf '%s' "$raw_name" | tr -cs 'A-Za-z0-9._-' '-'
}

# Environment model paths take precedence over bash/models.env.
if [[ -z "${ACTOR_MODEL_PATH:-}" || -z "${REWARD_MODEL_PATH:-}" ]]; then
    [[ -f "${SCRIPT_DIR}/../models.env" ]] || {
        echo "ERROR: ACTOR_MODEL_PATH/REWARD_MODEL_PATH are unset and bash/models.env does not exist." >&2
        echo "       Copy bash/models.env.example to bash/models.env, or export both variables." >&2
        exit 2
    }
    source "${SCRIPT_DIR}/../models.env"
fi
: "${ACTOR_MODEL_PATH:?ACTOR_MODEL_PATH is not set}"
: "${REWARD_MODEL_PATH:?REWARD_MODEL_PATH is not set}"

CKPT_ROOT="${RUN_TOP_DIR}/checkpoints"
if [[ -z "$RAW_MODEL_PATH" ]]; then
    [[ -d "$CKPT_ROOT" ]] || { echo "ERROR: checkpoints directory not found: $CKPT_ROOT" >&2; exit 2; }
fi
echo "Checkpoint 根目录: ${CKPT_ROOT}"
if [ -n "$EVAL_STEP_FILTER" ]; then
    echo "仅评测 step: ${EVAL_STEP_FILTER}"
fi

# 基础模型目录 (用于拷贝 config/tokenizer 到合并后的模型)
BASE_MODEL_PATH="$ACTOR_MODEL_PATH"
[[ -z "$RAW_MODEL_PATH" ]] || BASE_MODEL_PATH="$RAW_MODEL_PATH"

# 推理 & 评测工作目录
EVAL_DIR="${OPD_ROOT}/scripts/val/eval"

# 测试数据集目录 (统一使用 datasets/test/)
EVAL_DATA_DIR="${OPD_ROOT}/datasets/test"
EVAL_TASK_NAMES=(AIME24 AIME25 AMC23 HMMT24 HMMT25 MATH-500)
EVAL_TASK_PATHS=(
    "${EVAL_DATA_DIR}/AIME24/test.parquet"
    "${EVAL_DATA_DIR}/AIME25/test.parquet"
    "${EVAL_DATA_DIR}/AMC23/test.parquet"
    "${EVAL_DATA_DIR}/HMMT24/test.parquet"
    "${EVAL_DATA_DIR}/HMMT25/test.parquet"
    "${EVAL_DATA_DIR}/MATH-500/test.parquet"
)

# verl 脚本目录 (legacy_model_merger.py 所在)
VERL_SCRIPTS="${OPD_ROOT}/verl/scripts"
VERL_ROOT="${OPD_ROOT}/verl"

# Make the vendored verl package available to the checkpoint merger.
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

MERGED_MODELS_DIR="${MERGED_MODELS_DIR:-${RUN_TOP_DIR}/eval_models}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${RUN_TOP_DIR}/eval_results}"
PIPELINE_TAG="${PIPELINE_TAG:-}"
mkdir -p "$MERGED_MODELS_DIR" "$EVAL_RESULTS_DIR"
echo "评测模型输出目录: ${MERGED_MODELS_DIR}"
echo "评测结果输出目录: ${EVAL_RESULTS_DIR}"

# 推理参数
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=31744
# Avg@16/Pass@16 协议：每道题在一次评测中采样 16 条 response。
N_SAMPLES=16

# 推理固定使用单个 base seed 42；第 j 条 rollout 的采样 seed 为 42+j
# （即 rollout seeds 42..57），保证 16 条采样相互独立。
# 默认自动检测 GPU；EVAL_GPU_IDS 可覆盖。
EVAL_SEED_LIST=(42)

detect_eval_gpus() {
    local detected
    if [[ -n "${EVAL_GPU_IDS:-}" ]]; then
        printf '%s\n' "$EVAL_GPU_IDS"
        return
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "-1" ]]; then
        printf '%s\n' "${CUDA_VISIBLE_DEVICES//,/ }"
        return
    fi
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "ERROR: nvidia-smi is unavailable; set EVAL_GPU_IDS explicitly" >&2
        return 1
    }
    detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd ' ' -)"
    [[ -n "$detected" ]] || {
        echo "ERROR: no CUDA GPU detected" >&2
        return 1
    }
    printf '%s\n' "$detected"
}

read -r -a AVAILABLE_EVAL_GPUS <<< "$(detect_eval_gpus)"
if [ "${#AVAILABLE_EVAL_GPUS[@]}" -eq 0 ]; then
    echo "ERROR: EVAL_GPU_IDS did not contain any GPU" >&2
    exit 2
fi

# 每张 GPU 启动一个持久 vLLM worker；样本在 GPU 间分片，每道题采样 16 条 response。
declare -a PREFLIGHT_ARGS
PREFLIGHT_ARGS=(preflight --base-model "$BASE_MODEL_PATH")
for dataset_path in "${EVAL_TASK_PATHS[@]}"; do
    PREFLIGHT_ARGS+=(--dataset "$dataset_path")
done
for gpu_id in "${AVAILABLE_EVAL_GPUS[@]}"; do
    PREFLIGHT_ARGS+=(--gpu-id "$gpu_id")
done
echo "运行评测环境预检 ..."
python3 "${EVAL_DIR}/eval_integrity.py" "${PREFLIGHT_ARGS[@]}"
echo "[OK] 数据集、模型依赖和 GPU 预检通过"
echo "Evaluation base seed: ${EVAL_SEED_LIST[*]} (rollout seeds ${EVAL_SEED_LIST[0]}..$((EVAL_SEED_LIST[0] + N_SAMPLES - 1)))"
echo "检测到的 GPU IDs: ${AVAILABLE_EVAL_GPUS[*]}"
echo "调度方式: 每卡一个模型 worker，每道题采样 ${N_SAMPLES} 条 response (Avg@16/Pass@16)"
EVAL_SEED_TAG="$(opd_eval_seed_tag "${EVAL_SEED_LIST[@]}")-n${N_SAMPLES}"
if [[ "${EVAL_ARTIFACT_LAYOUT:-legacy}" == "canonical" ]]; then
    EVAL_RESULTS_SET_DIR="$EVAL_RESULTS_DIR"
    EVAL_LOG_SET_DIR="$RUN_TOP_DIR/logs/eval"
else
    EVAL_RESULTS_SET_DIR="$EVAL_RESULTS_DIR/$EVAL_SEED_TAG"
    EVAL_LOG_SET_DIR="$RUN_TOP_DIR/logs/eval/$EVAL_SEED_TAG"
fi
mkdir -p "$EVAL_RESULTS_SET_DIR" "$EVAL_LOG_SET_DIR"

# 是否启用 CompassVerifier 模型验证器（占用额外显存）
ENABLE_MODEL_VERIFIER=false

# ──────────────────────────── 扫描所有 checkpoint 运行 ────────────────────────────
echo "=============================================="
echo "扫描 checkpoint 运行目录 ..."
echo "=============================================="

list_checkpoint_steps() {
    local run_dir="$1"
    find "$run_dir" -mindepth 3 -maxdepth 3 -type f \
        -path "$run_dir/global_step_*/actor/model_world_size_*_rank_0.pt" \
        -print | sed 's|.*/global_step_\([0-9][0-9]*\)/.*|\1|' | sort -n -u
}

load_filtered_steps() {
    local run_dir="$1" step
    RUN_STEPS=()
    if [[ -n "$RAW_MODEL_PATH" ]]; then
        RUN_STEPS=(0)
        return
    fi
    while IFS= read -r step; do
        [[ -n "$step" ]] || continue
        if [[ -z "$EVAL_STEP_FILTER" || "$step" = "$EVAL_STEP_FILTER" ]]; then
            RUN_STEPS+=("$step")
        fi
    done < <(list_checkpoint_steps "$run_dir")
}

# 智能检测: 如果 CKPT_ROOT 本身直接包含 global_step_* 则视为单个运行
DIRECT_STEPS=""
[[ -n "$RAW_MODEL_PATH" ]] || DIRECT_STEPS="$(list_checkpoint_steps "$CKPT_ROOT")"
if [[ -n "$RAW_MODEL_PATH" ]]; then
    RUN_DIRS=("$CKPT_ROOT")
    mkdir -p "$CKPT_ROOT"
    echo "原始 Hugging Face 模型评测: $RAW_MODEL_PATH"
elif [ -n "$DIRECT_STEPS" ]; then
    # 用户直接指定了单个运行目录
    RUN_DIRS=("$CKPT_ROOT")
    echo "检测到单个运行目录: $(basename "$CKPT_ROOT")"
else
    # Only include immediate children that actually contain actor checkpoints.
    RUN_DIRS=()
    while IFS= read -r run_dir; do
        if [ -n "$(list_checkpoint_steps "$run_dir")" ]; then
            RUN_DIRS+=("$run_dir")
        fi
    done < <(find "$CKPT_ROOT" -mindepth 1 -maxdepth 1 -type d -print | sort)
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "错误: 在 ${CKPT_ROOT} 下未找到任何 checkpoint 运行目录!"
    echo "  (需要有 global_step_*/actor/model_world_size_*_rank_0.pt)"
    exit 1
fi

echo "找到 ${#RUN_DIRS[@]} 个训练运行:"
for RUN_DIR in "${RUN_DIRS[@]}"; do
    RUN_NAME=$(basename "$RUN_DIR")
    echo "  - ${RUN_NAME}"
done
echo ""

# ──────────────────────────── 遍历所有运行和 checkpoint step ────────────────────────────
TOTAL_CKPTS=0
CURRENT=0

# 先计算总数
for RUN_DIR in "${RUN_DIRS[@]}"; do
    load_filtered_steps "$RUN_DIR"
    TOTAL_CKPTS=$((TOTAL_CKPTS + ${#RUN_STEPS[@]}))
done

echo "总共 ${TOTAL_CKPTS} 个 checkpoint 待评测"
echo ""
if [ "$TOTAL_CKPTS" -eq 0 ]; then
    echo "ERROR: 没有与 --step 过滤条件匹配的 checkpoint" >&2
    exit 1
fi

# 汇总结果文件
SUMMARY_FILE="${EVAL_RESULTS_SET_DIR}/eval_summary.txt"
mkdir -p "$(dirname "$SUMMARY_FILE")"
echo "OPD Checkpoint 评测结果汇总 - $(date)" > "$SUMMARY_FILE"
echo "==============================================" >> "$SUMMARY_FILE"
echo "Protocol: Avg@16/Pass@16 (base seed ${EVAL_SEED_LIST[*]}, ${N_SAMPLES} samples per problem)" >> "$SUMMARY_FILE"
echo "GPU IDs: ${AVAILABLE_EVAL_GPUS[*]}" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for RUN_DIR in "${RUN_DIRS[@]}"; do
    RUN_NAME=$(basename "$RUN_DIR")
    RUN_SHORT="$(eval_run_short "$RUN_DIR")"

    load_filtered_steps "$RUN_DIR"

    if [ ${#RUN_STEPS[@]} -eq 0 ]; then
        echo "[WARN] 运行 ${RUN_NAME} 没有有效 checkpoint, 跳过"
        continue
    fi

    echo ""
    echo "################################################################"
    echo "  运行: ${RUN_NAME}"
    echo "  Checkpoint steps: ${RUN_STEPS[*]}"
    echo "################################################################"
    echo "  运行: ${RUN_NAME}" >> "$SUMMARY_FILE"
    echo "  Checkpoint steps: ${RUN_STEPS[*]}" >> "$SUMMARY_FILE"

    for STEP_NUM in "${RUN_STEPS[@]}"; do
        CURRENT=$((CURRENT + 1))
        CKPT_STEP="global_step_${STEP_NUM}"
        ACTOR_DIR="${RUN_DIR}/${CKPT_STEP}/actor"
        if [[ "${EVAL_ARTIFACT_LAYOUT:-legacy}" == "canonical" ]]; then
            MERGE_NAME="$(opd_eval_checkpoint_name "$STEP_NUM")"
        else
            MERGE_NAME="${RUN_SHORT}-$(opd_eval_checkpoint_name "$STEP_NUM")"
        fi
        MERGED_MODEL_DIR="${MERGED_MODELS_DIR}/${MERGE_NAME}"
        [[ -z "$RAW_MODEL_PATH" ]] || MERGED_MODEL_DIR="$RAW_MODEL_PATH"

        echo ""
        echo "################################################################"
        echo "  [${CURRENT}/${TOTAL_CKPTS}] 评测: ${RUN_SHORT}/${CKPT_STEP}"
        echo "################################################################"

        # ──────────────────── Step 1: 准备 HuggingFace 格式模型 ────────────────────
        echo "=============================================="
        echo "Step 1: 准备 HuggingFace 格式模型"
        echo "=============================================="

        [[ -n "$RAW_MODEL_PATH" ]] || mkdir -p "$MERGED_MODEL_DIR"

        # 检查是否已有 huggingface 子目录（verl 新版自动保存）
        HF_SUBDIR="${ACTOR_DIR}/huggingface"
        if [[ -n "$RAW_MODEL_PATH" ]]; then
            echo "[SKIP] 使用原始 Hugging Face 模型: $RAW_MODEL_PATH"
        elif [ -d "$HF_SUBDIR" ] && [ -f "${HF_SUBDIR}/config.json" ]; then
            # 有 huggingface 子目录：合并 FSDP 权重，legacy_model_merger 会自动检测
            if python3 "${EVAL_DIR}/eval_integrity.py" validate-model \
                "$MERGED_MODEL_DIR" >/dev/null 2>&1; then
                echo "[SKIP] 合并后的模型已存在: ${MERGED_MODEL_DIR}"
            else
                echo "检测到 huggingface 子目录，使用 FSDP merger 合并权重..."
                python3 "${VERL_SCRIPTS}/legacy_model_merger.py" merge \
                    --backend fsdp \
                    --local_dir "${ACTOR_DIR}" \
                    --target_dir "${MERGED_MODEL_DIR}"
                echo "[OK] 合并完成 → ${MERGED_MODEL_DIR}"
            fi
        else
            # 没有 huggingface 子目录：合并权重 + 从基础模型拷贝 config/tokenizer
            if python3 "${EVAL_DIR}/eval_integrity.py" validate-model \
                "$MERGED_MODEL_DIR" >/dev/null 2>&1; then
                echo "[SKIP] 合并后的模型已存在: ${MERGED_MODEL_DIR}"
            else
                echo "未检测到 huggingface 子目录，先合并 FSDP 权重..."
                python3 "${VERL_SCRIPTS}/legacy_model_merger.py" merge \
                    --backend fsdp \
                    --local_dir "${ACTOR_DIR}" \
                    --target_dir "${MERGED_MODEL_DIR}"

                # 从基础模型拷贝 config/tokenizer 文件
                echo "从基础模型 ${BASE_MODEL_PATH} 拷贝 config 和 tokenizer 文件..."
                for f in config.json generation_config.json tokenizer_config.json \
                         tokenizer.json vocab.json merges.txt added_tokens.json \
                         special_tokens_map.json chat_template.jinja; do
                    if [ -f "${BASE_MODEL_PATH}/${f}" ] && [ ! -f "${MERGED_MODEL_DIR}/${f}" ]; then
                        cp "${BASE_MODEL_PATH}/${f}" "${MERGED_MODEL_DIR}/"
                    fi
                done
                echo "[OK] 合并完成 → ${MERGED_MODEL_DIR}"
            fi
        fi

        # The merger output is not considered reusable until every referenced
        # safetensors shard and the tokenizer/config prerequisites are present.
        python3 "${EVAL_DIR}/eval_integrity.py" validate-model "$MERGED_MODEL_DIR"

        # ──────────────────── Step 2: vLLM 推理 ────────────────────
        echo ""
        echo "=============================================="
        echo "Step 2: vLLM 推理生成"
        echo "=============================================="

        cd "${EVAL_DIR}"

        SEED_LOG_DIR="${EVAL_LOG_SET_DIR}/${MERGE_NAME}"
        mkdir -p "$SEED_LOG_DIR"
        validate_seed_outputs() {
            local eval_seed="$1"
            local task_index task_name task_path task_file
            for task_index in "${!EVAL_TASK_NAMES[@]}"; do
                task_name="${EVAL_TASK_NAMES[$task_index]}"
                task_path="${EVAL_TASK_PATHS[$task_index]}"
                task_file="${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/$(printf '%s' "$task_name" | tr '[:upper:]' '[:lower:]')_t${TEMPERATURE}_p${TOP_P}_n${N_SAMPLES}-MNT${MAX_TOKENS}_seed${eval_seed}.jsonl"
                python3 "${EVAL_DIR}/eval_integrity.py" validate-output \
                    --path "$task_file" --task "$task_name" --dataset "$task_path" \
                    --n "$N_SAMPLES" --seed "$eval_seed"
            done
        }

        validate_all_seed_outputs() {
            local eval_seed
            for eval_seed in "${EVAL_SEED_LIST[@]}"; do
                validate_seed_outputs "$eval_seed"
            done
        }

        run_eval_all_seeds() {
            EVAL_SEEDS="${EVAL_SEED_LIST[*]}" \
            EVAL_GPU_IDS="${AVAILABLE_EVAL_GPUS[*]}" python3 -c "
import os, sys
sys.path.insert(0, '.')
import gen_vllm

gen_vllm.MODEL_NAMES = ['${MERGED_MODEL_DIR}']
gen_vllm.DATA_DIR = '${EVAL_DATA_DIR}'
gen_vllm.OUTPUT_BASE = '${EVAL_RESULTS_SET_DIR}'
gen_vllm.OUTPUT_NAME = '${MERGE_NAME}'
gen_vllm.TEMPERATURE = ${TEMPERATURE}
gen_vllm.TOP_P = ${TOP_P}
gen_vllm.MAX_TOKENS = ${MAX_TOKENS}
gen_vllm.N_ROLLOUTS = ${N_SAMPLES}
gen_vllm.REPLACE = False

gen_vllm.TASKS = [
    {'name': 'AIME24',   'path': f'{gen_vllm.DATA_DIR}/AIME24/test.parquet',   'N': ${N_SAMPLES}},
    {'name': 'AIME25',   'path': f'{gen_vllm.DATA_DIR}/AIME25/test.parquet',   'N': ${N_SAMPLES}},
    {'name': 'AMC23',    'path': f'{gen_vllm.DATA_DIR}/AMC23/test.parquet',    'N': ${N_SAMPLES}},
    {'name': 'HMMT24',   'path': f'{gen_vllm.DATA_DIR}/HMMT24/test.parquet',   'N': ${N_SAMPLES}},
    {'name': 'HMMT25',   'path': f'{gen_vllm.DATA_DIR}/HMMT25/test.parquet',   'N': ${N_SAMPLES}},
    {'name': 'MATH-500', 'path': f'{gen_vllm.DATA_DIR}/MATH-500/test.parquet', 'N': ${N_SAMPLES}},
]

print(f'模型: {gen_vllm.MODEL_NAMES[0]}')
print(f'base_seed: {os.environ[\"EVAL_SEEDS\"]}, GPUs: {os.environ[\"EVAL_GPU_IDS\"]}')
print(f'任务: {[t[\"name\"] for t in gen_vllm.TASKS]}')
print(f'温度={gen_vllm.TEMPERATURE}  TopP={gen_vllm.TOP_P}  MaxTokens={gen_vllm.MAX_TOKENS}  每题采样={gen_vllm.N_ROLLOUTS}')

gen_vllm.main()
"
        }

        EVAL_MAX_ATTEMPTS="${EVAL_MAX_ATTEMPTS:-2}"
        [[ "$EVAL_MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: EVAL_MAX_ATTEMPTS must be a positive integer" >&2
            exit 2
        }
        cleanup_stale_inference_processes() {
            # Clean up workers before retrying evaluation.
            local pattern
            for pattern in \
                'import gen_vllm' 'gen_vllm\.py' \
                'vllm\.v1\.engine' 'vllm\.engine' 'VLLM::EngineCore' \
                'from multiprocessing\.spawn import spawn_main'; do
                pkill -TERM -f "$pattern" 2>/dev/null || true
            done
            sleep 5
            for pattern in \
                'import gen_vllm' 'gen_vllm\.py' \
                'vllm\.v1\.engine' 'vllm\.engine' 'VLLM::EngineCore' \
                'from multiprocessing\.spawn import spawn_main'; do
                pkill -KILL -f "$pattern" 2>/dev/null || true
            done
        }
        run_eval_with_retry() {
            local attempt
            local retry_wait_secs="${EVAL_RETRY_WAIT_SECS:-30}"
            for ((attempt = 1; attempt <= EVAL_MAX_ATTEMPTS; attempt++)); do
                echo "[ATTEMPT ${attempt}/${EVAL_MAX_ATTEMPTS}] base_seed=${EVAL_SEED_LIST[*]}, n=${N_SAMPLES}, GPUs=${AVAILABLE_EVAL_GPUS[*]}"
                if run_eval_all_seeds && validate_all_seed_outputs; then
                    return 0
                fi
                echo "[WARN] 推理第 ${attempt} 次尝试失败" >&2
                if ((attempt < EVAL_MAX_ATTEMPTS)); then
                    echo "[CLEANUP] 清理上一轮残留的 vLLM/EngineCore 进程..." >&2
                    cleanup_stale_inference_processes
                    echo "[WAIT] 等待 ${retry_wait_secs}s 让端口与显存释放后重试..." >&2
                    sleep "${retry_wait_secs}"
                fi
            done
            return 1
        }

        EVAL_LOG="${SEED_LOG_DIR}/multi_seed.log"
        echo "[START] ${#AVAILABLE_EVAL_GPUS[@]} 个 GPU 各加载一份模型；每道题采样 ${N_SAMPLES} 条 response"
        if validate_all_seed_outputs >/dev/null 2>&1; then
            # 幂等: 全部 task 的推理输出已存在且校验通过，无需重新推理。
            echo "[SKIP] ${#EVAL_TASK_NAMES[@]} 个任务的 n=${N_SAMPLES} 推理输出均已存在且完整，跳过 vLLM 推理"
        else
            if ! run_eval_with_retry > "$EVAL_LOG" 2>&1; then
                echo "[ERROR] 多 GPU 推理失败，检查 ${EVAL_LOG}" >&2
                exit 1
            fi

            validate_all_seed_outputs
            echo "[OK] n=${N_SAMPLES} 推理完成（每卡一份模型，按样本分片）"
        fi

        # ──────────────────── Step 3: 评测打分 ────────────────────
        echo ""
        echo "=============================================="
        echo "Step 3: 评测打分"
        echo "=============================================="

        GRADING_RESULTS_JSON="${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/grading_results.json"
        GRADING_SUMMARY_TXT="${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/benchmark_summary.txt"
        if [[ -f "$GRADING_RESULTS_JSON" && -f "$GRADING_SUMMARY_TXT" ]]; then
            echo "[SKIP] 打分结果已存在，跳过评测打分: $GRADING_RESULTS_JSON"
        else
        USE_VERIFIER="false"
        if [ "${ENABLE_MODEL_VERIFIER}" = "true" ]; then
            USE_VERIFIER="true"
            echo "启用 CompassVerifier-3B 模型验证器"
        else
            echo "仅使用规则匹配（sympy 数学等价性检查）"
        fi

        python3 -c "
import os, sys, json
from pathlib import Path

sys.path.insert(0, '.')
import grade
from eval_integrity import validate_benchmark_seed_coverage

grade.NAME = '${MERGE_NAME}'
grade.EVAL_DIR = Path('${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}')
grade.OUTPUT_FILE = grade.EVAL_DIR / 'grading_results.json'
summary_text_file = grade.EVAL_DIR / 'benchmark_summary.txt'

print(f'评测目录: {grade.EVAL_DIR}')
print(f'输出文件: {grade.OUTPUT_FILE}')

use_verifier = '${USE_VERIFIER}' == 'true'

if use_verifier:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    print('加载 CompassVerifier-3B...')
    grade.model_tokenizer = AutoTokenizer.from_pretrained(grade.MODEL_NAME)
    grade.vllm_model = LLM(model=grade.MODEL_NAME, tensor_parallel_size=8)
    grade.sampling_params = SamplingParams(temperature=0.0, max_tokens=2048)

active_eval_seeds = {int(value) for value in '${EVAL_SEED_LIST[*]}'.split()}
if len(active_eval_seeds) != 1:
    raise RuntimeError(
        f'Avg@16/Pass@16 协议要求恰好一个 base seed, got {sorted(active_eval_seeds)}'
    )
base_seed = next(iter(active_eval_seeds))
n_samples = ${N_SAMPLES}
task_names = ['aime24', 'aime25', 'amc23', 'hmmt24', 'hmmt25', 'math-500']
all_results = []
for task_name in task_names:
    file_path = grade.EVAL_DIR / (
        f'{task_name}_t${TEMPERATURE}_p${TOP_P}_n${N_SAMPLES}'
        f'-MNT${MAX_TOKENS}_seed{base_seed}.jsonl'
    )
    if not file_path.is_file():
        raise RuntimeError(f'缺少本次评测输出: {file_path}')
    print(f'评测: {file_path.name}')
    file_result = grade.grade_file(file_path, use_model_verifier=use_verifier)
    if file_result is None:
        raise RuntimeError(f'无法解析评测输出文件名: {file_path.name}')
    all_results.append(file_result)

validate_benchmark_seed_coverage(
    all_results,
    ['AIME24', 'AIME25', 'AMC23', 'HMMT24', 'HMMT25', 'MATH-500'],
    active_eval_seeds,
)

by_task = {}
for result in all_results:
    hp = result.get('hyperparameters', {})
    task_name = hp.get('task_name', 'unknown')
    by_task[task_name] = result

display_names = {
    'aime24': 'AIME24',
    'aime25': 'AIME25',
    'amc23': 'AMC23',
    'hmmt24': 'HMMT24',
    'hmmt25': 'HMMT25',
    'math-500': 'MATH-500',
}
benchmark_summary = []
summary_lines = [
    '=' * 70,
    f'Avg@{n_samples}/Pass@{n_samples} '
    f'(每题随机采样 {n_samples} 条, rollout seeds {base_seed}..{base_seed + n_samples - 1})',
    '=' * 70,
]
for task_name in task_names:
    result = by_task.get(task_name)
    if result is None:
        raise RuntimeError(f'{task_name}: missing grading result')
    # grade_file 把同一题的 n_samples 条 response 聚合:
    #   mean_score = 16 条中的平均正确率  -> Avg@16
    #   best_score = 16 条中至少 1 条正确  -> Pass@16
    avg_at_n = float(result.get('mean_score', 0.0))
    pass_at_n = float(result.get('best_score', 0.0))
    display_name = display_names[task_name]
    summary_lines.append(
        f'{display_name:10s}: avg@{n_samples}={avg_at_n:.4f} pass@{n_samples}={pass_at_n:.4f}'
    )
    benchmark_summary.append({
        'benchmark': display_name,
        'num_samples': n_samples,
        'base_seed': base_seed,
        'rollout_seeds': [base_seed + i for i in range(n_samples)],
        f'avg_at_{n_samples}': avg_at_n,
        f'pass_at_{n_samples}': pass_at_n,
    })
summary_lines.append('=' * 70)

payload = {
    'protocol': f'avg@{n_samples}/pass@{n_samples}',
    'base_seed': base_seed,
    'rollout_seeds': [base_seed + i for i in range(n_samples)],
    'eval_seeds': sorted(active_eval_seeds),
    'samples_per_problem': n_samples,
    'per_task_results': all_results,
    'benchmark_summary': benchmark_summary,
}
temporary_json = grade.OUTPUT_FILE.with_name(f'.{grade.OUTPUT_FILE.name}.tmp-{os.getpid()}')
with temporary_json.open('w', encoding='utf-8') as f:
    json.dump(payload, f, indent=4)
temporary_json.replace(grade.OUTPUT_FILE)
summary_text = '\n'.join(summary_lines) + '\n'
temporary_summary = summary_text_file.with_name(f'.{summary_text_file.name}.tmp-{os.getpid()}')
temporary_summary.write_text(summary_text, encoding='utf-8')
temporary_summary.replace(summary_text_file)
print()
print(summary_text, end='')
print(f'评测结果已保存: {grade.OUTPUT_FILE}')
print(f'Avg@16/Pass@16 汇总已保存: {summary_text_file}')
"
        fi

        echo "[OK] ${CKPT_STEP} 评测完成"

        # 回到脚本目录
        cd "$SCRIPT_DIR"

        # 追加到汇总文件
        echo "" >> "$SUMMARY_FILE"
        echo "  ${MERGE_NAME}:" >> "$SUMMARY_FILE"
        echo "    推理输出: ${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/" >> "$SUMMARY_FILE"
        echo "    评测分数: ${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/grading_results.json" >> "$SUMMARY_FILE"
        cat "${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/benchmark_summary.txt" >> "$SUMMARY_FILE"
    done
done

# ──────────────────────────── 汇总 ────────────────────────────
echo ""
echo "=============================================="
echo "全部完成! 共评测 ${TOTAL_CKPTS} 个 checkpoint"
echo "=============================================="
echo ""
echo "模型目录: ${MERGED_MODELS_DIR}"
echo "评测输出: ${EVAL_RESULTS_DIR}/"
echo "汇总文件: ${SUMMARY_FILE}"
echo ""
echo "评测结果汇总:"
for RUN_DIR in "${RUN_DIRS[@]}"; do
    RUN_NAME=$(basename "$RUN_DIR")
    RUN_SHORT="$(eval_run_short "$RUN_DIR")"
    load_filtered_steps "$RUN_DIR"
    for STEP_NUM in "${RUN_STEPS[@]}"; do
        if [[ "${EVAL_ARTIFACT_LAYOUT:-legacy}" == "canonical" ]]; then
            MERGE_NAME="$(opd_eval_checkpoint_name "$STEP_NUM")"
        else
            MERGE_NAME="${RUN_SHORT}-$(opd_eval_checkpoint_name "$STEP_NUM")"
        fi
        echo "  ${MERGE_NAME}:"
        echo "    推理输出: ${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/"
        echo "    评测分数: ${EVAL_RESULTS_SET_DIR}/${MERGE_NAME}/grading_results.json"
    done
done
echo "=============================================="
