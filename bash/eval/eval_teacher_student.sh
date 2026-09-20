#!/usr/bin/env bash
# 老师 / 学生模型补推理入口
#
# 扫描 RUNS_ROOT(默认 runs/) 下所有 "<teacher>__to__<student>" 模型对目录,
# 对尚未产出评测结果的目标依次调用 bash/eval/eval.sh:
#
#   1. 学生 base 模型  → <pair>/base/base-student    (--model-path <学生>, step 0)
#   2. 老师 base 模型  → <pair>/teacher/base-teacher (--model-path <老师>, step 0)
#   3. 各训练 run 的最终学生 checkpoint
#      (full_opd / ta_opd / tlr_opd / sr_opd / ...)
#
# “已评测”的判定与新协议 (Avg@16/Pass@16) 的 eval.sh 产出一致:
#   最新的 eval_results/seeds-*/eval_summary.txt 中含 avg@16=/pass@16=
#   指标行即视为完成并跳过; 旧版 5-seed mean±std 结果视为未评测并重新评测。
#   因此本脚本幂等, 可安全重复运行逐步补齐缺失项。
#
# 用法:
#   bash bash/eval/eval_teacher_student.sh [options]
#
# 选项:
#   --runs-root PATH        runs 根目录 (默认: <repo>/runs)
#   --models-root PATH      HuggingFace 模型根目录
#                           (默认: 由 bash/models.env 的 ACTOR_MODEL_PATH 推导)
#   --eval-gpu-ids "IDS"    评测 GPU IDs (默认: "0 1 2 3 4 5 6 7")
#   --stop-on-error         任一目标失败即停止 (默认: 继续跑完其余目标, 最后汇总)
#   --dry-run               仅打印评测计划, 不实际调用 eval.sh
#   -h, --help              显示帮助
#
# 环境变量等价物:
#   RUNS_ROOT / MODELS_ROOT / EVAL_GPU_IDS / EVAL_MAX_ATTEMPTS / EVAL_SCRIPT
#
# 示例:
#   bash bash/eval/eval_teacher_student.sh --dry-run
#   EVAL_GPU_IDS="0 1 2 3" bash bash/eval/eval_teacher_student.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRIPT_DIR/eval.sh}"
RUNS_ROOT="${RUNS_ROOT:-$OPD_ROOT/runs}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
STOP_ON_ERROR=false
DRY_RUN=false

usage() {
    sed -n '2,/^set -Eeuo/p' "$0" | sed 's/^# \{0,1\}//; /^set -Eeuo/d' | head -n -1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs-root) RUNS_ROOT="${2:?--runs-root requires a path}"; shift 2 ;;
        --models-root) MODELS_ROOT="${2:?--models-root requires a path}"; shift 2 ;;
        --eval-gpu-ids) EVAL_GPU_IDS="${2:?--eval-gpu-ids requires a value}"; shift 2 ;;
        --stop-on-error) STOP_ON_ERROR=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

# 模型根目录默认从 bash/models.env 推导 (与训练/评测现有配置保持一致)
default_models_root() {
    local env_file="$OPD_ROOT/bash/models.env" actor
    if [[ -f "$env_file" ]]; then
        actor="$(grep -E '^[[:space:]]*ACTOR_MODEL_PATH=' "$env_file" | tail -n 1 | cut -d= -f2-)"
        actor="${actor%\"}"; actor="${actor#\"}"
        if [[ -n "$actor" ]]; then
            dirname "$actor"
            return 0
        fi
    fi
    echo "ERROR: cannot derive the model root: bash/models.env is missing or has no ACTOR_MODEL_PATH." >&2
    echo "       Set MODELS_ROOT explicitly, or create bash/models.env from bash/models.env.example." >&2
    return 2
}
MODELS_ROOT="${MODELS_ROOT:-$(default_models_root)}"

EVAL_GPU_IDS="${EVAL_GPU_IDS//,/ }"
read -r -a GPU_ID_LIST <<< "$EVAL_GPU_IDS"
[[ ${#GPU_ID_LIST[@]} -gt 0 ]] || { echo "ERROR: no evaluation GPU IDs" >&2; exit 2; }
[[ -d "$RUNS_ROOT" ]] || { echo "ERROR: runs root not found: $RUNS_ROOT" >&2; exit 2; }
[[ -f "$EVAL_SCRIPT" ]] || { echo "ERROR: eval script not found: $EVAL_SCRIPT" >&2; exit 2; }

# ──────────────────────────── 工具函数 ────────────────────────────

checkpoint_is_usable() {
    local checkpoints_dir="$1" step="$2"
    [[ "$step" =~ ^[0-9]+$ ]] || return 1
    find "$checkpoints_dir/global_step_${step}/actor" \
        -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' \
        -print -quit 2>/dev/null | grep -q .
}

# 优先采用 latest_checkpointed_iteration.txt, 否则取最大的可用 global_step_N
latest_usable_step() {
    local checkpoints_dir="$1"
    local metadata_file="$checkpoints_dir/latest_checkpointed_iteration.txt"
    local recorded_step="" discovered_step=""
    if [[ -f "$metadata_file" ]]; then
        recorded_step="$(tr -d '[:space:]' < "$metadata_file")"
        recorded_step="${recorded_step#global_step_}"
        if checkpoint_is_usable "$checkpoints_dir" "$recorded_step"; then
            printf '%s\n' "$recorded_step"
            return 0
        fi
    fi
    discovered_step="$(
        find "$checkpoints_dir" -mindepth 3 -maxdepth 3 -type f \
            -path "$checkpoints_dir/global_step_*/actor/model_world_size_*_rank_0.pt" \
            -print 2>/dev/null \
        | sed 's|.*/global_step_\([0-9][0-9]*\)/.*|\1|' \
        | sort -n -u | tail -n 1
    )"
    [[ "$discovered_step" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$discovered_step"
}

# 与新协议 (Avg@16/Pass@16) 的 eval.sh 产出一致:
# 最新的 eval_summary.txt 中含 avg@16=/pass@16= 指标行即视为完成;
# 仅存在旧版 5-seed 结果的 run 会被重新评测。
run_is_evaluated() {
    local run_dir="$1"
    local results_dir="$run_dir/eval_results"
    local summary=""
    if compgen -G "$results_dir/seeds-*/eval_summary.txt" >/dev/null; then
        summary="$(ls -t "$results_dir"/seeds-*/eval_summary.txt | head -n 1)"
    elif [[ -f "$results_dir/eval_summary.txt" ]]; then
        summary="$results_dir/eval_summary.txt"
    fi
    [[ -n "$summary" ]] || return 1
    grep -qE '^[A-Za-z0-9-]+[[:space:]]*:[[:space:]]*avg@16=[0-9.]+[[:space:]]+pass@16=' "$summary"
}

# ──────────────────────────── 评测执行器 ────────────────────────────

declare -a OK_ITEMS=() FAIL_ITEMS=() SKIP_DONE_ITEMS=() NO_CKPT_ITEMS=()
PAIR_TEACHER_PATH=""
PAIR_STUDENT_PATH=""

# run_eval <label> <run_dir> <step|""> <model_path|"">
#   model_path 非空 → --model-path 模式 (base 老师/学生模型, 固定 step 0)
#   否则            → checkpoint 模式, 评测指定 step
run_eval() {
    local label="$1" run_dir="$2" step="$3" model_path="$4"
    if run_is_evaluated "$run_dir"; then
        echo "[SKIP] 已有评测结果, 跳过: $label"
        SKIP_DONE_ITEMS+=("$label")
        return 0
    fi

    local -a cmd=(bash "$EVAL_SCRIPT" --run-dir "$run_dir")
    if [[ -n "$model_path" ]]; then
        cmd+=(--model-path "$model_path")
    elif [[ -n "$step" ]]; then
        cmd+=(--step "$step")
    fi

    echo ""
    echo "==================== 待推理: $label ===================="
    printf '  cmd: ACTOR_MODEL_PATH=%q REWARD_MODEL_PATH=%q EVAL_GPU_IDS=%q' \
        "$PAIR_STUDENT_PATH" "$PAIR_TEACHER_PATH" "$EVAL_GPU_IDS"
    printf ' %q' "${cmd[@]}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        OK_ITEMS+=("$label [dry-run]")
        return 0
    fi

    if ACTOR_MODEL_PATH="$PAIR_STUDENT_PATH" \
       REWARD_MODEL_PATH="$PAIR_TEACHER_PATH" \
       EVAL_GPU_IDS="$EVAL_GPU_IDS" \
       "${cmd[@]}"; then
        OK_ITEMS+=("$label")
    else
        FAIL_ITEMS+=("$label")
        echo "[ERROR] 评测失败: $label" >&2
        if [[ "$STOP_ON_ERROR" == true ]]; then
            echo "[FATAL] --stop-on-error 生效, 终止剩余任务" >&2
            exit 1
        fi
    fi
}

# ──────────────────────────── 主循环: 遍历师生模型对 ────────────────────────────

echo "=============================================="
echo "老师/学生模型补推理"
echo "  runs 根目录:    $RUNS_ROOT"
echo "  模型根目录:     $MODELS_ROOT"
echo "  评测 GPU:       ${GPU_ID_LIST[*]}"
echo "  dry-run:        $DRY_RUN"
echo "  stop-on-error:  $STOP_ON_ERROR"
echo "=============================================="

shopt -s nullglob
pair_found=false
for pair_dir in "$RUNS_ROOT"/*__to__*; do
    [[ -d "$pair_dir" ]] || continue
    pair_found=true
    pair_name="$(basename "$pair_dir")"
    teacher_name="${pair_name%%__to__*}"
    student_name="${pair_name##*__to__}"
    PAIR_TEACHER_PATH="$MODELS_ROOT/$teacher_name"
    PAIR_STUDENT_PATH="$MODELS_ROOT/$student_name"

    echo ""
    echo "################################################################"
    echo "  模型对: $teacher_name  (老师)  →  $student_name  (学生)"
    echo "################################################################"

    pair_models_ok=true
    if [[ ! -d "$PAIR_STUDENT_PATH" ]]; then
        echo "[WARN] 学生模型目录不存在: $PAIR_STUDENT_PATH" >&2
        pair_models_ok=false
    fi
    if [[ ! -d "$PAIR_TEACHER_PATH" ]]; then
        echo "[WARN] 老师模型目录不存在: $PAIR_TEACHER_PATH" >&2
        pair_models_ok=false
    fi
    if [[ "$pair_models_ok" != true ]]; then
        echo "[SKIP] 模型对 $pair_name 缺少 HF 模型目录, 整组跳过"
        continue
    fi

    # 1. 学生 base 模型 (原始 HF 权重, step 0)
    run_eval "$pair_name :: base-student ($student_name)" \
        "$pair_dir/base/base-student" "" "$PAIR_STUDENT_PATH"

    # 2. 老师 base 模型 (原始 HF 权重, step 0)
    run_eval "$pair_name :: base-teacher ($teacher_name)" \
        "$pair_dir/teacher/base-teacher" "" "$PAIR_TEACHER_PATH"

    # 3. 该模型对下所有训练 run 的最终学生 checkpoint
    while IFS= read -r -d '' run_dir; do
        case "$run_dir" in
            */base/*|*/teacher/*) continue ;;  # base 模型目录已在上面处理
        esac
        rel_name="${run_dir#"$pair_dir"/}"
        ckpt_dir="$run_dir/checkpoints"
        step=""
        if [[ -d "$ckpt_dir" ]]; then
            step="$(latest_usable_step "$ckpt_dir" || true)"
        fi
        if [[ -z "$step" ]]; then
            echo "[SKIP] 无可用学生 checkpoint (无法推理): $rel_name"
            NO_CKPT_ITEMS+=("$pair_name :: $rel_name")
            continue
        fi
        run_eval "$pair_name :: $rel_name (step $step)" "$run_dir" "$step" ""
    done < <(find "$pair_dir" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)
done

if [[ "$pair_found" != true ]]; then
    echo "ERROR: 在 $RUNS_ROOT 下未找到任何 '*__to__*' 模型对目录" >&2
    exit 1
fi

# ──────────────────────────── 汇总 ────────────────────────────

echo ""
echo "==================== 补推理汇总 ===================="
print_section() {
    local title="$1" marker="$2"; shift 2
    local -a items=("$@")
    local item
    echo "$title (${#items[@]}):"
    for item in "${items[@]}"; do
        printf '  [%s] %s\n' "$marker" "$item"
    done
}
print_section "本次完成评测" "OK" "${OK_ITEMS[@]}"
print_section "已评测跳过" "SKIP" "${SKIP_DONE_ITEMS[@]}"
print_section "无 checkpoint 无法推理" "NO-CKPT" "${NO_CKPT_ITEMS[@]}"
print_section "评测失败" "FAIL" "${FAIL_ITEMS[@]}"
echo "==================================================="

if ((${#FAIL_ITEMS[@]} > 0)); then
    exit 1
fi
