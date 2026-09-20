#!/usr/bin/env bash
# Recursively discover OPD training runs and evaluate every usable checkpoint
# in each run's checkpoints/ folder with the existing eval.sh pipeline
# (Avg@16/Pass@16 protocol: 16 independent samples per problem in a single
# evaluation run). eval.sh is invoked once per run without --step so it walks
# all global_step_* checkpoints; steps that already carry Avg@16/Pass@16
# results are skipped by the per-step checks here and by eval.sh's internal
# idempotency (merged models, inference outputs, grading).

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${EVAL_SCRIPT:-$SCRIPT_DIR/eval.sh}"
# Default to this checkout's runs/ tree; override with --root or FF_OPD_DIR.
OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SEARCH_ROOT="${FF_OPD_DIR:-$OPD_ROOT/runs}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:-0 1 2 3 4 5 6 7}"
CONTINUE_ON_ERROR=false
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: bash bash/eval/eval_recursive.sh [options]

  --root PATH             Root directory searched recursively for runs
  --eval-gpu-ids "IDS"    Space/comma-separated GPU IDs (default: 0..7)
  --continue-on-error     Continue evaluating remaining runs after a failure
  --dry-run               Print discovered runs without invoking eval.sh
  -h, --help              Show this help

Environment equivalents:
  FF_OPD_DIR, EVAL_GPU_IDS, EVAL_SCRIPT

A usable run contains checkpoints/global_step_N/actor/
model_world_size_*_rank_0.pt. Every usable checkpoint step of every run is
evaluated (one eval.sh call per run; eval.sh iterates all steps internally).
Steps whose benchmark_summary.txt already has Avg@16/Pass@16 lines are
skipped; runs without any usable checkpoint are skipped. The value of
checkpoints/latest_checkpointed_iteration.txt is only used to report the
run's latest step.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) SEARCH_ROOT="${2:?--root requires a path}"; shift 2 ;;
        --eval-gpu-ids) EVAL_GPU_IDS="${2:?--eval-gpu-ids requires a value}"; shift 2 ;;
        --continue-on-error) CONTINUE_ON_ERROR=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -d "$SEARCH_ROOT" ]] || {
    echo "ERROR: search root does not exist: $SEARCH_ROOT" >&2
    exit 2
}
[[ -x "$EVAL_SCRIPT" || -f "$EVAL_SCRIPT" ]] || {
    echo "ERROR: eval script not found: $EVAL_SCRIPT" >&2
    exit 2
}

# eval.sh accepts space-separated IDs. Normalize the common comma-separated
# form here so both CLI styles behave identically.
EVAL_GPU_IDS="${EVAL_GPU_IDS//,/ }"
read -r -a GPU_ID_LIST <<< "$EVAL_GPU_IDS"
[[ ${#GPU_ID_LIST[@]} -gt 0 ]] || {
    echo "ERROR: no evaluation GPU IDs were provided" >&2
    exit 2
}
export EVAL_GPU_IDS

checkpoint_is_usable() {
    local checkpoints_dir="$1"
    local step="$2"
    [[ "$step" =~ ^[0-9]+$ ]] || return 1
    find "$checkpoints_dir/global_step_${step}/actor" \
        -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' \
        -print -quit 2>/dev/null | grep -q .
}

# 列出 checkpoints 目录下所有可用 step（升序）。
list_usable_steps() {
    local checkpoints_dir="$1"
    find "$checkpoints_dir" -mindepth 3 -maxdepth 3 -type f \
        -path "$checkpoints_dir/global_step_*/actor/model_world_size_*_rank_0.pt" \
        -print 2>/dev/null \
    | sed 's|.*/global_step_\([0-9][0-9]*\)/.*|\1|' \
    | sort -n -u
}

# 元数据首选 step，仅用于报告 run 的最新 step（不再限定评测范围）。
latest_usable_step() {
    local checkpoints_dir="$1"
    local metadata_file="$checkpoints_dir/latest_checkpointed_iteration.txt"
    local recorded_step=""
    local discovered_step=""

    if [[ -f "$metadata_file" ]]; then
        recorded_step="$(tr -d '[:space:]' < "$metadata_file")"
        recorded_step="${recorded_step#global_step_}"
        if checkpoint_is_usable "$checkpoints_dir" "$recorded_step"; then
            printf '%s\n' "$recorded_step"
            return 0
        fi
        echo "[WARN] Ignoring stale checkpoint metadata: $metadata_file -> ${recorded_step:-empty}" >&2
    fi

    discovered_step="$(list_usable_steps "$checkpoints_dir" | tail -n 1)"
    [[ "$discovered_step" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$discovered_step"
}

TOTAL_RUNS=0
SUCCEEDED_RUNS=0
FAILED_RUNS=0
SKIPPED_DIRS=0
SKIPPED_EVALUATED=0

# 匹配 eval_summary.txt 中新协议指标行: "AIME24    : avg@16=0.3125 pass@16=0.6333"
AVG_PASS_LINE_RE='^[A-Za-z0-9-]+[[:space:]]*:[[:space:]]*avg@16=[0-9.]+[[:space:]]+pass@16=[0-9.]+'

# 定位某个 run 最新生成的 eval_summary.txt（legacy seeds-* 布局或 canonical 布局）。
latest_eval_summary() {
    local run_dir="$1"
    local results_dir="${EVAL_RESULTS_DIR:-$run_dir/eval_results}"
    local candidate=""
    if compgen -G "$results_dir/seeds-*/eval_summary.txt" > /dev/null; then
        candidate="$(ls -t "$results_dir"/seeds-*/eval_summary.txt | head -n 1)"
    elif [[ -f "$results_dir/eval_summary.txt" ]]; then
        candidate="$results_dir/eval_summary.txt"
    fi
    [[ -n "$candidate" ]] || return 1
    printf '%s\n' "$candidate"
}

# 打印某个 run 已保存的 Avg@16/Pass@16 指标（无新协议指标时不输出）。
print_run_metrics() {
    local summary_file="$1"
    grep -E "$AVG_PASS_LINE_RE" "$summary_file" 2>/dev/null | sed 's/^/    /' || true
}

# 定位某个 run 里某个 step 的评测结果目录，与 eval.sh 的 MERGE_NAME /
# eval_run_short 规则保持一致：canonical 布局直接位于 eval_results/ 下；
# legacy 布局位于 eval_results/seeds-*/<run_short>-step_%04d/ 下。
step_result_dirs() {
    local run_dir="$1" step="$2"
    local results_dir step_name run_short raw_short dir
    results_dir="${EVAL_RESULTS_DIR:-$run_dir/eval_results}"
    step_name="$(printf 'step_%04d' "$step")"
    if [[ "${EVAL_ARTIFACT_LAYOUT:-legacy}" == "canonical" ]]; then
        [[ -d "$results_dir/$step_name" ]] || return 0
        printf '%s\n' "$results_dir/$step_name"
        return 0
    fi
    raw_short="$(basename "$run_dir")"
    if [[ -n "${PIPELINE_TAG:-}" && "$PIPELINE_TAG" != "$raw_short" ]]; then
        run_short="${PIPELINE_TAG}-${raw_short}"
    else
        run_short="${PIPELINE_TAG:-$raw_short}"
    fi
    run_short="$(printf '%s' "$run_short" | tr -cs 'A-Za-z0-9._-' '-')"
    for dir in "$results_dir"/seeds-*/"${run_short}-${step_name}"; do
        [[ -d "$dir" ]] || continue
        printf '%s\n' "$dir"
    done
}

# 该 step 是否已有 Avg@16/Pass@16 协议的打分结果。
step_evaluated() {
    local run_dir="$1" step="$2" dir
    while IFS= read -r dir; do
        [[ -n "$dir" ]] || continue
        [[ -f "$dir/grading_results.json" ]] || continue
        [[ -f "$dir/benchmark_summary.txt" ]] || continue
        if grep -qE "$AVG_PASS_LINE_RE" "$dir/benchmark_summary.txt" 2>/dev/null; then
            return 0
        fi
    done < <(step_result_dirs "$run_dir" "$step")
    return 1
}

echo "Recursive evaluation root: $SEARCH_ROOT"
echo "Evaluation GPUs: ${GPU_ID_LIST[*]}"
echo "Evaluation script: $EVAL_SCRIPT"
echo "Protocol: Avg@16/Pass@16 · 每个 run 的全部 checkpoint 都会被评测"

while IFS= read -r -d '' checkpoints_dir; do
    run_dir="${checkpoints_dir%/checkpoints}"

    mapfile -t run_steps < <(list_usable_steps "$checkpoints_dir")
    if [[ ${#run_steps[@]} -eq 0 ]]; then
        SKIPPED_DIRS=$((SKIPPED_DIRS + 1))
        echo "[SKIP] No usable actor checkpoint: $run_dir"
        continue
    fi
    final_step="$(latest_usable_step "$checkpoints_dir" || true)"

    # 已有新协议 (Avg@16/Pass@16) 评测结果的 checkpoint 逐个跳过：判断条件是
    # 该 step 的 benchmark_summary.txt 中存在 avg@16=/pass@16= 指标行
    # (eval.sh 打分后写入 <results>/<merge_name>/benchmark_summary.txt)。
    # 旧版 5-seed mean±std 结果没有该协议指标行，视为未评测并重新打分。
    pending_steps=()
    evaluated_note=""
    for step in "${run_steps[@]}"; do
        if step_evaluated "$run_dir" "$step"; then
            [[ -z "$evaluated_note" ]] || evaluated_note+=", "
            evaluated_note+="$step"
        else
            pending_steps+=("$step")
        fi
    done

    if [[ ${#pending_steps[@]} -eq 0 ]]; then
        SKIPPED_EVALUATED=$((SKIPPED_EVALUATED + 1))
        echo "[SKIP] 所有 ${#run_steps[@]} 个 checkpoint 已有 Avg@16/Pass@16 评测结果，跳过推理: $run_dir"
        if existing_summary="$(latest_eval_summary "$run_dir")"; then
            echo "  已保存指标 ($existing_summary):"
            print_run_metrics "$existing_summary"
        fi
        continue
    fi

    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    run_name="$(basename "$run_dir")"
    echo "==== [$TOTAL_RUNS] Evaluating $run_name ===="
    echo "Run directory: $run_dir"
    echo "Checkpoint steps: ${run_steps[*]} (latest: $final_step)"
    if [[ -n "$evaluated_note" ]]; then
        echo "Already evaluated: $evaluated_note"
    fi
    echo "Pending steps:     ${pending_steps[*]}"

    if [[ "$DRY_RUN" == true ]]; then
        printf 'DRY-RUN: EVAL_GPU_IDS=%q bash %q --run-dir %q\n' \
            "$EVAL_GPU_IDS" "$EVAL_SCRIPT" "$run_dir"
        SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + 1))
        continue
    fi

    # 不传 --step：eval.sh 会按 step 依次处理该 run 的全部 checkpoint，
    # 已完成 step 在其内部幂等跳过（合并不重复、推理不重复、打分不重复）。
    if bash "$EVAL_SCRIPT" --run-dir "$run_dir"; then
        SUCCEEDED_RUNS=$((SUCCEEDED_RUNS + 1))
        # 评测完成后立即打印该 run 的 Avg@16/Pass@16 指标
        if new_summary="$(latest_eval_summary "$run_dir")"; then
            echo "[METRICS] $run_name Avg@16/Pass@16 ($new_summary):"
            print_run_metrics "$new_summary"
        fi
    else
        FAILED_RUNS=$((FAILED_RUNS + 1))
        echo "[ERROR] Evaluation failed: $run_dir (pending steps: ${pending_steps[*]})" >&2
        if [[ "$CONTINUE_ON_ERROR" != true ]]; then
            exit 1
        fi
    fi
done < <(find "$SEARCH_ROOT" -type d -name checkpoints -print0)

echo "==== Recursive evaluation summary ===="
echo "Discovered runs : $TOTAL_RUNS"
echo "Succeeded       : $SUCCEEDED_RUNS"
echo "Failed          : $FAILED_RUNS"
echo "Skipped (fully evaluated): $SKIPPED_EVALUATED"
echo "Skipped dirs    : $SKIPPED_DIRS"

if [[ "$TOTAL_RUNS" -eq 0 && "$SKIPPED_EVALUATED" -eq 0 ]]; then
    echo "ERROR: no usable checkpoints found under $SEARCH_ROOT" >&2
    exit 1
fi
[[ "$FAILED_RUNS" -eq 0 ]]