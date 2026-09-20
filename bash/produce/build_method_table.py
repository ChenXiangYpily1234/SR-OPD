#!/usr/bin/env python3
"""Build the paper-level comparison table across all runs of one teacher->student pair.

Accepts one or more root directories (positional args, or ``--root``) such as
``main`` and ``runs`` and recursively discovers every run inside them. A run is
any directory that directly contains ``eval_results/seeds-*/eval_summary.txt``
and/or ``metrics/step_metrics.csv`` (``artifacts/metrics/`` is also honored),
no matter how deeply it is nested.

For every run:

* accuracy: recomputed from the raw rollout ``*.jsonl`` files under
  ``eval_results/seeds-*/<run>-step_XXXX/`` (the smallest step wins, i.e. the
  epoch-1 checkpoint, when a run was evaluated at multiple epochs):

    - every rollout is re-graded with the rule-based ``grade_answer_verl``
      (sympy math equivalence, the same grader used by ``bash/eval/eval.sh``);
    - each rollout seed is ranked by its macro accuracy (mean over the six
      benchmarks); the main table reports three protocols side by side, all
      re-graded from the raw jsonl: **Avg@16** (mean ± std over every rollout
      seed), **Top-5** and **Top-2** (mean ± std over the top-ranked seeds);
    - every benchmark in those protocols is reported as **mean ± std** of the
      per-seed accuracies (sample std, ddof=1); the Avg. column is the
      mean ± std of the per-seed macro accuracies.

  Results are cached under ``~/.cache/build_method_table_top5/`` (keyed by
  directory path + file sizes/mtimes) so re-running is instant; pass
  ``--no-cache`` to disable.

* cost: aggregates ``step_metrics.csv`` (``artifacts/metrics/`` preferred over
  ``metrics/``) with per-step de-duplication, restricted to the steps of the
  selected epoch-1 checkpoint (``step <= step of the chosen eval directory``,
  so cost and accuracy describe the same model):
    - Teacher Queries = sum(teacher_queried_rollouts)  # rollouts actually scored by the teacher
    - Teacher Tokens  = sum(teacher_input_tokens)      # tokens actually processed by the teacher forward pass
    - Teacher GPU-h   = sum(teacher_gpu_hours)
    - E2E GPU-h       = sum(e2e_gpu_hours)

Outputs Markdown tables:
Method | AIME24 | ... | MATH-500 | Avg@16 | Top-5 Avg | Top-2 Avg | Teacher Queries | Teacher Tokens | Teacher GPU-h | E2E GPU-h   (main table: three Avg protocols side by side, all re-graded from raw jsonl)
Method | AIME24 | ... | MATH-500 | Pass@16 | Pass Top-5 | Pass Top-2                                                                               (union-pass, computed from raw jsonl)

The display label of a run is simply its directory name. To customize a label
without touching this script, drop a flat JSON object into ``labels.json``
inside a scanned root (picked up automatically) and/or pass ``--label-map``.
Keys are either ``"run-dir-name"`` or ``"method/run-dir-name"`` (the latter
wins on ties), e.g.::

    {"boundary-contrast-full-m8": "FF-OPD (boundary-contrast, m=8)"}

Example:
     python bash/produce/build_method_table.py main runs runs_tlr_bon8_3epoch --output method_table.md
   python bash/produce/build_method_table.py runs --label-map my_labels.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

BENCHMARKS = ("AIME24", "AIME25", "AMC23", "HMMT24", "HMMT25", "MATH-500")

TOP_K_SEEDS = 5
TOP_1_SEEDS = 1
TOP_2_SEEDS = 2

# Known method directory names, used to infer the method from a run's
# relative path no matter how deeply the run is nested (only needed to match
# "method/run-dir-name" keys in a label map).
KNOWN_METHODS: tuple[str, ...] = (
    "teacher",
    "base",
    "full_opd",
    "ta_opd",
    "tlr_opd",
    "ff_opd",
    "sr_opd",
    "strategy",
)

COST_COLUMNS = (
    "teacher_queried_rollouts",
    "teacher_input_tokens",
    "teacher_scored_tokens",
    "final_supervised_tokens",
    "teacher_gpu_hours",
    "e2e_gpu_hours",
)

# --------------------------------------------------------------------------- #
#                    Top-5 seed accuracy from raw rollout jsonl               #
# --------------------------------------------------------------------------- #

EVAL_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "val" / "eval"
if EVAL_SCRIPTS_DIR.is_dir() and str(EVAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_SCRIPTS_DIR))

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "build_method_table_top5"

_STEP_SUFFIX = re.compile(r"step_(\d+)$")

_grade_answer_verl = None


def _load_grader():
    """Lazily import the rule-based grader used by bash/eval/eval.sh."""
    global _grade_answer_verl
    if _grade_answer_verl is None:
        try:
            from utils import grade_answer_verl as fn
        except Exception as exc:  # pragma: no cover - environment issue
            raise RuntimeError(
                f"无法导入 grade_answer_verl（搜索路径 {EVAL_SCRIPTS_DIR}）：{exc}"
            )
        _grade_answer_verl = fn
    return _grade_answer_verl


def find_eval_step_dir(run_dir: Path) -> Path | None:
    """Pick the jsonl step directory with the smallest step for a run.

    Runs evaluated at multiple epochs (e.g. step_0280 and step_0560) use the
    epoch-1 checkpoint only. Any ``eval_results/seeds-*/<name>-step_XXXX/``
    directory containing at least one ``*.jsonl`` qualifies.
    """
    eval_root = run_dir / "eval_results"
    if not eval_root.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for seed_dir in eval_root.glob("seeds-*/"):
        if not seed_dir.is_dir():
            continue
        for sub in seed_dir.iterdir():
            if not sub.is_dir() or not any(sub.glob("*.jsonl")):
                continue
            match = _STEP_SUFFIX.search(sub.name)
            step = int(match.group(1)) if match else -1
            if best is None or step < best[0]:
                best = (step, sub)
    return best[1] if best else None


def grade_eval_dir(eval_dir: Path, use_cache: bool = True) -> dict | None:
    """Grade every rollout in ``eval_dir`` -> ``{bench: {example_id: {seed: bool}}}``.

    Results are cached on disk keyed by the directory path plus every jsonl's
    size/mtime, so re-running the script skips the (slow) sympy grading.
    """
    files = sorted(eval_dir.glob("*.jsonl"))
    if not files:
        return None
    key = [[f.name, f.stat().st_size, f.stat().st_mtime_ns] for f in files]
    cache_path = CACHE_DIR / (hashlib.sha1(str(eval_dir.resolve()).encode()).hexdigest() + ".json")
    if use_cache and cache_path.is_file():
        try:
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            if blob.get("key") == key:
                return blob["grades"]
        except Exception:
            pass  # corrupt cache: fall through and recompute

    grade = _load_grader()
    grades: dict[str, dict[str, dict[str, bool]]] = {}
    for f in files:
        bench = f.name.split("_t0.7")[0].upper()
        if bench not in BENCHMARKS:
            continue
        bench_g = grades.setdefault(bench, {})
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ok = bool(grade(str(r.get("response", "")), str(r.get("answer", ""))))
                bench_g.setdefault(str(r["example_id"]), {})[str(r["rollout_seed"])] = ok

    if use_cache:
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(f".tmp-{os.getpid()}")
            tmp.write_text(json.dumps({"key": key, "grades": grades}), encoding="utf-8")
            tmp.replace(cache_path)
        except Exception:
            pass  # caching is best-effort
    return grades


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def top5_seed_stats(grades: dict | None) -> dict | None:
    """Select top-K rollout seeds by macro accuracy; report mean ± std.

    Returns ``{seeds, n_seeds, per_bench: {bench: (mean, std)}, avg: (mean, std),
    union_pass: {bench: float}, top1: {...}, top2: {...}, all16: {...}}`` or
    ``None`` when data is insufficient. ``all16`` is the same mean ± std
    structure computed over every rollout seed (the Avg@16 protocol), computed
    directly from the raw jsonl just like the top-K protocols.
    """
    if not grades or not all(b in grades and grades[b] for b in BENCHMARKS):
        return None
    seeds = sorted({s for b in grades.values() for ex in b.values() for s in ex}, key=int)
    if not seeds:
        return None

    acc: dict[str, dict[str, float]] = {s: {} for s in seeds}
    for s in seeds:
        for bench in BENCHMARKS:
            vals = [ex[s] for ex in grades[bench].values() if s in ex]
            acc[s][bench] = sum(vals) / len(vals) if vals else 0.0

    def macro(s: str) -> float:
        return sum(acc[s][b] for b in BENCHMARKS) / len(BENCHMARKS)

    ranked = sorted(seeds, key=lambda s: -macro(s))

    def top_k_stats(k: int) -> dict | None:
        if len(ranked) < k:
            return None
        top = ranked[:k]
        per_bench = {bench: _mean_std([acc[s][bench] for s in top]) for bench in BENCHMARKS}
        avg = _mean_std([macro(s) for s in top])
        union_pass: dict[str, float] = {}
        for bench in BENCHMARKS:
            solved = [any(ex.get(s, False) for s in top) for ex in grades[bench].values()]
            union_pass[bench] = sum(solved) / len(solved) if solved else 0.0
        return {
            "seeds": [int(s) for s in top],
            "per_bench": per_bench,
            "avg": avg,
            "union_pass": union_pass,
        }

    top = ranked[:TOP_K_SEEDS]

    return {
        "seeds": [int(s) for s in top],
        "n_seeds": len(seeds),
        "per_bench": {bench: _mean_std([acc[s][bench] for s in top]) for bench in BENCHMARKS},
        "avg": _mean_std([macro(s) for s in top]),
        "union_pass": {
            bench: (
                sum(any(ex.get(s, False) for s in top) for ex in grades[bench].values())
                / len(grades[bench].values())
            )
            if grades[bench]
            else 0.0
            for bench in BENCHMARKS
        },
        "top1": top_k_stats(TOP_1_SEEDS),
        "top2": top_k_stats(TOP_2_SEEDS),
        "all16": top_k_stats(len(seeds)),
    }


# --------------------------------------------------------------------------- #
#                              Argument parsing                               #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate accuracy and cost metrics across all runs into one Markdown table. "
        "Runs are discovered recursively under every given root directory."
    )
    parser.add_argument(
        "roots",
        type=Path,
        nargs="*",
        help="One or more root directories to scan recursively (e.g. main runs). "
        "Defaults to runs/Qwen3-4B__to__Qwen3-1.7B when omitted.",
    )
    parser.add_argument(
        "--root",
        dest="opt_roots",
        type=Path,
        nargs="+",
        help="Same as the positional roots (kept for backward compatibility).",
    )
    parser.add_argument("--output", type=Path, help="Write the Markdown report to this file.")
    parser.add_argument(
        "--label-map",
        type=Path,
        help="JSON file mapping run directory names (or 'method/run-name') to "
        "display labels. A 'labels.json' inside each root is picked up "
        "automatically; this explicit file takes precedence on key conflicts.",
    )
    parser.add_argument(
        "--token-column",
        choices=("teacher_input_tokens", "teacher_scored_tokens", "final_supervised_tokens"),
        default="teacher_input_tokens",
        help="CSV column reported as 'Teacher Tokens' in the main table.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute rollout grading even when a cached result exists.",
    )
    return parser.parse_args()


def read_label_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Label map must be a flat JSON object: {path}")
    return {str(key): str(value) for key, value in data.items()}


def load_label_map(explicit: Path | None, roots: list[Path]) -> dict[str, str]:
    """Merge labels.json found under each root, then the explicit file on top."""
    mapping: dict[str, str] = {}
    for root in roots:
        candidate = root / "labels.json"
        if candidate.is_file():
            mapping.update(read_label_map(candidate))
    if explicit is not None:
        mapping.update(read_label_map(explicit))
    return mapping


def label_for(method: str, run_name: str, label_map: dict[str, str] | None = None) -> str:
    """Display label for a run: the label-map entry if one names the run
    (exact ``method/run`` key first, then the bare directory name), otherwise
    the run directory name itself."""
    key = f"{method}/{run_name}"
    if label_map:
        if key in label_map:
            return label_map[key]
        if run_name in label_map:
            return label_map[run_name]
    return run_name


def aggregate_cost(path: Path, max_step: int | None = None) -> dict[str, float]:
    """Sum cost columns; de-duplicate steps keeping the last row (resumed runs).

    When ``max_step`` is given, only rows with ``step <= max_step`` are
    accumulated (epoch-1 accounting aligned with the selected checkpoint).
    """
    by_step: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_step = (row.get("step") or "").strip()
            if not raw_step:
                continue
            step = int(float(raw_step))
            if max_step is not None and step > max_step:
                continue
            record: dict[str, float] = {}
            for column in COST_COLUMNS:
                value = (row.get(column) or "").strip()
                record[column] = float(value) if value else 0.0
            by_step[step] = record
    totals = {column: 0.0 for column in COST_COLUMNS}
    for record in by_step.values():
        for column in COST_COLUMNS:
            totals[column] += record[column]
    totals["num_steps"] = float(len(by_step))
    return totals


def infer_method(root: Path, run_dir: Path) -> str:
    """Infer the method name from the run's path relative to the scanned root."""
    try:
        relative = run_dir.relative_to(root)
    except ValueError:
        return run_dir.name
    for part in relative.parts:
        if part in KNOWN_METHODS:
            return part
    return relative.parts[0] if relative.parts else run_dir.name


def is_skipped_run(root: Path, run_dir: Path) -> bool:
    """Skip runs living under bookkeeping dirs such as ``_shared``/``_suites``."""
    try:
        relative = run_dir.relative_to(root)
    except ValueError:
        return True
    return any(part.startswith("_") for part in relative.parts)


def discover_runs(root: Path) -> dict[Path, dict]:
    """Recursively find every run directory under ``root``.

    A directory counts as a run when it directly contains
    ``eval_results/seeds-*/eval_summary.txt`` and/or ``metrics/step_metrics.csv``
    (``artifacts/metrics/step_metrics.csv`` is recognized as well).
    """
    runs: dict[Path, dict] = {}

    for summary in root.rglob("eval_results/seeds-*/eval_summary.txt"):
        run_dir = summary.parents[2]
        if is_skipped_run(root, run_dir):
            continue
        entry = runs.setdefault(run_dir, {"summaries": [], "metrics": None})
        entry["summaries"].append(summary)

    for metrics in root.rglob("metrics/step_metrics.csv"):
        if metrics.parent.parent.name == "artifacts":
            run_dir = metrics.parents[2]
        else:
            run_dir = metrics.parents[1]
        if is_skipped_run(root, run_dir):
            continue
        entry = runs.setdefault(run_dir, {"summaries": [], "metrics": None})
        current = entry["metrics"]
        # Prefer artifacts/metrics/ over metrics/ when both exist for one run.
        if current is None or (
            current.parent.parent.name != "artifacts" and metrics.parent.parent.name == "artifacts"
        ):
            entry["metrics"] = metrics

    return runs


def collect(roots: list[Path], label_map: dict[str, str] | None = None, use_cache: bool = True) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        for run_dir, found in sorted(discover_runs(root).items()):
            candidates = sorted(
                found["summaries"], key=lambda path: path.stat().st_mtime
            )
            summary_path = candidates[-1] if candidates else None
            metrics_path = found["metrics"]
            method = infer_method(root, run_dir)

            step_dir = find_eval_step_dir(run_dir)
            top5 = None
            max_step: int | None = None
            if step_dir is not None:
                print(f"[grade] {run_dir.name} -> {step_dir.name}", file=sys.stderr, flush=True)
                top5 = top5_seed_stats(grade_eval_dir(step_dir, use_cache=use_cache))
                step_match = _STEP_SUFFIX.search(step_dir.name)
                if step_match:
                    max_step = int(step_match.group(1))

            row: dict = {
                "label": label_for(method, run_dir.name, label_map),
                "method": method,
                "run": run_dir.name,
                "path": str(run_dir),
                "root": str(root),
                "root_name": root.name,
                "summary_path": str(summary_path) if summary_path else None,
                "metrics_path": str(metrics_path) if metrics_path else None,
                "cost": aggregate_cost(metrics_path, max_step=max_step) if metrics_path else None,
                "top5": top5,
                "step_dir": str(step_dir) if step_dir else None,
            }
            rows.append(row)
    rows.sort(key=lambda item: (item["label"], item["path"]))
    return rows


def fmt_pct(value: float | None) -> str:
    return f"{value * 100:.2f}" if value is not None else "—"


def _mean_pass(union_pass: dict[str, float] | None) -> float | None:
    """Macro mean of per-benchmark union-pass fractions (0-1) over the six benchmarks."""
    if not union_pass:
        return None
    vals = [union_pass[b] for b in BENCHMARKS if b in union_pass]
    return sum(vals) / len(vals) if len(vals) == len(BENCHMARKS) else None


def fmt_mean_std(ms: tuple[float, float] | None) -> str:
    """``(mean, std)`` fractions -> ``"28.00 ± 1.20"`` (percent)."""
    if ms is None:
        return "—"
    mean, std = ms
    return f"{mean * 100:.2f} ± {std * 100:.2f}"


def fmt_tokens(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    return f"{value / 1e6:.1f}M"


def fmt_queries(value: float | None) -> str:
    """Teacher query count: ``—`` when the run never recorded selective queries
    (full/TA score every rollout, so no per-rollout query count; TLR-OPD
    selects one BoN-4 trajectory per prompt for the Teacher and does record a
    count)."""
    if value is None or value <= 0:
        return "—"
    if value >= 1e6:
        return f"{value / 1e6:.2f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def fmt_hours(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


def render_markdown(rows: list[dict], roots: list[Path], token_column: str) -> str:
    lines: list[str] = []
    lines.append(f"# 实验结果汇总：{' + '.join(root.name for root in roots)}")
    lines.append("")
    lines.append(
        "- 主表三个 Avg 指标并排：**Avg@16**、**Top-5 Avg**、**Top-2 Avg**；"
        "主表每个 benchmark 单元格内三行依次为 `@16`（全部 16 个 seed 正确率的均值 ± 样本标准差，直接由 rollout jsonl 重算）、"
        "`T5`（前 5 个 seed 同口径）、`T2`（前 2 个 seed 同口径）；"
    )
    lines.append(
        "- 精度协议：每道题在一次评测中随机采样 16 条 response"
        "（base seed 42，第 j 条 rollout 的采样 seed 为 42+j，即 42..57），"
        "逐 rollout 重新打分后，按单个 rollout seed 的 6-benchmark 宏平均正确率排名取前 K 个 seed；"
        "Top-5 / Top-2 口径下每个 benchmark 报告这 K 个 seed 正确率的均值 ± 标准差（样本标准差）；"
        "Avg@16 口径为全部 16 个 seed 正确率的均值 ± 标准差（与 Top-K 同口径，直接由 rollout jsonl 重算）；"
    )
    lines.append(
        "- 打分：规则匹配（`grade_answer_verl`，sympy 数学等价），与 eval.sh 一致；"
        "老 5-seed 协议（n1）的 run 取全部 5 个 seed 计算同一口径；"
    )
    lines.append(
        "- Pass 表三个指标并排：**Pass@16**、**Pass Top-5**、**Pass Top-2**，"
        "均直接由 rollout jsonl 计算（该 seed 组内至少 1 条正确的题目占比，即 union-pass）；"
    )
    lines.append(
        "- Teacher Queries：训练全程 Teacher 实际评分的 rollout 条数 "
        "(`teacher_queried_rollouts` 求和)；full/TA 对全部 rollout 评分、"
        "不按条数计量，记为 —；TLR-OPD 每 prompt 从 BoN-4 中选 1 条送 Teacher、"
        "按选中条数计量；"
    )
    lines.append(
        "- Teacher Tokens：训练全程 Teacher 前向处理的输入 token 总量 "
        f"(`{token_column}` 求和)；"
    )
    lines.append(
        "- 成本口径：与精度对齐到同一 checkpoint——只累计 `step ≤ 所选评测 step` 的行"
        "（多 epoch 的 run 仅计 epoch 1；无评测数据的 run 计全程）；"
    )
    lines.append(
        "- Teacher GPU-h / E2E GPU-h：`step_metrics.csv` 中 "
        "`teacher_gpu_hours` / `e2e_gpu_hours` 逐步求和（8×GPU，含 resume 去重）；"
    )
    lines.append("- Teacher / Student 行为直接评测，无训练成本，记为 —。")
    lines.append("")

    # Main table: three Avg protocols side by side (Avg@16 / Top-5 / Top-2 seed) + training cost
    lines.append("## 主表：Avg@16 / Top-5 Avg / Top-2 Avg 三指标并排（%，benchmark 单元格三行：@16 / T5 / T2）")
    lines.append("")
    header = (
        "| Method | AIME24 | AIME25 | AMC23 | HMMT24 | HMMT25 | MATH-500 "
        "| Avg@16 ↑ | Top-5 Avg ↑ | Top-2 Avg ↑ | Teacher Queries ↓ | Teacher Tokens ↓ | Teacher GPU-h ↓ | E2E GPU-h ↓ |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 14)
    for row in rows:
        top5 = row.get("top5")
        cost = row["cost"]
        cells = [row["label"]]
        for b in BENCHMARKS:
            parts = [
                f"@16 {fmt_mean_std(top5['all16']['per_bench'][b]) if top5 and top5['all16'] else '—'}"
            ]
            parts.append(f"T5 {fmt_mean_std(top5['per_bench'][b]) if top5 else '—'}")
            parts.append(
                f"T2 {fmt_mean_std(top5['top2']['per_bench'][b]) if top5 and top5['top2'] else '—'}"
            )
            cells.append("<br>".join(parts))
        cells.append(fmt_mean_std(top5["all16"]["avg"]) if top5 and top5["all16"] else "—")
        cells.append(fmt_mean_std(top5["avg"]) if top5 else "—")
        cells.append(fmt_mean_std(top5["top2"]["avg"]) if top5 and top5["top2"] else "—")
        cells.append(fmt_queries(cost["teacher_queried_rollouts"]) if cost else "—")
        cells.append(fmt_tokens(cost[token_column]) if cost else "—")
        cells.append(fmt_hours(cost["teacher_gpu_hours"]) if cost else "—")
        cells.append(fmt_hours(cost["e2e_gpu_hours"]) if cost else "—")
        lines.append("| " + " | ".join(cells) + " |")

    # Second table: Pass@16 / Pass Top-5 / Pass Top-2 (union-pass, computed from raw jsonl)
    lines.append("")
    lines.append("## Pass@16 / Pass Top-5 / Pass Top-2 三指标并排（%，benchmark 单元格三行：p16 / pT5 / pT2；直接由 rollout jsonl 计算）")
    lines.append("")
    lines.append(
        "| Method | AIME24 | AIME25 | AMC23 | HMMT24 | HMMT25 | MATH-500 "
        "| Pass@16 ↑ | Pass Top-5 ↑ | Pass Top-2 ↑ |"
    )
    lines.append("|" + "---|" * 10)
    for row in rows:
        top5 = row.get("top5")
        cells = [row["label"]]
        for b in BENCHMARKS:
            p16 = top5["all16"]["union_pass"][b] if top5 and top5["all16"] else None
            p5 = top5["union_pass"][b] if top5 else None
            p2 = top5["top2"]["union_pass"][b] if top5 and top5["top2"] else None
            cells.append(
                "<br>".join(
                    [f"p16 {fmt_pct(p16)}", f"pT5 {fmt_pct(p5)}", f"pT2 {fmt_pct(p2)}"]
                )
            )
        cells.append(
            fmt_pct(_mean_pass(top5["all16"]["union_pass"]) if top5 and top5["all16"] else None)
        )
        cells.append(fmt_pct(_mean_pass(top5["union_pass"]) if top5 else None))
        cells.append(
            fmt_pct(_mean_pass(top5["top2"]["union_pass"]) if top5 and top5["top2"] else None)
        )
        lines.append("| " + " | ".join(cells) + " |")

    # Appendix 1: top-K seed details (selected seeds + per-benchmark mean ± std)
    lines.append("")
    lines.append("## 附录 A：Top-K seed 明细（均值 ± 标准差，小数）")
    lines.append("")
    lines.append(
        "| Method | Top-1 seed | Top-1 Avg | Top-2 seeds | Top-2 Avg "
        "| Top-5 seeds | " + " | ".join(BENCHMARKS) + " | Top-5 Avg |"
    )
    lines.append("|" + "---|" * (len(BENCHMARKS) + 8))
    for row in rows:
        top5 = row.get("top5")
        cells = [row["label"]]
        if not top5:
            cells += ["—"] * (len(BENCHMARKS) + 7)
        else:
            top1 = top5.get("top1")
            top2 = top5.get("top2")
            cells.append(",".join(str(s) for s in top1["seeds"]) if top1 else "—")
            if top1:
                mean, std = top1["avg"]
                cells.append(f"{mean:.4f} ± {std:.4f}")
            else:
                cells.append("—")
            cells.append(",".join(str(s) for s in top2["seeds"]) if top2 else "—")
            if top2:
                mean, std = top2["avg"]
                cells.append(f"{mean:.4f} ± {std:.4f}")
            else:
                cells.append("—")
            cells.append(",".join(str(s) for s in top5["seeds"]))
            for bench in BENCHMARKS:
                mean, std = top5["per_bench"][bench]
                cells.append(f"{mean:.4f} ± {std:.4f}")
            mean, std = top5["avg"]
            cells.append(f"{mean:.4f} ± {std:.4f}")
        lines.append("| " + " | ".join(cells) + " |")

    # Appendix 2: cost details with alternative token accounting
    lines.append("")
    lines.append("## 附录 B：成本明细（epoch 1 求和，step ≤ 所选评测 checkpoint）")
    lines.append("")
    lines.append(
        "| Method | Steps | Teacher Queries | Teacher Input Tokens | Teacher Scored Tokens "
        "| Final Supervised Tokens | Teacher GPU-h | E2E GPU-h |"
    )
    lines.append("|" + "---|" * 8)
    for row in rows:
        cost = row["cost"]
        if cost is None:
            lines.append(f"| {row['label']} | — | — | — | — | — | — | — |")
            continue
        queries = (
            f"{cost['teacher_queried_rollouts']:,.0f}"
            if cost["teacher_queried_rollouts"] > 0
            else "—"
        )
        lines.append(
            "| {label} | {steps} | {queries} | {inp} | {scored} | {sup} | {tgh} | {e2e} |".format(
                label=row["label"],
                steps=int(cost["num_steps"]),
                queries=queries,
                inp=f"{cost['teacher_input_tokens']:,.0f}",
                scored=f"{cost['teacher_scored_tokens']:,.0f}",
                sup=f"{cost['final_supervised_tokens']:,.0f}",
                tgh=f"{cost['teacher_gpu_hours']:.2f}",
                e2e=f"{cost['e2e_gpu_hours']:.2f}",
            )
        )

    # Appendix 3: data sources
    lines.append("")
    lines.append("## 附录 C：数据来源")
    lines.append("")
    lines.append("| Method | Root | Run 目录 | 评测 jsonl 目录 | eval_summary | step_metrics |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        row_root = Path(row["root"])
        step_dir = (
            f"`{Path(row['step_dir']).relative_to(row_root)}`"
            if row.get("step_dir")
            else "—"
        )
        summary = (
            f"`{Path(row['summary_path']).relative_to(row_root)}`"
            if row["summary_path"]
            else "—"
        )
        metrics = (
            f"`{Path(row['metrics_path']).relative_to(row_root)}`"
            if row["metrics_path"]
            else "—"
        )
        run_rel = Path(row["path"]).relative_to(row_root)
        lines.append(
            f"| {row['label']} | {row['root_name']} | `{run_rel}` | {step_dir} | {summary} | {metrics} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    roots = args.opt_roots or args.roots or [Path("runs/Qwen3-4B__to__Qwen3-1.7B")]
    label_map = load_label_map(args.label_map, roots)
    rows = collect(roots, label_map, use_cache=not args.no_cache)
    markdown = render_markdown(rows, roots, args.token_column)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(markdown)


if __name__ == "__main__":
    main()
