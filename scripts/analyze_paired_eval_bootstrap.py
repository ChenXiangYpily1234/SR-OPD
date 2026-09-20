#!/usr/bin/env python3
"""Stratified paired bootstrap between two OPD runs' evaluations.

Pairs the two methods by (benchmark, example_id, rollout_seed) — the decode
replication index is ``rollout_seed`` (42..57 under the current one-eval,
16-rollouts-per-problem protocol), NOT ``eval_seed``.

Bootstrap (preregistered):
1. resample questions with replacement within each benchmark;
2. within each question, resample the paired rollout seeds with replacement;
3. compute the per-benchmark paired difference;
4. equal-weight macro average over the six benchmarks;
5. report Delta(A - B), its 95% CI, and P(Delta > 0).

Inputs are the raw rollout jsonl trees each eval produces under
``eval_results/seeds-*/<run>-step_*/``. Rollouts are re-graded with the same
rule-based grader as ``bash/eval/eval.sh`` (``grade_answer_verl``, sympy
equivalence), so no old ``eval_seed``-protocol data or xiang mount is needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "verl" / "val" / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "val" / "eval"))
import grade  # noqa: E402

TASKS = ["aime24", "aime25", "amc23", "hmmt24", "hmmt25", "math-500"]


def find_eval_dir(run_dir: Path) -> Path:
    """Locate the single <run>-step_* dir holding the rollout jsonl files."""
    candidates = sorted(
        (d for d in (run_dir / "eval_results").glob("seeds-*/*") if d.is_dir()),
        key=lambda d: d.name,
    )
    if not candidates:
        raise SystemExit(f"no eval_results/seeds-*/<run>-step_* under {run_dir}")
    if len(candidates) > 1:
        names = ", ".join(d.name for d in candidates)
        raise SystemExit(f"multiple eval dirs under {run_dir}: {names} — pass the one to use")
    return candidates[0]


def load_per_rollout(eval_dir: Path) -> dict[str, dict[str, dict[int, bool]]]:
    """Grade every rollout: {task: {example_id: {rollout_seed: correct}}}."""
    per_task: dict[str, dict[str, dict[int, bool]]] = {}
    for task in TASKS:
        files = sorted(eval_dir.glob(f"{task}_*.jsonl"))
        if not files:
            raise SystemExit(f"{task}: no rollout jsonl under {eval_dir}")
        if len(files) > 1:
            raise SystemExit(f"{task}: multiple jsonl files {files} — ambiguous")
        records: dict[str, dict[int, bool]] = defaultdict(dict)
        with files[0].open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                example_id = str(row["example_id"])
                seed = int(row["rollout_seed"])
                correct = bool(grade.grade_answer_verl(str(row["response"]), row["answer"]))
                if seed in records[example_id]:
                    raise SystemExit(
                        f"{task}/{example_id}: duplicate rollout_seed {seed} — pairing key is not unique"
                    )
                records[example_id][seed] = correct
        per_task[task] = dict(records)
    return per_task


def check_pairing(left: dict, right: dict, left_name: str, right_name: str) -> None:
    for task in TASKS:
        left_ids, right_ids = set(left[task]), set(right[task])
        if left_ids != right_ids:
            raise SystemExit(
                f"{task}: question sets differ ({len(left_ids)} vs {len(right_ids)}); "
                f"example: only-{left_name}={sorted(left_ids - right_ids)[:2]} "
                f"only-{right_name}={sorted(right_ids - left_ids)[:2]}"
            )
        for example_id in left_ids:
            left_seeds, right_seeds = set(left[task][example_id]), set(right[task][example_id])
            if left_seeds != right_seeds:
                raise SystemExit(
                    f"{task}/{example_id}: rollout_seed sets differ "
                    f"({sorted(left_seeds)} vs {sorted(right_seeds)}) — cannot pair"
                )


def macro_difference(left: dict, right: dict, resampled: dict[str, list[tuple[str, int]]]) -> float:
    """Delta(macro avg of A) - Delta(macro avg of B) over the resampled data."""
    task_deltas = []
    for task in TASKS:
        numerator = 0.0
        denominator = 0
        for example_id, seeds in resampled[task]:
            left_hits = sum(1 for s in seeds if left[task][example_id][s])
            right_hits = sum(1 for s in seeds if right[task][example_id][s])
            numerator += (left_hits - right_hits) / len(seeds)
            denominator += 1
        task_deltas.append(numerator / denominator)
    return sum(task_deltas) / len(TASKS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path, help="run dir of method A (e.g. DTW-PDA)")
    parser.add_argument("run_b", type=Path, help="run dir of method B (e.g. Relative PDA)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print(f"[load] {args.label_a}: {args.run_a}")
    left = load_per_rollout(find_eval_dir(args.run_a))
    print(f"[load] {args.label_b}: {args.run_b}")
    right = load_per_rollout(find_eval_dir(args.run_b))
    check_pairing(left, right, args.label_a, args.label_b)

    questions = {task: sorted(left[task]) for task in TASKS}
    seeds_of = {
        task: {example_id: sorted(left[task][example_id]) for example_id in questions[task]}
        for task in TASKS
    }

    # Point estimate (no resampling).
    full = {
        task: [(example_id, seeds_of[task][example_id]) for example_id in questions[task]]
        for task in TASKS
    }
    point = macro_difference(left, right, full)

    rng = random.Random(args.seed)
    bootstrap = []
    for _ in range(args.repeats):
        resampled = {}
        for task in TASKS:
            picked = []
            for _ in range(len(questions[task])):
                example_id = rng.choice(questions[task])
                seeds = seeds_of[task][example_id]
                picked.append((example_id, [rng.choice(seeds) for _ in range(len(seeds))]))
            resampled[task] = picked
        bootstrap.append(macro_difference(left, right, resampled))
    bootstrap.sort()

    ci_lo, ci_hi = bootstrap[int(0.025 * args.repeats)], bootstrap[int(0.975 * args.repeats)]
    positive = sum(1 for value in bootstrap if value > 0) / len(bootstrap)

    lines = [
        f"paired bootstrap: {args.label_a} - {args.label_b}",
        f"pairing key: benchmark + example_id + rollout_seed",
        f"repeats={args.repeats}, seed={args.seed}",
        "",
        f"Delta (macro Avg@16, percentage points): {point * 100:+.2f}",
        f"95% CI: [{ci_lo * 100:+.2f}, {ci_hi * 100:+.2f}]",
        f"P(Delta > 0): {positive:.3f}",
        "",
        "per-benchmark point deltas (pp):",
    ]
    for task in TASKS:
        num = sum(
            (sum(left[task][q][s] for s in seeds_of[task][q]) - sum(right[task][q][s] for s in seeds_of[task][q]))
            / len(seeds_of[task][q])
            for q in questions[task]
        )
        lines.append(f"  {task:9s} {num / len(questions[task]) * 100:+.2f}")

    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nwritten: {args.output}")


if __name__ == "__main__":
    main()
