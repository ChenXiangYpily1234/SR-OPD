#!/usr/bin/env python3
"""Combine the existing 5-seed evaluator and cost analyzer outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.search(r"(?:global_step_|step_|-step-?)(\d+)(?:/|$)", str(path))
    return int(match.group(1)) if match else -1


def select_grading_result(run_dir: Path, requested_step: int | None) -> Path:
    evaluation_dir = run_dir / "eval_results"
    candidates = sorted(evaluation_dir.glob("**/grading_results.json"))
    if requested_step is not None:
        matching = [path for path in candidates if checkpoint_step(path) == requested_step]
        if not matching:
            raise FileNotFoundError(
                f"No grading_results.json for checkpoint step {requested_step} under "
                f"{evaluation_dir}"
            )
        candidates = matching
    if not candidates:
        raise FileNotFoundError(f"No grading_results.json found under {evaluation_dir}")
    return max(candidates, key=lambda path: (checkpoint_step(path), path.stat().st_mtime))


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summarize_legacy_results(results: list[dict]) -> tuple[list[int], list[dict]]:
    display_names = {
        "aime24": "AIME24",
        "aime25": "AIME25",
        "amc23": "AMC23",
        "hmmt24": "HMMT24",
        "hmmt25": "HMMT25",
        "math-500": "MATH-500",
        "math500": "MATH-500",
    }
    by_task: dict[str, dict[int, float]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        hyperparameters = result.get("hyperparameters", {})
        task = str(hyperparameters.get("task_name", result.get("task_name", ""))).lower()
        task = task.replace("_", "-")
        if task not in display_names:
            continue
        raw_seed = hyperparameters.get("eval_seed", result.get("eval_seed"))
        if raw_seed in (None, ""):
            raise ValueError(
                "Legacy grading_results.json has no eval_seed. Rerun evaluation without --skip-eval."
            )
        seed = int(raw_seed)
        if seed in by_task.setdefault(task, {}):
            raise ValueError(f"Duplicate {task} evaluation result for seed {seed}")
        by_task[task][seed] = float(result.get("mean_score", 0.0))

    if not by_task:
        raise ValueError("No supported benchmark results found in grading_results.json")
    seeds = sorted({seed for scores in by_task.values() for seed in scores})
    if len(seeds) != 5:
        raise ValueError(f"Expected five evaluation seeds, found {seeds}")

    benchmark_summary = []
    for task in ("aime24", "aime25", "amc23", "hmmt24", "hmmt25", "math-500"):
        scores_by_seed = by_task.get(task, {})
        if sorted(scores_by_seed) != seeds:
            raise ValueError(
                f"{display_names[task]} expected seeds {seeds}, found {sorted(scores_by_seed)}"
            )
        scores = [scores_by_seed[seed] for seed in seeds]
        benchmark_summary.append(
            {
                "benchmark": display_names[task],
                "num_seeds": 5,
                "accuracy_mean": statistics.mean(scores),
                "accuracy_sample_std": statistics.stdev(scores),
                "seed_scores": {str(seed): scores_by_seed[seed] for seed in seeds},
            }
        )
    return seeds, benchmark_summary


def normalize_grading(grading) -> tuple[list[int], list[dict]]:
    if isinstance(grading, list):
        return summarize_legacy_results(grading)
    if not isinstance(grading, dict):
        raise ValueError("grading_results.json must contain a JSON object or list")

    benchmark_summary = grading.get("benchmark_summary")
    if isinstance(benchmark_summary, list) and benchmark_summary:
        if benchmark_summary[0].get("num_samples") is not None or any(
            key.startswith("avg_at_") for key in benchmark_summary[0]
        ):
            raise ValueError(
                "grading_results.json uses the Avg@16/Pass@16 protocol; "
                "summarize_produce.py only supports the legacy 5-seed protocol. "
                "Use bash/produce/build_method_table.py instead."
            )
        seeds = [int(seed) for seed in grading.get("eval_seeds", [])]
        if not seeds:
            seeds = sorted(
                {
                    int(seed)
                    for benchmark in benchmark_summary
                    for seed in benchmark.get("seed_scores", {})
                }
            )
        if len(seeds) != 5:
            raise ValueError(f"Expected five evaluation seeds, found {seeds}")
        return seeds, benchmark_summary

    legacy_results = grading.get("per_seed_results")
    if isinstance(legacy_results, list):
        return summarize_legacy_results(legacy_results)
    raise ValueError("No benchmark results found in grading_results.json")


def compute_macro_summary(benchmarks: list[dict]) -> dict | None:
    by_seed: dict[str, list[float]] = {}
    for benchmark in benchmarks:
        for seed, score in benchmark.get("seed_scores", {}).items():
            by_seed.setdefault(str(seed), []).append(float(score))
    if not by_seed:
        return None
    seed_scores = {seed: statistics.mean(scores) for seed, scores in sorted(by_seed.items())}
    values = list(seed_scores.values())
    return {
        "benchmark": "Macro Average",
        "num_seeds": len(values),
        "accuracy_mean": statistics.mean(values),
        "accuracy_sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "seed_scores": seed_scores,
    }


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "artifacts" / "paper").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Schema v1 produces one canonical artifact. Remove only files emitted by
    # older versions of this same summarizer.
    for legacy_name in ("paper_results.md", "paper_results.csv"):
        legacy_path = output_dir / legacy_name
        if legacy_path.is_file():
            legacy_path.unlink()

    grading_path = select_grading_result(run_dir, args.checkpoint_step)
    cost_path = run_dir / "cost_analysis" / "cost_metrics.json"
    if not cost_path.is_file():
        raise FileNotFoundError(f"Cost result not found: {cost_path}")

    grading = load_json(grading_path)
    cost = load_json(cost_path)
    eval_seeds, benchmarks = normalize_grading(grading)
    macro = compute_macro_summary(benchmarks)
    if macro:
        benchmarks.append(macro)
    paper_cost = cost

    benchmark_results = {
        item["benchmark"]: {
            "mean": item["accuracy_mean"],
            "sample_std": item["accuracy_sample_std"],
            "per_seed": item.get("seed_scores", {}),
        }
        for item in benchmarks
    }
    cost_keys = (
        "student_generated_tokens",
        "teacher_input_tokens",
        "teacher_scored_tokens",
        "final_supervised_tokens",
        "student_generation_time_sec",
        "actor_update_time_sec",
        "teacher_gpu_hours",
        "e2e_wall_time_hours",
        "e2e_gpu_hours",
        "e2e_response_throughput_tokens_per_sec",
    )
    final_cost = {key: paper_cost.get(key) for key in cost_keys}
    final_cost["measurement"] = {
        "teacher_gpu_count": paper_cost.get("measurement", {}).get("teacher_gpu_count"),
        "allocated_gpu_count": paper_cost.get("measurement", {}).get("allocated_gpu_count"),
        "e2e_wall_time_source": paper_cost.get("measurement", {}).get("e2e_wall_time_source"),
        "first_step": paper_cost.get("measurement", {}).get("first_step"),
        "last_step": paper_cost.get("measurement", {}).get("last_step"),
        "expected_last_step": paper_cost.get("measurement", {}).get("expected_last_step"),
        "is_partial_training_log": paper_cost.get("measurement", {}).get("is_partial_training_log"),
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "checkpoint_step": args.checkpoint_step or checkpoint_step(grading_path),
        "evaluation": {
            "num_seeds": 5,
            "seeds": eval_seeds,
            "benchmarks": benchmark_results,
        },
        "cost": final_cost,
        "sources": {
            "evaluation": str(grading_path),
            "cost": str(cost_path),
        },
    }
    temporary_output = output_dir / f".paper_results.json.tmp-{os.getpid()}"
    temporary_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary_output.replace(output_dir / "paper_results.json")
    print(f"Combined paper result written to {output_dir / 'paper_results.json'}")


if __name__ == "__main__":
    main()
