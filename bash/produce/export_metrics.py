#!/usr/bin/env python3
"""Publish the stable CSV artifact contract for one OPD production run."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def read_env(path: Path) -> dict[str, str]:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    env = read_env(args.run_dir / "metadata" / "run.env")
    rows = []
    for benchmark, values in summary.get("evaluation", {}).get("benchmarks", {}).items():
        rows.append(
            {
                "run_id": env.get("RUN_ID", args.run_dir.name),
                "method": env.get("METHOD", env.get("QUERY_METHOD", "unknown")),
                "training_seed": env.get("TRAINING_SEED", ""),
                "checkpoint_step": summary.get("checkpoint_step", ""),
                "benchmark": benchmark,
                "score_mean": values.get("mean", ""),
                "score_sample_std": values.get("sample_std", ""),
                "eval_seeds": json.dumps(values.get("per_seed", {}), sort_keys=True),
            }
        )
    write_csv(
        args.paper_dir / "run_summary.csv",
        [
            "run_id", "method", "training_seed", "checkpoint_step", "benchmark",
            "score_mean", "score_sample_std", "eval_seeds",
        ],
        rows,
    )

    manifest = {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "available": {
            "step_metrics": (args.metrics_dir / "step_metrics.csv").is_file(),
            "validation_metrics": (args.metrics_dir / "validation_metrics.csv").is_file(),
            "communication_time_s": False,
        },
        "unavailable_reasons": {
            "communication_time_s": "requires explicit distributed dispatch instrumentation",
        },
    }
    temporary_manifest = args.metrics_dir / f".metrics_manifest.json.tmp-{os.getpid()}"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(args.metrics_dir / "metrics_manifest.json")

    artifact_manifest = {
        "schema_version": 1,
        "run_dir": str(args.run_dir.resolve()),
        "model": "../eval_models",
        "evaluation": "../eval_results",
        "metrics": "metrics",
        "cost": "../cost_analysis/cost_metrics.json",
        "paper_summary": "paper/run_summary.csv",
        "paper_results": "paper/paper_results.json",
    }
    artifact_root = args.metrics_dir.parent
    temporary_artifact_manifest = artifact_root / f".manifest.json.tmp-{os.getpid()}"
    temporary_artifact_manifest.write_text(
        json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_artifact_manifest.replace(artifact_root / "manifest.json")


if __name__ == "__main__":
    main()
