"""Integrity checks shared by the evaluation launcher and generator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REQUIRED_DATASET_COLUMNS = {"prompt", "reward_model"}


def load_eval_samples(path: str | Path) -> list[dict]:
    """Load the canonical VERL evaluation schema without formatting prompts."""
    frame = pd.read_parquet(path)
    missing = REQUIRED_DATASET_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    samples = []
    for index in range(len(frame)):
        prompt = frame.at[index, "prompt"]
        reward_model = frame.at[index, "reward_model"]
        try:
            prompt_text = prompt[0]["content"].strip()
            answer = reward_model["ground_truth"].strip()
        except (IndexError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{path}: invalid VERL schema at row {index}") from exc
        samples.append({"example_id": index, "prompt": prompt_text, "answer": answer})
    if not samples:
        raise ValueError(f"{path}: dataset is empty")
    return samples


def model_directory_errors(model_dir: str | Path) -> list[str]:
    """Return missing/corrupt Hugging Face model artifacts."""
    model_dir = Path(model_dir)
    errors = []
    if not model_dir.is_dir():
        return [f"model directory does not exist: {model_dir}"]
    if not (model_dir / "config.json").is_file():
        errors.append("missing config.json")
    if not any((model_dir / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        errors.append("missing tokenizer.json or tokenizer.model")

    single_weight = model_dir / "model.safetensors"
    index_path = model_dir / "model.safetensors.index.json"
    if single_weight.is_file() and single_weight.stat().st_size > 0:
        return errors
    if not index_path.is_file():
        errors.append("missing model.safetensors or model.safetensors.index.json")
        return errors
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(f"invalid model.safetensors.index.json: {exc}")
        return errors
    if not shard_names:
        errors.append("model.safetensors.index.json contains no shards")
    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file() or shard_path.stat().st_size == 0:
            errors.append(f"missing or empty referenced shard: {shard_name}")
    return errors


def validate_jsonl_output(
    path: str | Path,
    task_name: str,
    samples: Sequence[dict],
    rollout_ids: Iterable[int],
    eval_seed: int,
) -> tuple[bool, str]:
    """Validate a completed task output before resume or grading."""
    path = Path(path)
    if not path.is_file():
        return False, "file does not exist"
    if not path.name.lower().startswith(f"{task_name.lower()}_"):
        return False, f"filename does not match task {task_name}"
    expected_samples = {int(sample["example_id"]): sample for sample in samples}
    expected_keys = {
        (example_id, int(rollout_id))
        for example_id in expected_samples
        for rollout_id in rollout_ids
    }
    observed_keys = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    return False, f"blank line at {line_number}"
                item = json.loads(line)
                required = {
                    "example_id", "prompt", "answer", "eval_seed",
                    "rollout_id", "rollout_seed", "student_enable_thinking", "response",
                }
                missing = required - set(item)
                if missing:
                    return False, f"line {line_number} missing fields {sorted(missing)}"
                if item.get("task_name", task_name) != task_name:
                    return False, f"line {line_number} has wrong task_name"
                example_id = int(item["example_id"])
                rollout_id = int(item["rollout_id"])
                if example_id not in expected_samples:
                    return False, f"line {line_number} has unknown example_id {example_id}"
                if int(item["eval_seed"]) != int(eval_seed):
                    return False, f"line {line_number} has wrong eval_seed"
                if int(item["rollout_seed"]) != int(eval_seed) + rollout_id:
                    return False, f"line {line_number} has wrong rollout_seed"
                if item["student_enable_thinking"] is not False:
                    return False, f"line {line_number} enabled Student thinking"
                sample = expected_samples[example_id]
                if item["answer"] != sample["answer"]:
                    return False, f"line {line_number} has wrong answer metadata"
                if not isinstance(item["response"], str):
                    return False, f"line {line_number} response is not a string"
                observed_keys.append((example_id, rollout_id))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return False, f"cannot parse output: {exc}"

    observed_set = set(observed_keys)
    if len(observed_keys) != len(observed_set):
        return False, "duplicate example_id/rollout_id pairs"
    if observed_set != expected_keys:
        return False, f"expected {len(expected_keys)} generations, found {len(observed_set)}"
    return True, "complete"


def validate_benchmark_seed_coverage(
    results: Sequence[dict], task_names: Sequence[str], eval_seeds: Iterable[int]
) -> None:
    """Require exactly one grading result for every task/seed pair."""
    expected = {
        (task_name.lower(), int(seed)) for task_name in task_names for seed in eval_seeds
    }
    observed = []
    for result in results:
        hyperparameters = result.get("hyperparameters", {})
        observed.append(
            (
                str(hyperparameters.get("task_name", "")).lower(),
                int(hyperparameters["eval_seed"]),
            )
        )
    if len(observed) != len(set(observed)):
        raise RuntimeError("duplicate task/seed grading results")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        unexpected = sorted(observed_set - expected)
        raise RuntimeError(
            f"incomplete grading coverage: missing={missing}, unexpected={unexpected}"
        )


def _require_imports(module_names: Sequence[str]) -> None:
    missing = [name for name in module_names if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f"missing Python modules: {', '.join(missing)}")


def preflight(
    dataset_paths: Sequence[str],
    base_model: str,
    gpu_ids: Sequence[str],
    allow_gpu_reuse: bool = False,
) -> None:
    _require_imports(("pandas", "pyarrow", "torch", "transformers", "vllm"))
    for dataset_path in dataset_paths:
        load_eval_samples(dataset_path)
    errors = model_directory_errors(base_model)
    if errors:
        raise RuntimeError(f"base model is incomplete: {'; '.join(errors)}")
    if not gpu_ids or (not allow_gpu_reuse and len(gpu_ids) != len(set(gpu_ids))):
        qualifier = "non-empty" if allow_gpu_reuse else "non-empty and unique"
        raise RuntimeError(f"GPU IDs must be {qualifier}: {list(gpu_ids)}")
    if not all(gpu_id.isdigit() for gpu_id in gpu_ids):
        raise RuntimeError(f"GPU IDs must be non-negative integer indices: {list(gpu_ids)}")
    import torch

    visible_count = torch.cuda.device_count()
    visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(",", " ").split()
    if visibility and visibility != ["-1"]:
        invalid = [gpu_id for gpu_id in gpu_ids if gpu_id not in visibility]
        visibility_description = f"CUDA_VISIBLE_DEVICES={visibility}"
    else:
        invalid = [gpu_id for gpu_id in gpu_ids if int(gpu_id) >= visible_count]
        visibility_description = f"torch sees {visible_count} CUDA devices"
    if invalid:
        raise RuntimeError(
            f"GPU IDs {invalid} are invalid; {visibility_description}"
        )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    model_parser = subparsers.add_parser("validate-model")
    model_parser.add_argument("model_dir")

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--dataset", action="append", required=True)
    preflight_parser.add_argument("--base-model", required=True)
    preflight_parser.add_argument("--gpu-id", action="append", required=True)
    preflight_parser.add_argument("--allow-gpu-reuse", action="store_true")

    output_parser = subparsers.add_parser("validate-output")
    output_parser.add_argument("--path", required=True)
    output_parser.add_argument("--task", required=True)
    output_parser.add_argument("--dataset", required=True)
    output_parser.add_argument("--n", type=int, required=True)
    output_parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()
    if args.command == "validate-model":
        errors = model_directory_errors(args.model_dir)
        if errors:
            raise SystemExit("; ".join(errors))
    elif args.command == "preflight":
        preflight(args.dataset, args.base_model, args.gpu_id, args.allow_gpu_reuse)
    else:
        samples = load_eval_samples(args.dataset)
        valid, reason = validate_jsonl_output(
            args.path, args.task, samples, range(args.n), args.seed
        )
        if not valid:
            raise SystemExit(reason)


if __name__ == "__main__":
    _main()
