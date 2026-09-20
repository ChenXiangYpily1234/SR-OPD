import json
from pathlib import Path

import pytest

from eval_integrity import (
    load_eval_samples,
    model_directory_errors,
    validate_benchmark_seed_coverage,
    validate_jsonl_output,
)


def _samples():
    return [
        {"example_id": 0, "prompt": "p0", "answer": "a0"},
        {"example_id": 1, "prompt": "p1", "answer": "a1"},
    ]


def _write_output(path: Path, seed: int = 42, omit_last: bool = False):
    rows = []
    for sample in _samples():
        for rollout_id in range(2):
            rows.append(
                {
                    "task_name": "AIME24",
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "eval_seed": seed,
                    "rollout_id": rollout_id,
                    "rollout_seed": seed + rollout_id,
                    "student_enable_thinking": False,
                    "response": "answer",
                }
            )
    if omit_last:
        rows.pop()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_jsonl_resume_accepts_only_complete_output(tmp_path):
    output = tmp_path / "aime24_t0.7_p0.95_n2-MNT24_seed42.jsonl"
    _write_output(output)
    assert validate_jsonl_output(output, "AIME24", _samples(), range(2), 42) == (
        True,
        "complete",
    )

    _write_output(output, omit_last=True)
    valid, reason = validate_jsonl_output(output, "AIME24", _samples(), range(2), 42)
    assert not valid
    assert "expected 4 generations" in reason


def test_jsonl_resume_rejects_corruption_and_wrong_seed(tmp_path):
    output = tmp_path / "aime24_results.jsonl"
    output.write_text("{not-json}\n", encoding="utf-8")
    assert not validate_jsonl_output(output, "AIME24", _samples(), range(2), 42)[0]

    _write_output(output, seed=43)
    assert not validate_jsonl_output(output, "AIME24", _samples(), range(2), 42)[0]


def test_model_directory_accepts_indexed_shards(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    assert model_directory_errors(tmp_path) == []
    (tmp_path / "model-00002-of-00002.safetensors").unlink()
    assert "missing or empty referenced shard" in model_directory_errors(tmp_path)[0]


def test_grading_requires_six_tasks_by_five_seeds():
    tasks = ["AIME24", "AIME25", "AMC23", "HMMT24", "HMMT25", "MATH-500"]
    results = [
        {"hyperparameters": {"task_name": task.lower(), "eval_seed": str(seed)}}
        for task in tasks
        for seed in range(42, 47)
    ]
    validate_benchmark_seed_coverage(results, tasks, range(42, 47))
    with pytest.raises(RuntimeError, match="incomplete grading coverage"):
        validate_benchmark_seed_coverage(results[:-1], tasks, range(42, 47))


def test_default_six_datasets_exist_and_use_verl_schema():
    repository_root = Path(__file__).resolve().parents[3]
    for task in ("AIME24", "AIME25", "AMC23", "HMMT24", "HMMT25", "MATH-500"):
        samples = load_eval_samples(repository_root / "datasets" / "test" / task / "test.parquet")
        assert samples
        assert set(samples[0]) == {"example_id", "prompt", "answer"}
