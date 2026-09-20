import numpy as np
import torch

from verl.utils.csv_utils import build_metric_csv_rows, numeric_scalar_for_csv


def test_numeric_scalar_for_csv_accepts_python_numpy_and_tensor_scalars():
    assert numeric_scalar_for_csv(1) == 1
    assert numeric_scalar_for_csv(1.5) == 1.5
    assert numeric_scalar_for_csv(np.int64(2)) == 2
    assert numeric_scalar_for_csv(np.float32(2.5)) == 2.5
    assert numeric_scalar_for_csv(torch.tensor(3.5)) == 3.5


def test_numeric_scalar_for_csv_rejects_non_metrics():
    assert numeric_scalar_for_csv(True) is None
    assert numeric_scalar_for_csv("1.0") is None
    assert numeric_scalar_for_csv(np.array([1.0, 2.0])) is None
    assert numeric_scalar_for_csv(torch.tensor([1.0, 2.0])) is None


def test_build_metric_csv_rows_is_lossless_sorted_and_prefix_filterable():
    metrics = {
        "training/loss": torch.tensor(1.25),
        "val-core/math/acc/mean@1": np.float32(0.75),
        "val-aux/num_turns/max": np.int64(4),
        "metadata/name": "run",
    }

    paper_rows = build_metric_csv_rows(metrics, step=7)
    assert [row["metric"] for row in paper_rows] == [
        "training/loss",
        "val-aux/num_turns/max",
        "val-core/math/acc/mean@1",
    ]
    assert all(row["step"] == 7 for row in paper_rows)

    validation_rows = build_metric_csv_rows(
        metrics, step=7, prefixes=("val-core/", "val-aux/")
    )
    assert [row["metric"] for row in validation_rows] == [
        "val-aux/num_turns/max",
        "val-core/math/acc/mean@1",
    ]
