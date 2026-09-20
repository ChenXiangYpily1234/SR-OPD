"""Small helpers for lossless metric CSV serialization."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np


def numeric_scalar_for_csv(value: Any) -> Optional[int | float]:
    """Convert Python, NumPy, or tensor numeric scalars to a Python scalar."""
    if isinstance(value, np.generic):
        value = value.item()
    elif hasattr(value, "numel") and hasattr(value, "detach"):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def build_metric_csv_rows(
    metrics: dict[str, Any], step: int, prefixes: Optional[Iterable[str]] = None
) -> list[dict[str, Any]]:
    """Build sorted long-form rows, retaining every supported numeric scalar."""
    prefixes_tuple = tuple(prefixes) if prefixes is not None else None
    rows = []
    for key, value in sorted(metrics.items()):
        if prefixes_tuple is not None and not key.startswith(prefixes_tuple):
            continue
        scalar = numeric_scalar_for_csv(value)
        if scalar is not None:
            rows.append({"step": step, "metric": key, "value": scalar})
    return rows
