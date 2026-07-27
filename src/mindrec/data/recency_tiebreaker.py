from __future__ import annotations

from typing import Any

import numpy as np


def recency_tiebreaker_config(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict(cfg.get("posthoc_recency", {}))
    out = {
        "enabled": bool(raw.get("enabled", False)),
        "alpha": float(raw.get("alpha", 0.02)),
        "age_dataset_name": str(
            raw.get(
                "age_dataset_name",
                cfg.get("data", {}).get("dataset_name", ""),
            )
        ),
        "max_age_hours": float(raw.get("max_age_hours", 720.0)),
    }
    if out["enabled"] and out["alpha"] != 0.02:
        raise ValueError(
            "The retained recency submission supports only alpha=0.02."
        )
    if out["enabled"] and not out["age_dataset_name"]:
        raise ValueError("posthoc_recency.age_dataset_name is required.")
    if out["max_age_hours"] <= 0.0:
        raise ValueError("posthoc_recency.max_age_hours must be positive.")
    return out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def freshness_percentile(item_age_log1p: np.ndarray) -> np.ndarray:
    """Return tie-aware within-impression freshness in [-1, 1]."""
    ages = np.asarray(item_age_log1p, dtype=np.float64)
    if len(ages) <= 1:
        return np.zeros(ages.shape, dtype=np.float64)
    age_ranks = _average_ranks(ages)
    return 1.0 - 2.0 * age_ranks / float(len(ages) - 1)


def apply_recency_tiebreaker(
    baseline_scores: np.ndarray,
    item_age_log1p: np.ndarray,
    alpha: float = 0.02,
) -> np.ndarray:
    """Apply the tested scale-free alpha=0.02 freshness adjustment."""
    if float(alpha) != 0.02:
        raise ValueError("The retained recency tiebreaker supports only alpha=0.02.")
    scores = np.asarray(baseline_scores, dtype=np.float64)
    ages = np.asarray(item_age_log1p, dtype=np.float64)
    if ages.shape != scores.shape:
        raise ValueError("Item ages must match baseline scores.")
    if len(scores) <= 1:
        return scores.copy()
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        standardized = np.zeros(scores.shape, dtype=np.float64)
    else:
        standardized = (scores - float(np.mean(scores))) / scale
    return standardized + 0.02 * freshness_percentile(ages)
