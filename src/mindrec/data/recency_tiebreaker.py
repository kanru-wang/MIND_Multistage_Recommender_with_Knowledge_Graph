from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def recency_tiebreaker_config(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict(cfg.get("posthoc_recency", {}))
    alpha_grid = [
        float(value)
        for value in raw.get(
            "alpha_grid",
            [-0.02, -0.01, -0.005, -0.0025, 0.0, 0.0025, 0.005, 0.01, 0.02],
        )
    ]
    if 0.0 not in alpha_grid:
        alpha_grid.append(0.0)
    alpha_grid = sorted(set(alpha_grid))
    out = {
        "enabled": bool(raw.get("enabled", False)),
        "alpha": float(raw.get("alpha", 0.0)),
        "alpha_path": raw.get("alpha_path"),
        "alpha_grid": alpha_grid,
        "min_auc_improvement": float(raw.get("min_auc_improvement", 1.0e-5)),
        "require_all_dates_nonnegative": bool(
            raw.get("require_all_dates_nonnegative", True)
        ),
        "age_dataset_name": str(
            raw.get("age_dataset_name", cfg.get("data", {}).get("dataset_name", ""))
        ),
        "candidate_buffer_size": int(raw.get("candidate_buffer_size", 65_536)),
        "batch_size": int(raw.get("batch_size", 8_192)),
        "exact_rank_guard_threshold": float(
            raw.get("exact_rank_guard_threshold", 1.0e-5)
        ),
        "reference_batch_size": int(raw.get("reference_batch_size", 2_048)),
    }
    if out["candidate_buffer_size"] < out["batch_size"]:
        raise ValueError(
            "posthoc_recency.candidate_buffer_size must be at least batch_size."
        )
    if out["min_auc_improvement"] < 0.0:
        raise ValueError("posthoc_recency.min_auc_improvement must be non-negative.")
    if out["exact_rank_guard_threshold"] < 0.0:
        raise ValueError(
            "posthoc_recency.exact_rank_guard_threshold must be non-negative."
        )
    if out["reference_batch_size"] < 1:
        raise ValueError("posthoc_recency.reference_batch_size must be positive.")
    if out["enabled"] and not out["age_dataset_name"]:
        raise ValueError("posthoc_recency.age_dataset_name is required.")
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
    # Smaller age rank means fresher.
    return 1.0 - 2.0 * age_ranks / float(len(ages) - 1)


def recency_tiebreaker_components(
    baseline_scores: np.ndarray,
    item_age_log1p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare standardized baseline scores and freshness once per impression."""
    scores = np.asarray(baseline_scores, dtype=np.float64)
    ages = np.asarray(item_age_log1p, dtype=np.float64)
    if ages.shape != scores.shape:
        raise ValueError("Item ages must match baseline scores.")
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        standardized = np.zeros(scores.shape, dtype=np.float64)
    else:
        standardized = (scores - float(np.mean(scores))) / scale
    return standardized, freshness_percentile(ages)


def apply_recency_tiebreaker(
    baseline_scores: np.ndarray,
    item_age_log1p: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply a scale-free recency adjustment while preserving alpha=0 exactly."""
    scores = np.asarray(baseline_scores, dtype=np.float64)
    if float(alpha) == 0.0 or len(scores) <= 1:
        return scores.copy()
    standardized, freshness = recency_tiebreaker_components(
        scores,
        item_age_log1p,
    )
    return standardized + float(alpha) * freshness


def resolve_recency_alpha(cfg: dict[str, Any]) -> float:
    recency_cfg = recency_tiebreaker_config(cfg)
    alpha_path = recency_cfg["alpha_path"]
    if not alpha_path:
        return float(recency_cfg["alpha"])
    path = Path(str(alpha_path))
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tuned recency coefficient: {path}. Run "
            "tune_recency_tiebreaker first."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["selected_alpha"])
