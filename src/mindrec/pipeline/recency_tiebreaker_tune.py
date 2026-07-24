from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.data.item_trend import ItemTrendIndex, item_trend_artifact_path
from mindrec.data.recency_tiebreaker import (
    recency_tiebreaker_components,
    recency_tiebreaker_config,
)
from mindrec.pipeline.evaluate import _attach_behavior_time, _load_model
from mindrec.pipeline.submission import (
    _has_near_tied_scores,
    _precompute_item_semantics,
    _score_prepared_groups,
    _score_reference_group,
)
from mindrec.utils import (
    behavior_artifact_path,
    impression_artifact_path,
    log_device,
    resolve_device,
    save_json,
    validation_split_name,
)


def _impression_aucs(
    labels: np.ndarray,
    score_matrix: np.ndarray,
) -> np.ndarray | None:
    """Compute impression AUC for every score row in one vectorized comparison."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError("Score matrix must have one column per label.")
    positive = scores[:, labels == 1]
    negative = scores[:, labels == 0]
    if positive.shape[1] == 0 or negative.shape[1] == 0:
        return None
    differences = positive[:, :, None] - negative[:, None, :]
    wins = np.count_nonzero(differences > 0.0, axis=(1, 2))
    ties = np.count_nonzero(differences == 0.0, axis=(1, 2))
    return (wins + 0.5 * ties) / float(
        positive.shape[1] * negative.shape[1]
    )


def _select_alpha(
    results: list[dict[str, Any]],
    *,
    min_auc_improvement: float,
    require_all_dates_nonnegative: bool,
) -> tuple[float, str]:
    baseline = next(result for result in results if result["alpha"] == 0.0)
    baseline_auc = float(baseline["auc"])
    baseline_dates = baseline["auc_by_date"]
    feasible: list[dict[str, Any]] = []
    for result in results:
        if result["alpha"] == 0.0:
            continue
        if float(result["auc"]) - baseline_auc < min_auc_improvement:
            continue
        if require_all_dates_nonnegative and any(
            float(value) < float(baseline_dates[date])
            for date, value in result["auc_by_date"].items()
        ):
            continue
        feasible.append(result)
    if not feasible:
        return 0.0, "no_nonzero_alpha_passed_the_overall_and_date_guardrails"
    best = max(
        feasible,
        key=lambda result: (
            float(result["auc"]),
            -abs(float(result["alpha"])),
        ),
    )
    return float(best["alpha"]), "best_guardrail_feasible_validation_auc"


def run_tune_recency_tiebreaker(cfg: dict[str, Any]) -> None:
    recency_cfg = recency_tiebreaker_config(cfg)
    if not recency_cfg["enabled"]:
        raise ValueError("posthoc_recency.enabled must be true for tuning.")

    device = resolve_device(cfg["ranker"].get("device", "cuda"))
    log_device(device, "Recency tiebreaker tuning")
    proc_root = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / "tuning")
    split_name = validation_split_name(cfg)

    model, item_base, _ = _load_model(
        cfg,
        proc_root,
        runs_root,
        device,
        load_teacher_item=False,
    )
    impressions = pd.read_parquet(impression_artifact_path(proc_root, split_name))
    behaviors = pd.read_parquet(
        behavior_artifact_path(proc_root, split_name),
        columns=["impression_id", "time"],
    )
    impressions["impression_id"] = impressions["impression_id"].astype(str)
    behaviors["impression_id"] = behaviors["impression_id"].astype(str)
    impressions = _attach_behavior_time(impressions, behaviors)
    parsed_time = pd.to_datetime(
        impressions["time"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="raise",
    )
    impressions["recency_date"] = parsed_time.dt.strftime("%Y-%m-%d")
    del behaviors

    age_root = (
        Path(cfg["data"]["processed_root"]) / recency_cfg["age_dataset_name"]
    )
    age_maps = IdMaps.load(age_root / "id_maps.json")
    age_index = ItemTrendIndex.load(
        item_trend_artifact_path(age_root),
        use_burst=False,
    )
    age_lookup = age_maps.news2idx.get

    encoded_items, item_semantics = _precompute_item_semantics(
        model=model,
        item_base=item_base,
        device=device,
        batch_size=recency_cfg["batch_size"],
    )

    alpha_grid = recency_cfg["alpha_grid"]
    alpha_values = np.asarray(alpha_grid, dtype=np.float64)
    zero_alpha_index = alpha_grid.index(0.0)
    auc_sums = {alpha: 0.0 for alpha in alpha_grid}
    auc_counts = {alpha: 0 for alpha in alpha_grid}
    date_sums: dict[str, dict[float, float]] = {}
    date_counts: dict[str, dict[float, int]] = {}
    pending_groups: list[dict[str, Any]] = []
    n_pending_candidates = 0
    n_reference_rescored = 0

    def flush() -> None:
        nonlocal n_pending_candidates, n_reference_rescored
        if not pending_groups:
            return
        scores_by_group = _score_prepared_groups(
            model=model,
            groups=pending_groups,
            encoded_items=encoded_items,
            item_semantics=item_semantics,
            batch_size=recency_cfg["batch_size"],
            device=device,
        )
        for group, baseline_scores in zip(pending_groups, scores_by_group):
            if _has_near_tied_scores(
                baseline_scores,
                recency_cfg["exact_rank_guard_threshold"],
            ):
                baseline_scores = _score_reference_group(
                    model=model,
                    group=group,
                    item_base=item_base,
                    batch_size=recency_cfg["reference_batch_size"],
                    device=device,
                )
                n_reference_rescored += 1
            date = group["recency_date"]
            if date not in date_sums:
                date_sums[date] = {alpha: 0.0 for alpha in alpha_grid}
                date_counts[date] = {alpha: 0 for alpha in alpha_grid}
            standardized, freshness = recency_tiebreaker_components(
                baseline_scores,
                group["item_age_log1p"],
            )
            adjusted = (
                standardized[None, :]
                + alpha_values[:, None] * freshness[None, :]
            )
            # Alpha zero is the exact frozen baseline, including degenerate
            # impressions whose logits all have the same value.
            adjusted[zero_alpha_index] = baseline_scores
            values = _impression_aucs(group["labels"], adjusted)
            if values is None:
                continue
            for alpha, value in zip(alpha_grid, values):
                value = float(value)
                auc_sums[alpha] += value
                auc_counts[alpha] += 1
                date_sums[date][alpha] += value
                date_counts[date][alpha] += 1
        pending_groups.clear()
        n_pending_candidates = 0

    for row in tqdm(
        impressions.itertuples(index=False),
        total=len(impressions),
        desc=f"Tune recency tiebreaker ({split_name})",
    ):
        cand_news_id = list(row.cand_news_id)
        age_news_idx = [int(age_lookup(str(news_id), 0)) for news_id in cand_news_id]
        item_age_log1p, _ = age_index.features(age_news_idx, row.time)
        clicks = np.asarray(row.cand_item_clicks_log1p, dtype=np.float32)
        dense = np.stack(
            [
                np.full(clicks.shape, float(row.history_len), dtype=np.float32),
                clicks,
            ],
            axis=1,
        )
        group = {
            "user_idx": int(row.user_idx),
            "hist_news_idx": [int(value) for value in row.hist_news_idx],
            "cand_news_idx": np.asarray(row.cand_news_idx, dtype=np.int64),
            "cand_cat_idx": np.asarray(row.cand_cat_idx, dtype=np.int64),
            "cand_subcat_idx": np.asarray(row.cand_subcat_idx, dtype=np.int64),
            "cand_is_new": np.asarray(row.cand_is_new_item, dtype=np.int64),
            "dense": dense,
            "item_age_log1p": item_age_log1p,
            "labels": np.asarray(row.cand_label, dtype=np.int8),
            "recency_date": str(row.recency_date),
        }
        pending_groups.append(group)
        n_pending_candidates += len(cand_news_id)
        if n_pending_candidates >= recency_cfg["candidate_buffer_size"]:
            flush()
    flush()

    results: list[dict[str, Any]] = []
    for alpha in alpha_grid:
        auc = auc_sums[alpha] / max(auc_counts[alpha], 1)
        auc_by_date = {
            date: date_sums[date][alpha] / max(date_counts[date][alpha], 1)
            for date in sorted(date_sums)
        }
        results.append(
            {
                "alpha": float(alpha),
                "auc": float(auc),
                "auc_by_date": auc_by_date,
                "n_impressions": int(auc_counts[alpha]),
            }
        )

    selected_alpha, reason = _select_alpha(
        results,
        min_auc_improvement=recency_cfg["min_auc_improvement"],
        require_all_dates_nonnegative=recency_cfg[
            "require_all_dates_nonnegative"
        ],
    )
    payload = {
        "selected_alpha": selected_alpha,
        "selection_reason": reason,
        "formula": (
            "zscore_within_impression(baseline_logit) + "
            "alpha * freshness_percentile"
        ),
        "eval_split": split_name,
        "ranker_run_name": cfg.get("artifacts", {}).get(
            "ranker_run_name", cfg["run_name"]
        ),
        "age_dataset_name": recency_cfg["age_dataset_name"],
        "min_auc_improvement": recency_cfg["min_auc_improvement"],
        "require_all_dates_nonnegative": recency_cfg[
            "require_all_dates_nonnegative"
        ],
        "n_reference_rescored_impressions": n_reference_rescored,
        "results": results,
    }
    output_path = out_root / "item_age_tiebreaker.json"
    save_json(output_path, payload)
    print(f"Selected recency alpha: {selected_alpha:+.6f} ({reason})")
    print(f"Saved recency tuning result: {output_path}")
