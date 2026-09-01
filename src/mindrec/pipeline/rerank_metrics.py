from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mindrec.metrics.diversity import category_coverage, entropy, ild_from_similarity
from mindrec.metrics.fairness import (
    catalog_target,
    exposure_from_ranking,
    gini,
    kl_divergence,
    normalize_dist,
    uniform_target,
)
from mindrec.metrics.ranking import ndcg_from_order, recall_from_order
from mindrec.rerank.greedy import cosine_sim_matrix, greedy_rerank
from mindrec.utils import position_bias_weights


METRIC_NAMES = (
    "ndcg@k",
    "recall@k",
    "ild",
    "category_coverage",
    "category_entropy",
    "fairness_kl_full",
    "fairness_kl_pool",
    "fairness_gini",
    "new_item_exposure_frac",
)


@dataclass
class ScoredRerankImpression:
    labels: np.ndarray
    cand_news_id: list[str]
    cand_news_idx: np.ndarray
    cand_is_new: list[int]
    scores: np.ndarray


@dataclass
class MetricAccumulator:
    sums: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in METRIC_NAMES}
    )
    count: int = 0

    def add(self, metrics: dict[str, float]) -> None:
        for name in METRIC_NAMES:
            self.sums[name] += float(metrics[name])
        self.count += 1

    def mean(self) -> dict[str, float]:
        if self.count == 0:
            return {name: 0.0 for name in METRIC_NAMES}
        return {name: value / self.count for name, value in self.sums.items()}


def _cat_idx(news_meta: dict[str, Any], news_id: str) -> int:
    meta = news_meta.get(news_id)
    return int(meta.cat_idx) if meta is not None else 0


def _category_reference(
    cand_news_id: list[str], news_meta: dict[str, Any]
) -> list[int]:
    return [
        cat_idx
        for cat_idx in (_cat_idx(news_meta, nid) for nid in cand_news_id)
        if cat_idx != 0
    ]


def _category_target_dist(
    category_target: str, reference_cats: list[int]
) -> dict[int, float]:
    target = (
        uniform_target(reference_cats)
        if category_target == "uniform"
        else catalog_target(reference_cats)
    )
    return normalize_dist(target)


def _new_item_exposure_frac(
    weights: np.ndarray, ranking_idx: list[int], cand_is_new: list[int]
) -> float:
    new_exposure = sum(
        float(weight)
        for weight, idx in zip(weights.tolist(), ranking_idx)
        if int(cand_is_new[idx]) == 1
    )
    return float(new_exposure / (float(weights.sum()) + 1e-12))


def _gini_over_reference(
    exposure: dict[int, float], target: dict[int, float]
) -> float:
    keys = sorted(set(exposure) | set(target))
    return gini([float(exposure.get(key, 0.0)) for key in keys])


def _metrics_for_order(
    row: ScoredRerankImpression,
    order: list[int],
    pool_order: np.ndarray,
    teacher_item: np.ndarray,
    news_meta: dict[str, Any],
    k_out: int,
    position_bias: str,
    category_target: str,
) -> dict[str, float]:
    order_array = np.asarray(order, dtype=np.int64)
    ranked_ids = [row.cand_news_id[idx] for idx in order]
    ranked_categories = [_cat_idx(news_meta, news_id) for news_id in ranked_ids]
    full_reference = _category_reference(row.cand_news_id, news_meta)
    pool_reference = [
        category
        for category in (
            _cat_idx(news_meta, row.cand_news_id[idx])
            for idx in pool_order.tolist()
        )
        if category != 0
    ]

    ranked_embeddings = teacher_item[row.cand_news_idx[order_array]]
    ranked_embeddings = ranked_embeddings / (
        np.linalg.norm(ranked_embeddings, axis=1, keepdims=True) + 1e-12
    )
    weights = position_bias_weights(len(order), mode=position_bias)
    exposure = normalize_dist(exposure_from_ranking(ranked_categories, weights))
    target_full = _category_target_dist(category_target, full_reference)
    target_pool = _category_target_dist(category_target, pool_reference)

    return {
        "ndcg@k": ndcg_from_order(row.labels, order_array, k_out),
        "recall@k": recall_from_order(row.labels, order_array, k_out),
        "ild": ild_from_similarity(cosine_sim_matrix(ranked_embeddings)),
        "category_coverage": category_coverage(
            [category for category in ranked_categories if category != 0]
        ),
        "category_entropy": entropy(
            [category for category in ranked_categories if category != 0]
        ),
        "fairness_kl_full": kl_divergence(exposure, target_full),
        "fairness_kl_pool": kl_divergence(exposure, target_pool),
        "fairness_gini": _gini_over_reference(exposure, target_pool),
        "new_item_exposure_frac": _new_item_exposure_frac(
            weights, order, row.cand_is_new
        ),
    }


def baseline_metrics_for_impression(
    row: ScoredRerankImpression,
    teacher_item: np.ndarray,
    news_meta: dict[str, Any],
    k_out: int,
    pool_size: int,
    position_bias: str,
    category_target: str,
) -> dict[str, float]:
    score_order = np.argsort(-row.scores, kind="stable")
    pool_order = score_order[:pool_size]
    return _metrics_for_order(
        row=row,
        order=score_order[:k_out].tolist(),
        pool_order=pool_order,
        teacher_item=teacher_item,
        news_meta=news_meta,
        k_out=k_out,
        position_bias=position_bias,
        category_target=category_target,
    )


def candidate_metrics_for_impression(
    row: ScoredRerankImpression,
    teacher_item: np.ndarray,
    news_meta: dict[str, Any],
    k_out: int,
    pool_size: int,
    position_bias: str,
    coverage_cfg: dict[str, Any],
    fairness_cfg: dict[str, Any],
    relevance_weight: float,
    novelty_weight: float,
    coverage_weight: float,
    novelty_sim: str,
    relevance_normalization: str,
) -> dict[str, float]:
    pool_order = np.argsort(-row.scores, kind="stable")[:pool_size]
    reranked = greedy_rerank(
        cand_news_id=row.cand_news_id,
        cand_scores=row.scores,
        cand_is_new=row.cand_is_new,
        news_meta=news_meta,
        item_teacher_emb=teacher_item[row.cand_news_idx[pool_order]],
        k_out=k_out,
        pool_size=pool_size,
        relevance_weight=relevance_weight,
        novelty_weight=novelty_weight,
        coverage_weight=coverage_weight,
        novelty_sim=novelty_sim,
        coverage_cfg=coverage_cfg,
        fairness_cfg=fairness_cfg,
        relevance_normalization=relevance_normalization,
    )
    return _metrics_for_order(
        row=row,
        order=reranked["ranked_indices"],
        pool_order=pool_order,
        teacher_item=teacher_item,
        news_meta=news_meta,
        k_out=k_out,
        position_bias=position_bias,
        category_target=str(fairness_cfg.get("category_target", "catalog")),
    )


def evaluate_baseline(
    scored_impressions: list[ScoredRerankImpression],
    **kwargs: Any,
) -> dict[str, float]:
    accumulator = MetricAccumulator()
    for row in scored_impressions:
        accumulator.add(baseline_metrics_for_impression(row=row, **kwargs))
    return accumulator.mean()


def evaluate_candidate(
    scored_impressions: list[ScoredRerankImpression],
    **kwargs: Any,
) -> dict[str, Any]:
    accumulator = MetricAccumulator()
    for row in scored_impressions:
        accumulator.add(candidate_metrics_for_impression(row=row, **kwargs))
    metrics: dict[str, Any] = accumulator.mean()
    metrics["weights"] = {
        "relevance": kwargs["relevance_weight"],
        "novelty": kwargs["novelty_weight"],
        "coverage": kwargs["coverage_weight"],
    }
    fairness_cfg = kwargs["fairness_cfg"]
    metrics["fairness"] = {
        "penalty_weight": float(fairness_cfg.get("penalty_weight", 0.0)),
        "new_item_floor": float(fairness_cfg.get("new_item_floor", 0.0)),
        "category_target": fairness_cfg.get("category_target", "catalog"),
    }
    metrics["novelty_sim"] = kwargs["novelty_sim"]
    return metrics
