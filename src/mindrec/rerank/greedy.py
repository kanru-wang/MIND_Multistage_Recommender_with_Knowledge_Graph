from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from mindrec.metrics.fairness import (
    catalog_target,
    normalize_dist,
    uniform_target,
)
from mindrec.utils import position_bias_weights


def _stable_entity_id(name: str) -> int:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _parse_entities(s: str) -> set[int]:
    # MIND entities are JSON-like; keep robust.
    if not isinstance(s, str) or not s.strip():
        return set()
    try:
        data = json.loads(s)
    except Exception:
        return set()
    out = set()
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                name = str(e.get("Label") or e.get("WikidataId") or e.get("Type") or "")
                if name:
                    out.add(_stable_entity_id(name))
    return out


@dataclass
class NewsMeta:
    cat_idx: int
    subcat_idx: int
    ent: set[int]


def build_news_meta(news_df) -> dict[str, NewsMeta]:
    # Example output:
    # {
    #     "N12345": NewsMeta(cat_idx=4, subcat_idx=12, ent={101, 202}),
    #     "N67890": NewsMeta(cat_idx=7, subcat_idx=19, ent=set()),
    # }
    meta = {}
    for _, r in news_df.iterrows():
        ent = _parse_entities(r.get("title_entities", "")) | _parse_entities(
            r.get("abstract_entities", "")
        )
        meta[str(r["news_id"])] = NewsMeta(
            cat_idx=int(r.get("cat_idx", 0)),
            subcat_idx=int(r.get("subcat_idx", 0)),
            ent=ent,
        )
    return meta


def _normalize_relevance(scores: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if mode == "none":
        return values
    if mode == "minmax":
        lo = float(values.min()) if len(values) else 0.0
        hi = float(values.max()) if len(values) else 0.0
        if hi - lo <= 1e-12:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)
    raise ValueError(
        f"Unknown relevance_normalization: {mode!r}; expected 'minmax' or 'none'."
    )


DEFAULT_OBJECTIVE_WEIGHTS = {
    "relevance_weight": 0.95,
    "coverage_weight": 0.05,
}


def validate_rerank_config(rr_cfg: dict[str, Any]) -> None:
    """Fail early on invalid or silently ineffective reranker settings."""

    k_out = int(rr_cfg.get("k_out", 0))
    pool_size = int(rr_cfg.get("pool_size", 0))
    if k_out < 1:
        raise ValueError("rerank.k_out must be at least 1.")
    if pool_size < k_out:
        raise ValueError("rerank.pool_size must be greater than or equal to k_out.")

    position_bias = str(rr_cfg.get("position_bias", "log"))
    if position_bias not in {"log", "linear"}:
        raise ValueError("rerank.position_bias must be 'log' or 'linear'.")
    obsolete = {"novelty_weight", "novelty_sim"}.intersection(rr_cfg)
    if obsolete:
        raise ValueError("Semantic novelty is diagnostic only; remove rerank fields: " + ", ".join(sorted(obsolete)))
    relevance_normalization = str(
        rr_cfg.get("relevance_normalization", "none")
    )
    if relevance_normalization not in {"minmax", "none"}:
        raise ValueError(
            "rerank.relevance_normalization must be 'minmax' or 'none'."
        )

    weights = [float(rr_cfg.get(name, default)) for name, default in DEFAULT_OBJECTIVE_WEIGHTS.items()]
    if not all(np.isfinite(weight) and weight >= 0.0 for weight in weights):
        raise ValueError(
            "Reranker relevance/coverage weights must be finite and "
            "non-negative."
        )
    if sum(weights) <= 0.0:
        raise ValueError("At least one reranker objective weight must be positive.")

    coverage_cfg = dict(rr_cfg.get("coverage", {}))
    if int(coverage_cfg.get("max_new_entities_per_item", 3)) < 0:
        raise ValueError(
            "rerank.coverage.max_new_entities_per_item cannot be negative."
        )
    for key in ("category_bonus", "entity_bonus"):
        value = float(coverage_cfg.get(key, 0.0))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"rerank.coverage.{key} must be finite and non-negative.")

    fairness_cfg = dict(rr_cfg.get("fairness", {}))
    if str(fairness_cfg.get("category_target", "catalog")) not in {
        "catalog",
        "uniform",
    }:
        raise ValueError(
            "rerank.fairness.category_target must be 'catalog' or 'uniform'."
        )
    penalty = float(fairness_cfg.get("penalty_weight", 0.0))
    if "new_item_floor" in fairness_cfg:
        raise ValueError("Remove rerank.fairness.new_item_floor; reranking now uses a KL-only penalty.")
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError(
            "rerank.fairness.penalty_weight must be finite and non-negative."
        )


def greedy_rerank(
    cand_news_id: list[str],
    cand_scores: np.ndarray,
    news_meta: dict[str, NewsMeta],
    k_out: int,
    pool_size: int,
    relevance_weight: float,
    coverage_weight: float,
    coverage_cfg: dict[str, Any],
    fairness_cfg: dict[str, Any],
    relevance_normalization: str = "none",
) -> dict[str, Any]:
    if "new_item_floor" in fairness_cfg:
        raise ValueError("Remove new_item_floor; reranking now uses a KL-only penalty.")
    cand_scores = np.asarray(cand_scores, dtype=np.float32)
    n_candidates = len(cand_news_id)
    if cand_scores.ndim != 1 or len(cand_scores) != n_candidates:
        raise ValueError("cand_scores must be one-dimensional and align with cand_news_id.")
    if not np.isfinite(cand_scores).all():
        raise ValueError("cand_scores must contain only finite values.")
    if k_out < 1 or pool_size < k_out:
        raise ValueError("Expected pool_size >= k_out >= 1.")

    # Work on top pool_size by relevance
    order = np.argsort(-cand_scores, kind="stable")[:pool_size]
    pool = [cand_news_id[i] for i in order]
    pool_scores = _normalize_relevance(
        cand_scores[order], relevance_normalization
    )
    # news_meta example {"N12345": NewsMeta(cat_idx=4, subcat_idx=12, ent={101, 202}), ...}
    # pool_cats is a list of such news_meta.
    pool_cats = [news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx for nid in pool]
    chosen = []
    chosen_idx = []
    chosen_mask = np.zeros(len(pool), dtype=np.bool_)
    chosen_cats = set()
    chosen_ents = set()
    chosen_exp_by_cat: dict[int, float] = {}
    chosen_cat_counts: Counter[int] = Counter()
    chosen_cat_pos_sums: dict[int, int] = {}
    max_new_ent = int(coverage_cfg.get("max_new_entities_per_item", 3))
    cat_bonus = float(coverage_cfg.get("category_bonus", 1.0))
    ent_bonus = float(coverage_cfg.get("entity_bonus", 0.3))
    pos_mode = fairness_cfg.get("position_bias", "log")
    weights_by_len = {
        k: position_bias_weights(k, mode=pos_mode) for k in range(1, k_out + 1)
    }
    total_exp_by_len = {
        k: float(weights.sum()) for k, weights in weights_by_len.items()
    }

    target_mode = fairness_cfg.get("category_target", "catalog")
    # For the fairness penalty, we need a target distribution over categories.
    # Either use the distribution in the candidate pool (catalog) or a uniform distribution over categories.
    if target_mode == "uniform":
        target_dist = normalize_dist(uniform_target([c for c in pool_cats if c != 0]))
    else:
        target_dist = normalize_dist(catalog_target([c for c in pool_cats if c != 0]))
    # Match evaluation: unknown-category exposure has zero target probability
    # and participates in the epsilon-smoothed KL, but earns no coverage bonus.
    target_keys = list(target_dist.keys())
    if 0 in pool_cats:
        target_keys.append(0)

    def coverage(i: int) -> float:
        """
        coverage = category bonus for a new category + entity bonus for new entities.
        """
        m = news_meta.get(pool[i], NewsMeta(0, 0, set()))
        bonus = 0.0
        if m.cat_idx not in chosen_cats and m.cat_idx != 0:
            bonus += cat_bonus
        if m.ent:
            new_ents = list(m.ent - chosen_ents)
            bonus += ent_bonus * float(min(len(new_ents), max_new_ent))
        return bonus

    def fairness_penalty(i: int) -> float:
        """Pool KL of the prospective prefix; no L1 or new-item floor."""
        if not fairness_cfg.get("enabled", False):
            return 0.0
        k = len(chosen) + 1
        cat_i = pool_cats[i]
        total_exp = total_exp_by_len[k]
        kl = 0.0
        for gid in target_keys:
            if pos_mode == "linear":
                count = chosen_cat_counts.get(gid, 0)
                pos_sum = chosen_cat_pos_sums.get(gid, 0)
                if gid == cat_i:
                    count += 1
                    pos_sum += len(chosen)
                raw_exp = float(count) - float(pos_sum) / float(k)
            else:
                raw_exp = chosen_exp_by_cat.get(gid, 0.0)
                if gid == cat_i:
                    raw_exp += float(weights_by_len[k][-1])
            pk = raw_exp / total_exp if total_exp > 0 else 0.0
            if pk > 0.0:
                qk = float(target_dist.get(gid, 0.0))
                kl += pk * np.log((pk + 1e-12) / (qk + 1e-12))
        return float(kl)

    # Greedy selection
    for _ in range(min(k_out, len(pool))):  # in case the pool size < k_out
        best = None
        best_i = None
        best_val = -1e18
        # Among all items in the pool, examine every candidate that has not already been chosen,
        # score it for the current position, and find the best one.
        for i, nid in enumerate(pool):
            if bool(chosen_mask[i]):
                continue
            rel = float(pool_scores[i])
            val = (
                relevance_weight * rel
                + coverage_weight * coverage(i)
            )

            if fairness_cfg.get("enabled", False):
                val -= float(fairness_cfg.get("penalty_weight", 0.0)) * fairness_penalty(i)

            if val > best_val:
                best_val = val
                best = nid
                best_i = i

        if best is None:
            break
        chosen.append(best)
        chosen_idx.append(int(best_i))
        chosen_mask[int(best_i)] = True
        m = news_meta.get(best, NewsMeta(0, 0, set()))
        if m.cat_idx != 0:
            chosen_cats.add(m.cat_idx)
        chosen_cat_counts[m.cat_idx] += 1
        chosen_cat_pos_sums[m.cat_idx] = chosen_cat_pos_sums.get(m.cat_idx, 0) + (
            len(chosen) - 1
        )
        chosen_ents |= m.ent
        if pos_mode == "log":
            pos_w = float(weights_by_len[len(chosen)][-1])
            chosen_exp_by_cat[m.cat_idx] = chosen_exp_by_cat.get(m.cat_idx, 0.0) + pos_w

    return {
        "ranked_news_id": chosen,
        "ranked_indices": [int(order[i]) for i in chosen_idx],
    }
