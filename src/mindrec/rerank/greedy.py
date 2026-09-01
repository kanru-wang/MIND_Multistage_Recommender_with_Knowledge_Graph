from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from mindrec.metrics.diversity import jaccard
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


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    # x: [K,D] assumed normalized
    return x @ x.T


def _build_novelty_similarity(
    novelty_sim: str,
    pool: list[str],
    news_meta: dict[str, NewsMeta],
    item_teacher_emb: np.ndarray | None,
) -> np.ndarray | None:
    if novelty_sim == "teacher_cosine":
        if item_teacher_emb is None:
            raise ValueError(
                "item_teacher_emb is required when novelty_sim='teacher_cosine'."
            )
        x = item_teacher_emb.astype(np.float32)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return cosine_sim_matrix(x)
    if novelty_sim == "category":
        cats = np.array(
            [news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx for nid in pool],
            dtype=np.int64,
        )
        sim = (cats[:, None] == cats[None, :]).astype(np.float32)
        # Treat unknown category 0 as missing signal rather than a real shared category.
        unknown_mask = cats == 0
        sim[unknown_mask, :] = 0.0
        sim[:, unknown_mask] = 0.0
        np.fill_diagonal(sim, 1.0)
        return sim
    if novelty_sim == "entity_jaccard":
        n = len(pool)
        sim = np.eye(n, dtype=np.float32)
        ents = [news_meta.get(nid, NewsMeta(0, 0, set())).ent for nid in pool]
        for i in range(n):
            for j in range(i + 1, n):
                # Empty annotations are missing evidence, not proof that two
                # articles cover exactly the same entities.
                score = (
                    float(jaccard(ents[i], ents[j]))
                    if ents[i] or ents[j]
                    else 0.0
                )
                sim[i, j] = score
                sim[j, i] = score
        return sim
    raise ValueError(f"Unknown novelty_sim: {novelty_sim}")


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
    novelty_sim = str(rr_cfg.get("novelty_sim", "teacher_cosine"))
    if novelty_sim not in {"teacher_cosine", "category", "entity_jaccard"}:
        raise ValueError(
            "rerank.novelty_sim must be 'teacher_cosine', 'category', or "
            "'entity_jaccard'."
        )
    relevance_normalization = str(
        rr_cfg.get("relevance_normalization", "none")
    )
    if relevance_normalization not in {"minmax", "none"}:
        raise ValueError(
            "rerank.relevance_normalization must be 'minmax' or 'none'."
        )

    weights = [
        float(rr_cfg.get("relevance_weight", 0.85)),
        float(rr_cfg.get("novelty_weight", 0.10)),
        float(rr_cfg.get("coverage_weight", 0.05)),
    ]
    if not all(np.isfinite(weight) and weight >= 0.0 for weight in weights):
        raise ValueError(
            "Reranker relevance/novelty/coverage weights must be finite and "
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
    floor = float(fairness_cfg.get("new_item_floor", 0.0))
    penalty = float(fairness_cfg.get("penalty_weight", 0.0))
    if not np.isfinite(floor) or not 0.0 <= floor <= 1.0:
        raise ValueError("rerank.fairness.new_item_floor must be between 0 and 1.")
    if not np.isfinite(penalty) or penalty < 0.0:
        raise ValueError(
            "rerank.fairness.penalty_weight must be finite and non-negative."
        )


def greedy_rerank(
    cand_news_id: list[str],
    cand_scores: np.ndarray,
    cand_is_new: list[int],
    news_meta: dict[str, NewsMeta],
    item_teacher_emb: np.ndarray | None,
    k_out: int,
    pool_size: int,
    relevance_weight: float,
    novelty_weight: float,
    coverage_weight: float,
    novelty_sim: str,
    coverage_cfg: dict[str, Any],
    fairness_cfg: dict[str, Any],
    relevance_normalization: str = "none",
) -> dict[str, Any]:
    cand_scores = np.asarray(cand_scores, dtype=np.float32)
    n_candidates = len(cand_news_id)
    if cand_scores.ndim != 1 or len(cand_scores) != n_candidates:
        raise ValueError("cand_scores must be one-dimensional and align with cand_news_id.")
    if len(cand_is_new) != n_candidates:
        raise ValueError("cand_is_new must align with cand_news_id.")
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
    pool_is_new = [int(cand_is_new[i]) for i in order]
    # news_meta example {"N12345": NewsMeta(cat_idx=4, subcat_idx=12, ent={101, 202}), ...}
    # pool_cats is a list of such news_meta.
    pool_cats = [news_meta.get(nid, NewsMeta(0, 0, set())).cat_idx for nid in pool]
    # pool is a list of news IDs truncated to pool_size. We will select from this pool.
    # _build_novelty_similarity returns a similarity matrix for items in the pool based on the specified novelty_sim method.
    sim_mat = _build_novelty_similarity(
        novelty_sim=novelty_sim,
        pool=pool,
        news_meta=news_meta,
        item_teacher_emb=item_teacher_emb,
    )
    if sim_mat is not None and sim_mat.shape != (len(pool), len(pool)):
        raise ValueError(
            "item_teacher_emb must contain one row per candidate in the relevance pool."
        )

    chosen = []
    chosen_idx = []
    chosen_mask = np.zeros(len(pool), dtype=np.bool_)
    chosen_cats = set()
    chosen_ents = set()
    max_sim_to_chosen = np.zeros(len(pool), dtype=np.float32)
    chosen_exp_by_cat: dict[int, float] = {}
    chosen_cat_counts: Counter[int] = Counter()
    chosen_cat_pos_sums: dict[int, int] = {}
    chosen_new_count = 0
    chosen_new_pos_sum = 0
    chosen_new_exp = 0.0
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
    # target_keys are the category IDs that appear in the target distribution.
    target_keys = list(target_dist.keys())

    def novelty(i: int) -> float:
        """
        0 if nothing has been selected yet.
        Otherwise, - max similarity to anything already selected.
        Why the minus sign? Because higher similarity means lower novelty.
        """
        if not chosen_idx:
            return 0.0
        return -float(max_sim_to_chosen[i])

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

    def fairness_penalty_log(cat_i: int, is_new_i: int, k: int) -> float:
        """
        cat_i is the category index of the candidate item at pool[i].
        is_new_i indicates whether this candidate item is a "new item".
        """
        # Look up the position bias weight for the next position k.
        next_w = float(weights_by_len[k][-1])
        total_exp = total_exp_by_len[k]
        kl = 0.0
        l1 = 0.0
        # Loop through every category in the target distribution.
        for gid in target_keys:
            # chosen_exp_by_cat keeps track of the accumulated position-bias-weighted exposure
            # for each category among the already chosen items.
            # An exmaple value of chosen_exp_by_cat is {4: 1.5, 7: 0.8}, meaning category 4 
            # has accumulated exposure of 1.5, and category 7 has 0.8 so far.
            # Instead of recomputing category exposure from scratch over the whole chosen list
            # every time, having chosen_exp_by_cat would allow efficient reranking.
            raw_exp = chosen_exp_by_cat.get(gid, 0.0)

            # Only one category's exposure will be updated, but all categories in the
            # target distribution will be used for the KL and L1 calculations.
            if gid == cat_i and gid != 0:
                raw_exp += next_w
            pk = (raw_exp / total_exp) if total_exp > 0 else 0.0
            qk = float(target_dist.get(gid, 0.0))
            if pk > 0.0:
                kl += pk * np.log((pk + 1e-12) / (qk + 1e-12))
            l1 += abs(pk - qk)

        pen = 0.5 * float(kl) + 0.5 * float(l1)
        floor = float(fairness_cfg.get("new_item_floor", 0.0))
        if floor > 0.0:
            new_exp = chosen_new_exp + (next_w if is_new_i == 1 else 0.0)
            frac = (new_exp / total_exp) if total_exp > 0 else 0.0
            if frac < floor:
                pen += (floor - frac) * 2.0
        return pen

    def fairness_penalty_linear(cat_i: int, is_new_i: int, k: int) -> float:
        total_exp = total_exp_by_len[k]
        kl = 0.0
        l1 = 0.0
        for gid in target_keys:
            count = chosen_cat_counts.get(gid, 0)
            pos_sum = chosen_cat_pos_sums.get(gid, 0)
            if gid == cat_i and gid != 0:
                count += 1
                pos_sum += len(chosen)
            raw_exp = float(count) - (float(pos_sum) / float(k))
            pk = (raw_exp / total_exp) if total_exp > 0 else 0.0
            qk = float(target_dist.get(gid, 0.0))
            if pk > 0.0:
                kl += pk * np.log((pk + 1e-12) / (qk + 1e-12))
            l1 += abs(pk - qk)

        pen = 0.5 * float(kl) + 0.5 * float(l1)
        floor = float(fairness_cfg.get("new_item_floor", 0.0))
        if floor > 0.0:
            new_count = chosen_new_count + int(is_new_i == 1)
            new_pos_sum = chosen_new_pos_sum + (len(chosen) if is_new_i == 1 else 0)
            new_exp = float(new_count) - (float(new_pos_sum) / float(k))
            frac = (new_exp / total_exp) if total_exp > 0 else 0.0
            if frac < floor:
                pen += (floor - frac) * 2.0
        return pen

    def fairness_penalty(i: int) -> float:
        """
        Pretend candidate i is added at the next rank position k, and compute
        how much (i.e. penalty) would the top-k exposure distribution deviate
        from the desired category mix (either log or linear position-bias mode).
        """
        if not fairness_cfg.get("enabled", False):
            return 0.0
        k = len(chosen) + 1
        cat_i = pool_cats[i]
        is_new_i = int(pool_is_new[i])
        if pos_mode == "linear":
            return float(fairness_penalty_linear(cat_i, is_new_i, k))
        return float(fairness_penalty_log(cat_i, is_new_i, k))

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
                + novelty_weight * novelty(i)
                + coverage_weight * coverage(i)
            )

            if fairness_cfg.get("enabled", False):
                val -= float(fairness_cfg.get("penalty_weight", 0.5)) * fairness_penalty(i)

            if val > best_val:
                best_val = val
                best = nid
                best_i = i

        if best is None:
            break
        chosen.append(best)
        chosen_idx.append(int(best_i))
        chosen_mask[int(best_i)] = True
        if sim_mat is not None and best_i is not None:
            # np.maximum compares two arrays and return a new array of element-wise larger values.
            # To the not yet chosen items, their similarities to the chosen-set-as-a-whole are represented by max_sim_to_chosen.
            max_sim_to_chosen = np.maximum(max_sim_to_chosen, sim_mat[:, int(best_i)])
        m = news_meta.get(best, NewsMeta(0, 0, set()))
        if m.cat_idx != 0:
            chosen_cats.add(m.cat_idx)
            chosen_cat_counts[m.cat_idx] += 1
            chosen_cat_pos_sums[m.cat_idx] = chosen_cat_pos_sums.get(m.cat_idx, 0) + (
                len(chosen) - 1
            )
        chosen_ents |= m.ent
        if pos_mode == "linear":
            if int(pool_is_new[int(best_i)]) == 1:
                chosen_new_count += 1
                chosen_new_pos_sum += len(chosen) - 1
        else:
            pos_w = float(weights_by_len[len(chosen)][-1])
            if m.cat_idx != 0:
                chosen_exp_by_cat[m.cat_idx] = chosen_exp_by_cat.get(m.cat_idx, 0.0) + pos_w
            if int(pool_is_new[int(best_i)]) == 1:
                chosen_new_exp += pos_w

    return {
        "ranked_news_id": chosen,
        "ranked_indices": [int(order[i]) for i in chosen_idx],
    }
