from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mindrec.pipeline.evaluate import _load_model
from mindrec.pipeline.ranker_scoring import (
    precompute_item_semantics,
    score_prepared_groups,
)
from mindrec.utils import test_split_name, validation_split_name


@dataclass
class RerankScoringAssets:
    """Cached ranker state shared by reranker search and evaluation."""

    model: Any
    encoded_items: torch.Tensor
    item_semantics: torch.Tensor
    teacher_item: np.ndarray
    score_batch_size: int
    impression_batch_size: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RerankProtocol:
    search_split: str
    reporting_split: str
    selection: dict[str, Any]


def resolve_rerank_protocol(
    cfg: dict[str, Any],
    *,
    require_frozen: bool = False,
) -> RerankProtocol:
    rr_cfg = dict(cfg.get("rerank", {}))
    search_split = str(
        rr_cfg.get("search", {}).get("split", validation_split_name(cfg))
    )
    reporting_split = str(rr_cfg.get("eval_split", test_split_name(cfg)))
    selection = dict(rr_cfg.get("selection", {}))

    if not search_split or not reporting_split:
        raise ValueError("Reranker search and reporting split names must be non-empty.")
    if bool(selection.get("require_distinct_splits", False)) and (
        search_split == reporting_split
    ):
        raise ValueError(
            "Reranker search and reporting splits must differ; both resolve to "
            f"{search_split!r}."
        )

    configured_source = selection.get("source_split")
    if configured_source is not None and str(configured_source) != search_split:
        raise ValueError(
            "rerank.selection.source_split does not match rerank.search.split: "
            f"{configured_source!r} != {search_split!r}."
        )
    configured_reporting = selection.get("reporting_split")
    if configured_reporting is not None and str(configured_reporting) != reporting_split:
        raise ValueError(
            "rerank.selection.reporting_split does not match rerank.eval_split: "
            f"{configured_reporting!r} != {reporting_split!r}."
        )

    if require_frozen and bool(selection.get("require_frozen_for_eval", False)):
        if not bool(selection.get("frozen", False)):
            raise RuntimeError(
                "Final reranker evaluation is blocked because "
                "rerank.selection.frozen is false."
            )
        if bool(selection.get("require_provenance_for_eval", False)):
            missing = [
                key
                for key in ("search_artifact", "decision_note")
                if selection.get(key) is None
                or not str(selection.get(key, "")).strip()
            ]
            if missing:
                raise RuntimeError(
                    "Final reranker evaluation requires frozen-selection provenance; "
                    "missing rerank.selection fields: " + ", ".join(missing)
                )

    return RerankProtocol(
        search_split=search_split,
        reporting_split=reporting_split,
        selection=selection,
    )


def rerank_eval_readiness(cfg: dict[str, Any]) -> tuple[bool, str]:
    try:
        protocol = resolve_rerank_protocol(cfg, require_frozen=True)
    except (RuntimeError, ValueError) as exc:
        return False, str(exc)
    if protocol.search_split == protocol.reporting_split:
        return (
            False,
            "Reranker search and reporting use the same split; automated final "
            "evaluation is not ready. Invoke rerank_eval explicitly only for a "
            "diagnostic report.",
        )
    return True, "ready"


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be at least 1.")
    return parsed


def load_rerank_scoring_assets(
    cfg: dict[str, Any],
    proc_root: Path,
    device: torch.device,
) -> RerankScoringAssets:
    """Load the frozen ranker and cache the semantics used by current scoring."""

    rr_scoring_cfg = dict(cfg.get("rerank", {}).get("scoring", {}))
    eval_cfg = dict(cfg.get("eval", {}))
    score_batch_size = _positive_int(
        rr_scoring_cfg.get("batch_size", eval_cfg.get("batch_size", 2048)),
        "rerank.scoring.batch_size",
    )
    item_encoding_batch_size = _positive_int(
        rr_scoring_cfg.get(
            "item_encoding_batch_size",
            eval_cfg.get("item_encoding_batch_size", max(score_batch_size, 8192)),
        ),
        "rerank.scoring.item_encoding_batch_size",
    )
    impression_batch_size = _positive_int(
        rr_scoring_cfg.get("impression_batch_size", 128),
        "rerank.scoring.impression_batch_size",
    )

    model, item_base, teacher_item = _load_model(cfg, proc_root, device)
    if teacher_item is None:
        raise RuntimeError("Reranking requires the teacher item embedding artifact.")
    encoded_items, item_semantics = precompute_item_semantics(
        model=model,
        item_base=item_base,
        device=device,
        batch_size=item_encoding_batch_size,
        description="Pre-encode reranker items",
    )

    return RerankScoringAssets(
        model=model,
        encoded_items=encoded_items,
        item_semantics=item_semantics,
        teacher_item=teacher_item,
        score_batch_size=score_batch_size,
        impression_batch_size=impression_batch_size,
        metadata={
            "mode": "cached_item_and_history_semantics",
            "history_pooling": str(model.history_pooling),
            "score_batch_size": score_batch_size,
            "item_encoding_batch_size": item_encoding_batch_size,
            "impression_batch_size": impression_batch_size,
        },
    )


def prepare_rerank_score_group(row: Any) -> dict[str, Any]:
    """Convert one processed impression row to the shared ranker scoring shape."""

    cand_news_idx = np.asarray(row["cand_news_idx"], dtype=np.int64)
    cand_clicks_log1p = np.asarray(
        row["cand_item_clicks_log1p"], dtype=np.float32
    )
    history_len = float(row["history_len"])
    dense = np.stack(
        [
            np.full_like(cand_clicks_log1p, history_len),
            cand_clicks_log1p,
        ],
        axis=1,
    )
    return {
        "user_idx": int(row["user_idx"]),
        "hist_news_idx": [int(x) for x in list(row["hist_news_idx"])],
        "cand_news_idx": cand_news_idx,
        "cand_cat_idx": np.asarray(row["cand_cat_idx"], dtype=np.int64),
        "cand_subcat_idx": np.asarray(row["cand_subcat_idx"], dtype=np.int64),
        "cand_is_new": np.asarray(row["cand_is_new_item"], dtype=np.int64),
        "dense": dense,
    }


def score_rerank_groups(
    assets: RerankScoringAssets,
    groups: list[dict[str, Any]],
    device: torch.device,
) -> list[np.ndarray]:
    return score_prepared_groups(
        model=assets.model,
        groups=groups,
        encoded_items=assets.encoded_items,
        item_semantics=assets.item_semantics,
        batch_size=assets.score_batch_size,
        device=device,
    )
