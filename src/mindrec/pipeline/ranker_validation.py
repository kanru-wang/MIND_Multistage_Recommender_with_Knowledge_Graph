from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mindrec.metrics.ranking import auc
from mindrec.models.dlrm import DLRMStudent
from mindrec.pipeline.ranker_scoring import (
    precompute_item_semantics,
    prepare_impression_group,
    score_prepared_groups,
)
from mindrec.utils import to_device


@torch.inference_mode()
def evaluate_sampled_pair_auc(model, item_base_tensor, loader, device, *, description):
    """Retain the historical pooled validation-pair metric as a second selector."""
    model.eval()
    labels, scores = [], []
    for batch in tqdm(loader, desc=description, unit="batch"):
        batch = to_device(batch, device)
        logits, _ = model(
            user_idx=batch["user_idx"], news_idx=batch["news_idx"],
            cat_idx=batch["cat_idx"], subcat_idx=batch["subcat_idx"],
            dense=batch["dense"], item_base=item_base_tensor[batch["news_idx"]],
            history_item_base=item_base_tensor[batch["hist_news_idx"]],
            history_mask=batch["hist_mask"], is_new_item=batch["is_new_item"],
            return_repr=False,
        )
        labels.append(batch["label"].cpu().numpy())
        scores.append(torch.sigmoid(logits).cpu().numpy())
    if not labels:
        raise ValueError("Sampled validation pairs must not be empty.")
    y, p = np.concatenate(labels), np.concatenate(scores)
    value = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.0
    return {"auc": value, "n_pairs": len(y)}


@torch.inference_mode()
def evaluate_mean_impression_auc(
    model: DLRMStudent,
    item_base: np.ndarray,
    impressions: pd.DataFrame,
    device: torch.device,
    *,
    score_batch_size: int = 2048,
    item_encoding_batch_size: int = 8192,
    impression_batch_size: int = 128,
    description: str = "Validation mean impression AUC",
) -> dict[str, Any]:
    """Score full candidate sets, weighting each impression equally as in evaluate.

    Match the existing evaluator: omit impressions without clicks and assign
    zero AUC to single-class (all-clicked) impressions. Item semantics must be
    recomputed on every call because the ranker's weights change each epoch.
    """
    if min(score_batch_size, item_encoding_batch_size, impression_batch_size) < 1:
        raise ValueError("Validation batch sizes must be at least 1.")
    model.eval()
    encoded_items, item_semantics = precompute_item_semantics(
        model, item_base, device, item_encoding_batch_size,
        description="Validation: encode items",
    )
    auc_sum = 0.0
    n_impressions = 0
    n_candidates = 0
    n_skipped = 0
    with tqdm(total=len(impressions), desc=description, unit="impr") as progress:
        for start in range(0, len(impressions), impression_batch_size):
            chunk = impressions.iloc[start : start + impression_batch_size]
            groups = []
            labels_by_group = []
            for row in chunk.itertuples(index=False):
                labels = np.asarray(row.cand_label, dtype=np.int32)
                if labels.sum() <= 0:
                    n_skipped += 1
                    continue
                groups.append(prepare_impression_group(row._asdict()))
                labels_by_group.append(labels)
            scores_by_group = score_prepared_groups(
                model, groups, encoded_items, item_semantics,
                score_batch_size, device,
            )
            if len(scores_by_group) != len(labels_by_group):
                raise ValueError("Validation requires scores for every impression.")
            for labels, scores in zip(labels_by_group, scores_by_group):
                if len(scores) != len(labels) or not np.isfinite(scores).all():
                    raise ValueError("Validation requires a finite score for every candidate.")
                auc_sum += auc(labels, scores)
                n_impressions += 1
                n_candidates += len(labels)
            if n_impressions:
                progress.set_postfix(auc=f"{auc_sum / n_impressions:.6f}", refresh=False)
            progress.update(len(chunk))
    if n_impressions == 0:
        raise ValueError("No clicked validation impressions available for checkpoint selection.")
    return {
        "auc": auc_sum / n_impressions,
        "n_impressions": n_impressions,
        "n_candidates": n_candidates,
        "n_skipped_no_click": n_skipped,
    }
