from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from mindrec.models.dlrm import DLRMStudent


@torch.inference_mode()
def precompute_item_semantics(
    model: DLRMStudent,
    item_base: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    description: str = "Pre-encode ranker items",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache shared and candidate-specific semantic vectors for every item."""
    if batch_size < 1:
        raise ValueError("Item-encoding batch size must be at least 1.")

    n_items = int(item_base.shape[0])
    emb_dim = int(model.item_base_proj.out_features)
    dtype = model.item_base_proj.weight.dtype
    encoded_items = torch.empty((n_items, emb_dim), dtype=dtype, device=device)
    item_semantics = torch.empty_like(encoded_items)

    for start in tqdm(range(0, n_items, batch_size), desc=description):
        stop = min(start + batch_size, n_items)
        item_batch = torch.as_tensor(
            item_base[start:stop],
            dtype=dtype,
            device=device,
        )
        encoded_batch = model.encode_item_base_semantics(item_batch)
        encoded_items[start:stop].copy_(encoded_batch)
        item_semantics[start:stop].copy_(
            model.project_item_semantics(encoded_batch)
        )

    return encoded_items, item_semantics


@torch.inference_mode()
def score_prepared_groups(
    model: DLRMStudent,
    groups: list[dict[str, Any]],
    encoded_items: torch.Tensor,
    item_semantics: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Score groups while reusing item encodings and refined history states."""
    if not groups:
        return []
    if batch_size < 1:
        raise ValueError("Scoring batch size must be at least 1.")

    lengths = np.asarray(
        [len(group["cand_news_idx"]) for group in groups], dtype=np.int64
    )
    n_candidates = int(lengths.sum())
    if n_candidates == 0:
        return [np.empty(0, dtype=np.float32) for _ in groups]

    max_history = max(1, max(len(group["hist_news_idx"]) for group in groups))
    history_idx_array = np.zeros((len(groups), max_history), dtype=np.int64)
    history_mask_array = np.zeros((len(groups), max_history), dtype=np.bool_)
    for group_idx, group in enumerate(groups):
        history = group["hist_news_idx"]
        if history:
            history_idx_array[group_idx, : len(history)] = history
            history_mask_array[group_idx, : len(history)] = True

    history_idx = torch.as_tensor(
        history_idx_array, dtype=torch.long, device=device
    )
    history_mask = torch.as_tensor(
        history_mask_array, dtype=torch.bool, device=device
    )

    group_history_semantics = model.encode_history_semantics(
        encoded_items[history_idx]
    )
    group_user_semantics = None
    if model.history_pooling == "mean":
        group_user_semantics = model.pool_history_semantics(
            group_history_semantics,
            history_mask,
        )

    candidate_group_idx = np.repeat(np.arange(len(groups), dtype=np.int64), lengths)
    user_idx = np.concatenate(
        [
            np.full(length, int(group["user_idx"]), dtype=np.int64)
            for group, length in zip(groups, lengths)
        ]
    )
    news_idx = np.concatenate([group["cand_news_idx"] for group in groups])
    cat_idx = np.concatenate([group["cand_cat_idx"] for group in groups])
    subcat_idx = np.concatenate([group["cand_subcat_idx"] for group in groups])
    is_new_item = np.concatenate([group["cand_is_new"] for group in groups])
    dense = np.concatenate([group["dense"] for group in groups], axis=0)

    t_group_idx = torch.as_tensor(candidate_group_idx, dtype=torch.long, device=device)
    t_user_idx = torch.as_tensor(user_idx, dtype=torch.long, device=device)
    t_news_idx = torch.as_tensor(news_idx, dtype=torch.long, device=device)
    t_cat_idx = torch.as_tensor(cat_idx, dtype=torch.long, device=device)
    t_subcat_idx = torch.as_tensor(subcat_idx, dtype=torch.long, device=device)
    t_is_new_item = torch.as_tensor(is_new_item, dtype=torch.long, device=device)
    t_dense = torch.as_tensor(dense, dtype=torch.float32, device=device)

    score_tensor = torch.empty(
        n_candidates,
        dtype=model.item_base_proj.weight.dtype,
        device=device,
    )
    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)
        sl = slice(start, stop)
        batch_group_idx = t_group_idx[sl]
        batch_item_semantics = item_semantics[t_news_idx[sl]]
        if model.history_pooling == "candidate_attention":
            user_semantics = model.pool_history_semantics(
                group_history_semantics[batch_group_idx],
                history_mask[batch_group_idx],
                candidate_item_semantics=batch_item_semantics,
            )
        else:
            assert group_user_semantics is not None
            user_semantics = group_user_semantics[batch_group_idx]
        logit, _ = model.score_from_semantics(
            user_idx=t_user_idx[sl],
            news_idx=t_news_idx[sl],
            cat_idx=t_cat_idx[sl],
            subcat_idx=t_subcat_idx[sl],
            dense=t_dense[sl],
            user_sem=user_semantics,
            item_sem=batch_item_semantics,
            is_new_item=t_is_new_item[sl],
        )
        score_tensor[sl].copy_(logit)

    scores = score_tensor.cpu().numpy()
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    return [
        scores[int(offsets[i]) : int(offsets[i + 1])]
        for i in range(len(groups))
    ]
