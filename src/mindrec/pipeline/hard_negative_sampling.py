from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.data.featurize import IdMaps, is_cold_user
from mindrec.models.teacher import TeacherTwoTower


def _choose_negative_offsets(
    scores: np.ndarray,
    n_select: int,
    hard_fraction: float,
    rng: np.random.Generator,
    hard_eligible_mask: np.ndarray | None = None,
) -> tuple[list[int], int, int]:
    all_offsets = np.arange(len(scores), dtype=np.int64)
    if hard_eligible_mask is None:
        hard_eligible_offsets = all_offsets
    else:
        hard_eligible_offsets = all_offsets[np.asarray(hard_eligible_mask, dtype=bool)]
    ranked = hard_eligible_offsets[
        np.argsort(-scores[hard_eligible_offsets], kind="stable")
    ]
    n_hard = min(n_select, int(round(n_select * hard_fraction)))
    chosen = ranked[:n_hard].tolist()
    n_hard = len(chosen)
    n_random = n_select - len(chosen)
    if n_random:
        remaining = np.setdiff1d(all_offsets, np.asarray(chosen, dtype=np.int64))
        chosen.extend(rng.choice(remaining, size=n_random, replace=False).tolist())
    return chosen, n_hard, n_random


def build_teacher_hard_negative_pairs(
    beh: pd.DataFrame,
    news_idx_df: pd.DataFrame,
    maps: IdMaps,
    item_clicks_train: dict[str, int],
    min_user_hist_for_warm: int,
    min_item_train_clicks_for_warm: int,
    max_history: int,
    negatives_per_positive: int,
    pool_size: int,
    hard_fraction: float,
    teacher_consistent_hard_only: bool,
    max_score_above_positive: float,
    hard_for_cold_users_only: bool,
    seed: int,
    teacher_model: TeacherTwoTower,
    teacher_item_tensor: torch.Tensor,
    device: torch.device,
    group_batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mine teacher-hard negatives without materializing the larger candidate pool."""
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1.")
    if pool_size < negatives_per_positive:
        raise ValueError("pool_size must be at least negatives_per_positive.")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1.")
    if group_batch_size < 1:
        raise ValueError("group_batch_size must be at least 1.")

    rng = np.random.default_rng(seed)
    news_lookup = news_idx_df.set_index("news_id")[
        ["news_idx", "cat_idx", "subcat_idx"]
    ].to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    n_groups = 0
    n_teacher_scored_pool_negatives = 0
    n_random_only_selected_negatives = 0
    n_hard_eligible_pool_negatives = 0
    n_hard_ineligible_pool_negatives = 0
    n_hard_mining_groups = 0
    n_random_only_groups = 0
    n_zero_history_random_only_groups = 0
    n_hard_negatives = 0
    n_random_negatives = 0

    def append_row(
        group: dict[str, Any],
        candidate_position: int,
        *,
        is_hard_negative: int = 0,
    ) -> None:
        nid = group["cand_ids"][candidate_position]
        meta = news_lookup.get(
            nid, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0}
        )
        clicks = int(item_clicks_train.get(nid, 0))
        rows.append(
            {
                "user_id": group["user_id"],
                "news_id": nid,
                "user_idx": group["user_idx"],
                "news_idx": int(meta["news_idx"]),
                "cat_idx": int(meta["cat_idx"]),
                "subcat_idx": int(meta["subcat_idx"]),
                "hist_news_idx": group["hist_news_idx"],
                "history_len": float(len(group["hist_news_idx"])),
                "item_clicks": float(clicks),
                "item_clicks_log1p": float(np.log1p(clicks)),
                "label": int(group["labels"][candidate_position]),
                "is_cold_user": group["is_cold_user"],
                "is_new_item": (
                    1 if clicks < min_item_train_clicks_for_warm else 0
                ),
                "is_hard_negative": int(is_hard_negative),
            }
        )

    def flush_pending() -> None:
        nonlocal n_hard_negatives, n_random_negatives
        nonlocal n_hard_eligible_pool_negatives, n_hard_ineligible_pool_negatives
        nonlocal n_hard_mining_groups
        if not pending:
            return
        histories = [group["hist_news_idx"] for group in pending]
        max_batch_history = max(len(history) for history in histories)
        history_idx = torch.zeros(
            (len(pending), max_batch_history), dtype=torch.long, device=device
        )
        history_mask = torch.zeros(
            (len(pending), max_batch_history), dtype=torch.bool, device=device
        )
        for row_idx, history in enumerate(histories):
            history_idx[row_idx, : len(history)] = torch.as_tensor(
                history, dtype=torch.long, device=device
            )
            history_mask[row_idx, : len(history)] = True

        with torch.no_grad():
            user_vectors = teacher_model.encode_user_from_item_vectors(
                teacher_item_tensor[history_idx],
                history_mask,
            )

            max_pool_size = max(len(group["negative_pool"]) for group in pending)
            pool_news_idx = torch.zeros(
                (len(pending), max_pool_size), dtype=torch.long, device=device
            )
            positive_news_idx = torch.zeros(
                len(pending), dtype=torch.long, device=device
            )
            for row_idx, group in enumerate(pending):
                positive_news_idx[row_idx] = int(
                    news_lookup.get(
                        group["cand_ids"][group["positive_position"]],
                        {"news_idx": 0},
                    )["news_idx"]
                )
                pool_positions = group["negative_pool"]
                pool_news_idx[row_idx, : len(pool_positions)] = torch.as_tensor(
                    [
                        int(
                            news_lookup.get(
                                group["cand_ids"][position],
                                {"news_idx": 0},
                            )["news_idx"]
                        )
                        for position in pool_positions
                    ],
                    dtype=torch.long,
                    device=device,
                )

            pool_scores = (
                teacher_item_tensor[pool_news_idx] * user_vectors.unsqueeze(1)
            ).sum(dim=2)
            positive_scores = (
                teacher_item_tensor[positive_news_idx] * user_vectors
            ).sum(dim=1)
            pool_lengths = torch.as_tensor(
                [len(group["negative_pool"]) for group in pending],
                dtype=torch.long,
                device=device,
            )
            pool_mask = (
                torch.arange(max_pool_size, device=device).unsqueeze(0)
                < pool_lengths.unsqueeze(1)
            )
            pool_scores = pool_scores.masked_fill(~pool_mask, float("-inf"))
            pool_scores_array = pool_scores.detach().cpu().numpy()

            for row_idx, group in enumerate(pending):
                pool_positions = group["negative_pool"]
                scores = pool_scores_array[row_idx, : len(pool_positions)]
                hard_eligible_mask = None
                n_hard_mining_groups += 1

                if teacher_consistent_hard_only:
                    hard_eligible_mask = (
                        scores
                        <= float(positive_scores[row_idx].detach().cpu())
                        + max_score_above_positive
                    )
                    n_hard_eligible_pool_negatives += int(
                        np.count_nonzero(hard_eligible_mask)
                    )
                    n_hard_ineligible_pool_negatives += int(
                        len(hard_eligible_mask) - np.count_nonzero(hard_eligible_mask)
                    )
                chosen_offsets, n_hard, n_random = _choose_negative_offsets(
                    scores=scores,
                    n_select=min(negatives_per_positive, len(pool_positions)),
                    hard_fraction=hard_fraction,
                    rng=rng,
                    hard_eligible_mask=hard_eligible_mask,
                )
                n_hard_negatives += n_hard
                n_random_negatives += n_random

                append_row(group, group["positive_position"])
                for choice_rank, offset in enumerate(chosen_offsets):
                    append_row(
                        group,
                        pool_positions[int(offset)],
                        is_hard_negative=1 if choice_rank < n_hard else 0,
                    )
        pending.clear()

    teacher_model.eval()
    for _, behavior in tqdm(
        beh.iterrows(), total=len(beh), desc="Build hard-negative pools"
    ):
        cand_ids = list(behavior["cand_news_id"])
        labels = list(behavior["cand_label"])
        positive_positions = [
            position for position, label in enumerate(labels) if label == 1
        ]
        negative_positions = np.asarray(
            [position for position, label in enumerate(labels) if label == 0],
            dtype=np.int64,
        )
        if not positive_positions or len(negative_positions) == 0:
            continue

        history = list(behavior["history"])
        hist_news_idx = [
            maps.news2idx[nid]
            for nid in history[-max_history:]
            if nid in maps.news2idx and maps.news2idx[nid] != 0
        ]
        cold_u = 1 if is_cold_user(history, min_user_hist_for_warm) else 0
        for positive_position in positive_positions:
            group = {
                "user_id": str(behavior["user_id"]),
                "user_idx": maps.user2idx.get(str(behavior["user_id"]), 0),
                "hist_news_idx": hist_news_idx,
                "is_cold_user": cold_u,
                "cand_ids": cand_ids,
                "labels": labels,
                "positive_position": positive_position,
            }
            n_groups += 1

            has_usable_history = bool(hist_news_idx)
            use_hard_mining = (
                hard_fraction > 0.0
                and has_usable_history
                and (not hard_for_cold_users_only or cold_u == 1)
            )
            if not use_hard_mining:
                n_random_only_groups += 1
                if not has_usable_history:
                    n_zero_history_random_only_groups += 1
                sample_size = min(negatives_per_positive, len(negative_positions))
                chosen_negatives = rng.choice(
                    negative_positions, size=sample_size, replace=False
                ).tolist()
                n_random_only_selected_negatives += len(chosen_negatives)
                n_random_negatives += len(chosen_negatives)
                append_row(group, positive_position)
                for negative_position in chosen_negatives:
                    append_row(group, int(negative_position))
                continue

            sample_size = min(pool_size, len(negative_positions))
            negative_pool = rng.choice(
                negative_positions, size=sample_size, replace=False
            ).tolist()
            n_teacher_scored_pool_negatives += len(negative_pool)
            group["negative_pool"] = negative_pool
            pending.append(group)
            if len(pending) >= group_batch_size:
                flush_pending()
    flush_pending()

    n_selected_negatives = n_hard_negatives + n_random_negatives
    stats: dict[str, Any] = {
        "n_groups": int(n_groups),
        "n_teacher_scored_pool_negatives": int(
            n_teacher_scored_pool_negatives
        ),
        "n_random_only_selected_negatives": int(
            n_random_only_selected_negatives
        ),
        "n_hard_eligible_pool_negatives": int(n_hard_eligible_pool_negatives),
        "n_hard_ineligible_pool_negatives": int(n_hard_ineligible_pool_negatives),
        "n_hard_mining_groups": int(n_hard_mining_groups),
        "n_random_only_groups": int(n_random_only_groups),
        "n_zero_history_random_only_groups": int(
            n_zero_history_random_only_groups
        ),
        "n_selected_rows": int(len(rows)),
        "n_selected_negatives": int(n_selected_negatives),
        "n_hard_negatives": int(n_hard_negatives),
        "n_random_negatives": int(n_random_negatives),
        "teacher_consistent_hard_only": bool(teacher_consistent_hard_only),
        "max_score_above_positive": float(max_score_above_positive),
        "hard_for_cold_users_only": bool(hard_for_cold_users_only),
        "hard_fraction_requested": float(hard_fraction),
        "hard_fraction_actual": (
            float(n_hard_negatives / n_selected_negatives)
            if n_selected_negatives
            else 0.0
        ),
    }
    return pd.DataFrame(rows), stats
