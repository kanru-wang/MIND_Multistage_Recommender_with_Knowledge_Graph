from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.data.mind_io import count_behavior_rows, iter_behaviors_tsv
from mindrec.models.dlrm import DLRMStudent
from mindrec.pipeline.evaluate import _load_model
from mindrec.utils import load_json, log_device, resolve_device, save_json


def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return ranks


def _has_near_tied_scores(scores: np.ndarray, threshold: float) -> bool:
    """Return whether FP32 perturbations could change an impression's ordering."""
    if len(scores) < 2:
        return False
    if not np.isfinite(scores).all():
        return True
    ordered = np.sort(scores)
    return bool(np.any(np.diff(ordered) <= threshold))


def _write_prediction_zip(prediction_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname="prediction.txt")


@torch.inference_mode()
def _precompute_item_semantics(
    model: DLRMStudent,
    item_base: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the shared and candidate-specific semantic vectors for every item."""
    if batch_size < 1:
        raise ValueError("submission.item_encoding_batch_size must be at least 1.")

    n_items = int(item_base.shape[0])
    emb_dim = int(model.item_base_proj.out_features)
    dtype = model.item_base_proj.weight.dtype
    encoded_items = torch.empty((n_items, emb_dim), dtype=dtype, device=device)
    item_semantics = torch.empty_like(encoded_items)

    for start in tqdm(
        range(0, n_items, batch_size),
        desc="Pre-encode submission items",
    ):
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
def _score_prepared_groups(
    model: DLRMStudent,
    groups: list[dict[str, Any]],
    encoded_items: torch.Tensor,
    item_semantics: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    """Score buffered impressions while encoding each distinct history only once."""
    if not groups:
        return []
    if batch_size < 1:
        raise ValueError("submission.batch_size must be at least 1.")

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

    group_user_semantics = model.encode_user_semantics_from_history(
        encoded_items[history_idx],
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
        logit, _ = model.score_from_semantics(
            user_idx=t_user_idx[sl],
            news_idx=t_news_idx[sl],
            cat_idx=t_cat_idx[sl],
            subcat_idx=t_subcat_idx[sl],
            dense=t_dense[sl],
            user_sem=group_user_semantics[t_group_idx[sl]],
            item_sem=item_semantics[t_news_idx[sl]],
            is_new_item=t_is_new_item[sl],
        )
        score_tensor[sl].copy_(logit)

    scores = score_tensor.cpu().numpy()
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    return [
        scores[int(offsets[i]) : int(offsets[i + 1])]
        for i in range(len(groups))
    ]


@torch.inference_mode()
def _score_reference_group(
    model: DLRMStudent,
    group: dict[str, Any],
    item_base: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Reproduce the original per-impression FP32 submission scoring path."""
    if batch_size < 1:
        raise ValueError("submission.reference_batch_size must be at least 1.")

    news_idx = group["cand_news_idx"]
    n_candidates = len(news_idx)
    if n_candidates == 0:
        return np.empty(0, dtype=np.float32)

    history = group["hist_news_idx"]
    logits: list[np.ndarray] = []
    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)
        sl = slice(start, stop)
        current_batch_size = stop - start
        if history:
            history_array = item_base[np.asarray(history, dtype=np.int64)]
            history_base = torch.tensor(
                history_array, dtype=torch.float32, device=device
            ).unsqueeze(0)
            history_base = history_base.repeat(current_batch_size, 1, 1)
            history_mask = torch.ones(
                (current_batch_size, len(history)),
                dtype=torch.bool,
                device=device,
            )
        else:
            history_base = torch.zeros(
                (current_batch_size, 1, item_base.shape[1]),
                dtype=torch.float32,
                device=device,
            )
            history_mask = torch.zeros(
                (current_batch_size, 1), dtype=torch.bool, device=device
            )

        logit, _ = model(
            user_idx=torch.tensor(
                [group["user_idx"]] * current_batch_size,
                dtype=torch.long,
                device=device,
            ),
            news_idx=torch.tensor(news_idx[sl], dtype=torch.long, device=device),
            cat_idx=torch.tensor(
                group["cand_cat_idx"][sl], dtype=torch.long, device=device
            ),
            subcat_idx=torch.tensor(
                group["cand_subcat_idx"][sl], dtype=torch.long, device=device
            ),
            dense=torch.tensor(
                group["dense"][sl], dtype=torch.float32, device=device
            ),
            item_base=torch.tensor(
                item_base[news_idx[sl]], dtype=torch.float32, device=device
            ),
            history_item_base=history_base,
            history_mask=history_mask,
            is_new_item=torch.tensor(
                group["cand_is_new"][sl], dtype=torch.long, device=device
            ),
        )
        logits.append(logit.cpu().numpy())

    return np.concatenate(logits)


def run_write_submission(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / "submission")

    submission_cfg = dict(cfg.get("submission", {}))
    split_name = str(submission_cfg.get("split_name", "submission_test"))
    batch_size = int(submission_cfg.get("batch_size", 2048))
    candidate_buffer_size = int(
        submission_cfg.get("candidate_buffer_size", batch_size * 8)
    )
    item_encoding_batch_size = int(
        submission_cfg.get("item_encoding_batch_size", max(batch_size, 8192))
    )
    exact_rank_guard_threshold = float(
        submission_cfg.get("exact_rank_guard_threshold", 1.0e-5)
    )
    reference_batch_size = int(submission_cfg.get("reference_batch_size", 2048))
    save_scores = bool(submission_cfg.get("save_scores", False))
    if batch_size < 1:
        raise ValueError("submission.batch_size must be at least 1.")
    if candidate_buffer_size < batch_size:
        raise ValueError(
            "submission.candidate_buffer_size must be at least submission.batch_size."
        )
    if exact_rank_guard_threshold < 0.0:
        raise ValueError(
            "submission.exact_rank_guard_threshold must be non-negative."
        )
    if reference_batch_size < 1:
        raise ValueError("submission.reference_batch_size must be at least 1.")

    device = resolve_device(cfg["ranker"].get("device", "cuda"))
    device_str = str(device)
    log_device(device, "Submission")

    model, item_base, _ = _load_model(cfg, proc_root, runs_root, device)
    maps = IdMaps.load(proc_root / "id_maps.json")
    news = pd.read_parquet(proc_root / "news.parquet")
    news_lookup = news.set_index("news_id")[["news_idx", "cat_idx", "subcat_idx"]].to_dict(
        orient="index"
    )
    click_counts = load_json(proc_root / "item_click_counts.json")
    raw_root = Path(cfg["data"]["raw_root"])
    test_dir = raw_root / cfg["data"]["test_dir"]
    behavior_path = test_dir / "behaviors.tsv"
    n_impressions = count_behavior_rows(behavior_path)

    prediction_path = out_root / "prediction.txt"
    scored_path = out_root / f"{split_name}_scores.parquet"
    zip_path = out_root / "prediction.zip"
    scored_rows: list[dict[str, Any]] = []

    encoded_items, item_semantics = _precompute_item_semantics(
        model=model,
        item_base=item_base,
        device=device,
        batch_size=item_encoding_batch_size,
    )

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        pending_groups: list[dict[str, Any]] = []
        n_pending_candidates = 0
        n_reference_rescored_impressions = 0

        def flush_pending_groups() -> None:
            nonlocal n_pending_candidates, n_reference_rescored_impressions
            if not pending_groups:
                return
            scores_by_group = _score_prepared_groups(
                model=model,
                groups=pending_groups,
                encoded_items=encoded_items,
                item_semantics=item_semantics,
                batch_size=batch_size,
                device=device,
            )
            for group, scores in zip(pending_groups, scores_by_group):
                if _has_near_tied_scores(scores, exact_rank_guard_threshold):
                    scores = _score_reference_group(
                        model=model,
                        group=group,
                        item_base=item_base,
                        batch_size=reference_batch_size,
                        device=device,
                    )
                    n_reference_rescored_impressions += 1
                ranks = _scores_to_ranks(scores)
                rank_json = json.dumps(ranks.tolist(), separators=(",", ":"))
                f.write(f"{group['impression_id']} {rank_json}\n")
                if save_scores:
                    scored_rows.append(
                        {
                            "impression_id": str(group["impression_id"]),
                            "cand_news_id": group["cand_news_id"],
                            "score": scores.astype(float).tolist(),
                            "rank": ranks.astype(int).tolist(),
                        }
                    )
            pending_groups.clear()
            n_pending_candidates = 0

        for r in tqdm(
            iter_behaviors_tsv(behavior_path),
            total=n_impressions,
            desc=f"Score submission ({split_name})",
        ):
            user_id = str(r["user_id"])
            history = list(r["history"])
            user_idx = maps.user2idx.get(user_id, 0)
            hist_news_idx = [
                maps.news2idx[h]
                for h in history[-int(cfg["data"]["max_history"]) :]
                if h in maps.news2idx and maps.news2idx[h] != 0
            ]
            cand_news_id = [str(x) for x in list(r["cand_news_id"])]

            cand_news_idx = []
            cand_cat_idx = []
            cand_subcat_idx = []
            cand_is_new = []
            cand_clicks_log1p = []
            for news_id in cand_news_id:
                meta = news_lookup.get(
                    news_id, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0}
                )
                clicks = int(click_counts.get(news_id, 0))
                cand_news_idx.append(int(meta["news_idx"]))
                cand_cat_idx.append(int(meta["cat_idx"]))
                cand_subcat_idx.append(int(meta["subcat_idx"]))
                cand_is_new.append(
                    1
                    if clicks < int(cfg["data"]["min_item_train_clicks_for_warm"])
                    else 0
                )
                cand_clicks_log1p.append(float(np.log1p(clicks)))

            cand_news_idx = np.asarray(cand_news_idx, dtype=np.int64)
            cand_cat_idx = np.asarray(cand_cat_idx, dtype=np.int64)
            cand_subcat_idx = np.asarray(cand_subcat_idx, dtype=np.int64)
            cand_is_new = np.asarray(cand_is_new, dtype=np.int64)
            cand_clicks_log1p = np.asarray(cand_clicks_log1p, dtype=np.float32)

            hlen = float(len(hist_news_idx))
            dense = np.stack(
                [np.full_like(cand_clicks_log1p, hlen), cand_clicks_log1p],
                axis=1,
            )

            pending_groups.append(
                {
                    "impression_id": str(r["impression_id"]),
                    "cand_news_id": cand_news_id,
                    "user_idx": int(user_idx),
                    "hist_news_idx": hist_news_idx,
                    "cand_news_idx": cand_news_idx,
                    "cand_cat_idx": cand_cat_idx,
                    "cand_subcat_idx": cand_subcat_idx,
                    "cand_is_new": cand_is_new,
                    "dense": dense,
                }
            )
            n_pending_candidates += len(cand_news_idx)
            if n_pending_candidates >= candidate_buffer_size:
                flush_pending_groups()

        flush_pending_groups()

    if save_scores:
        pd.DataFrame(scored_rows).to_parquet(scored_path, index=False)
    _write_prediction_zip(prediction_path, zip_path)
    save_json(
        out_root / "submission_meta.json",
        {
            "split_name": split_name,
            "n_impressions": int(n_impressions),
            "prediction_path": str(prediction_path),
            "zip_path": str(zip_path),
            "scores_path": str(scored_path) if save_scores else None,
            "save_scores": save_scores,
            "behavior_path": str(behavior_path),
            "device": device_str,
            "batch_size": batch_size,
            "candidate_buffer_size": candidate_buffer_size,
            "item_encoding_batch_size": item_encoding_batch_size,
            "exact_rank_guard_threshold": exact_rank_guard_threshold,
            "reference_batch_size": reference_batch_size,
            "n_reference_rescored_impressions": n_reference_rescored_impressions,
            "scoring_mode": "cached_item_semantics_with_exact_rank_guard",
            "format": "MIND leaderboard prediction.txt: impression_id compact_json_ranks",
        },
    )
