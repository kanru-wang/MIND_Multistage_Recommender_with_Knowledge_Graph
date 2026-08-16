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
from mindrec.data.item_age import ItemAgeIndex, item_age_artifact_path
from mindrec.data.mind_io import count_behavior_rows, iter_behaviors_tsv
from mindrec.data.recency_tiebreaker import (
    apply_recency_tiebreaker,
    recency_tiebreaker_config,
)
from mindrec.models.dlrm import DLRMStudent
from mindrec.pipeline.evaluate import _load_model
from mindrec.pipeline.ranker_scoring import (
    precompute_item_semantics,
    score_prepared_groups,
)
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

    # Submission output policy:
    # By default, score the hidden test file as a stream and do not materialize
    # the much larger per-candidate scores parquet. Set submission.save_scores
    # to true only when that parquet is explicitly needed for debugging. For
    # the standard Large submission run, its path is:
    # runs/mind_large_submission_hard_neg_v4/submission/
    # submission_test_scores.parquet
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

    model, item_base, _ = _load_model(
        cfg,
        proc_root,
        device,
        load_teacher_item=False,
    )
    maps = IdMaps.load(proc_root / "id_maps.json")
    news = pd.read_parquet(proc_root / "news.parquet")
    news_lookup = news.set_index("news_id")[["news_idx", "cat_idx", "subcat_idx"]].to_dict(
        orient="index"
    )
    recency_cfg = recency_tiebreaker_config(cfg)
    recency_alpha = float(recency_cfg["alpha"]) if recency_cfg["enabled"] else 0.0
    recency_age_index: ItemAgeIndex | None = None
    recency_age_lookup = None
    recency_uses_model_news_indices = False
    if recency_cfg["enabled"]:
        recency_age_root = (
            Path(cfg["data"]["processed_root"])
            / recency_cfg["age_dataset_name"]
        )
        recency_age_maps = IdMaps.load(recency_age_root / "id_maps.json")
        if recency_age_root.resolve() == proc_root.resolve():
            recency_uses_model_news_indices = True
        else:
            recency_age_lookup = recency_age_maps.news2idx.get
        recency_age_index = ItemAgeIndex.load(
            item_age_artifact_path(recency_age_root),
            expected_max_age_hours=recency_cfg["max_age_hours"],
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

    encoded_items, item_semantics = precompute_item_semantics(
        model=model,
        item_base=item_base,
        device=device,
        batch_size=item_encoding_batch_size,
        description="Pre-encode submission items",
    )

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        pending_groups: list[dict[str, Any]] = []
        n_pending_candidates = 0
        n_reference_rescored_impressions = 0

        def flush_pending_groups() -> None:
            nonlocal n_pending_candidates, n_reference_rescored_impressions
            if not pending_groups:
                return
            scores_by_group = score_prepared_groups(
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
                if recency_cfg["enabled"]:
                    scores = apply_recency_tiebreaker(
                        scores,
                        group["recency_item_age_log1p"],
                        recency_alpha,
                    )
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
            if recency_age_index is not None:
                if recency_uses_model_news_indices:
                    recency_news_idx = cand_news_idx
                else:
                    assert recency_age_lookup is not None
                    recency_news_idx = np.asarray(
                        [
                            int(recency_age_lookup(news_id, 0))
                            for news_id in cand_news_id
                        ],
                        dtype=np.int64,
                    )
                recency_item_age_log1p = recency_age_index.ages(
                    recency_news_idx,
                    r.get("time"),
                )
            else:
                recency_item_age_log1p = np.zeros(
                    len(cand_news_idx), dtype=np.float32
                )
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
                    "recency_item_age_log1p": recency_item_age_log1p,
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
            "scoring_mode": (
                "cached_item_and_history_semantics_with_"
                "candidate_attention_and_exact_rank_guard"
                if model.history_pooling == "candidate_attention"
                else "cached_item_semantics_with_exact_rank_guard"
            ),
            "history_pooling": model.history_pooling,
            "posthoc_recency": {
                "enabled": recency_cfg["enabled"],
                "alpha": recency_alpha,
                "age_dataset_name": recency_cfg["age_dataset_name"],
                "formula": (
                    "zscore_within_impression(baseline_logit) + "
                    "alpha * freshness_percentile"
                ),
            },
            "format": "MIND leaderboard prediction.txt: impression_id compact_json_ranks",
        },
    )
