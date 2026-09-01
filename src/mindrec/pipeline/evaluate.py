from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.metrics.benchmark import official_mind_benchmark_view
from mindrec.metrics.calibration import brier_score, expected_calibration_error
from mindrec.metrics.ranking import (
    auc,
    average_precision_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from mindrec.models.calibration import TemperatureScaler
from mindrec.models.dlrm import DLRMStudent
from mindrec.pipeline.ranker_scoring import (
    precompute_item_semantics,
    score_prepared_groups,
)
from mindrec.utils import (
    behavior_artifact_path,
    impression_artifact_path,
    log_device,
    resolve_device,
    save_json,
    teacher_artifact_root,
    test_split_name,
    validation_split_name,
)


def _load_model(
    cfg: dict[str, Any],
    proc_root: Path,
    device: torch.device,
    *,
    load_teacher_item: bool = True,
) -> tuple[DLRMStudent, np.ndarray, np.ndarray | None]:
    maps = IdMaps.load(proc_root / "id_maps.json")
    news = pd.read_parquet(proc_root / "news.parquet")
    n_users = max(maps.user2idx.values()) + 1
    n_news = int(news["news_idx"].max()) + 1
    # Match training/checkpoint shapes even though fit-unseen taxonomy keys
    # deliberately resolve to the neutral row 0.
    n_cats = len(maps.cat2idx) + 1
    n_subcats = len(maps.subcat2idx) + 1

    ranker_run_name = str(
        cfg.get("artifacts", {}).get("ranker_run_name", cfg["run_name"])
    )
    checkpoint_path = Path("runs") / ranker_run_name / "ranker" / "best.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Post-hoc runs reuse the frozen ranker's architecture and persisted
    # taxonomy masks directly from its checkpoint.
    model_cfg = ckpt["cfg"]

    ranker_base_name = (
        "item_ranker_base_emb.npy"
        if bool(cfg.get("knowledge_graph", {}).get("enabled", False))
        else "item_base_emb.npy"
    )
    teacher_root = teacher_artifact_root(cfg)
    ranker_base_path = teacher_root / ranker_base_name
    item_base = np.load(ranker_base_path)
    teacher_item = (
        np.load(teacher_root / "item_teacher_emb.npy")
        if load_teacher_item
        else None
    )

    dlrm_cfg = model_cfg["ranker"]["dlrm"]
    model = DLRMStudent(
        n_users=n_users,
        n_news=n_news,
        n_cats=n_cats,
        n_subcats=n_subcats,
        dense_dim=2,
        item_base_dim=int(item_base.shape[1]),
        emb_dim=int(dlrm_cfg["emb_dim"]),
        id_emb_dim=int(dlrm_cfg.get("id_emb_dim", dlrm_cfg["emb_dim"])),
        bottom_mlp=[int(x) for x in dlrm_cfg["bottom_mlp"]],
        top_mlp=[int(x) for x in dlrm_cfg["top_mlp"]],
        dropout=float(dlrm_cfg.get("dropout", 0.0)),
        fusion_heads=4,
        semantic_ff_mult=int(dlrm_cfg.get("semantic_ff_mult", 1)),
        semantic_dropout=float(
            dlrm_cfg.get("semantic_dropout", dlrm_cfg.get("dropout", 0.0))
        ),
        history_pooling=str(dlrm_cfg.get("history_pooling", "mean")),
        candidate_attention_heads=int(
            dlrm_cfg.get("candidate_attention_heads", 4)
        ),
        candidate_attention_dropout=float(
            dlrm_cfg.get("candidate_attention_dropout", 0.0)
        ),
        news_id_warm_scale=float(dlrm_cfg.get("news_id_warm_scale", 1.0)),
        news_id_cold_scale=float(dlrm_cfg.get("news_id_cold_scale", 1.0)),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, item_base, teacher_item


def _sanitize_slice_value(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    chars = [ch if ch.isalnum() else "_" for ch in text]
    text = "".join(chars)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


def _history_len_bucket(history_len: float) -> str:
    h = int(round(float(history_len)))
    if h <= 0:
        return "0"
    if h <= 4:
        return "1_4"
    if h <= 20:
        return "5_20"
    return "21_plus"


def _click_count_from_log1p(clicks_log1p: float) -> int:
    return max(0, int(round(float(np.expm1(clicks_log1p)))))


def _popularity_bucket(click_count: int) -> str:
    if click_count <= 0:
        return "0"
    if click_count <= 4:
        return "1_4"
    if click_count <= 19:
        return "5_19"
    return "20_plus"


def _resolve_eval_splits(cfg: dict[str, Any]) -> list[str]:
    raw_splits = cfg.get("eval", {}).get("report_splits", ["test"])
    resolved: list[str] = []
    for split in raw_splits:
        split_name = str(split)
        if split_name == "val":
            split_name = validation_split_name(cfg)
        elif split_name == "test":
            split_name = test_split_name(cfg)
        if split_name not in resolved:
            resolved.append(split_name)
    return resolved


def _attach_time_periods(impr: pd.DataFrame, n_periods: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = impr.copy()
    out["time_period"] = "unknown"
    meta: list[dict[str, Any]] = []

    if n_periods <= 0 or "time" not in out.columns:
        return out, meta

    parsed = pd.to_datetime(out["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    valid_idx = np.flatnonzero(parsed.notna().to_numpy())
    if len(valid_idx) == 0:
        return out, meta

    order = np.argsort(parsed.iloc[valid_idx].to_numpy(dtype="datetime64[ns]"), kind="stable")
    ordered_valid_idx = valid_idx[order]
    period_chunks = np.array_split(ordered_valid_idx, min(n_periods, len(ordered_valid_idx)))

    for i, chunk in enumerate(period_chunks, start=1):
        if len(chunk) == 0:
            continue
        label = f"period_{i}_of_{len(period_chunks)}"
        out.iloc[chunk, out.columns.get_loc("time_period")] = label
        times = parsed.iloc[chunk]
        meta.append(
            {
                "name": label,
                "n_impressions": int(len(chunk)),
                "time_min": str(times.min()),
                "time_max": str(times.max()),
            }
        )
    return out, meta


def _attach_behavior_time(impr: pd.DataFrame, beh: pd.DataFrame) -> pd.DataFrame:
    out = impr.copy()
    if "time" in out.columns:
        return out

    if len(out) == len(beh):
        out["time"] = beh["time"].to_numpy()
        return out

    beh_time = beh[["impression_id", "time"]].drop_duplicates(
        "impression_id",
        keep="first",
    )
    return out.merge(
        beh_time,
        on="impression_id",
        how="left",
        validate="many_to_one",
    )


def _metric_keys(ks: list[int]) -> list[str]:
    return (
        [f"ndcg@{k}" for k in ks]
        + [f"recall@{k}" for k in ks]
        + [f"map@{k}" for k in ks]
        + ["mrr", "auc"]
    )


def _make_metric_accumulator(keys: list[str]) -> dict[str, list[float]]:
    return {key: [] for key in keys}


def _append_metrics(
    acc: dict[str, list[float]], metric_values: dict[str, float], keys: list[str]
) -> None:
    for key in keys:
        acc[key].append(float(metric_values[key]))


def _finalize_metric_accumulator(acc: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values) if values else 0.0) for key, values in acc.items()}


def _evaluate_split(
    cfg: dict[str, Any],
    proc_root: Path,
    out_root: Path,
    split_name: str,
    device: torch.device,
    device_str: str,
    model: DLRMStudent,
    encoded_items: torch.Tensor,
    item_semantics: torch.Tensor,
    score_batch_size: int,
    scaler: TemperatureScaler | None,
) -> dict[str, Any]:
    news = pd.read_parquet(proc_root / "news.parquet")
    cat_lookup = (
        news[["cat_idx", "category"]]
        .drop_duplicates("cat_idx")
        .set_index("cat_idx")["category"]
        .to_dict()
    )
    subcat_lookup = (
        news[["subcat_idx", "subcategory"]]
        .drop_duplicates("subcat_idx")
        .set_index("subcat_idx")["subcategory"]
        .to_dict()
    )
    cat_lookup[0] = "unknown"
    subcat_lookup[0] = "unknown"

    impr = pd.read_parquet(impression_artifact_path(proc_root, split_name))
    beh = pd.read_parquet(behavior_artifact_path(proc_root, split_name))
    impr["impression_id"] = impr["impression_id"].astype(str)
    beh["impression_id"] = beh["impression_id"].astype(str)
    impr = _attach_behavior_time(impr, beh)
    time_periods = int(cfg.get("eval", {}).get("time_periods", 4))
    impr, time_period_meta = _attach_time_periods(impr, time_periods)
    ks = [int(k) for k in cfg["eval"]["ks"]]
    metric_keys = _metric_keys(ks)

    agg = _make_metric_accumulator(metric_keys)
    all_scores: list[float] = []
    all_probs: list[float] = []
    all_labels: list[float] = []

    slice_aggs: dict[str, dict[str, list[float]]] = {}
    slice_counts: dict[str, int] = {}

    def add_to_slice(name: str, metric_values: dict[str, float]) -> None:
        if name not in slice_aggs:
            slice_aggs[name] = _make_metric_accumulator(metric_keys)
            slice_counts[name] = 0
        _append_metrics(slice_aggs[name], metric_values, metric_keys)
        slice_counts[name] += 1

    with torch.no_grad():
        for _, r in tqdm(impr.iterrows(), total=len(impr), desc=f"Eval ranker ({split_name})"):
            labels = np.array(r["cand_label"], dtype=np.int32)
            if labels.sum() <= 0:
                continue
            user_idx = int(r["user_idx"])
            hist_news_idx = [int(x) for x in list(r["hist_news_idx"])]
            cand_news_idx = np.array(r["cand_news_idx"], dtype=np.int64)
            cand_cat_idx = np.array(r["cand_cat_idx"], dtype=np.int64)
            cand_subcat_idx = np.array(r["cand_subcat_idx"], dtype=np.int64)
            cand_is_new = np.array(r["cand_is_new_item"], dtype=np.int64)
            cand_clicks_log1p = np.array(r["cand_item_clicks_log1p"], dtype=np.float32)

            hlen = float(r["history_len"])
            dense = np.stack(
                [np.full_like(cand_clicks_log1p, hlen), cand_clicks_log1p], axis=1
            )

            scores = score_prepared_groups(
                model=model,
                groups=[
                    {
                        "user_idx": user_idx,
                        "hist_news_idx": hist_news_idx,
                        "cand_news_idx": cand_news_idx,
                        "cand_cat_idx": cand_cat_idx,
                        "cand_subcat_idx": cand_subcat_idx,
                        "cand_is_new": cand_is_new,
                        "dense": dense,
                    }
                ],
                encoded_items=encoded_items,
                item_semantics=item_semantics,
                batch_size=score_batch_size,
                device=device,
            )[0]
            probs_raw = 1.0 / (1.0 + np.exp(-scores))
            probs = scaler.predict_proba(scores) if scaler is not None else probs_raw

            all_scores.extend(scores.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.astype(float).tolist())

            metric_values = {
                "mrr": mrr(labels, scores),
                "auc": auc(labels, scores),
            }
            for k in ks:
                metric_values[f"ndcg@{k}"] = ndcg_at_k(labels, scores, k)
                metric_values[f"recall@{k}"] = recall_at_k(labels, scores, k)
                metric_values[f"map@{k}"] = average_precision_at_k(labels, scores, k)
            _append_metrics(agg, metric_values, metric_keys)

            add_to_slice("overall", metric_values)
            add_to_slice("cold_user" if int(r["is_cold_user"]) == 1 else "warm_user", metric_values)
            add_to_slice(
                f"history_len_bucket__{_history_len_bucket(hlen)}",
                metric_values,
            )
            add_to_slice(
                f"time_period__{_sanitize_slice_value(r['time_period'])}",
                metric_values,
            )

            clicked_mask = labels == 1
            clicked_is_new = cand_is_new[clicked_mask]
            clicked_click_counts = {
                _click_count_from_log1p(x) for x in cand_clicks_log1p[clicked_mask].tolist()
            }
            clicked_pop_buckets = {
                _popularity_bucket(click_count) for click_count in clicked_click_counts
            }
            clicked_cat_names = {
                _sanitize_slice_value(cat_lookup.get(int(cat_idx), "unknown"))
                for cat_idx in np.unique(cand_cat_idx[clicked_mask]).tolist()
            }
            clicked_subcat_names = {
                _sanitize_slice_value(subcat_lookup.get(int(sub_idx), "unknown"))
                for sub_idx in np.unique(cand_subcat_idx[clicked_mask]).tolist()
            }

            if bool((clicked_is_new == 1).any()):
                add_to_slice("impressions_with_clicked_new_item", metric_values)
            if bool((clicked_is_new == 0).any()):
                add_to_slice("impressions_with_clicked_warm_item", metric_values)
            for bucket in clicked_pop_buckets:
                add_to_slice(
                    f"impressions_with_clicked_popularity_bucket__{bucket}",
                    metric_values,
                )
            for cat_name in clicked_cat_names:
                add_to_slice(
                    f"impressions_with_clicked_category__{cat_name}",
                    metric_values,
                )
            for subcat_name in clicked_subcat_names:
                add_to_slice(
                    f"impressions_with_clicked_subcategory__{subcat_name}",
                    metric_values,
                )

    y = np.array(all_labels, dtype=np.float32)
    s = np.array(all_scores, dtype=np.float32)
    p_raw = 1.0 / (1.0 + np.exp(-s))
    p = np.array(all_probs, dtype=np.float32)

    ranking = _finalize_metric_accumulator(agg)
    out = {
        "ranking": ranking,
        "official_mind": official_mind_benchmark_view(ranking),
        "calibration": {
            "method": "temperature" if scaler is not None else "sigmoid",
            "temperature": float(scaler.temperature) if scaler is not None else 1.0,
            "raw_brier": brier_score(y, p_raw),
            "raw_ece_15": expected_calibration_error(y, p_raw, n_bins=15),
            "brier": brier_score(y, p),
            "ece_15": expected_calibration_error(y, p, n_bins=15),
        },
        "slices": {
            name: {
                **_finalize_metric_accumulator(values),
                "n_impressions": int(slice_counts[name]),
            }
            for name, values in slice_aggs.items()
        },
        "time_periods": time_period_meta,
        "n_impressions": int(len(impr)),
        "n_scored_pairs": int(len(y)),
        "device": device_str,
        "eval_split": split_name,
        "scoring_mode": "cached_item_and_history_semantics",
        "score_batch_size": score_batch_size,
    }
    save_json(out_root / f"ranker_eval_{split_name}.json", out)
    return out


def run_evaluate(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / "eval")

    device = resolve_device(cfg["ranker"].get("device", "cuda"))
    device_str = str(device)
    log_device(device, "Evaluate")

    model, item_base, _ = _load_model(
        cfg,
        proc_root,
        device,
        load_teacher_item=False,
    )
    calib_path = runs_root / "ranker" / "calibration.json"
    scaler = TemperatureScaler.load(calib_path) if calib_path.exists() else None
    eval_splits = _resolve_eval_splits(cfg)
    if not eval_splits:
        return
    eval_cfg = dict(cfg.get("eval", {}))
    score_batch_size = int(eval_cfg.get("batch_size", 2048))
    item_encoding_batch_size = int(
        eval_cfg.get("item_encoding_batch_size", max(score_batch_size, 8192))
    )
    if score_batch_size < 1:
        raise ValueError("eval.batch_size must be at least 1.")
    encoded_items, item_semantics = precompute_item_semantics(
        model=model,
        item_base=item_base,
        device=device,
        batch_size=item_encoding_batch_size,
        description="Pre-encode evaluation items",
    )

    # Temporal-evaluation consistency check:
    # ranker_eval_val.json:n_impressions should equal
    # preprocess_meta.json:n_validation_eval_impressions. A mismatch means the
    # evaluation artifact is stale and must be regenerated before its metrics
    # are treated as a baseline.
    split_results = {}
    for split_name in eval_splits:
        split_results[split_name] = _evaluate_split(
            cfg=cfg,
            proc_root=proc_root,
            out_root=out_root,
            split_name=split_name,
            device=device,
            device_str=device_str,
            model=model,
            encoded_items=encoded_items,
            item_semantics=item_semantics,
            score_batch_size=score_batch_size,
            scaler=scaler,
        )
