from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.data.taxonomy_support import (
    TaxonomySupport,
    taxonomy_support_path,
    validate_taxonomy_mapping_artifact,
)
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


def _validate_ranker_checkpoint_config(
    actual: dict[str, Any],
    expected: dict[str, Any],
    checkpoint_path: Path,
    prefix: str = "",
) -> None:
    """Require an optional nested config subset to match a frozen checkpoint."""
    mismatches: list[str] = []

    def compare(
        actual_value: Any,
        expected_value: Any,
        path: str,
    ) -> None:
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                mismatches.append(
                    f"{path}: expected a mapping, found {actual_value!r}"
                )
                return
            for key, child_expected in expected_value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key not in actual_value:
                    mismatches.append(f"{child_path}: missing from checkpoint config")
                    continue
                compare(actual_value[key], child_expected, child_path)
            return
        if actual_value != expected_value:
            mismatches.append(
                f"{path}: expected {expected_value!r}, found {actual_value!r}"
            )

    compare(actual, expected, prefix)
    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(
            f"Ranker checkpoint does not match the required experiment baseline: "
            f"{checkpoint_path}. {details}. Retrain the required baseline before "
            "writing this submission."
        )


def _load_model(
    cfg: dict[str, Any],
    proc_root: Path,
    runs_root: Path,
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
    # The checkpoint is authoritative for architecture-affecting settings. This
    # also lets a post-hoc output run reuse a frozen ranker without copying it.
    model_cfg = ckpt.get("cfg", cfg)
    expected_checkpoint_cfg = cfg.get("artifacts", {}).get(
        "ranker_expected_config"
    )
    if expected_checkpoint_cfg:
        _validate_ranker_checkpoint_config(
            model_cfg,
            expected_checkpoint_cfg,
            checkpoint_path,
        )
    require_taxonomy_support = bool(
        cfg.get("artifacts", {}).get("require_taxonomy_support", False)
    )
    if require_taxonomy_support:
        validate_taxonomy_mapping_artifact(proc_root, maps)
    support_path = taxonomy_support_path(ranker_run_name)
    taxonomy_support = (
        TaxonomySupport.load(
            support_path,
            maps=maps,
            checkpoint_path=checkpoint_path,
            ranker_run_name=ranker_run_name,
            dataset_name=str(cfg["data"]["dataset_name"]),
        )
        if support_path.exists()
        else None
    )
    if require_taxonomy_support and taxonomy_support is None:
        raise FileNotFoundError(
            f"Missing taxonomy-support artifact: {support_path}. "
            "Build it from the exact ranker training pairs before scoring."
        )

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
        news_id_warm_scale=float(dlrm_cfg.get("news_id_warm_scale", 1.0)),
        news_id_cold_scale=float(dlrm_cfg.get("news_id_cold_scale", 1.0)),
        supported_cat_ids=(
            taxonomy_support.supported_cat_ids
            if taxonomy_support is not None
            else None
        ),
        supported_subcat_ids=(
            taxonomy_support.supported_subcat_ids
            if taxonomy_support is not None
            else None
        ),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, item_base, teacher_item


def _expand_history_base(
    item_base: np.ndarray, hist_idx: list[int], batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if hist_idx:
        hist_np = item_base[np.asarray(hist_idx, dtype=np.int64)]
        hist = torch.tensor(hist_np, dtype=torch.float32, device=device).unsqueeze(0)
        hist = hist.repeat(batch_size, 1, 1)
        mask = torch.ones((batch_size, len(hist_idx)), dtype=torch.bool, device=device)
        return hist, mask
    hist = torch.zeros((batch_size, 1, item_base.shape[1]), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    return hist, mask


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
    runs_root: Path,
    out_root: Path,
    split_name: str,
    device: torch.device,
    device_str: str,
    model: DLRMStudent,
    item_base: np.ndarray,
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

            logits = []
            bs = 2048
            for i in range(0, len(cand_news_idx), bs):
                sl = slice(i, i + bs)
                batch_size = len(cand_news_idx[sl])
                b_user = torch.tensor(
                    [user_idx] * batch_size, dtype=torch.long, device=device
                )
                b_news = torch.tensor(
                    cand_news_idx[sl], dtype=torch.long, device=device
                )
                b_cat = torch.tensor(cand_cat_idx[sl], dtype=torch.long, device=device)
                b_sub = torch.tensor(
                    cand_subcat_idx[sl], dtype=torch.long, device=device
                )
                b_is_new = torch.tensor(
                    cand_is_new[sl], dtype=torch.long, device=device
                )
                b_dense = torch.tensor(dense[sl], dtype=torch.float32, device=device)
                b_item_base = torch.tensor(
                    item_base[cand_news_idx[sl]], dtype=torch.float32, device=device
                )
                b_hist_base, b_hist_mask = _expand_history_base(
                    item_base=item_base,
                    hist_idx=hist_news_idx,
                    batch_size=batch_size,
                    device=device,
                )
                logit, _ = model(
                    user_idx=b_user,
                    news_idx=b_news,
                    cat_idx=b_cat,
                    subcat_idx=b_sub,
                    dense=b_dense,
                    item_base=b_item_base,
                    history_item_base=b_hist_base,
                    history_mask=b_hist_mask,
                    is_new_item=b_is_new,
                )
                logits.append(logit.detach().cpu().numpy())
            scores = np.concatenate(logits, axis=0)
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
        runs_root,
        device,
        load_teacher_item=False,
    )
    calib_path = runs_root / "ranker" / "calibration.json"
    scaler = TemperatureScaler.load(calib_path) if calib_path.exists() else None

    # Temporal-evaluation consistency check:
    # ranker_eval_val.json:n_impressions should equal
    # preprocess_meta.json:n_validation_eval_impressions. A mismatch means the
    # evaluation artifact is stale and must be regenerated before its metrics
    # are treated as a baseline.
    split_results = {}
    for split_name in _resolve_eval_splits(cfg):
        split_results[split_name] = _evaluate_split(
            cfg=cfg,
            proc_root=proc_root,
            runs_root=runs_root,
            out_root=out_root,
            split_name=split_name,
            device=device,
            device_str=device_str,
            model=model,
            item_base=item_base,
            scaler=scaler,
        )
