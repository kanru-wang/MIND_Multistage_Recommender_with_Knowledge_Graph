from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps, add_indices, build_id_maps, is_cold_user
from mindrec.data.mind_io import (
    count_behavior_rows,
    read_behaviors_tsv,
    read_news_tsv,
    sub_sample_behaviors,
)
from mindrec.utils import (
    behavior_artifact_path,
    impression_artifact_path,
    load_json,
    pair_artifact_path,
    save_json,
    set_seed,
)


def build_pairs(
    beh: pd.DataFrame,
    news_idx_df: pd.DataFrame,
    maps: IdMaps,
    item_clicks_train: dict[str, int],
    min_user_hist_for_warm: int,
    min_item_train_clicks_for_warm: int,
    max_history: int,
    neg_per_pos: int = 4,
    seed: int = 13,
) -> pd.DataFrame:
    set_seed(seed, seed_cuda=False)
    news_lookup = news_idx_df.set_index("news_id")[
        ["news_idx", "cat_idx", "subcat_idx"]
    ].to_dict(orient="index")

    rows = []
    for _, r in tqdm(beh.iterrows(), total=len(beh), desc="Build pairs"):
        user_id = str(r["user_id"])
        user_idx = maps.user2idx.get(user_id, 0)
        hist = r["history"]
        hist_news_idx = [
            maps.news2idx[h]
            for h in hist[-max_history:]
            if h in maps.news2idx and maps.news2idx[h] != 0
        ]
        cold_u = 1 if is_cold_user(hist, min_user_hist_for_warm) else 0

        cand_ids = list(r["cand_news_id"])
        labels = list(r["cand_label"])
        if not cand_ids:
            continue
        pos = [i for i, l in enumerate(labels) if l == 1]
        neg = [i for i, l in enumerate(labels) if l == 0]
        if not pos or not neg:
            continue

        # For each positive, sample negatives
        for pi in pos:
            pos_id = cand_ids[pi]
            neg_idx = np.random.choice(
                neg, size=min(neg_per_pos, len(neg)), replace=False
            )
            for j in [pi] + neg_idx.tolist():
                nid = cand_ids[j]
                lab = int(labels[j])
                meta = news_lookup.get(
                    nid, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0}
                )
                clicks = int(item_clicks_train.get(nid, 0))
                is_new = 1 if clicks < min_item_train_clicks_for_warm else 0
                rows.append(
                    {
                        "user_id": user_id,
                        "news_id": nid,
                        "user_idx": user_idx,
                        "news_idx": int(meta["news_idx"]),
                        "cat_idx": int(meta["cat_idx"]),
                        "subcat_idx": int(meta["subcat_idx"]),
                        "hist_news_idx": hist_news_idx,
                        "history_len": float(len(hist_news_idx)),
                        "item_clicks": float(clicks),
                        "item_clicks_log1p": float(np.log1p(clicks)),
                        "label": lab,
                        "is_cold_user": cold_u,
                        "is_new_item": is_new,
                    }
                )
    return pd.DataFrame(rows)


def build_impressions_for_eval(
    beh: pd.DataFrame,
    news_idx_df: pd.DataFrame,
    maps: IdMaps,
    item_clicks_train: dict[str, int],
    min_user_hist_for_warm: int,
    min_item_train_clicks_for_warm: int,
    max_history: int,
    require_positive_label: bool = True,
) -> pd.DataFrame:
    news_lookup = news_idx_df.set_index("news_id")[
        ["news_idx", "cat_idx", "subcat_idx"]
    ].to_dict(orient="index")
    rows = []
    for _, r in tqdm(beh.iterrows(), total=len(beh), desc="Build eval impressions"):
        user_id = str(r["user_id"])
        user_idx = maps.user2idx.get(user_id, 0)
        hist = r["history"]
        hist_news_idx = [
            maps.news2idx[h]
            for h in hist[-max_history:]
            if h in maps.news2idx and maps.news2idx[h] != 0
        ]
        cold_u = 1 if is_cold_user(hist, min_user_hist_for_warm) else 0

        cand_ids = list(r["cand_news_id"])
        labels = list(r["cand_label"])
        if not cand_ids:
            continue
        if require_positive_label and sum(labels) == 0:
            continue

        cand_news_idx = []
        cand_cat_idx = []
        cand_subcat_idx = []
        cand_is_new = []
        cand_clicks_log1p = []
        for nid in cand_ids:
            meta = news_lookup.get(nid, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0})
            clicks = int(item_clicks_train.get(nid, 0))
            is_new = 1 if clicks < min_item_train_clicks_for_warm else 0
            cand_news_idx.append(int(meta["news_idx"]))
            cand_cat_idx.append(int(meta["cat_idx"]))
            cand_subcat_idx.append(int(meta["subcat_idx"]))
            cand_is_new.append(is_new)
            cand_clicks_log1p.append(float(np.log1p(clicks)))

        row = {
            "impression_id": str(r["impression_id"]),
            "user_id": user_id,
            "user_idx": user_idx,
            "hist_news_idx": hist_news_idx,
            "history_len": float(len(hist_news_idx)),
            "is_cold_user": cold_u,
            "cand_news_id": cand_ids,
            "cand_label": labels,
            "cand_news_idx": cand_news_idx,
            "cand_cat_idx": cand_cat_idx,
            "cand_subcat_idx": cand_subcat_idx,
            "cand_is_new_item": cand_is_new,
            "cand_item_clicks_log1p": cand_clicks_log1p,
        }
        if "time" in r.index:
            row["time"] = r["time"]
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_click_counts(beh: pd.DataFrame) -> dict[str, int]:
    click_counts: dict[str, int] = {}
    for ids, labs in zip(beh["cand_news_id"], beh["cand_label"]):
        for nid, lab in zip(ids, labs):
            if lab == 1:
                click_counts[nid] = click_counts.get(nid, 0) + 1
    return click_counts


def _split_holdout_behaviors(
    beh: pd.DataFrame,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(beh) < 2:
        raise ValueError("Need at least 2 holdout impressions to create val/test splits.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be between 0 and 1, got {validation_fraction}."
        )

    n_val = int(np.floor(len(beh) * validation_fraction))
    n_val = min(max(n_val, 1), len(beh) - 1)

    meta: dict[str, Any] = {
        "strategy": "time",
        "validation_fraction": float(validation_fraction),
        "n_holdout_impressions": int(len(beh)),
        "n_val_impressions": int(n_val),
        "n_test_impressions": int(len(beh) - n_val),
    }

    ordered = beh.copy()
    ordered["_parsed_time"] = pd.to_datetime(
        ordered["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    ordered["_sort_impression_id"] = pd.to_numeric(
        ordered["impression_id"], errors="coerce"
    )
    ordered = ordered.sort_values(
        by=["_parsed_time", "_sort_impression_id", "impression_id"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)
    val = ordered.iloc[:n_val].drop(
        columns=["_parsed_time", "_sort_impression_id"]
    )
    test = ordered.iloc[n_val:].drop(
        columns=["_parsed_time", "_sort_impression_id"]
    )
    meta["val_time_min"] = str(val["time"].iloc[0])
    meta["val_time_max"] = str(val["time"].iloc[-1])
    meta["test_time_min"] = str(test["time"].iloc[0])
    meta["test_time_max"] = str(test["time"].iloc[-1])
    return val.reset_index(drop=True), test.reset_index(drop=True), meta


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def _split_behaviors_from_time(
    beh: pd.DataFrame,
    validation_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    parsed = pd.to_datetime(
        beh["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    if parsed.isna().any():
        raise ValueError("Temporal split found unparseable behavior timestamps.")

    validation_start_ts = pd.Timestamp(validation_start)
    if validation_start_ts.tzinfo is not None:
        validation_start_ts = validation_start_ts.tz_convert(None)

    train_mask = parsed < validation_start_ts
    val_mask = parsed >= validation_start_ts
    train = beh.loc[train_mask].reset_index(drop=True)
    val = beh.loc[val_mask].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError(
            "Temporal split produced an empty train or validation set. "
            f"validation_start={validation_start!r}, "
            f"n_train={len(train)}, n_val={len(val)}"
        )

    meta = {
        "strategy": "train_tail_time_split",
        "validation_start": str(validation_start_ts),
        "n_source_impressions": int(len(beh)),
        "n_train_impressions": int(len(train)),
        "n_validation_impressions": int(len(val)),
        "train_time_min": str(train["time"].iloc[0]),
        "train_time_max": str(train["time"].iloc[-1]),
        "val_time_min": str(val["time"].iloc[0]),
        "val_time_max": str(val["time"].iloc[-1]),
    }
    return train, val, meta


def _split_temporal_rerank_holdout(
    beh: pd.DataFrame,
    reporting_start: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split a labeled temporal holdout into reranker tuning and reporting rows."""

    parsed = pd.to_datetime(
        beh["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    if parsed.isna().any():
        raise ValueError("Reranker holdout split found unparseable timestamps.")

    reporting_start_ts = pd.Timestamp(reporting_start)
    if reporting_start_ts.tzinfo is not None:
        reporting_start_ts = reporting_start_ts.tz_convert(None)

    tuning = beh.loc[parsed < reporting_start_ts].reset_index(drop=True)
    reporting = beh.loc[parsed >= reporting_start_ts].reset_index(drop=True)
    if tuning.empty or reporting.empty:
        raise ValueError(
            "Reranker holdout split produced an empty tuning or reporting set. "
            f"reporting_start={reporting_start!r}, "
            f"n_tuning={len(tuning)}, n_reporting={len(reporting)}"
        )

    tuning_times = parsed.loc[parsed < reporting_start_ts]
    reporting_times = parsed.loc[parsed >= reporting_start_ts]
    meta = {
        "strategy": "chronological_reranker_tuning_reporting",
        "reporting_start": str(reporting_start_ts),
        "n_tuning_impressions": int(len(tuning)),
        "n_reporting_impressions": int(len(reporting)),
        "tuning_time_min": str(tuning_times.min()),
        "tuning_time_max": str(tuning_times.max()),
        "reporting_time_min": str(reporting_times.min()),
        "reporting_time_max": str(reporting_times.max()),
    }
    return tuning, reporting, meta


def _materialize_train_pairs(cfg: dict[str, Any]) -> bool:
    explicit = cfg.get("data", {}).get("materialize_train_pairs")
    if explicit is not None:
        return bool(explicit)
    # Preprocessing artifacts must not change merely because the active ranker
    # protocol changes. Hard-negative training may ignore this file, while a
    # baseline run sharing the same processed root still needs it.
    return True


def _resolve_temporal_rerank_holdout_settings(
    cfg: dict[str, Any],
) -> tuple[str, str, str] | None:
    temporal_cfg = dict(cfg["data"].get("temporal_validation", {}))
    reporting_start = temporal_cfg.get("rerank_reporting_start")
    if reporting_start is None:
        return None

    tuning_split = str(
        temporal_cfg.get("rerank_tuning_split_name", "rerank_tune")
    )
    reporting_split = str(
        temporal_cfg.get("rerank_reporting_split_name", "rerank_test")
    )
    reserved_split_names = {
        "train",
        "val",
        "submission",
        "submission_test",
        str(cfg.get("submission", {}).get("split_name", "submission_test")),
    }
    if (
        not tuning_split
        or not reporting_split
        or tuning_split == reporting_split
        or tuning_split in reserved_split_names
        or reporting_split in reserved_split_names
    ):
        raise ValueError(
            "Temporal reranker split names must be non-empty, distinct, and "
            "different from train, val, and the submission split."
        )
    return str(reporting_start), tuning_split, reporting_split


def run_prepare_rerank_holdout(cfg: dict[str, Any]) -> None:
    """Backfill reranker day splits from an existing combined temporal val."""

    if str(cfg["data"].get("mode", "standard")) != "temporal_tune":
        raise ValueError("prepare_rerank_holdout requires data.mode: temporal_tune.")
    rerank_settings = _resolve_temporal_rerank_holdout_settings(cfg)
    if rerank_settings is None:
        raise ValueError(
            "prepare_rerank_holdout requires "
            "data.temporal_validation.rerank_reporting_start."
        )
    reporting_start, tuning_split, reporting_split = rerank_settings

    proc_root = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    val_behaviors_path = behavior_artifact_path(proc_root, "val")
    val_impressions_path = impression_artifact_path(proc_root, "val")
    meta_path = proc_root / "preprocess_meta.json"
    for required_path in (val_behaviors_path, val_impressions_path, meta_path):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required combined validation artifact not found: {required_path}. "
                "Run preprocess first."
            )

    val_behaviors = pd.read_parquet(val_behaviors_path)
    _, _, split_meta = _split_temporal_rerank_holdout(
        val_behaviors, reporting_start=str(reporting_start)
    )
    val_impressions = pd.read_parquet(val_impressions_path)
    if "time" not in val_impressions.columns:
        behavior_ids = (
            val_behaviors["impression_id"].astype(str).reset_index(drop=True)
        )
        impression_ids = (
            val_impressions["impression_id"].astype(str).reset_index(drop=True)
        )
        if len(behavior_ids) != len(impression_ids) or not behavior_ids.equals(
            impression_ids
        ):
            raise ValueError(
                "Cannot backfill impression timestamps because the historical val "
                "behavior and impression artifacts are not row-aligned. Rerun full "
                "preprocessing instead."
            )
        val_impressions["time"] = val_behaviors["time"].reset_index(drop=True)
        if val_impressions["time"].isna().any():
            raise ValueError(
                "Could not recover every validation impression timestamp from "
                "val_behaviors.parquet. Rerun full preprocessing instead."
            )
    tuning_impr, reporting_impr, _ = _split_temporal_rerank_holdout(
        val_impressions, reporting_start=str(reporting_start)
    )
    tuning_impr.to_parquet(
        impression_artifact_path(proc_root, tuning_split), index=False
    )
    reporting_impr.to_parquet(
        impression_artifact_path(proc_root, reporting_split), index=False
    )

    split_meta.update(
        {
            "tuning_split_name": tuning_split,
            "reporting_split_name": reporting_split,
            "n_tuning_eval_impressions": int(len(tuning_impr)),
            "n_reporting_eval_impressions": int(len(reporting_impr)),
        }
    )
    meta = dict(load_json(meta_path))
    train_pairs_materialized = (proc_root / "train_pairs.parquet").exists()
    holdout_meta = dict(meta.get("holdout", {}))
    holdout_meta["rerank_holdout"] = split_meta
    meta.update(
        {
            "rerank_tuning_split_name": tuning_split,
            "rerank_reporting_split_name": reporting_split,
            "n_rerank_tuning_eval_impressions": int(len(tuning_impr)),
            "n_rerank_reporting_eval_impressions": int(len(reporting_impr)),
            "train_pairs_materialized": train_pairs_materialized,
            "ranker_train_pair_source": (
                "train_pairs.parquet"
                if train_pairs_materialized
                else "train_behaviors.parquet"
            ),
            "holdout": holdout_meta,
        }
    )
    save_json(meta_path, meta)
    print(
        "Prepared reranker holdout artifacts: "
        f"{tuning_split}={len(tuning_impr)}, {reporting_split}={len(reporting_impr)}"
    )


def _run_standard_preprocess(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed, seed_cuda=False)

    raw_root = Path(cfg["data"]["raw_root"])
    ds = cfg["data"]["dataset_name"]
    train_dir = raw_root / cfg["data"]["train_dir"]
    dev_dir = raw_root / cfg["data"]["dev_dir"]

    news_train = read_news_tsv(train_dir / "news.tsv")
    beh_train = read_behaviors_tsv(train_dir / "behaviors.tsv")

    news_dev = read_news_tsv(dev_dir / "news.tsv")
    beh_dev = read_behaviors_tsv(dev_dir / "behaviors.tsv")

    # Optionally sub-sample behaviors (faster laptop iteration)
    ss = cfg["data"].get("sub_sample", {})
    if ss.get("enabled", False):
        beh_train = sub_sample_behaviors(beh_train, int(ss["train_impressions"]), seed)
        beh_dev = sub_sample_behaviors(beh_dev, int(ss["dev_impressions"]), seed)

    news_all = (
        pd.concat([news_train, news_dev], axis=0)
        .drop_duplicates("news_id")
        .reset_index(drop=True)
    )

    maps = build_id_maps(news_all, beh_train)
    proc_root = ensure_dir(Path(cfg["data"]["processed_root"]) / ds)
    maps_path = proc_root / "id_maps.json"
    maps.save(maps_path)

    news_idx_df = add_indices(news_all, maps)
    news_idx_df.to_parquet(proc_root / "news.parquet", index=False)

    click_counts = _compute_click_counts(beh_train)
    save_json(proc_root / "item_click_counts.json", click_counts)

    holdout_cfg = dict(cfg["data"].get("holdout", {}))
    val_name = "val"
    test_name = "test"
    val_beh, test_beh, holdout_meta = _split_holdout_behaviors(
        beh=beh_dev,
        validation_fraction=float(holdout_cfg.get("validation_fraction", 0.8)),
    )
    ranker_neg_per_pos = int(cfg["data"].get("ranker_negatives_per_positive", 4))

    train_pairs_path = pair_artifact_path(proc_root, "train")
    pairs_train = None
    if _materialize_train_pairs(cfg):
        pairs_train = build_pairs(
            beh=beh_train,
            news_idx_df=news_idx_df,
            maps=maps,
            item_clicks_train=click_counts,
            min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
            min_item_train_clicks_for_warm=int(
                cfg["data"]["min_item_train_clicks_for_warm"]
            ),
            max_history=int(cfg["data"]["max_history"]),
            neg_per_pos=ranker_neg_per_pos,
            seed=seed,
        )
        pairs_train.to_parquet(train_pairs_path, index=False)
    else:
        _remove_if_exists(train_pairs_path)
    pairs_val = build_pairs(
        beh=val_beh,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
        max_history=int(cfg["data"]["max_history"]),
        neg_per_pos=ranker_neg_per_pos,
        seed=seed + 1,
    )
    pairs_test = build_pairs(
        beh=test_beh,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
        max_history=int(cfg["data"]["max_history"]),
        neg_per_pos=ranker_neg_per_pos,
        seed=seed + 2,
    )

    pairs_val.to_parquet(pair_artifact_path(proc_root, val_name), index=False)
    pairs_test.to_parquet(pair_artifact_path(proc_root, test_name), index=False)

    impr_val = build_impressions_for_eval(
        beh=val_beh,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
        max_history=int(cfg["data"]["max_history"]),
    )
    impr_test = build_impressions_for_eval(
        beh=test_beh,
        news_idx_df=news_idx_df,
        maps=maps,
        item_clicks_train=click_counts,
        min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
        min_item_train_clicks_for_warm=int(
            cfg["data"]["min_item_train_clicks_for_warm"]
        ),
        max_history=int(cfg["data"]["max_history"]),
    )
    impr_val.to_parquet(impression_artifact_path(proc_root, val_name), index=False)
    impr_test.to_parquet(impression_artifact_path(proc_root, test_name), index=False)
    beh_train.to_parquet(behavior_artifact_path(proc_root, "train"), index=False)
    val_beh.to_parquet(behavior_artifact_path(proc_root, val_name), index=False)
    test_beh.to_parquet(behavior_artifact_path(proc_root, test_name), index=False)

    meta = {
        "dataset": ds,
        "mode": "standard",
        "n_news": int(len(news_idx_df)),
        "n_train_impressions": int(len(beh_train)),
        "n_holdout_source_impressions": int(len(beh_dev)),
        "validation_split_name": val_name,
        "test_split_name": test_name,
        "n_validation_impressions": int(len(val_beh)),
        "n_test_impressions": int(len(test_beh)),
        "n_train_pairs": int(len(pairs_train)) if pairs_train is not None else 0,
        "train_pairs_materialized": pairs_train is not None,
        "ranker_train_pair_source": (
            "train_pairs.parquet"
            if pairs_train is not None
            else "train_behaviors.parquet"
        ),
        "n_validation_pairs": int(len(pairs_val)),
        "n_test_pairs": int(len(pairs_test)),
        "ranker_negatives_per_positive": ranker_neg_per_pos,
        "n_validation_eval_impressions": int(len(impr_val)),
        "n_test_eval_impressions": int(len(impr_test)),
        "holdout": holdout_meta,
    }
    save_json(proc_root / "preprocess_meta.json", meta)


def _run_multi_source_preprocess(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed, seed_cuda=False)

    raw_root = Path(cfg["data"]["raw_root"])
    ds = cfg["data"]["dataset_name"]
    train_dir = raw_root / cfg["data"]["train_dir"]
    dev_dir = raw_root / cfg["data"]["dev_dir"]
    mode = str(cfg["data"].get("mode", "leaderboard_submission"))
    test_dir = (
        raw_root / cfg["data"]["test_dir"]
        if "test_dir" in cfg["data"]
        else None
    )

    news_train = read_news_tsv(train_dir / "news.tsv")
    beh_train = read_behaviors_tsv(train_dir / "behaviors.tsv")
    news_dev = None
    beh_dev = None
    if mode in {
        "leaderboard_tune",
        "temporal_tune",
        "leaderboard_submission",
    }:
        news_dev = read_news_tsv(dev_dir / "news.tsv")
        beh_dev = read_behaviors_tsv(dev_dir / "behaviors.tsv")

    ss = cfg["data"].get("sub_sample", {})
    if ss.get("enabled", False):
        beh_train = sub_sample_behaviors(beh_train, int(ss["train_impressions"]), seed)
        if beh_dev is not None:
            beh_dev = sub_sample_behaviors(beh_dev, int(ss["dev_impressions"]), seed)

    news_parts = [news_train]
    n_submission_impressions = 0
    rerank_tuning_beh = None
    rerank_reporting_beh = None
    rerank_holdout_meta = None
    rerank_tuning_split = None
    rerank_reporting_split = None
    if mode == "leaderboard_tune":
        assert news_dev is not None and beh_dev is not None
        news_parts.append(news_dev)
        fit_beh = beh_train.reset_index(drop=True)
        val_beh = beh_dev.reset_index(drop=True)
        holdout_meta = {
            "strategy": "leaderboard_model_selection",
            "train_source": cfg["data"]["train_dir"],
            "validation_source": cfg["data"]["dev_dir"],
            "test_source": None,
        }
    elif mode == "temporal_tune":
        assert news_dev is not None and beh_dev is not None
        news_parts.append(news_dev)
        temporal_cfg = dict(cfg["data"].get("temporal_validation", {}))
        validation_start = str(
            temporal_cfg.get("validation_start", "2019-11-14 00:00:00")
        )
        fit_beh, train_tail_val, holdout_meta = _split_behaviors_from_time(
            beh=beh_train,
            validation_start=validation_start,
        )
        val_beh = pd.concat([train_tail_val, beh_dev], axis=0).reset_index(drop=True)
        val_times = pd.to_datetime(
            val_beh["time"],
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        if val_times.isna().any():
            raise ValueError("Large Temporal Val contains unparseable timestamps.")
        rerank_settings = _resolve_temporal_rerank_holdout_settings(cfg)
        if rerank_settings is not None:
            (
                rerank_reporting_start,
                rerank_tuning_split,
                rerank_reporting_split,
            ) = rerank_settings
            (
                rerank_tuning_beh,
                rerank_reporting_beh,
                rerank_holdout_meta,
            ) = _split_temporal_rerank_holdout(
                val_beh,
                reporting_start=str(rerank_reporting_start),
            )
            rerank_holdout_meta.update(
                {
                    "tuning_split_name": rerank_tuning_split,
                    "reporting_split_name": rerank_reporting_split,
                }
            )
        holdout_meta.update(
            {
                "strategy": "temporal_model_selection",
                "train_source": cfg["data"]["train_dir"],
                "validation_sources": [
                    f"{cfg['data']['train_dir']} tail",
                    cfg["data"]["dev_dir"],
                ],
                "n_train_tail_validation_impressions": int(len(train_tail_val)),
                "n_dev_validation_impressions": int(len(beh_dev)),
                "n_validation_impressions": int(len(val_beh)),
                "val_time_min": str(val_times.min()),
                "val_time_max": str(val_times.max()),
                "test_source": None,
            }
        )
        if rerank_holdout_meta is not None:
            holdout_meta["rerank_holdout"] = rerank_holdout_meta
    elif mode == "leaderboard_submission":
        if test_dir is None:
            raise ValueError("leaderboard_submission mode requires data.test_dir.")
        assert news_dev is not None and beh_dev is not None
        news_test = read_news_tsv(test_dir / "news.tsv")
        n_submission_impressions = count_behavior_rows(test_dir / "behaviors.tsv")
        news_parts.append(news_dev)
        news_parts.append(news_test)
        fit_beh = pd.concat([beh_train, beh_dev], axis=0).reset_index(drop=True)
        val_beh = None
        holdout_meta = {
            "strategy": "leaderboard_final_fit",
            "train_sources": [cfg["data"]["train_dir"], cfg["data"]["dev_dir"]],
            "validation_source": None,
            "test_source": cfg["data"]["test_dir"],
        }
    else:
        raise ValueError(f"Unsupported multi-source mode: {mode}")

    news_all = (
        pd.concat(news_parts, axis=0)
        .drop_duplicates("news_id")
        .reset_index(drop=True)
    )

    maps = build_id_maps(news_all, fit_beh)
    proc_root = ensure_dir(Path(cfg["data"]["processed_root"]) / ds)
    submission_split = str(cfg.get("submission", {}).get("split_name", "submission_test"))
    artifact_splits = {
        "val",
        submission_split,
        "rerank_tune",
        "rerank_test",
    }
    if rerank_tuning_split is not None:
        artifact_splits.add(rerank_tuning_split)
    if rerank_reporting_split is not None:
        artifact_splits.add(rerank_reporting_split)
    for split_name in artifact_splits:
        _remove_if_exists(pair_artifact_path(proc_root, split_name))
        _remove_if_exists(impression_artifact_path(proc_root, split_name))
        _remove_if_exists(behavior_artifact_path(proc_root, split_name))
    maps.save(proc_root / "id_maps.json")

    news_idx_df = add_indices(news_all, maps)
    news_idx_df.to_parquet(proc_root / "news.parquet", index=False)

    click_counts = _compute_click_counts(fit_beh)
    save_json(proc_root / "item_click_counts.json", click_counts)

    ranker_neg_per_pos = int(cfg["data"].get("ranker_negatives_per_positive", 4))
    train_pairs_path = pair_artifact_path(proc_root, "train")
    pairs_train = None
    if _materialize_train_pairs(cfg):
        pairs_train = build_pairs(
            beh=fit_beh,
            news_idx_df=news_idx_df,
            maps=maps,
            item_clicks_train=click_counts,
            min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
            min_item_train_clicks_for_warm=int(
                cfg["data"]["min_item_train_clicks_for_warm"]
            ),
            max_history=int(cfg["data"]["max_history"]),
            neg_per_pos=ranker_neg_per_pos,
            seed=seed,
        )
        pairs_train.to_parquet(train_pairs_path, index=False)
    else:
        _remove_if_exists(train_pairs_path)

    pairs_val = None
    impr_val = None
    impr_rerank_tuning = None
    impr_rerank_reporting = None
    if val_beh is not None:
        pairs_val = build_pairs(
            beh=val_beh,
            news_idx_df=news_idx_df,
            maps=maps,
            item_clicks_train=click_counts,
            min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
            min_item_train_clicks_for_warm=int(
                cfg["data"]["min_item_train_clicks_for_warm"]
            ),
            max_history=int(cfg["data"]["max_history"]),
            neg_per_pos=ranker_neg_per_pos,
            seed=seed + 1,
        )
        pairs_val.to_parquet(pair_artifact_path(proc_root, "val"), index=False)
        impr_val = build_impressions_for_eval(
            beh=val_beh,
            news_idx_df=news_idx_df,
            maps=maps,
            item_clicks_train=click_counts,
            min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
            min_item_train_clicks_for_warm=int(
                cfg["data"]["min_item_train_clicks_for_warm"]
            ),
            max_history=int(cfg["data"]["max_history"]),
        )
        impr_val.to_parquet(impression_artifact_path(proc_root, "val"), index=False)
        val_beh.to_parquet(behavior_artifact_path(proc_root, "val"), index=False)
        if rerank_tuning_beh is not None and rerank_reporting_beh is not None:
            assert rerank_tuning_split is not None
            assert rerank_reporting_split is not None
            assert rerank_holdout_meta is not None
            (
                impr_rerank_tuning,
                impr_rerank_reporting,
                _,
            ) = _split_temporal_rerank_holdout(
                impr_val,
                reporting_start=str(rerank_holdout_meta["reporting_start"]),
            )
            impr_rerank_tuning.to_parquet(
                impression_artifact_path(proc_root, rerank_tuning_split), index=False
            )
            impr_rerank_reporting.to_parquet(
                impression_artifact_path(proc_root, rerank_reporting_split), index=False
            )
            rerank_holdout_meta.update(
                {
                    "n_tuning_eval_impressions": int(len(impr_rerank_tuning)),
                    "n_reporting_eval_impressions": int(len(impr_rerank_reporting)),
                }
            )

    fit_beh.to_parquet(behavior_artifact_path(proc_root, "train"), index=False)

    meta = {
        "dataset": ds,
        "mode": mode,
        "n_news": int(len(news_idx_df)),
        "n_original_train_impressions": int(len(beh_train)),
        "n_original_dev_impressions": int(len(beh_dev)),
        "n_train_impressions": int(len(fit_beh)),
        "validation_split_name": "val" if val_beh is not None else None,
        "submission_split_name": submission_split,
        "n_validation_impressions": int(len(val_beh)) if val_beh is not None else 0,
        "n_submission_impressions": int(n_submission_impressions),
        "n_train_pairs": int(len(pairs_train)) if pairs_train is not None else 0,
        "train_pairs_materialized": pairs_train is not None,
        "ranker_train_pair_source": (
            "train_pairs.parquet"
            if pairs_train is not None
            else "train_behaviors.parquet"
        ),
        "n_validation_pairs": int(len(pairs_val)) if pairs_val is not None else 0,
        "ranker_negatives_per_positive": ranker_neg_per_pos,
        "n_validation_eval_impressions": int(len(impr_val)) if impr_val is not None else 0,
        "rerank_tuning_split_name": rerank_tuning_split,
        "rerank_reporting_split_name": rerank_reporting_split,
        "n_rerank_tuning_eval_impressions": (
            int(len(impr_rerank_tuning)) if impr_rerank_tuning is not None else 0
        ),
        "n_rerank_reporting_eval_impressions": (
            int(len(impr_rerank_reporting)) if impr_rerank_reporting is not None else 0
        ),
        "n_submission_eval_impressions": 0,
        "submission_eval_mode": "stream_raw_behaviors",
        "holdout": holdout_meta,
    }
    save_json(proc_root / "preprocess_meta.json", meta)


def run_preprocess(cfg: dict[str, Any]) -> None:
    mode = str(cfg["data"].get("mode", "standard"))
    if mode in {
        "leaderboard_tune",
        "temporal_tune",
        "leaderboard_submission",
    }:
        _run_multi_source_preprocess(cfg)
        return
    if mode != "standard":
        raise ValueError(f"Unknown data.mode: {mode}")
    _run_standard_preprocess(cfg)
