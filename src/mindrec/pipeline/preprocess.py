from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps, add_indices, build_id_maps, is_cold_user
from mindrec.data.item_trend import (
    ItemTrendIndex,
    build_item_trend_index,
    item_trend_artifact_path,
    item_trend_config,
)
from mindrec.data.mind_io import (
    count_behavior_rows,
    read_behaviors_tsv,
    read_news_tsv,
    sub_sample_behaviors,
)
from mindrec.utils import (
    behavior_artifact_path,
    impression_artifact_path,
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
    item_trend_index: ItemTrendIndex | None = None,
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
        cand_meta = [
            news_lookup.get(
                nid, {"news_idx": 0, "cat_idx": 0, "subcat_idx": 0}
            )
            for nid in cand_ids
        ]
        if item_trend_index is not None:
            cand_item_age_log1p, cand_item_burst = item_trend_index.features(
                [int(meta["news_idx"]) for meta in cand_meta],
                r.get("time"),
            )
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
                meta = cand_meta[j]
                clicks = int(item_clicks_train.get(nid, 0))
                is_new = 1 if clicks < min_item_train_clicks_for_warm else 0
                row = {
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
                if item_trend_index is not None:
                    row["item_age_log1p"] = float(cand_item_age_log1p[j])
                    if item_trend_index.use_burst:
                        row["item_burst"] = float(cand_item_burst[j])
                rows.append(row)
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
    item_trend_index: ItemTrendIndex | None = None,
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

        if item_trend_index is not None:
            cand_item_age_log1p, cand_item_burst = item_trend_index.features(
                cand_news_idx,
                r.get("time"),
            )

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
        if item_trend_index is not None:
            row["cand_item_age_log1p"] = cand_item_age_log1p.tolist()
            if item_trend_index.use_burst:
                row["cand_item_burst"] = cand_item_burst.tolist()
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


def _materialize_train_pairs(cfg: dict[str, Any]) -> bool:
    hard_cfg = cfg.get("ranker", {}).get("hard_negative_sampling", {})
    return not bool(hard_cfg.get("enabled", False))


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

    item_trend_index = build_item_trend_index(
        cfg,
        [train_dir / "behaviors.tsv", dev_dir / "behaviors.tsv"],
        maps.news2idx,
    )
    if item_trend_index is not None:
        item_trend_index.save(item_trend_artifact_path(proc_root))

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
            item_trend_index=item_trend_index,
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
        item_trend_index=item_trend_index,
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
        item_trend_index=item_trend_index,
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
        item_trend_index=item_trend_index,
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
        item_trend_index=item_trend_index,
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
        "item_trend": item_trend_config(cfg),
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
                "test_source": None,
            }
        )
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
    for split_name in ["val", submission_split]:
        _remove_if_exists(pair_artifact_path(proc_root, split_name))
        _remove_if_exists(impression_artifact_path(proc_root, split_name))
        _remove_if_exists(behavior_artifact_path(proc_root, split_name))
    maps.save(proc_root / "id_maps.json")

    news_idx_df = add_indices(news_all, maps)
    news_idx_df.to_parquet(proc_root / "news.parquet", index=False)

    trend_behavior_paths = [
        train_dir / "behaviors.tsv",
        dev_dir / "behaviors.tsv",
    ]
    if mode == "leaderboard_submission":
        assert test_dir is not None
        trend_behavior_paths.append(test_dir / "behaviors.tsv")
    item_trend_index = build_item_trend_index(
        cfg,
        trend_behavior_paths,
        maps.news2idx,
    )
    if item_trend_index is not None:
        item_trend_index.save(item_trend_artifact_path(proc_root))

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
            item_trend_index=item_trend_index,
        )
        pairs_train.to_parquet(train_pairs_path, index=False)
    else:
        _remove_if_exists(train_pairs_path)

    pairs_val = None
    impr_val = None
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
            item_trend_index=item_trend_index,
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
            item_trend_index=item_trend_index,
        )
        impr_val.to_parquet(impression_artifact_path(proc_root, "val"), index=False)
        val_beh.to_parquet(behavior_artifact_path(proc_root, "val"), index=False)

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
        "item_trend": item_trend_config(cfg),
        "n_validation_eval_impressions": int(len(impr_val)) if impr_val is not None else 0,
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
