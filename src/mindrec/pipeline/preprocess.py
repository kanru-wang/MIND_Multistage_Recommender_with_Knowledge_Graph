from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps, add_indices, build_id_maps, is_cold_user
from mindrec.data.mind_io import read_behaviors_tsv, read_news_tsv, sub_sample_behaviors
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
) -> pd.DataFrame:
    set_seed(seed)
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
        if sum(labels) == 0:
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

        rows.append(
            {
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
        )
    return pd.DataFrame(rows)


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


def run_preprocess(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

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

    # Item clicks in train from impressions
    click_counts = {}
    for ids, labs in zip(beh_train["cand_news_id"], beh_train["cand_label"]):
        for nid, lab in zip(ids, labs):
            if lab == 1:
                click_counts[nid] = click_counts.get(nid, 0) + 1
    save_json(proc_root / "item_click_counts.json", click_counts)

    holdout_cfg = dict(cfg["data"].get("holdout", {}))
    val_name = "val"
    test_name = "test"
    val_beh, test_beh, holdout_meta = _split_holdout_behaviors(
        beh=beh_dev,
        validation_fraction=float(holdout_cfg.get("validation_fraction", 0.8)),
    )
    ranker_neg_per_pos = int(cfg["data"].get("ranker_negatives_per_positive", 4))

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

    pairs_train.to_parquet(pair_artifact_path(proc_root, "train"), index=False)
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
    val_beh.to_parquet(behavior_artifact_path(proc_root, val_name), index=False)
    test_beh.to_parquet(behavior_artifact_path(proc_root, test_name), index=False)

    meta = {
        "dataset": ds,
        "n_news": int(len(news_idx_df)),
        "n_train_impressions": int(len(beh_train)),
        "n_holdout_source_impressions": int(len(beh_dev)),
        "validation_split_name": val_name,
        "test_split_name": test_name,
        "n_validation_impressions": int(len(val_beh)),
        "n_test_impressions": int(len(test_beh)),
        "n_train_pairs": int(len(pairs_train)),
        "n_validation_pairs": int(len(pairs_val)),
        "n_test_pairs": int(len(pairs_test)),
        "ranker_negatives_per_positive": ranker_neg_per_pos,
        "n_validation_eval_impressions": int(len(impr_val)),
        "n_test_eval_impressions": int(len(impr_test)),
        "holdout": holdout_meta,
    }
    save_json(proc_root / "preprocess_meta.json", meta)
