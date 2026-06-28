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
from mindrec.data.mind_io import count_behavior_rows, iter_behaviors_tsv, time_feature_indices
from mindrec.pipeline.evaluate import _expand_history_base, _load_model
from mindrec.utils import load_json, log_device, resolve_device, save_json


def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.int32)
    return ranks


def _write_prediction_zip(prediction_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(prediction_path, arcname="prediction.txt")


def run_write_submission(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / "submission")

    submission_cfg = dict(cfg.get("submission", {}))
    split_name = str(submission_cfg.get("split_name", "submission_test"))
    batch_size = int(submission_cfg.get("batch_size", 2048))
    save_scores = bool(submission_cfg.get("save_scores", False))

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

    with prediction_path.open("w", encoding="utf-8", newline="\n") as f:
        with torch.no_grad():
            for r in tqdm(
                iter_behaviors_tsv(behavior_path),
                total=n_impressions,
                desc=f"Score submission ({split_name})",
            ):
                user_id = str(r["user_id"])
                history = list(r["history"])
                hour_idx, weekday_idx = time_feature_indices(r["time"])
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

                logits = []
                for i in range(0, len(cand_news_idx), batch_size):
                    sl = slice(i, i + batch_size)
                    current_batch_size = len(cand_news_idx[sl])
                    b_user = torch.tensor(
                        [user_idx] * current_batch_size,
                        dtype=torch.long,
                        device=device,
                    )
                    b_news = torch.tensor(
                        cand_news_idx[sl], dtype=torch.long, device=device
                    )
                    b_cat = torch.tensor(
                        cand_cat_idx[sl], dtype=torch.long, device=device
                    )
                    b_sub = torch.tensor(
                        cand_subcat_idx[sl], dtype=torch.long, device=device
                    )
                    b_is_new = torch.tensor(
                        cand_is_new[sl], dtype=torch.long, device=device
                    )
                    b_hour = torch.full(
                        (current_batch_size,), hour_idx, dtype=torch.long, device=device
                    )
                    b_weekday = torch.full(
                        (current_batch_size,), weekday_idx, dtype=torch.long, device=device
                    )
                    b_dense = torch.tensor(
                        dense[sl], dtype=torch.float32, device=device
                    )
                    b_item_base = torch.tensor(
                        item_base[cand_news_idx[sl]],
                        dtype=torch.float32,
                        device=device,
                    )
                    b_hist_base, b_hist_mask = _expand_history_base(
                        item_base=item_base,
                        hist_idx=hist_news_idx,
                        batch_size=current_batch_size,
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
                        hour_idx=b_hour,
                        weekday_idx=b_weekday,
                    )
                    logits.append(logit.detach().cpu().numpy())

                scores = np.concatenate(logits, axis=0)
                ranks = _scores_to_ranks(scores)
                rank_json = json.dumps(ranks.tolist(), separators=(",", ":"))
                f.write(f"{r['impression_id']} {rank_json}\n")
                if save_scores:
                    scored_rows.append(
                        {
                            "impression_id": str(r["impression_id"]),
                            "cand_news_id": cand_news_id,
                            "score": scores.astype(float).tolist(),
                            "rank": ranks.astype(int).tolist(),
                        }
                    )

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
            "format": "MIND leaderboard prediction.txt: impression_id compact_json_ranks",
        },
    )
