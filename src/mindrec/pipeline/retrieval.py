from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.models.teacher import TeacherTwoTower
from mindrec.utils import (
    behavior_artifact_path,
    impression_artifact_path,
    load_json,
    save_json,
    test_split_name,
    validation_split_name,
)


def _build_index(item_emb: np.ndarray, index_type: str, ivf_nlist: int) -> faiss.Index:
    dim = item_emb.shape[1]
    xb = item_emb.astype(np.float32)
    if index_type == "flat_ip":
        index = faiss.IndexFlatIP(dim)
        index.add(xb)
        return index

    if index_type == "ivf_flat_ip":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, ivf_nlist, faiss.METRIC_INNER_PRODUCT
        )
        index.train(xb)
        index.add(xb)
        return index

    raise ValueError(f"Unknown index_type: {index_type}")


def run_build_index(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "retrieval")

    teacher_root = runs_root / "teacher"
    item_emb = np.load(teacher_root / "item_teacher_emb.npy")
    item_base = np.load(teacher_root / "item_base_emb.npy")

    index = _build_index(
        item_emb=item_emb,
        index_type=cfg["retrieval"]["index_type"],
        ivf_nlist=int(cfg["retrieval"].get("ivf_nlist", 2048)),
    )
    faiss.write_index(index, str(art_root / "faiss.index"))
    base_index = _build_index(
        item_emb=item_base,
        index_type=cfg["retrieval"].get("base_index_type", "flat_ip"),
        ivf_nlist=int(cfg["retrieval"].get("ivf_nlist", 2048)),
    )
    faiss.write_index(base_index, str(art_root / "base_faiss.index"))
    save_json(
        art_root / "meta.json",
        {
            "index_type": cfg["retrieval"]["index_type"],
            "base_index_type": cfg["retrieval"].get("base_index_type", "flat_ip"),
            "ivf_nlist": int(cfg["retrieval"].get("ivf_nlist", 2048)),
            "n_items": int(item_emb.shape[0]),
            "dim": int(item_emb.shape[1]),
        },
    )


def _dedupe_settings(settings: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    seen: set[tuple[float, int]] = set()
    out: list[dict[str, int | float]] = []
    for setting in settings:
        key = (
            float(setting["hybrid_base_weight"]),
            int(setting["hybrid_oversample"]),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "hybrid_base_weight": key[0],
                "hybrid_oversample": key[1],
            }
        )
    return out


def _retrieve_topk(
    teacher_scores: np.ndarray,
    teacher_idx: np.ndarray,
    base_scores: np.ndarray | None,
    base_idx: np.ndarray | None,
    topk: int,
    hybrid_base_weight: float,
) -> set[int]:
    if (
        hybrid_base_weight <= 0.0
        or base_scores is None
        or base_idx is None
    ):
        return {int(idx) for idx in teacher_idx[:topk].tolist()}

    combined: dict[int, float] = {}
    for idx, score in zip(teacher_idx.tolist(), teacher_scores.tolist()):
        combined[int(idx)] = combined.get(int(idx), 0.0) + (
            (1.0 - hybrid_base_weight) * float(score)
        )
    for idx, score in zip(base_idx.tolist(), base_scores.tolist()):
        combined[int(idx)] = combined.get(int(idx), 0.0) + (
            hybrid_base_weight * float(score)
        )
    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    return {idx for idx, _ in ranked[:topk]}


def _sanitize_slice_value(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    chars = [ch if ch.isalnum() else "_" for ch in text]
    text = "".join(chars)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


def _history_len_bucket(history_len: int) -> str:
    if history_len <= 0:
        return "0"
    if history_len <= 4:
        return "1_4"
    if history_len <= 20:
        return "5_20"
    return "21_plus"


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


def _attach_time_periods(df: pd.DataFrame, n_periods: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
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
    chunks = np.array_split(ordered_valid_idx, min(n_periods, len(ordered_valid_idx)))

    for i, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        label = f"period_{i}_of_{len(chunks)}"
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


def _evaluate_single_retrieval_split(
    cfg: dict[str, Any],
    split_name: str,
    model: TeacherTwoTower,
    item_emb_tensor: torch.Tensor,
    item_base_tensor: torch.Tensor,
    index: faiss.Index,
    base_index: faiss.Index | None,
    maps: IdMaps,
    item_click_counts: dict[str, int],
    hybrid_base_weight: float,
    hybrid_oversample: int,
) -> dict[str, Any]:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    news = pd.read_parquet(proc_root / "news.parquet")
    news_lookup = (
        news[["news_id", "category", "subcategory"]]
        .copy()
        .assign(news_id=lambda df: df["news_id"].astype(str))
        .set_index("news_id")[["category", "subcategory"]]
        .to_dict(orient="index")
    )

    beh_eval = pd.read_parquet(behavior_artifact_path(proc_root, split_name))
    time_periods = int(cfg.get("eval", {}).get("time_periods", 4))
    beh_eval, time_period_meta = _attach_time_periods(beh_eval, time_periods)

    topk = int(cfg["retrieval"]["topk"])
    max_hist = int(cfg["data"]["max_history"])
    search_k = max(topk, topk * max(int(hybrid_oversample), 1))

    recalls: list[float] = []
    slice_sums: dict[str, float] = {}
    slice_counts: dict[str, int] = {}

    def add_to_slice(name: str, value: float) -> None:
        slice_sums[name] = slice_sums.get(name, 0.0) + float(value)
        slice_counts[name] = slice_counts.get(name, 0) + 1

    for _, r in tqdm(beh_eval.iterrows(), total=len(beh_eval), desc=f"Retrieval eval ({split_name})"):
        hist = [str(h) for h in r["history"][-max_hist:]]
        hist_idx = [
            maps.news2idx[h]
            for h in hist
            if h in maps.news2idx and maps.news2idx[h] != 0
        ]
        if not hist_idx:
            continue

        cand_ids = [str(n) for n in r["cand_news_id"]]
        labels = [int(x) for x in r["cand_label"]]
        clicked_ids = [nid for nid, lab in zip(cand_ids, labels) if lab == 1]
        clicked = [
            maps.news2idx.get(nid, 0)
            for nid in clicked_ids
            if maps.news2idx.get(nid, 0) != 0
        ]
        if not clicked:
            continue

        with torch.no_grad():
            hist_z = item_emb_tensor[hist_idx].unsqueeze(0)
            hist_mask = torch.ones((1, len(hist_idx)), dtype=torch.bool)
            q = (
                model.encode_user_from_item_vectors(hist_z, hist_mask)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            q_base = item_base_tensor[hist_idx].mean(dim=0, keepdim=True)
            q_base = q_base / q_base.norm(dim=1, keepdim=True).clamp_min(1e-12)
            q_base_np = q_base.cpu().numpy().astype(np.float32)
        teacher_scores, teacher_idx = index.search(q, search_k)
        base_scores = base_idx = None
        if base_index is not None:
            base_scores, base_idx = base_index.search(q_base_np, search_k)
        teacher_limit = min(search_k, topk * max(int(hybrid_oversample), 1))
        retrieved = _retrieve_topk(
            teacher_scores=teacher_scores[0][:teacher_limit],
            teacher_idx=teacher_idx[0][:teacher_limit],
            base_scores=base_scores[0][:teacher_limit] if base_scores is not None else None,
            base_idx=base_idx[0][:teacher_limit] if base_idx is not None else None,
            topk=topk,
            hybrid_base_weight=hybrid_base_weight,
        )
        recall = sum(1 for c in clicked if c in retrieved) / len(clicked)
        recalls.append(recall)

        add_to_slice("overall", recall)
        add_to_slice(f"history_len_bucket__{_history_len_bucket(len(hist_idx))}", recall)
        add_to_slice(
            f"time_period__{_sanitize_slice_value(r['time_period'])}",
            recall,
        )

        clicked_pop_buckets = {
            _popularity_bucket(int(item_click_counts.get(nid, 0)))
            for nid in clicked_ids
        }
        clicked_cat_names = {
            _sanitize_slice_value(news_lookup.get(nid, {}).get("category", "unknown"))
            for nid in clicked_ids
        }
        clicked_subcat_names = {
            _sanitize_slice_value(news_lookup.get(nid, {}).get("subcategory", "unknown"))
            for nid in clicked_ids
        }
        for bucket in clicked_pop_buckets:
            add_to_slice(
                f"impressions_with_clicked_popularity_bucket__{bucket}",
                recall,
            )
        for cat_name in clicked_cat_names:
            add_to_slice(
                f"impressions_with_clicked_category__{cat_name}",
                recall,
            )
        for subcat_name in clicked_subcat_names:
            add_to_slice(
                f"impressions_with_clicked_subcategory__{subcat_name}",
                recall,
            )

    return {
        "recall_at_k": float(np.mean(recalls) if recalls else 0.0),
        "n_eval": int(len(recalls)),
        "k": topk,
        "eval_split": split_name,
        "hybrid_base_weight": float(hybrid_base_weight),
        "hybrid_oversample": int(hybrid_oversample),
        "slices": {
            name: {
                "recall_at_k": float(slice_sums[name] / slice_counts[name])
                if slice_counts[name]
                else 0.0,
                "n_impressions": int(slice_counts[name]),
            }
            for name in slice_sums
        },
        "time_periods": time_period_meta,
    }


def _evaluate_retrieval_settings(
    cfg: dict[str, Any],
    settings: list[dict[str, int | float]],
) -> list[dict[str, Any]]:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "retrieval")

    settings = _dedupe_settings(settings)
    if not settings:
        raise ValueError("At least one retrieval setting is required")

    teacher_root = runs_root / "teacher"
    item_emb = np.load(teacher_root / "item_teacher_emb.npy")
    item_base = np.load(teacher_root / "item_base_emb.npy")
    item_emb_tensor = torch.tensor(item_emb, dtype=torch.float32)
    item_base_tensor = torch.tensor(item_base, dtype=torch.float32)
    model_ckpt_path = teacher_root / "model.pt"

    index = faiss.read_index(str(art_root / "faiss.index"))
    base_index_path = art_root / "base_faiss.index"
    base_index = faiss.read_index(str(base_index_path)) if base_index_path.exists() else None
    topk = int(cfg["retrieval"]["topk"])
    max_hist = int(cfg["data"]["max_history"])
    search_k = max(
        topk,
        topk * max(int(setting["hybrid_oversample"]) for setting in settings),
    )

    eval_split = validation_split_name(cfg)
    recall_sums = [0.0 for _ in settings]
    recall_counts = [0 for _ in settings]
    if model_ckpt_path.exists():
        ckpt = torch.load(model_ckpt_path, map_location="cpu")
        model = TeacherTwoTower(
            item_dim=int(ckpt["item_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
            heads=int(ckpt["heads"]),
            dropout=float(ckpt.get("dropout", 0.1)),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        beh_eval = pd.read_parquet(behavior_artifact_path(proc_root, eval_split))
        maps = IdMaps.load(proc_root / "id_maps.json")

        for _, r in tqdm(
            beh_eval.iterrows(), total=len(beh_eval), desc="Retrieval eval"
        ):
            hist_idx = [
                maps.news2idx[h]
                for h in r["history"][-max_hist:]
                if h in maps.news2idx and maps.news2idx[h] != 0
            ]
            if not hist_idx:
                continue

            clicked = [
                maps.news2idx.get(str(n), 0)
                for n, l in zip(r["cand_news_id"], r["cand_label"])
                if int(l) == 1 and maps.news2idx.get(str(n), 0) != 0
            ]
            if not clicked:
                continue

            with torch.no_grad():
                hist_z = item_emb_tensor[hist_idx].unsqueeze(0)
                hist_mask = torch.ones((1, len(hist_idx)), dtype=torch.bool)
                q = (
                    model.encode_user_from_item_vectors(hist_z, hist_mask)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                q_base = item_base_tensor[hist_idx].mean(dim=0, keepdim=True)
                q_base = q_base / q_base.norm(dim=1, keepdim=True).clamp_min(1e-12)
                q_base_np = q_base.cpu().numpy().astype(np.float32)
            teacher_scores, teacher_idx = index.search(q, search_k)
            base_scores = base_idx = None
            if base_index is not None:
                base_scores, base_idx = base_index.search(q_base_np, search_k)
            teacher_scores_1d = teacher_scores[0]
            teacher_idx_1d = teacher_idx[0]
            base_scores_1d = base_scores[0] if base_scores is not None else None
            base_idx_1d = base_idx[0] if base_idx is not None else None
            for i, setting in enumerate(settings):
                oversample = max(int(setting["hybrid_oversample"]), 1)
                teacher_limit = min(search_k, topk * oversample)
                retrieved = _retrieve_topk(
                    teacher_scores=teacher_scores_1d[:teacher_limit],
                    teacher_idx=teacher_idx_1d[:teacher_limit],
                    base_scores=(
                        base_scores_1d[:teacher_limit]
                        if base_scores_1d is not None
                        else None
                    ),
                    base_idx=(
                        base_idx_1d[:teacher_limit]
                        if base_idx_1d is not None
                        else None
                    ),
                    topk=topk,
                    hybrid_base_weight=float(setting["hybrid_base_weight"]),
                )
                hit = sum(1 for c in clicked if c in retrieved)
                recall_sums[i] += hit / len(clicked)
                recall_counts[i] += 1
    else:
        impr = pd.read_parquet(impression_artifact_path(proc_root, eval_split))
        user_emb = np.load(teacher_root / "user_teacher_emb.npy")

        for _, r in tqdm(impr.iterrows(), total=len(impr), desc="Retrieval eval"):
            u = int(r["user_idx"])
            if u <= 0:
                continue
            q = user_emb[u : u + 1].astype(np.float32)
            _, I = index.search(q, topk)
            retrieved = set(I[0].tolist())
            clicked = [
                int(n) for n, l in zip(r["cand_news_idx"], r["cand_label"]) if int(l) == 1
            ]
            if not clicked:
                continue
            hit = sum(1 for c in clicked if c in retrieved)
            for i in range(len(settings)):
                recall_sums[i] += hit / len(clicked)
                recall_counts[i] += 1

    return [
        {
            "recall_at_k": float(recall_sums[i] / recall_counts[i])
            if recall_counts[i]
            else 0.0,
            "n_eval": int(recall_counts[i]),
            "k": topk,
            "eval_split": eval_split,
            "hybrid_base_weight": float(settings[i]["hybrid_base_weight"]),
            "hybrid_oversample": int(settings[i]["hybrid_oversample"]),
        }
        for i in range(len(settings))
    ]


def run_eval_retrieval(cfg: dict[str, Any]) -> None:
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "retrieval")
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    teacher_root = runs_root / "teacher"

    item_emb = np.load(teacher_root / "item_teacher_emb.npy")
    item_base = np.load(teacher_root / "item_base_emb.npy")
    item_emb_tensor = torch.tensor(item_emb, dtype=torch.float32)
    item_base_tensor = torch.tensor(item_base, dtype=torch.float32)
    index = faiss.read_index(str(art_root / "faiss.index"))
    base_index_path = art_root / "base_faiss.index"
    base_index = faiss.read_index(str(base_index_path)) if base_index_path.exists() else None

    ckpt = torch.load(teacher_root / "model.pt", map_location="cpu")
    model = TeacherTwoTower(
        item_dim=int(ckpt["item_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        heads=int(ckpt["heads"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    maps = IdMaps.load(proc_root / "id_maps.json")
    item_click_counts = load_json(proc_root / "item_click_counts.json")
    split_results: dict[str, dict[str, Any]] = {}
    for split_name in _resolve_eval_splits(cfg):
        split_results[split_name] = _evaluate_single_retrieval_split(
            cfg=cfg,
            split_name=split_name,
            model=model,
            item_emb_tensor=item_emb_tensor,
            item_base_tensor=item_base_tensor,
            index=index,
            base_index=base_index,
            maps=maps,
            item_click_counts=item_click_counts,
            hybrid_base_weight=float(cfg["retrieval"].get("hybrid_base_weight", 0.0)),
            hybrid_oversample=int(cfg["retrieval"].get("hybrid_oversample", 3)),
        )
        save_json(art_root / f"eval_{split_name}.json", split_results[split_name])


def run_eval_retrieval_sweep(cfg: dict[str, Any]) -> None:
    sweep_cfg = cfg.get("retrieval", {}).get("sweep", {})
    weights = sweep_cfg.get("hybrid_base_weights", [0.25, 0.5, 0.75])
    oversamples = sweep_cfg.get("hybrid_oversamples", [2, 3, 5])
    settings = [
        {
            "hybrid_base_weight": float(weight),
            "hybrid_oversample": int(oversample),
        }
        for weight, oversample in product(weights, oversamples)
    ]
    results = _evaluate_retrieval_settings(cfg, settings)
    best = max(results, key=lambda row: row["recall_at_k"])
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "retrieval")
    save_json(
        art_root / "sweep.json",
        {
            "eval_split": best["eval_split"],
            "k": best["k"],
            "n_settings": len(results),
            "settings": results,
            "best": best,
        },
    )
