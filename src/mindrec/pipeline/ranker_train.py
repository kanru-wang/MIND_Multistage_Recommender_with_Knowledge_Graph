from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.datasets import PairDataset, collate_batch
from mindrec.data.featurize import IdMaps
from mindrec.models.calibration import fit_temperature_scaler
from mindrec.models.dlrm import DLRMStudent
from mindrec.models.distill import (
    distillation_history_masks,
    representation_distillation_dims,
    select_representation_distillation_inputs,
)
from mindrec.models.teacher import TeacherTwoTower
from mindrec.pipeline.hard_negative_sampling import build_teacher_hard_negative_pairs
from mindrec.utils import (
    behavior_artifact_path,
    device_info,
    load_json,
    log_device,
    pair_artifact_path,
    resolve_device,
    save_json,
    set_seed,
    teacher_artifact_root,
    teacher_artifact_run_name,
    to_device,
    validation_split_name,
)


class StudentProjHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def run_train_ranker(cfg: dict[str, Any]) -> None:
    dist_cfg = dict(cfg.get("ranker", {}).get("distill", {}))
    representation_target = str(dist_cfg.get("representation_target", "full"))
    # Resolve this before data access so invalid experiment configs fail fast.
    representation_distillation_dims(1, 1, representation_target)

    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    maps = IdMaps.load(proc_root / "id_maps.json")

    hard_cfg = dict(cfg["ranker"].get("hard_negative_sampling", {}))
    hard_enabled = bool(hard_cfg.get("enabled", False))
    pairs_train = (
        None
        if hard_enabled
        else pd.read_parquet(proc_root / "train_pairs.parquet")
    )
    val_split = validation_split_name(cfg)
    val_pairs_path = pair_artifact_path(proc_root, val_split)
    data_mode = str(cfg["data"].get("mode", "standard"))
    validation_allowed = data_mode != "leaderboard_submission"
    has_validation = validation_allowed and val_pairs_path.exists()
    pairs_val = pd.read_parquet(val_pairs_path) if has_validation else None

    dense_cols = ["history_len", "item_clicks_log1p"]

    device = resolve_device(cfg["ranker"].get("device", "cuda"))
    device_str = str(device)
    log_device(device, "Ranker")

    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "ranker")

    ranker_base_name = (
        "item_ranker_base_emb.npy"
        if bool(cfg.get("knowledge_graph", {}).get("enabled", False))
        else "item_base_emb.npy"
    )
    teacher_root = teacher_artifact_root(cfg)
    ranker_base_path = teacher_root / ranker_base_name
    item_base = np.load(ranker_base_path)
    teacher_item = np.load(teacher_root / "item_teacher_emb.npy")
    item_base_tensor = torch.tensor(item_base, dtype=torch.float32, device=device)
    teacher_item_tensor = torch.tensor(teacher_item, dtype=torch.float32, device=device)
    teacher_dim = int(teacher_item.shape[1])
    teacher_ckpt = torch.load(teacher_root / "model.pt", map_location=device)
    teacher_model = TeacherTwoTower(
        item_dim=int(teacher_ckpt["item_dim"]),
        hidden_dim=int(teacher_ckpt["hidden_dim"]),
        heads=int(teacher_ckpt["heads"]),
        dropout=float(teacher_ckpt.get("dropout", 0.1)),
    ).to(device)
    teacher_model.load_state_dict(teacher_ckpt["state_dict"])
    teacher_model.eval()

    news = pd.read_parquet(proc_root / "news.parquet")
    hard_stats: dict[str, Any] = {"enabled": hard_enabled}
    if hard_enabled:
        negatives_per_positive = int(
            cfg["data"].get("ranker_negatives_per_positive", 4)
        )
        pool_size = int(
            hard_cfg.get("pool_size", max(negatives_per_positive * 5, 20))
        )
        hard_fraction = float(hard_cfg.get("hard_fraction", 0.75))
        score_batch_size = int(hard_cfg.get("score_batch_size", 512))
        teacher_consistent_hard_only = bool(
            hard_cfg.get("teacher_consistent_hard_only", True)
        )
        max_score_above_positive = float(hard_cfg.get("max_score_above_positive", 0.0))
        hard_for_cold_users_only = bool(
            hard_cfg.get("hard_for_cold_users_only", False)
        )
        if pool_size < negatives_per_positive:
            raise ValueError(
                "ranker.hard_negative_sampling.pool_size must be at least "
                "data.ranker_negatives_per_positive."
            )

        train_behaviors_path = behavior_artifact_path(proc_root, "train")
        if not train_behaviors_path.exists():
            raise FileNotFoundError(
                "Hard-negative sampling requires train_behaviors.parquet. "
                "Run preprocessing with the current pipeline first."
            )
        train_behaviors = pd.read_parquet(train_behaviors_path)
        click_counts = load_json(proc_root / "item_click_counts.json")
        pairs_train, selection_stats = build_teacher_hard_negative_pairs(
            beh=train_behaviors,
            news_idx_df=news,
            maps=maps,
            item_clicks_train=click_counts,
            min_user_hist_for_warm=int(cfg["data"]["min_user_hist_for_warm"]),
            min_item_train_clicks_for_warm=int(
                cfg["data"]["min_item_train_clicks_for_warm"]
            ),
            max_history=int(cfg["data"]["max_history"]),
            negatives_per_positive=negatives_per_positive,
            pool_size=pool_size,
            hard_fraction=hard_fraction,
            teacher_consistent_hard_only=teacher_consistent_hard_only,
            max_score_above_positive=max_score_above_positive,
            hard_for_cold_users_only=hard_for_cold_users_only,
            seed=seed,
            teacher_model=teacher_model,
            teacher_item_tensor=teacher_item_tensor,
            device=device,
            group_batch_size=score_batch_size,
        )
        hard_stats.update(
            {
                "pool_size": pool_size,
                "negatives_per_positive": negatives_per_positive,
                "score_batch_size": score_batch_size,
                "scorer": "teacher_history_item_dot_product",
                "teacher_consistent_hard_only": teacher_consistent_hard_only,
                "max_score_above_positive": max_score_above_positive,
                "hard_for_cold_users_only": hard_for_cold_users_only,
                **selection_stats,
            }
        )
        print(
            "Hard-negative sampling selected "
            f"{selection_stats['n_selected_negatives']:,} negatives "
            f"({selection_stats['n_hard_negatives']:,} hard, "
            f"{selection_stats['n_random_negatives']:,} random); teacher-scored "
            f"{selection_stats['n_teacher_scored_pool_negatives']:,} pooled "
            "candidates and directly selected "
            f"{selection_stats['n_random_only_selected_negatives']:,} negatives "
            "for random-only groups."
        )

    if pairs_train is None:
        raise RuntimeError("Ranker training pairs were not initialized.")
    supported_cat_ids = tuple(
        int(value)
        for value in np.unique(pairs_train["cat_idx"].to_numpy())
        if int(value) > 0
    )
    supported_subcat_ids = tuple(
        int(value)
        for value in np.unique(pairs_train["subcat_idx"].to_numpy())
        if int(value) > 0
    )
    train_ds = PairDataset(pairs_train, dense_cols=dense_cols)
    val_ds = (
        PairDataset(pairs_val, dense_cols=dense_cols)
        if pairs_val is not None
        else None
    )
    n_users = max(maps.user2idx.values()) + 1
    n_news = int(news["news_idx"].max()) + 1
    # Full vocabulary cardinality preserves stable checkpoint tensor shapes;
    # fit-unseen taxonomy keys are retained in the maps with value 0.
    n_cats = len(maps.cat2idx) + 1
    n_subcats = len(maps.subcat2idx) + 1

    dlrm_cfg = cfg["ranker"]["dlrm"]
    model = DLRMStudent(
        n_users=n_users,
        n_news=n_news,
        n_cats=n_cats,
        n_subcats=n_subcats,
        dense_dim=len(dense_cols),
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
        supported_cat_ids=supported_cat_ids,
        supported_subcat_ids=supported_subcat_ids,
    ).to(device)

    emb_dim = int(dlrm_cfg["emb_dim"])
    student_repr_dim, teacher_repr_dim = representation_distillation_dims(
        emb_dim,
        teacher_dim,
        representation_target,
    )
    proj = StudentProjHead(student_repr_dim, teacher_repr_dim).to(device)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(proj.parameters()),
        lr=float(cfg["ranker"]["lr"]),
        weight_decay=float(cfg["ranker"].get("weight_decay", 0.0)),
    )

    bsz = int(cfg["ranker"]["batch_size"])
    train_loader = DataLoader(
        train_ds, batch_size=bsz, shuffle=True, num_workers=0, collate_fn=collate_batch
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=bsz,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
        if val_ds is not None
        else None
    )

    temp = float(dist_cfg.get("temperature", 2.0))
    lam_logit = float(dist_cfg.get("lambda_logit", 1.0))
    lam_repr = float(dist_cfg.get("lambda_repr", 0.1))
    w_cold = float(dist_cfg.get("cold_weight", 2.0))
    w_warm = float(dist_cfg.get("warm_weight", 0.3))
    hard_negative_distill_weight = float(
        hard_cfg.get("hard_negative_distill_weight", 0.0)
    )
    es_cfg = dict(cfg["ranker"].get("early_stopping", {}))
    es_enabled = bool(es_cfg.get("enabled", True)) and has_validation
    es_patience = int(es_cfg.get("patience", 2))
    es_min_delta = float(es_cfg.get("min_delta", 1.0e-4))

    epochs = int(cfg["ranker"]["epochs"])

    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    epoch_metrics: list[dict[str, float | int]] = []
    for ep in range(1, epochs + 1):
        model.train()
        proj.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"Train ep {ep}"):
            batch = to_device(batch, device)
            item_base_batch = item_base_tensor[batch["news_idx"]]
            hist_base_batch = item_base_tensor[batch["hist_news_idx"]]
            has_hist = batch["hist_mask"].any(dim=1)
            with torch.no_grad():
                tu = torch.zeros(
                    (batch["user_idx"].size(0), teacher_dim),
                    dtype=torch.float32,
                    device=device,
                )
                if bool(has_hist.any().item()):
                    hist_teacher_batch = teacher_item_tensor[batch["hist_news_idx"][has_hist]]
                    tu[has_hist] = teacher_model.encode_user_from_item_vectors(
                        hist_teacher_batch, batch["hist_mask"][has_hist]
                    )
                ti = teacher_item_tensor[batch["news_idx"]]

            logits, rep = model(
                user_idx=batch["user_idx"],
                news_idx=batch["news_idx"],
                cat_idx=batch["cat_idx"],
                subcat_idx=batch["subcat_idx"],
                dense=batch["dense"],
                item_base=item_base_batch,
                history_item_base=hist_base_batch,
                history_mask=batch["hist_mask"],
                is_new_item=batch["is_new_item"],
                return_repr=True,
            )
            y = batch["label"]
            loss_rank = nn.functional.binary_cross_entropy_with_logits(logits, y)

            # teacher logit as cosine / inner product (embeddings are normalized)
            tlogit = (tu * ti).sum(dim=1)
            target = torch.sigmoid(tlogit / temp)
            loss_logit = nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )

            student_repr_input, t_repr = select_representation_distillation_inputs(
                rep,
                tu,
                ti,
                student_emb_dim=emb_dim,
                target=representation_target,
            )
            s_repr = proj(student_repr_input)
            loss_repr = ((s_repr - t_repr) ** 2).mean(dim=1)

            cold_mask = (
                (batch["is_cold_user"] == 1) | (batch["is_new_item"] == 1)
            ).float()
            w = (cold_mask * w_cold) + ((1.0 - cold_mask) * w_warm)
            if hard_enabled:
                hard_distill_scale = torch.where(
                    batch["is_hard_negative"] == 1,
                    torch.full_like(w, hard_negative_distill_weight),
                    torch.ones_like(w),
                )
                w = w * hard_distill_scale
            if representation_target == "full":
                # Preserve the verified full-distillation arithmetic exactly.
                w = (w * has_hist.float()).detach()
                distill_loss = (
                    w * (lam_logit * loss_logit + lam_repr * loss_repr)
                ).mean()
            else:
                logit_mask, representation_mask = distillation_history_masks(
                    has_hist,
                    representation_target,
                )
                logit_w = (w * logit_mask).detach()
                representation_w = (w * representation_mask).detach()
                distill_loss = (
                    (lam_logit * logit_w * loss_logit)
                    + (lam_repr * representation_w * loss_repr)
                ).mean()
            loss = loss_rank + distill_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(proj.parameters()), max_norm=5.0
            )
            opt.step()
            losses.append(float(loss.item()))

        # Validation AUC on the split returned by validation_split_name(cfg).
        model.eval()
        proj.eval()
        auc: float | None = None
        if val_loader is not None:
            ys = []
            ps = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val ep {ep}"):
                    batch = to_device(batch, device)
                    logits, _ = model(
                        user_idx=batch["user_idx"],
                        news_idx=batch["news_idx"],
                        cat_idx=batch["cat_idx"],
                        subcat_idx=batch["subcat_idx"],
                        dense=batch["dense"],
                        item_base=item_base_tensor[batch["news_idx"]],
                        history_item_base=item_base_tensor[batch["hist_news_idx"]],
                        history_mask=batch["hist_mask"],
                        is_new_item=batch["is_new_item"],
                        return_repr=False,
                    )
                    ys.extend(batch["label"].detach().cpu().numpy().tolist())
                    ps.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            try:
                auc = float(roc_auc_score(ys, ps))
            except ValueError:
                auc = 0.0

        epoch_metrics.append(
            {
                "epoch": ep,
                "train_loss_mean": float(np.mean(losses) if losses else 0.0),
                "val_auc": auc,
            }
        )
        save_json(art_root / "epochs.json", epoch_metrics)

        improved = True if not has_validation else auc is not None and auc > best_auc
        significant_improvement = (
            has_validation and auc is not None and (auc - best_auc) > es_min_delta
        )
        if improved:
            if has_validation and auc is not None:
                best_auc = auc
            best_epoch = ep
            torch.save(
                {
                    "model": model.state_dict(),
                    "proj": proj.state_dict(),
                    "cfg": cfg,
                    "epoch": ep,
                    "val_auc": auc,
                    "selection_mode": "validation_auc" if has_validation else "fixed_epoch",
                },
                art_root / "best.pt",
            )
        if not has_validation:
            epochs_without_improvement = 0
        elif significant_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if es_enabled and epochs_without_improvement >= es_patience:
            stop_reason = "early_stopping"
            break

    save_json(
        art_root / "train_summary.json",
        {
            "validation_split_name": val_split,
            "has_validation": has_validation,
            "best_val_auc": best_auc if has_validation else None,
            "best_epoch": best_epoch,
            "selection_mode": "validation_auc" if has_validation else "fixed_epoch",
            "early_stopping_enabled": es_enabled,
            "early_stopping_patience": es_patience,
            "early_stopping_min_delta": es_min_delta,
            "stop_reason": stop_reason,
            "stopped_epoch": ep,
            "device": device_str,
            "device_info": device_info(device),
            "teacher_artifact_run_name": teacher_artifact_run_name(cfg),
            "history_pooling": model.history_pooling,
            "representation_distillation_target": representation_target,
            "hard_negative_sampling": hard_stats,
            "taxonomy_masks": {
                "n_supported_categories": len(supported_cat_ids),
                "n_supported_subcategories": len(supported_subcat_ids),
                "n_train_pairs": int(len(pairs_train)),
            },
        },
    )

    checkpoint_path = art_root / "best.pt"
    cal_cfg = dict(cfg["ranker"].get("calibration", {}))
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    calibration_enabled = bool(cal_cfg.get("enabled", True)) and val_loader is not None
    if not calibration_enabled:
        calib_path = art_root / "calibration.json"
        if calib_path.exists():
            calib_path.unlink()
        save_json(
            art_root / "calibration_stats.json",
            {
                "enabled": False,
                "reason": "disabled_final_fit_or_no_validation_split",
            },
        )
        return

    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Fit temperature"):
            batch = to_device(batch, device)
            logits, _ = model(
                user_idx=batch["user_idx"],
                news_idx=batch["news_idx"],
                cat_idx=batch["cat_idx"],
                subcat_idx=batch["subcat_idx"],
                dense=batch["dense"],
                item_base=item_base_tensor[batch["news_idx"]],
                history_item_base=item_base_tensor[batch["hist_news_idx"]],
                history_mask=batch["hist_mask"],
                is_new_item=batch["is_new_item"],
                return_repr=False,
            )
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(batch["label"].detach().cpu().numpy())

    scaler, stats = fit_temperature_scaler(
        logits=np.concatenate(logits_all, axis=0),
        labels=np.concatenate(labels_all, axis=0),
        max_iter=int(cal_cfg.get("max_iter", 100)),
        lr=float(cal_cfg.get("lr", 0.05)),
    )
    scaler.save(
        art_root / "calibration.json",
        meta={
            "fit_split": f"{val_split}_pairs",
            "stats": stats,
        },
    )
    save_json(art_root / "calibration_stats.json", stats)
