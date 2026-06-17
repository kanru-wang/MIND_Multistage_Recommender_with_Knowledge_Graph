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
from mindrec.data.datasets import (
    RANKER_DENSE_COLUMNS,
    PairDataset,
    build_ranker_dense_matrix,
    collate_batch,
)
from mindrec.data.featurize import IdMaps
from mindrec.metrics.ranking import auc as impression_auc
from mindrec.models.calibration import fit_temperature_scaler
from mindrec.models.dlrm import DLRMStudent
from mindrec.models.teacher import TeacherTwoTower
from mindrec.pipeline.ranker_assets import (
    load_ranker_item_features,
    ranker_feature_fingerprints,
    ranker_score_batch_size,
)
from mindrec.utils import (
    impression_artifact_path,
    pair_artifact_path,
    save_json,
    set_seed,
    to_device,
    validation_split_name,
)


class StudentProjHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def final_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_user: torch.Tensor,
    teacher_item: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    teacher_logit = (teacher_user * teacher_item).sum(dim=1)
    teacher_target = torch.sigmoid(teacher_logit / temperature).detach()
    return nn.functional.binary_cross_entropy_with_logits(
        student_logits, teacher_target, reduction="none"
    )


def _expand_history_tensor(
    matrix: torch.Tensor,
    hist_idx: list[int],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = matrix.device
    if hist_idx:
        idx = torch.tensor(hist_idx, dtype=torch.long, device=device)
        hist = matrix[idx].unsqueeze(0)
        hist = hist.expand(batch_size, *hist.shape[1:])
        mask = torch.ones((batch_size, len(hist_idx)), dtype=torch.bool, device=device)
        return hist, mask
    hist = torch.zeros(
        (batch_size, 1, *matrix.shape[1:]), dtype=matrix.dtype, device=device
    )
    mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
    return hist, mask


def _validate_impression_auc(
    cfg: dict[str, Any],
    proc_root: Path,
    split_name: str,
    model: DLRMStudent,
    item_text_tensor: torch.Tensor,
    item_kg_tensor: torch.Tensor,
    item_kg_mask_tensor: torch.Tensor,
) -> float:
    impr = pd.read_parquet(impression_artifact_path(proc_root, split_name))
    values: list[float] = []
    bs = ranker_score_batch_size(cfg)

    model.eval()
    with torch.no_grad():
        for _, r in tqdm(
            impr.iterrows(),
            total=len(impr),
            desc=f"Val impressions ({split_name})",
        ):
            labels = np.array(r["cand_label"], dtype=np.int32)
            if labels.sum() <= 0:
                continue
            cand_news_idx = np.array(r["cand_news_idx"], dtype=np.int64)
            if len(cand_news_idx) <= 0:
                continue

            user_idx = int(r["user_idx"])
            hist_news_idx = [int(x) for x in list(r["hist_news_idx"])]
            cand_cat_idx = np.array(r["cand_cat_idx"], dtype=np.int64)
            cand_subcat_idx = np.array(r["cand_subcat_idx"], dtype=np.int64)
            cand_is_new = np.array(r["cand_is_new_item"], dtype=np.int64)
            cand_clicks_log1p = np.array(
                r["cand_item_clicks_log1p"], dtype=np.float32
            )
            dense = build_ranker_dense_matrix(float(r["history_len"]), cand_clicks_log1p)

            scores: list[np.ndarray] = []
            for i in range(0, len(cand_news_idx), bs):
                sl = slice(i, i + bs)
                batch_size = len(cand_news_idx[sl])
                device = item_text_tensor.device
                b_news = torch.tensor(
                    cand_news_idx[sl], dtype=torch.long, device=device
                )
                b_user = torch.full(
                    (batch_size,), user_idx, dtype=torch.long, device=device
                )
                b_cat = torch.tensor(cand_cat_idx[sl], dtype=torch.long, device=device)
                b_sub = torch.tensor(
                    cand_subcat_idx[sl], dtype=torch.long, device=device
                )
                b_is_new = torch.tensor(
                    cand_is_new[sl], dtype=torch.long, device=device
                )
                b_dense = torch.tensor(dense[sl], dtype=torch.float32, device=device)
                b_hist_text, b_hist_mask = _expand_history_tensor(
                    item_text_tensor, hist_news_idx, batch_size
                )
                b_hist_kg, _ = _expand_history_tensor(
                    item_kg_tensor, hist_news_idx, batch_size
                )
                b_hist_kg_mask, _ = _expand_history_tensor(
                    item_kg_mask_tensor, hist_news_idx, batch_size
                )
                logits, _ = model(
                    user_idx=b_user,
                    news_idx=b_news,
                    cat_idx=b_cat,
                    subcat_idx=b_sub,
                    dense=b_dense,
                    item_text_base=item_text_tensor[b_news],
                    history_item_text_base=b_hist_text,
                    item_kg_base=item_kg_tensor[b_news],
                    history_item_kg_base=b_hist_kg,
                    item_kg_mask=item_kg_mask_tensor[b_news],
                    history_item_kg_mask=b_hist_kg_mask,
                    history_mask=b_hist_mask,
                    is_new_item=b_is_new,
                    return_repr=False,
                )
                scores.append(logits.detach().cpu().numpy())
            values.append(impression_auc(labels, np.concatenate(scores, axis=0)))

    return float(np.mean(values) if values else 0.0)


def run_train_ranker(
    cfg: dict[str, Any], ranker_art_root: Path | None = None
) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    maps = IdMaps.load(proc_root / "id_maps.json")

    pairs_train = pd.read_parquet(proc_root / "train_pairs.parquet")
    val_split = validation_split_name(cfg)
    pairs_val = pd.read_parquet(pair_artifact_path(proc_root, val_split))

    dense_cols = list(RANKER_DENSE_COLUMNS)
    train_ds = PairDataset(pairs_train, dense_cols=dense_cols)
    val_ds = PairDataset(pairs_val, dense_cols=dense_cols)

    device_str = cfg["ranker"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(ranker_art_root or (runs_root / "ranker"))

    teacher_root = runs_root / "teacher"
    item_text_base, item_kg_base = load_ranker_item_features(cfg, teacher_root)
    feature_fingerprints = ranker_feature_fingerprints(
        teacher_root,
        kg_enabled=bool(cfg.get("knowledge_graph", {}).get("enabled", False)),
    )
    teacher_item = np.load(teacher_root / "item_teacher_emb.npy")
    item_text_tensor = torch.tensor(item_text_base, dtype=torch.float32, device=device)
    item_kg_tensor = torch.tensor(item_kg_base, dtype=torch.float32, device=device)
    item_kg_mask_tensor = torch.tensor(
        np.linalg.norm(item_kg_base, axis=-1) > 0.0, dtype=torch.bool, device=device
    )
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
    n_users = max(maps.user2idx.values()) + 1
    n_news = int(news["news_idx"].max()) + 1
    n_cats = int(news["cat_idx"].max()) + 1
    n_subcats = int(news["subcat_idx"].max()) + 1

    dlrm_cfg = cfg["ranker"]["dlrm"]
    model = DLRMStudent(
        n_users=n_users,
        n_news=n_news,
        n_cats=n_cats,
        n_subcats=n_subcats,
        dense_dim=len(dense_cols),
        item_text_dim=int(item_text_base.shape[1]),
        item_kg_dim=int(item_kg_base.shape[-1]),
        emb_dim=int(dlrm_cfg["emb_dim"]),
        id_emb_dim=int(dlrm_cfg.get("id_emb_dim", dlrm_cfg["emb_dim"])),
        bottom_mlp=[int(x) for x in dlrm_cfg["bottom_mlp"]],
        top_mlp=[int(x) for x in dlrm_cfg["top_mlp"]],
        dropout=float(dlrm_cfg.get("dropout", 0.0)),
        fusion_heads=int(dlrm_cfg.get("fusion_heads", 4)),
        semantic_ff_mult=int(dlrm_cfg.get("semantic_ff_mult", 1)),
        semantic_dropout=float(
            dlrm_cfg.get("semantic_dropout", dlrm_cfg.get("dropout", 0.0))
        ),
        news_id_warm_scale=float(dlrm_cfg.get("news_id_warm_scale", 1.0)),
        news_id_cold_scale=float(dlrm_cfg.get("news_id_cold_scale", 1.0)),
        kg_gate_init=float(dlrm_cfg.get("kg_gate_init", 0.15)),
        kg_gate_trainable=bool(dlrm_cfg.get("kg_gate_trainable", False)),
    ).to(device)

    emb_dim = int(dlrm_cfg["emb_dim"])
    student_repr_dim = 2 * emb_dim
    teacher_repr_dim = 2 * teacher_dim
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
    val_loader = DataLoader(
        val_ds, batch_size=bsz, shuffle=False, num_workers=0, collate_fn=collate_batch
    )
    epochs = int(cfg["ranker"]["epochs"])

    max_grad_norm = float(cfg["ranker"].get("max_grad_norm", 5.0))
    if max_grad_norm <= 0.0:
        raise ValueError("ranker.max_grad_norm must be positive")

    sched_cfg = dict(cfg["ranker"].get("lr_scheduler", {}))
    scheduler = None
    if bool(sched_cfg.get("enabled", False)):
        sched_type = str(sched_cfg.get("type", "cosine")).lower()
        if sched_type != "cosine":
            raise ValueError("Only ranker.lr_scheduler.type='cosine' is supported")
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.1))
        if min_lr_ratio < 0.0 or min_lr_ratio > 1.0:
            raise ValueError("ranker.lr_scheduler.min_lr_ratio must be in [0, 1]")
        total_steps = max(1, epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=total_steps,
            eta_min=float(cfg["ranker"]["lr"]) * min_lr_ratio,
        )

    dist_cfg = cfg["ranker"]["distill"]
    distill_temperature = float(dist_cfg.get("temperature", 2.0))
    lam_logit = float(dist_cfg.get("lambda_logit", 0.7))
    lam_repr = float(dist_cfg.get("lambda_repr", 0.05))
    if distill_temperature <= 0.0:
        raise ValueError("ranker.distill.temperature must be positive")
    if lam_logit < 0.0 or lam_repr < 0.0:
        raise ValueError("ranker distillation weights must be non-negative")
    w_cold = float(dist_cfg.get("cold_weight", 2.0))
    w_warm = float(dist_cfg.get("warm_weight", 0.5))
    es_cfg = dict(cfg["ranker"].get("early_stopping", {}))
    es_patience = int(es_cfg.get("patience", 2))
    es_min_delta = float(es_cfg.get("min_delta", 1.0e-4))
    monitor = str(es_cfg.get("monitor", "impression_auc")).lower()
    if monitor not in {"impression_auc", "pair_auc"}:
        raise ValueError(
            "ranker.early_stopping.monitor must be 'impression_auc' or 'pair_auc'"
        )

    best_monitor_value = -1.0
    best_pair_auc = -1.0
    best_impression_auc = -1.0
    best_epoch = 0
    best_kg_gate = 0.0
    epochs_without_improvement = 0
    stop_reason = "max_epochs"
    epoch_metrics: list[dict[str, float | int]] = []
    for ep in range(1, epochs + 1):
        model.train()
        proj.train()
        losses = []
        rank_losses = []
        logit_losses = []
        repr_losses = []
        distill_losses = []
        for batch in tqdm(train_loader, desc=f"Train ep {ep}"):
            batch = to_device(batch, device)
            item_text_batch = item_text_tensor[batch["news_idx"]]
            hist_text_batch = item_text_tensor[batch["hist_news_idx"]]
            item_kg_batch = item_kg_tensor[batch["news_idx"]]
            hist_kg_batch = item_kg_tensor[batch["hist_news_idx"]]
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
                item_text_base=item_text_batch,
                history_item_text_base=hist_text_batch,
                item_kg_base=item_kg_batch,
                history_item_kg_base=hist_kg_batch,
                item_kg_mask=item_kg_mask_tensor[batch["news_idx"]],
                history_item_kg_mask=item_kg_mask_tensor[batch["hist_news_idx"]],
                history_mask=batch["hist_mask"],
                is_new_item=batch["is_new_item"],
                return_repr=True,
            )
            y = batch["label"]
            loss_rank = nn.functional.binary_cross_entropy_with_logits(logits, y)

            # This soft target regularizes the complete final scorer. It does
            # not require any individual student feature branch to imitate a
            # teacher representation.
            loss_logit = final_logit_distillation_loss(
                student_logits=logits,
                teacher_user=tu,
                teacher_item=ti,
                temperature=distill_temperature,
            )

            t_repr = torch.cat([tu, ti], dim=1)
            s_repr = proj(rep)
            loss_repr = ((s_repr - t_repr) ** 2).mean(dim=1)

            cold_mask = (
                (batch["is_cold_user"] == 1) | (batch["is_new_item"] == 1)
            ).float()
            w = (cold_mask * w_cold) + ((1.0 - cold_mask) * w_warm)
            w = w * has_hist.float()
            w = w.detach()

            weighted_logit_loss = (w * loss_logit).mean()
            weighted_repr_loss = (w * loss_repr).mean()
            distill_loss = (
                lam_logit * weighted_logit_loss
                + lam_repr * weighted_repr_loss
            )
            loss = loss_rank + distill_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(proj.parameters()), max_norm=max_grad_norm
            )
            opt.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.item()))
            rank_losses.append(float(loss_rank.item()))
            logit_losses.append(float(weighted_logit_loss.item()))
            repr_losses.append(float(weighted_repr_loss.item()))
            distill_losses.append(float(distill_loss.item()))

        # Validation AUC on the split returned by validation_split_name(cfg).
        model.eval()
        proj.eval()
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
                    item_text_base=item_text_tensor[batch["news_idx"]],
                    history_item_text_base=item_text_tensor[batch["hist_news_idx"]],
                    item_kg_base=item_kg_tensor[batch["news_idx"]],
                    history_item_kg_base=item_kg_tensor[batch["hist_news_idx"]],
                    item_kg_mask=item_kg_mask_tensor[batch["news_idx"]],
                    history_item_kg_mask=item_kg_mask_tensor[batch["hist_news_idx"]],
                    history_mask=batch["hist_mask"],
                    is_new_item=batch["is_new_item"],
                    return_repr=False,
                )
                ys.extend(batch["label"].detach().cpu().numpy().tolist())
                ps.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
        try:
            pair_auc = float(roc_auc_score(ys, ps))
        except ValueError:
            pair_auc = 0.0
        val_impression_auc = _validate_impression_auc(
            cfg=cfg,
            proc_root=proc_root,
            split_name=val_split,
            model=model,
            item_text_tensor=item_text_tensor,
            item_kg_tensor=item_kg_tensor,
            item_kg_mask_tensor=item_kg_mask_tensor,
        )
        monitor_value = (
            val_impression_auc if monitor == "impression_auc" else pair_auc
        )

        epoch_metrics.append(
            {
                "epoch": ep,
                "train_loss_mean": float(np.mean(losses) if losses else 0.0),
                "train_rank_loss_mean": float(
                    np.mean(rank_losses) if rank_losses else 0.0
                ),
                "train_logit_distill_loss_mean": float(
                    np.mean(logit_losses) if logit_losses else 0.0
                ),
                "train_repr_distill_loss_mean": float(
                    np.mean(repr_losses) if repr_losses else 0.0
                ),
                "train_distill_loss_mean": float(
                    np.mean(distill_losses) if distill_losses else 0.0
                ),
                "val_pair_auc": pair_auc,
                "val_impression_auc": val_impression_auc,
                "val_auc": val_impression_auc,
                "monitor": monitor,
                "monitor_value": monitor_value,
                "lr": float(opt.param_groups[0]["lr"]),
                "kg_gate": (
                    float(model.kg_gate().detach().cpu().item())
                    if model.has_kg
                    else 0.0
                ),
            }
        )
        save_json(art_root / "epochs.json", epoch_metrics)

        improved = monitor_value > best_monitor_value
        significant_improvement = (monitor_value - best_monitor_value) > es_min_delta
        if improved:
            best_monitor_value = monitor_value
            best_pair_auc = pair_auc
            best_impression_auc = val_impression_auc
            best_epoch = ep
            best_kg_gate = (
                float(model.kg_gate().detach().cpu().item())
                if model.has_kg
                else 0.0
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "proj": proj.state_dict(),
                    "cfg": cfg,
                    "epoch": ep,
                    "val_auc": val_impression_auc,
                    "val_pair_auc": pair_auc,
                    "val_impression_auc": val_impression_auc,
                    "monitor": monitor,
                    "monitor_value": monitor_value,
                    "feature_fingerprints": feature_fingerprints,
                },
                art_root / "best.pt",
            )
        if significant_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= es_patience:
            stop_reason = "early_stopping"
            break

    save_json(
        art_root / "train_summary.json",
        {
            "validation_split_name": val_split,
            "selection_monitor": monitor,
            "best_monitor_value": best_monitor_value,
            "best_val_auc": best_impression_auc,
            "best_val_impression_auc": best_impression_auc,
            "best_val_pair_auc": best_pair_auc,
            "best_epoch": best_epoch,
            "best_kg_gate": best_kg_gate,
            "distill_temperature": distill_temperature,
            "lambda_logit": lam_logit,
            "lambda_repr": lam_repr,
            "max_grad_norm": max_grad_norm,
            "lr_scheduler": sched_cfg,
            "early_stopping_patience": es_patience,
            "early_stopping_min_delta": es_min_delta,
            "stop_reason": stop_reason,
            "stopped_epoch": ep,
        },
    )

    cal_cfg = dict(cfg["ranker"].get("calibration", {}))
    ckpt = torch.load(art_root / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

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
                item_text_base=item_text_tensor[batch["news_idx"]],
                history_item_text_base=item_text_tensor[batch["hist_news_idx"]],
                item_kg_base=item_kg_tensor[batch["news_idx"]],
                history_item_kg_base=item_kg_tensor[batch["hist_news_idx"]],
                item_kg_mask=item_kg_mask_tensor[batch["news_idx"]],
                history_item_kg_mask=item_kg_mask_tensor[batch["hist_news_idx"]],
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
