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
from mindrec.models.teacher import TeacherTwoTower
from mindrec.utils import pair_artifact_path, save_json, set_seed, to_device, validation_split_name


class StudentProjHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def run_train_ranker(cfg: dict[str, Any]) -> None:
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    maps = IdMaps.load(proc_root / "id_maps.json")

    pairs_train = pd.read_parquet(proc_root / "train_pairs.parquet")
    val_split = validation_split_name(cfg)
    pairs_val = pd.read_parquet(pair_artifact_path(proc_root, val_split))

    dense_cols = ["history_len", "item_clicks_log1p"]
    train_ds = PairDataset(pairs_train, dense_cols=dense_cols)
    val_ds = PairDataset(pairs_val, dense_cols=dense_cols)

    device_str = cfg["ranker"].get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    art_root = ensure_dir(runs_root / "ranker")

    item_base = np.load(runs_root / "teacher" / "item_base_emb.npy")
    teacher_item = np.load(runs_root / "teacher" / "item_teacher_emb.npy")
    item_base_tensor = torch.tensor(item_base, dtype=torch.float32, device=device)
    teacher_item_tensor = torch.tensor(teacher_item, dtype=torch.float32, device=device)
    teacher_dim = int(teacher_item.shape[1])
    teacher_ckpt = torch.load(runs_root / "teacher" / "model.pt", map_location=device)
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
    ).to(device)

    emb_dim = int(dlrm_cfg["emb_dim"])
    student_repr_dim = 3 * emb_dim
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

    dist_cfg = cfg["ranker"]["distill"]
    temp = float(dist_cfg.get("temperature", 2.0))
    lam_logit = float(dist_cfg.get("lambda_logit", 1.0))
    lam_repr = float(dist_cfg.get("lambda_repr", 0.1))
    w_cold = float(dist_cfg.get("cold_weight", 2.0))
    w_warm = float(dist_cfg.get("warm_weight", 0.3))
    es_cfg = dict(cfg["ranker"].get("early_stopping", {}))
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

            t_repr = torch.cat([tu, ti], dim=1)
            s_repr = proj(rep)
            loss_repr = ((s_repr - t_repr) ** 2).mean(dim=1)

            cold_mask = (
                (batch["is_cold_user"] == 1) | (batch["is_new_item"] == 1)
            ).float()
            w = (cold_mask * w_cold) + ((1.0 - cold_mask) * w_warm)
            w = w * has_hist.float()
            w = w.detach()

            distill_loss = (w * (lam_logit * loss_logit + lam_repr * loss_repr)).mean()
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

        improved = auc > best_auc
        significant_improvement = (auc - best_auc) > es_min_delta
        if improved:
            best_auc = auc
            best_epoch = ep
            torch.save(
                {
                    "model": model.state_dict(),
                    "proj": proj.state_dict(),
                    "cfg": cfg,
                    "epoch": ep,
                    "val_auc": auc,
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
            "best_val_auc": best_auc,
            "best_epoch": best_epoch,
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
