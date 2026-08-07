from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.data.featurize import IdMaps
from mindrec.pipeline.teacher_train import format_news_texts
from mindrec.pipeline.hard_negative_sampling import choose_negative_offsets
from mindrec.utils import (
    behavior_artifact_path,
    device_info,
    load_json,
    log_device,
    resolve_device,
    save_json,
    set_seed,
)


@dataclass(frozen=True)
class TextAdaptSample:
    history: list[int]
    positive: int
    negatives: list[int]


class TextAdaptIterableDataset(IterableDataset):
    """Generate samples per epoch without retaining MINDlarge sample objects."""

    def __init__(
        self,
        behaviors: pd.DataFrame,
        maps: IdMaps,
        baseline_embeddings: np.ndarray,
        sampling_kwargs: dict[str, Any],
    ) -> None:
        self.behaviors = behaviors
        self.maps = maps
        self.baseline_embeddings = baseline_embeddings
        self.sampling_kwargs = sampling_kwargs
        self.epoch = 0
        self.last_stats: dict[str, int | float] = {}

    def __iter__(self):
        seed = int(self.sampling_kwargs["seed"]) + self.epoch
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.behaviors))
        stats = _empty_sample_stats()
        self.last_stats = stats
        for row_index in order:
            row = self.behaviors.iloc[int(row_index)]
            yield from _samples_for_behavior(
                row,
                self.maps,
                self.baseline_embeddings,
                rng=rng,
                stats=stats,
                **{k: v for k, v in self.sampling_kwargs.items() if k != "seed"},
            )
        self.epoch += 1


def _empty_sample_stats() -> dict[str, int | float]:
    return {
        "samples": 0,
        "selected_hard_negatives": 0,
        "selected_random_negatives": 0,
        "hard_mining_groups": 0,
        "random_only_groups": 0,
    }


def _samples_for_behavior(
    row: pd.Series,
    maps: IdMaps,
    baseline_embeddings: np.ndarray,
    *,
    max_history: int,
    negatives_per_positive: int,
    hard_fraction: float,
    hard_pool_size: int,
    teacher_consistent_hard_only: bool,
    max_score_above_positive: float,
    hard_for_cold_users_only: bool,
    min_user_hist_for_warm: int,
    rng: np.random.Generator,
    stats: dict[str, int | float],
):
    raw_history = list(row["history"])
    history = [maps.news2idx.get(str(n), 0) for n in raw_history[-max_history:]]
    history = [n for n in history if n != 0]
    if not history:
        return
    negatives = [
        maps.news2idx.get(str(n), 0)
        for n, label in zip(row["cand_news_id"], row["cand_label"])
        if int(label) == 0
    ]
    negatives = list(dict.fromkeys(n for n in negatives if n != 0))
    positives = [
        maps.news2idx.get(str(n), 0)
        for n, label in zip(row["cand_news_id"], row["cand_label"])
        if int(label) == 1 and maps.news2idx.get(str(n), 0) != 0
    ]
    if not negatives or not positives:
        return
    history_vector = baseline_embeddings[history].mean(axis=0)
    history_vector /= max(float(np.linalg.norm(history_vector)), 1.0e-12)
    use_hard = not hard_for_cold_users_only or len(raw_history) < min_user_hist_for_warm
    for positive in positives:
        n_select = min(negatives_per_positive, len(negatives))
        if use_hard and hard_fraction > 0.0:
            pool_size = min(hard_pool_size, len(negatives))
            pool = rng.choice(negatives, size=pool_size, replace=False).astype(int)
            scores = baseline_embeddings[pool] @ history_vector
            eligible = None
            if teacher_consistent_hard_only:
                positive_score = float(baseline_embeddings[positive] @ history_vector)
                eligible = scores <= positive_score + max_score_above_positive
            offsets, n_hard, n_random = choose_negative_offsets(
                scores, n_select, hard_fraction, rng, eligible
            )
            selected = pool[offsets].tolist()
            stats["hard_mining_groups"] += 1
        else:
            selected = rng.choice(negatives, size=n_select, replace=False).tolist()
            n_hard, n_random = 0, len(selected)
            stats["random_only_groups"] += 1
        stats["samples"] += 1
        stats["selected_hard_negatives"] += n_hard
        stats["selected_random_negatives"] += n_random
        yield TextAdaptSample(history, positive, selected)


def build_text_adaptation_samples(
    behaviors: pd.DataFrame,
    maps: IdMaps,
    baseline_embeddings: np.ndarray,
    *,
    max_history: int,
    negatives_per_positive: int,
    hard_fraction: float,
    hard_pool_size: int,
    teacher_consistent_hard_only: bool = True,
    max_score_above_positive: float = 0.0,
    hard_for_cold_users_only: bool = False,
    min_user_hist_for_warm: int = 5,
    seed: int,
) -> tuple[list[TextAdaptSample], dict[str, int | float]]:
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1")
    if hard_pool_size < 1:
        raise ValueError("hard_pool_size must be at least 1")
    if hard_pool_size < negatives_per_positive:
        raise ValueError("hard_pool_size must be at least negatives_per_positive")
    rng = np.random.default_rng(seed)
    stats = _empty_sample_stats()
    samples = []
    for _, row in tqdm(behaviors.iterrows(), total=len(behaviors), desc="Build text-adaptation samples"):
        samples.extend(_samples_for_behavior(
            row, maps, baseline_embeddings, max_history=max_history,
            negatives_per_positive=negatives_per_positive,
            hard_fraction=hard_fraction, hard_pool_size=hard_pool_size,
            teacher_consistent_hard_only=teacher_consistent_hard_only,
            max_score_above_positive=max_score_above_positive,
            hard_for_cold_users_only=hard_for_cold_users_only,
            min_user_hist_for_warm=min_user_hist_for_warm, rng=rng, stats=stats,
        ))
    stats["hard_negatives_per_group"] = int(round(negatives_per_positive * hard_fraction))
    return samples, stats


def _collate(samples: list[TextAdaptSample]) -> dict[str, torch.Tensor]:
    max_history = max(len(s.history) for s in samples)
    max_negatives = max(len(s.negatives) for s in samples)
    history = torch.zeros((len(samples), max_history), dtype=torch.long)
    history_mask = torch.zeros_like(history, dtype=torch.bool)
    negatives = torch.zeros((len(samples), max_negatives), dtype=torch.long)
    negative_mask = torch.zeros_like(negatives, dtype=torch.bool)
    for i, sample in enumerate(samples):
        history[i, : len(sample.history)] = torch.as_tensor(sample.history)
        history_mask[i, : len(sample.history)] = True
        negatives[i, : len(sample.negatives)] = torch.as_tensor(sample.negatives)
        negative_mask[i, : len(sample.negatives)] = True
    return {"history": history, "history_mask": history_mask, "positive": torch.tensor([s.positive for s in samples]), "negatives": negatives, "negative_mask": negative_mask}


def _encode_indices(
    model: SentenceTransformer,
    indices: torch.Tensor,
    texts_by_idx: list[str],
    device: torch.device,
) -> torch.Tensor:
    texts = [texts_by_idx[int(i)] for i in indices.detach().cpu().tolist()]
    # SentenceTransformers 5.x includes non-tensor metadata such as
    # ``modality: "text"`` in preprocessing output. Preserve that metadata and
    # move only tensors to the training device.
    features = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in model.preprocess(texts).items()
    }
    return F.normalize(model(features)["sentence_embedding"], dim=-1)


def _batch_loss(
    model: SentenceTransformer,
    batch: dict[str, torch.Tensor],
    texts_by_idx: list[str],
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    history_shape = batch["history"].shape
    negative_shape = batch["negatives"].shape
    history_z = _encode_indices(model, batch["history"].reshape(-1), texts_by_idx, device).reshape(*history_shape, -1)
    history_mask = batch["history_mask"].to(device)
    user_z = (history_z * history_mask.unsqueeze(-1)).sum(1) / history_mask.sum(1, keepdim=True).clamp_min(1)
    user_z = F.normalize(user_z, dim=-1)
    positive_z = _encode_indices(model, batch["positive"], texts_by_idx, device)
    negative_z = _encode_indices(model, batch["negatives"].reshape(-1), texts_by_idx, device).reshape(*negative_shape, -1)
    positive_logits = (user_z * positive_z).sum(-1, keepdim=True)
    negative_logits = (user_z.unsqueeze(1) * negative_z).sum(-1)
    negative_logits = negative_logits.masked_fill(
        ~batch["negative_mask"].to(device), float("-inf")
    )
    local = torch.cat([positive_logits, negative_logits], dim=1) / temperature
    targets = torch.zeros(local.size(0), dtype=torch.long, device=device)
    in_batch = user_z @ positive_z.T / temperature
    diagonal = torch.arange(local.size(0), device=device)
    return F.cross_entropy(local, targets) + 0.5 * (F.cross_entropy(in_batch, diagonal) + F.cross_entropy(in_batch.T, diagonal))


def _referenced_news_indices(behaviors: pd.DataFrame, maps: IdMaps) -> list[int]:
    news_ids: set[str] = set()
    for row in behaviors.itertuples(index=False):
        news_ids.update(str(news_id) for news_id in row.history)
        news_ids.update(str(news_id) for news_id in row.cand_news_id)
    return sorted(
        {
            maps.news2idx[news_id]
            for news_id in news_ids
            if maps.news2idx.get(news_id, 0) != 0
        }
    )


def _binary_auc_fast(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact binary AUC via positive/negative comparisons, including ties."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _evaluate_impression_auc(
    model: SentenceTransformer,
    behaviors: pd.DataFrame,
    maps: IdMaps,
    texts_by_idx: list[str],
    *,
    max_history: int,
    encode_batch_size: int,
) -> tuple[float, int]:
    """Evaluate raw encoder ranking on held-out, impression-grouped clicks."""
    indices = _referenced_news_indices(behaviors, maps)
    embeddings = np.zeros(
        (len(texts_by_idx), int(model.get_embedding_dimension())), dtype=np.float32
    )
    embeddings[indices] = model.encode(
        [texts_by_idx[index] for index in indices],
        batch_size=encode_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    aucs: list[float] = []
    for row in tqdm(
        behaviors.itertuples(index=False),
        total=len(behaviors),
        desc="Validate adapted MiniLM",
    ):
        history = [
            maps.news2idx.get(str(news_id), 0)
            for news_id in row.history[-max_history:]
        ]
        history = [index for index in history if index != 0]
        if not history:
            continue
        candidate_indices: list[int] = []
        labels: list[int] = []
        for news_id, label in zip(row.cand_news_id, row.cand_label):
            index = maps.news2idx.get(str(news_id), 0)
            if index != 0:
                candidate_indices.append(index)
                labels.append(int(label))
        label_array = np.asarray(labels, dtype=np.int8)
        if len(candidate_indices) < 2 or len(np.unique(label_array)) < 2:
            continue
        history_vector = embeddings[history].mean(axis=0)
        history_vector /= max(float(np.linalg.norm(history_vector)), 1.0e-12)
        scores = embeddings[candidate_indices] @ history_vector
        aucs.append(_binary_auc_fast(label_array, scores))
    return float(np.mean(aucs) if aucs else 0.0), len(aucs)


def _resolve_max_updates(adaptation: dict[str, Any]) -> int:
    maximum = int(adaptation.get("max_optimizer_updates", 10_000))
    if maximum < 1:
        raise ValueError("max_optimizer_updates must be at least 1")
    return maximum


def _resolve_initial_model(
    cfg: dict[str, Any], adaptation: dict[str, Any]
) -> tuple[str, int]:
    initial_run = adaptation.get("initial_model_from_run")
    if initial_run is None:
        return str(cfg["teacher"]["model_name"]), 0
    source_root = Path("runs") / str(initial_run) / "text_encoder"
    source_model = source_root / "model"
    source_meta = source_root / "meta.json"
    if not source_model.exists() or not source_meta.exists():
        raise FileNotFoundError(
            f"Continuation source is incomplete under {source_root}. "
            "Complete Phase 1 first."
        )
    initial_update = int(load_json(source_meta).get("best_update", -1))
    expected_update = adaptation.get("expected_initial_update")
    if expected_update is not None and initial_update != int(expected_update):
        raise ValueError(
            f"Expected Phase 1 update {int(expected_update):,}, but "
            f"{source_meta} records {initial_update:,}."
        )
    return str(source_model), initial_update


def run_adapt_text_encoder(cfg: dict[str, Any]) -> None:
    adaptation = dict(cfg["teacher"].get("text_adaptation", {}))
    if not adaptation.get("enabled", False):
        raise ValueError("teacher.text_adaptation.enabled must be true")
    seed = int(cfg["data"].get("sub_sample", {}).get("seed", 13))
    set_seed(seed)
    device = resolve_device(adaptation.get("device", cfg["teacher"].get("device", "cuda")))
    log_device(device, "Text encoder adaptation")
    proc_root = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    art_root = ensure_dir(Path("runs") / cfg["run_name"] / "text_encoder")
    maps = IdMaps.load(proc_root / "id_maps.json")
    news = pd.read_parquet(proc_root / "news.parquet")
    include_prefix = bool(cfg["teacher"].get("text", {}).get("include_category_prefix", False))
    texts = format_news_texts(news, include_prefix)
    max_idx = max(maps.news2idx.values(), default=0)
    texts_by_idx = [""] * (max_idx + 1)
    for text, news_idx in zip(texts, news["news_idx"].astype(int)):
        texts_by_idx[news_idx] = text
    initial_model_source, initial_model_update = _resolve_initial_model(
        cfg, adaptation
    )
    model = SentenceTransformer(
        initial_model_source, device=str(device), local_files_only=True
    )
    encode_batch_size = int(adaptation.get("encode_batch_size", 256))
    baseline = np.zeros(
        (max_idx + 1, int(model.get_embedding_dimension())), dtype=np.float32
    )
    training_split = str(adaptation.get("training_split", "train"))
    training_behavior_path = behavior_artifact_path(proc_root, training_split)
    if not training_behavior_path.exists():
        raise FileNotFoundError(
            f"MiniLM training split {training_split!r} was not found at "
            f"{training_behavior_path}."
        )
    behaviors = pd.read_parquet(training_behavior_path)
    valid_indices = _referenced_news_indices(behaviors, maps)
    baseline[valid_indices] = model.encode(
        [texts_by_idx[index] for index in valid_indices],
        batch_size=encode_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    sampling_kwargs = {
        "max_history": int(adaptation.get("max_history", 10)),
        "negatives_per_positive": int(adaptation.get("negatives_per_positive", 4)),
        "hard_fraction": float(adaptation.get("hard_fraction", 0.25)),
        "hard_pool_size": int(adaptation.get("hard_pool_size", 20)),
        "teacher_consistent_hard_only": bool(
            adaptation.get("teacher_consistent_hard_only", True)
        ),
        "max_score_above_positive": float(
            adaptation.get("max_score_above_positive", 0.0)
        ),
        "hard_for_cold_users_only": bool(
            adaptation.get("hard_for_cold_users_only", False)
        ),
        "min_user_hist_for_warm": int(cfg["data"]["min_user_hist_for_warm"]),
        "seed": seed,
    }
    if sampling_kwargs["negatives_per_positive"] < 1:
        raise ValueError("negatives_per_positive must be at least 1")
    if not 0.0 <= sampling_kwargs["hard_fraction"] <= 1.0:
        raise ValueError("hard_fraction must be between 0 and 1")
    if sampling_kwargs["hard_pool_size"] < sampling_kwargs["negatives_per_positive"]:
        raise ValueError("hard_pool_size must be at least negatives_per_positive")
    sample_dataset = TextAdaptIterableDataset(
        behaviors, maps, baseline, sampling_kwargs
    )
    loader = DataLoader(
        sample_dataset,
        batch_size=int(adaptation.get("batch_size", 16)),
        collate_fn=_collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(adaptation.get("lr", 2.0e-5)),
        weight_decay=float(adaptation.get("weight_decay", 0.01)),
    )
    temperature = float(adaptation.get("temperature", 0.05))
    accumulation = int(adaptation.get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    max_updates = _resolve_max_updates(adaptation)
    early_cfg = dict(adaptation.get("early_stopping", {}))
    early_enabled = bool(early_cfg.get("enabled", False))
    validation_interval = int(early_cfg.get("validation_interval_updates", 1_000))
    patience = int(early_cfg.get("patience", 3))
    min_delta = float(early_cfg.get("min_delta", 1.0e-4))
    if validation_interval < 1 or patience < 1:
        raise ValueError("MiniLM validation interval and patience must be at least 1")
    val_path = behavior_artifact_path(proc_root, "val")
    if early_enabled and not val_path.exists():
        raise FileNotFoundError(
            f"Early stopping requires held-out behaviors at {val_path}. "
            "Use the temporal selection config for Phase 1."
        )
    val_behaviors = pd.read_parquet(val_path) if early_enabled else None
    history_rows: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_update = 0
    checks_without_improvement = 0
    stop_reason = "max_optimizer_updates"
    best_model_path = art_root / "best_model"

    def validate(update: int, train_loss: float | None) -> bool:
        nonlocal best_auc, best_update, checks_without_improvement
        assert val_behaviors is not None
        model.eval()
        val_auc, n_eval = _evaluate_impression_auc(
            model,
            val_behaviors,
            maps,
            texts_by_idx,
            max_history=int(sampling_kwargs["max_history"]),
            encode_batch_size=encode_batch_size,
        )
        improved = val_auc > best_auc + min_delta
        if improved:
            best_auc = val_auc
            best_update = update
            checks_without_improvement = 0
            model.save_pretrained(str(best_model_path))
        else:
            checks_without_improvement += 1
        history_rows.append(
            {
                "optimizer_update": update,
                "train_loss_since_validation": train_loss,
                "val_impression_auc": val_auc,
                "n_val_impressions": n_eval,
                "improved": improved,
                "checks_without_improvement": checks_without_improvement,
            }
        )
        save_json(art_root / "validation_history.json", history_rows)
        print(
            f"MiniLM validation at update {update:,}: AUC={val_auc:.6f} "
            f"(best={best_auc:.6f} at {best_update:,})"
        )
        model.train()
        return improved

    optimizer.zero_grad()
    model.train()
    if early_enabled:
        validate(0, None)
    optimizer_updates = 0
    microbatches = 0
    losses_since_validation: list[float] = []
    stop = False
    progress = tqdm(
        total=max_updates,
        desc="Adapt MiniLM optimizer updates",
        unit="update",
        dynamic_ncols=True,
    )
    while optimizer_updates < max_updates and not stop:
        yielded = False
        for batch in loader:
            yielded = True
            loss = _batch_loss(model, batch, texts_by_idx, temperature, device)
            (loss / accumulation).backward()
            microbatches += 1
            losses_since_validation.append(float(loss.detach().cpu()))
            if microbatches % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_updates += 1
                progress.update(1)
                progress.set_postfix(
                    loss=f"{np.mean(losses_since_validation[-20:]):.4f}",
                    best_auc=(f"{best_auc:.5f}" if np.isfinite(best_auc) else "n/a"),
                )
                if early_enabled and optimizer_updates % validation_interval == 0:
                    mean_loss = float(np.mean(losses_since_validation))
                    losses_since_validation.clear()
                    validate(optimizer_updates, mean_loss)
                    if checks_without_improvement >= patience:
                        stop_reason = "early_stopping"
                        stop = True
                if optimizer_updates >= max_updates or stop:
                    break
        if not yielded:
            raise ValueError("No text-encoder adaptation samples were built")
    progress.close()
    if early_enabled:
        if optimizer_updates % validation_interval != 0:
            validate(
                optimizer_updates,
                float(np.mean(losses_since_validation)) if losses_since_validation else None,
            )
        model = SentenceTransformer(
            str(best_model_path), device=str(device), local_files_only=True
        )
    else:
        best_update = optimizer_updates
    model.save_pretrained(str(art_root / "model"))
    cumulative_updates = initial_model_update + optimizer_updates
    save_json(
        art_root / "meta.json",
        {
            "base_model_name": cfg["teacher"]["model_name"],
            "objective": "history_mean_clicked_article_contrastive",
            "negative_policy": "shared_ranker_hard_random_policy",
            "hard_negative_scorer": "initial_encoder_snapshot_bootstrap",
            "sample_materialization": "lazy_per_epoch",
            "training_behavior_artifact": str(training_behavior_path),
            "training_split": training_split,
            "initial_model_source": initial_model_source,
            "initial_model_update": initial_model_update,
            "continuation_training": initial_model_update > 0,
            "validation_behavior_artifact": str(val_path) if early_enabled else None,
            "validation_metric": "history_mean_candidate_cosine_impression_auc" if early_enabled else None,
            "hidden_test_used": False,
            "device_info": device_info(device),
            "lr": float(adaptation.get("lr", 2.0e-5)),
            "temperature": temperature,
            "max_history": int(adaptation.get("max_history", 10)),
            "max_optimizer_updates": max_updates,
            "completed_optimizer_updates": optimizer_updates,
            "cumulative_optimizer_updates": cumulative_updates,
            "best_update": best_update,
            "best_val_impression_auc": best_auc if early_enabled else None,
            "early_stopping_enabled": early_enabled,
            "early_stopping_patience": patience if early_enabled else None,
            "validation_interval_updates": validation_interval if early_enabled else None,
            "stop_reason": stop_reason,
            **sample_dataset.last_stats,
        },
    )
    print(
        f"MiniLM adaptation complete: {optimizer_updates:,} local updates, "
        f"{cumulative_updates:,} cumulative updates; "
        f"model saved to {art_root / 'model'}"
    )
