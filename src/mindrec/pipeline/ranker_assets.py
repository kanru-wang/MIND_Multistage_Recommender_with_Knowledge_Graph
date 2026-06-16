from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from mindrec.utils import load_json, save_json


def ranker_score_batch_size(cfg: dict[str, Any]) -> int:
    batch_size = int(
        cfg.get("ranker", {}).get(
            "score_batch_size", cfg.get("ranker", {}).get("batch_size", 256)
        )
    )
    if batch_size <= 0:
        raise ValueError("ranker.score_batch_size must be positive")
    return batch_size


def _validate_kg_artifact_config(
    cfg: dict[str, Any], teacher_root: Path, item_kg_base: np.ndarray
) -> None:
    meta_path = teacher_root / "item_kg_base_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"KG artifact metadata was not found: {meta_path}. Run build_ranker_kg."
        )
    meta = load_json(meta_path)
    if meta.get("mode") != "kred_entity_slots":
        raise ValueError("KG artifact is not in KRED entity-slot format. Run build_ranker_kg.")
    if int(meta.get("max_entities_per_news", -1)) != int(item_kg_base.shape[1]):
        raise ValueError("KG artifact metadata does not match its entity-slot tensor.")
    if int(meta.get("kg_dim", -1)) != int(item_kg_base.shape[2]):
        raise ValueError("KG artifact metadata does not match its embedding dimension.")

    kg_cfg = dict(cfg.get("knowledge_graph", {}))
    built_cfg = dict(meta.get("kg", {}))
    checks = {
        "max_entities_per_news": int(kg_cfg.get("max_entities_per_news", 12)),
        "max_neighbors_per_entity": int(kg_cfg.get("max_neighbors_per_entity", 20)),
        "add_reverse_edges": bool(kg_cfg.get("add_reverse_edges", True)),
        "entity_weight": float(kg_cfg.get("entity_weight", 1.0)),
        "neighbor_weight": float(kg_cfg.get("neighbor_weight", 0.5)),
        "relation_weight": float(kg_cfg.get("relation_weight", 0.25)),
        "normalize": bool(kg_cfg.get("normalize", True)),
    }
    mismatches = [
        name
        for name, expected in checks.items()
        if built_cfg.get(name) != expected
    ]
    if mismatches:
        names = ", ".join(mismatches)
        raise ValueError(
            f"KG artifact does not match the current config ({names}). "
            "Run build_ranker_kg before training or evaluation."
        )


def load_ranker_item_features(
    cfg: dict[str, Any], teacher_root: Path
) -> tuple[np.ndarray, np.ndarray]:
    item_text_base = np.load(teacher_root / "item_base_emb.npy")
    if item_text_base.ndim != 2:
        raise ValueError("item_base_emb.npy must be a 2D matrix")

    if not bool(cfg.get("knowledge_graph", {}).get("enabled", False)):
        item_kg_base = np.zeros((item_text_base.shape[0], 0), dtype=np.float32)
        return item_text_base, item_kg_base

    kg_base_path = teacher_root / "item_kg_base_emb.npy"
    if not kg_base_path.exists():
        raise FileNotFoundError(
            f"KG is enabled, but ranker KG features were not found: {kg_base_path}. "
            "Run build_ranker_kg (or train_teacher) so item_kg_base_emb.npy is generated."
        )

    item_kg_base = np.load(kg_base_path)
    if item_kg_base.ndim != 3:
        raise ValueError(
            "item_kg_base_emb.npy must be a 3D [news, entity_slot, embedding] tensor"
        )
    if item_kg_base.shape[0] != item_text_base.shape[0]:
        raise ValueError(
            "Text and KG item feature matrices must contain the same number of items"
        )
    _validate_kg_artifact_config(cfg, teacher_root, item_kg_base)
    return item_text_base, item_kg_base


def save_ranker_kg_features(
    teacher_root: Path,
    item_kg_base: np.ndarray,
    item_kg_meta: dict[str, Any],
    *,
    update_teacher_meta: bool = False,
) -> None:
    np.save(teacher_root / "item_kg_base_emb.npy", item_kg_base.astype(np.float32))
    save_json(teacher_root / "item_kg_base_meta.json", item_kg_meta)

    teacher_meta_path = teacher_root / "meta.json"
    if update_teacher_meta and teacher_meta_path.exists():
        teacher_meta = load_json(teacher_meta_path)
        teacher_meta["item_kg_base_features"] = item_kg_meta
        item_base_meta = teacher_meta.get("item_base_features")
        if isinstance(item_base_meta, dict) and not bool(item_base_meta.get("use_kg")):
            item_base_meta.pop("kg", None)
        save_json(teacher_meta_path, teacher_meta)


def ranker_feature_fingerprints(
    teacher_root: Path, *, kg_enabled: bool
) -> dict[str, str]:
    names = ["item_base_emb.npy"]
    if kg_enabled:
        names.append("item_kg_base_emb.npy")

    fingerprints: dict[str, str] = {}
    for name in names:
        path = teacher_root / name
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[name] = digest.hexdigest()
    return fingerprints


def validate_ranker_feature_fingerprints(
    checkpoint: dict[str, Any],
    teacher_root: Path,
    *,
    kg_enabled: bool,
) -> None:
    expected = checkpoint.get("feature_fingerprints")
    if not expected:
        raise ValueError(
            "Ranker checkpoint does not record input feature fingerprints. "
            "Retrain the ranker before evaluation."
        )
    current = ranker_feature_fingerprints(teacher_root, kg_enabled=kg_enabled)
    if current != expected:
        raise ValueError(
            "Ranker input feature artifacts changed after this checkpoint was trained. "
            "Retrain the ranker before evaluation."
        )
