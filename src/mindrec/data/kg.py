from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeighborEdge:
    relation_id: str
    neighbor_id: str
    # 1 means current entity is the triple head; -1 means it is the triple tail.
    direction: int


def parse_wikidata_ids(*entity_columns: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in entity_columns:
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            rows = json.loads(value)
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            entity_id = str(row.get("WikidataId") or "").strip()
            if entity_id and entity_id not in seen:
                ids.append(entity_id)
                seen.add(entity_id)
    return ids


def read_embedding_vec(path: str | Path) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                vec = np.asarray([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            embeddings[parts[0]] = vec
    return embeddings


def read_embedding_vecs(paths: list[str | Path]) -> dict[str, np.ndarray]:
    embeddings: dict[str, np.ndarray] = {}
    for path in paths:
        embeddings.update(read_embedding_vec(path))
    return embeddings


def read_kg_triples(
    path: str | Path,
    max_neighbors_per_entity: int,
    add_reverse_edges: bool = True,
) -> dict[str, list[NeighborEdge]]:
    adjacency: dict[str, list[NeighborEdge]] = {}
    if max_neighbors_per_entity <= 0:
        return adjacency

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            head, relation, tail = parts[:3]
            if len(adjacency.get(head, ())) < max_neighbors_per_entity:
                adjacency.setdefault(head, []).append(
                    NeighborEdge(
                        relation_id=relation,
                        neighbor_id=tail,
                        direction=1,
                    )
                )
            if (
                add_reverse_edges
                and len(adjacency.get(tail, ())) < max_neighbors_per_entity
            ):
                adjacency.setdefault(tail, []).append(
                    NeighborEdge(
                        relation_id=relation,
                        neighbor_id=head,
                        direction=-1,
                    )
                )
    return adjacency


def resolve_mind_side_files(
    raw_root: str | Path,
    train_dir: str,
    dev_dir: str,
    filename: str,
    extra_dirs: list[str] | None = None,
) -> list[Path]:
    raw_root = Path(raw_root)
    side_dirs = [train_dir, dev_dir] + list(extra_dirs or [])
    candidates = [raw_root / side_dir / filename for side_dir in side_dirs]
    return [path for path in candidates if path.exists()]


def _configured_paths(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [Path(v) for v in value]
    return [Path(value)]


def _format_paths(paths: list[Path]) -> str | list[str] | None:
    if not paths:
        return None
    if len(paths) == 1:
        return str(paths[0])
    return [str(path) for path in paths]


def build_news_kg_feature_matrix(
    news: pd.DataFrame,
    entity_embeddings: dict[str, np.ndarray],
    relation_embeddings: dict[str, np.ndarray] | None = None,
    adjacency: dict[str, list[NeighborEdge]] | None = None,
    max_entities_per_news: int = 12,
    entity_weight: float = 1.0,
    neighbor_weight: float = 0.5,
    relation_weight: float = 0.25,
    normalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not entity_embeddings:
        raise ValueError("entity_embeddings must not be empty")

    dim = int(next(iter(entity_embeddings.values())).shape[0])
    n_items = int(news["news_idx"].max()) + 1
    features = np.zeros((n_items, dim), dtype=np.float32)
    relation_embeddings = relation_embeddings or {}
    adjacency = adjacency or {}

    n_with_entities = 0
    n_with_entity_vectors = 0
    n_with_neighbors = 0
    n_entity_mentions = 0
    n_entity_vectors = 0
    n_neighbor_vectors = 0

    for row in news.itertuples(index=False):
        news_idx = int(getattr(row, "news_idx"))
        entity_ids = parse_wikidata_ids(
            getattr(row, "title_entities", ""),
            getattr(row, "abstract_entities", ""),
        )[:max_entities_per_news]
        if entity_ids:
            n_with_entities += 1
            n_entity_mentions += len(entity_ids)

        entity_vecs = [
            entity_embeddings[entity_id]
            for entity_id in entity_ids
            if entity_id in entity_embeddings
        ]
        if entity_vecs:
            n_with_entity_vectors += 1
            n_entity_vectors += len(entity_vecs)

        neighbor_messages: list[np.ndarray] = []
        for entity_id in entity_ids:
            for edge in adjacency.get(entity_id, []):
                neighbor_vec = entity_embeddings.get(edge.neighbor_id)
                if neighbor_vec is None:
                    continue
                relation_vec = relation_embeddings.get(edge.relation_id)
                if relation_vec is None:
                    msg = neighbor_vec
                else:
                    # TransE-style relation embeddings satisfy head + relation ~= tail.
                    # Convert the neighbor back toward the current entity's space:
                    # head-side entity receives tail - relation; tail-side receives head + relation.
                    msg = neighbor_vec - (
                        edge.direction * relation_weight * relation_vec
                    )
                neighbor_messages.append(msg.astype(np.float32, copy=False))

        parts: list[np.ndarray] = []
        if entity_vecs:
            parts.append(entity_weight * np.mean(entity_vecs, axis=0))
        if neighbor_messages:
            n_with_neighbors += 1
            n_neighbor_vectors += len(neighbor_messages)
            parts.append(neighbor_weight * np.mean(neighbor_messages, axis=0))
        if parts:
            vec = np.sum(parts, axis=0).astype(np.float32, copy=False)
            if normalize:
                norm = float(np.linalg.norm(vec))
                if norm > 0.0:
                    vec = vec / norm
            features[news_idx] = vec

    meta = {
        "dim": dim,
        "n_items": n_items,
        "n_items_with_linked_entities": n_with_entities,
        "n_items_with_entity_vectors": n_with_entity_vectors,
        "n_items_with_kg_neighbors": n_with_neighbors,
        "n_entity_mentions": n_entity_mentions,
        "n_entity_vectors_used": n_entity_vectors,
        "n_neighbor_vectors_used": n_neighbor_vectors,
        "max_entities_per_news": int(max_entities_per_news),
        "entity_weight": float(entity_weight),
        "neighbor_weight": float(neighbor_weight),
        "relation_weight": float(relation_weight),
        "normalize": bool(normalize),
    }
    return features, meta


def build_news_kg_feature_matrix_from_config(
    cfg: dict[str, Any],
    news: pd.DataFrame,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    kg_cfg = dict(cfg.get("knowledge_graph", {}))
    if not bool(kg_cfg.get("enabled", False)):
        return None, {"enabled": False}

    raw_cfg = cfg["data"]
    extra_side_dirs = []
    if raw_cfg.get("test_dir") and raw_cfg.get("mode") == "leaderboard_submission":
        extra_side_dirs.append(str(raw_cfg["test_dir"]))
    entity_paths = _configured_paths(kg_cfg.get("entity_embedding_path"))
    if not entity_paths:
        entity_paths = resolve_mind_side_files(
            raw_root=raw_cfg["raw_root"],
            train_dir=raw_cfg["train_dir"],
            dev_dir=raw_cfg["dev_dir"],
            filename="entity_embedding.vec",
            extra_dirs=extra_side_dirs,
        )
    relation_paths = _configured_paths(kg_cfg.get("relation_embedding_path"))
    if not relation_paths:
        relation_paths = resolve_mind_side_files(
            raw_root=raw_cfg["raw_root"],
            train_dir=raw_cfg["train_dir"],
            dev_dir=raw_cfg["dev_dir"],
            filename="relation_embedding.vec",
            extra_dirs=extra_side_dirs,
        )
    triples_path = kg_cfg.get("triples_path")
    if not triples_path:
        raise ValueError(
            "knowledge_graph.enabled is true but knowledge_graph.triples_path is not set."
        )
    triples = Path(triples_path)
    if not triples.exists():
        raise FileNotFoundError(f"Configured KG triples file was not found: {triples}")

    missing_entity_paths = [path for path in entity_paths if not path.exists()]
    if not entity_paths or missing_entity_paths:
        raise FileNotFoundError(
            "knowledge_graph.enabled is true but entity_embedding.vec was not found. "
            "Place it in the MIND train/dev directory or set knowledge_graph.entity_embedding_path."
        )

    missing_relation_paths = [path for path in relation_paths if not path.exists()]
    if not relation_paths or missing_relation_paths:
        raise FileNotFoundError(
            "knowledge_graph.enabled is true but relation_embedding.vec was not found. "
            "Place it in the MIND train/dev directory or set knowledge_graph.relation_embedding_path."
        )

    entity_embeddings = read_embedding_vecs(entity_paths)
    relation_embeddings = read_embedding_vecs(relation_paths)
    if not relation_embeddings:
        raise ValueError("relation_embedding.vec files did not contain any usable vectors")
    adjacency = read_kg_triples(
        path=triples,
        max_neighbors_per_entity=int(kg_cfg.get("max_neighbors_per_entity", 20)),
        add_reverse_edges=bool(kg_cfg.get("add_reverse_edges", True)),
    )

    features, meta = build_news_kg_feature_matrix(
        news=news,
        entity_embeddings=entity_embeddings,
        relation_embeddings=relation_embeddings,
        adjacency=adjacency,
        max_entities_per_news=int(kg_cfg.get("max_entities_per_news", 12)),
        entity_weight=float(kg_cfg.get("entity_weight", 1.0)),
        neighbor_weight=float(kg_cfg.get("neighbor_weight", 0.5)),
        relation_weight=float(kg_cfg.get("relation_weight", 0.25)),
        normalize=bool(kg_cfg.get("normalize", True)),
    )
    meta.update(
        {
            "enabled": True,
            "entity_embedding_path": _format_paths(entity_paths),
            "relation_embedding_path": _format_paths(relation_paths),
            "triples_path": str(triples),
            "n_entity_embeddings": int(len(entity_embeddings)),
            "n_relation_embeddings": int(len(relation_embeddings)),
            "n_entities_with_neighbors": int(len(adjacency)),
        }
    )
    return features, meta
