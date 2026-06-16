from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mindrec.config import ensure_dir
from mindrec.data.kg import build_news_kg_entity_matrix_from_config
from mindrec.pipeline.ranker_assets import save_ranker_kg_features


def run_build_ranker_kg(cfg: dict[str, Any]) -> None:
    if not bool(cfg.get("knowledge_graph", {}).get("enabled", False)):
        raise ValueError("knowledge_graph.enabled must be true to build ranker KG features")

    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    news = pd.read_parquet(proc_root / "news.parquet")
    features, meta = build_news_kg_entity_matrix_from_config(cfg, news)
    if features is None:
        raise RuntimeError("KG feature builder returned no features while KG is enabled")

    teacher_root = ensure_dir(Path("runs") / cfg["run_name"] / "teacher")
    save_ranker_kg_features(
        teacher_root=teacher_root,
        item_kg_base=features,
        item_kg_meta={
            "mode": "kred_entity_slots",
            "max_entities_per_news": int(features.shape[1]),
            "kg_dim": int(features.shape[2]),
            "use_kg": True,
            "kg": meta,
        },
        update_teacher_meta=True,
    )
