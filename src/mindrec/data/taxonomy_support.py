"""Fit-aware taxonomy support and the neutral-OOV policy.

Category and subcategory values absent from fit candidates map to neutral OOV
ID 0. Each trained ranker is also bound to the exact taxonomy IDs present in
its selected training pairs, so candidate-seen but unselected taxonomy IDs are
neutral as well.

"Neutral OOV 0" means "provide no category/subcategory evidence," not "predict
a score of zero." ID 0 is the embedding padding row and is fixed to an all-zero
vector. Its taxonomy contribution and feature interactions are therefore zero,
while the ranker still uses the candidate's text/KG representation, user and
news signals, click count, and history. This is safer than feeding an unseen
taxonomy through a random embedding row that received no training updates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mindrec.data.featurize import IdMaps
from mindrec.utils import load_json, pair_artifact_path, save_json


TAXONOMY_MAPPING_VERSION = 1
TAXONOMY_MAPPING_POLICY = "fit_candidate_seen_else_oov"
TAXONOMY_SUPPORT_VERSION = 1
TAXONOMY_SUPPORT_FILENAME = "taxonomy_support.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def taxonomy_layout_sha256(maps: IdMaps) -> str:
    """Fingerprint stable embedding-row numbering, ignoring OOV remapping."""
    payload = json.dumps(
        {
            "categories": sorted(maps.cat2idx),
            "subcategories": sorted(maps.subcat2idx),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def taxonomy_mapping_metadata(maps: IdMaps) -> dict[str, Any]:
    return {
        "version": TAXONOMY_MAPPING_VERSION,
        "policy": TAXONOMY_MAPPING_POLICY,
        "layout_sha256": taxonomy_layout_sha256(maps),
        "n_category_keys": len(maps.cat2idx),
        "n_active_categories": sum(int(value) > 0 for value in maps.cat2idx.values()),
        "n_oov_categories": sum(int(value) == 0 for value in maps.cat2idx.values()),
        "n_subcategory_keys": len(maps.subcat2idx),
        "n_active_subcategories": sum(
            int(value) > 0 for value in maps.subcat2idx.values()
        ),
        "n_oov_subcategories": sum(
            int(value) == 0 for value in maps.subcat2idx.values()
        ),
    }


def validate_taxonomy_mapping_artifact(processed_root: Path, maps: IdMaps) -> None:
    meta_path = processed_root / "preprocess_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing metadata: {meta_path}. Run preprocess again."
        )
    meta = load_json(meta_path)
    actual = meta.get("taxonomy_mapping")
    expected = taxonomy_mapping_metadata(maps)
    if not isinstance(actual, dict):
        raise ValueError(
            "Processed data predates fit-aware taxonomy OOV mapping. "
            "Run preprocess again before scoring."
        )
    mismatches = [
        name
        for name, expected_value in expected.items()
        if actual.get(name) != expected_value
    ]
    if mismatches:
        details = ", ".join(mismatches)
        raise ValueError(
            "Processed taxonomy metadata is stale or incompatible "
            f"({details}). Run preprocess again before scoring."
        )


def taxonomy_support_path(ranker_run_name: str) -> Path:
    return Path("runs") / ranker_run_name / "ranker" / TAXONOMY_SUPPORT_FILENAME


@dataclass(frozen=True)
class TaxonomySupport:
    supported_cat_ids: tuple[int, ...]
    supported_subcat_ids: tuple[int, ...]
    n_train_pairs: int
    source: str

    @classmethod
    def from_pairs(
        cls,
        pairs: pd.DataFrame,
        *,
        source: str,
    ) -> "TaxonomySupport":
        required = {"cat_idx", "subcat_idx"}
        missing = required.difference(pairs.columns)
        if missing:
            raise ValueError(
                "Training pairs are missing taxonomy columns: "
                + ", ".join(sorted(missing))
            )
        return cls(
            supported_cat_ids=tuple(
                sorted(
                    int(value)
                    for value in np.unique(pairs["cat_idx"].to_numpy())
                    if int(value) > 0
                )
            ),
            supported_subcat_ids=tuple(
                sorted(
                    int(value)
                    for value in np.unique(pairs["subcat_idx"].to_numpy())
                    if int(value) > 0
                )
            ),
            n_train_pairs=int(len(pairs)),
            source=str(source),
        )

    def validate_ids(self, maps: IdMaps) -> None:
        n_cats = len(maps.cat2idx) + 1
        n_subcats = len(maps.subcat2idx) + 1
        if any(value >= n_cats for value in self.supported_cat_ids):
            raise ValueError("Taxonomy support contains an out-of-range category ID.")
        if any(value >= n_subcats for value in self.supported_subcat_ids):
            raise ValueError(
                "Taxonomy support contains an out-of-range subcategory ID."
            )

    def save(
        self,
        path: Path,
        *,
        maps: IdMaps,
        checkpoint_path: Path,
        ranker_run_name: str,
        dataset_name: str,
    ) -> None:
        self.validate_ids(maps)
        save_json(
            path,
            {
                "version": TAXONOMY_SUPPORT_VERSION,
                "ranker_run_name": str(ranker_run_name),
                "dataset_name": str(dataset_name),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "taxonomy_layout_sha256": taxonomy_layout_sha256(maps),
                "n_train_pairs": self.n_train_pairs,
                "source": self.source,
                "supported_cat_ids": list(self.supported_cat_ids),
                "supported_subcat_ids": list(self.supported_subcat_ids),
            },
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        maps: IdMaps,
        checkpoint_path: Path,
        ranker_run_name: str,
        dataset_name: str,
    ) -> "TaxonomySupport":
        raw = load_json(path)
        expected = {
            "version": TAXONOMY_SUPPORT_VERSION,
            "ranker_run_name": str(ranker_run_name),
            "dataset_name": str(dataset_name),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "taxonomy_layout_sha256": taxonomy_layout_sha256(maps),
        }
        mismatches = [
            name for name, value in expected.items() if raw.get(name) != value
        ]
        if mismatches:
            raise ValueError(
                "Taxonomy-support artifact does not match the ranker checkpoint "
                f"or processed catalog ({', '.join(mismatches)})."
            )
        support = cls(
            supported_cat_ids=tuple(int(x) for x in raw["supported_cat_ids"]),
            supported_subcat_ids=tuple(
                int(x) for x in raw["supported_subcat_ids"]
            ),
            n_train_pairs=int(raw["n_train_pairs"]),
            source=str(raw["source"]),
        )
        support.validate_ids(maps)
        return support


def run_build_taxonomy_support(cfg: dict[str, Any]) -> None:
    dataset_name = str(cfg["data"]["dataset_name"])
    processed_root = Path(cfg["data"]["processed_root"]) / dataset_name
    pair_path = pair_artifact_path(processed_root, "train")
    if not pair_path.exists():
        raise FileNotFoundError(
            f"Missing exact ranker training pairs: {pair_path}. "
            "Build support before preprocessing removes this legacy artifact."
        )
    ranker_run_name = str(
        cfg.get("artifacts", {}).get("ranker_run_name", cfg["run_name"])
    )
    checkpoint_path = Path("runs") / ranker_run_name / "ranker" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing ranker checkpoint: {checkpoint_path}")

    maps = IdMaps.load(processed_root / "id_maps.json")
    pairs = pd.read_parquet(pair_path, columns=["cat_idx", "subcat_idx"])
    support = TaxonomySupport.from_pairs(
        pairs,
        source=f"legacy_exact_pairs:{pair_path.as_posix()}",
    )
    output_path = taxonomy_support_path(ranker_run_name)
    support.save(
        output_path,
        maps=maps,
        checkpoint_path=checkpoint_path,
        ranker_run_name=ranker_run_name,
        dataset_name=dataset_name,
    )
    print(
        f"Saved taxonomy support: {output_path} "
        f"({len(support.supported_cat_ids)} categories, "
        f"{len(support.supported_subcat_ids)} subcategories)"
    )
