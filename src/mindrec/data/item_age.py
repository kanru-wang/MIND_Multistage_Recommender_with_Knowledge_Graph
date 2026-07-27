from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from mindrec.data.featurize import IdMaps


MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
ITEM_AGE_ARTIFACT = "item_age_index.npz"
_EPOCH = datetime(1970, 1, 1)
_UNSEEN_SECONDS = np.iinfo(np.int64).max


def item_age_artifact_path(processed_root: Path) -> Path:
    return processed_root / ITEM_AGE_ARTIFACT


def _parse_time_seconds(value: object) -> int | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(str(value), MIND_TIME_FORMAT)
    except (TypeError, ValueError):
        return None
    return int((parsed - _EPOCH).total_seconds())


def _news_id_from_token(token: str) -> str:
    if len(token) >= 3 and token[-2] == "-" and token[-1] in {"0", "1"}:
        return token[:-2]
    return token


@dataclass
class ItemAgeIndex:
    """First candidate-appearance time used by the fixed recency tiebreaker."""

    first_seen_seconds: np.ndarray
    max_age_hours: float

    @classmethod
    def build(
        cls,
        behavior_paths: Iterable[Path],
        news2idx: dict[str, int],
        *,
        max_age_hours: float = 24.0 * 30.0,
    ) -> "ItemAgeIndex":
        if max_age_hours <= 0.0:
            raise ValueError("posthoc_recency.max_age_hours must be positive.")
        first_seen = np.full(
            max(news2idx.values(), default=0) + 1,
            _UNSEEN_SECONDS,
            dtype=np.int64,
        )
        lookup = news2idx.get
        n_candidates = 0
        for behavior_path in behavior_paths:
            with Path(behavior_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 5:
                        continue
                    seconds = _parse_time_seconds(parts[2])
                    if seconds is None:
                        continue
                    indices = []
                    for token in parts[4].split():
                        index = lookup(_news_id_from_token(token))
                        if index is not None and int(index) > 0:
                            indices.append(int(index))
                    if indices:
                        np.minimum.at(
                            first_seen,
                            np.asarray(indices, dtype=np.int64),
                            seconds,
                        )
                        n_candidates += len(indices)
        print(
            "Built item-age index from "
            f"{n_candidates:,} candidate appearances."
        )
        return cls(
            first_seen_seconds=first_seen,
            max_age_hours=float(max_age_hours),
        )

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_max_age_hours: float | None = None,
    ) -> "ItemAgeIndex":
        with np.load(path, allow_pickle=False) as data:
            index = cls(
                first_seen_seconds=data["first_seen_seconds"],
                max_age_hours=float(data["max_age_hours"]),
            )
        if (
            expected_max_age_hours is not None
            and not np.isclose(index.max_age_hours, expected_max_age_hours)
        ):
            raise ValueError(
                "Item-age artifact max_age_hours does not match the config. "
                "Run build_item_age again."
            )
        return index

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            first_seen_seconds=self.first_seen_seconds,
            max_age_hours=np.asarray(self.max_age_hours, dtype=np.float64),
        )

    def ages(
        self,
        news_indices: np.ndarray | list[int],
        time_value: object,
    ) -> np.ndarray:
        indices = np.asarray(news_indices, dtype=np.int64)
        age = np.zeros(indices.shape, dtype=np.float32)
        seconds = _parse_time_seconds(time_value)
        if seconds is None or indices.size == 0:
            return age
        valid = (indices > 0) & (indices < len(self.first_seen_seconds))
        if not np.any(valid):
            return age
        seen = self.first_seen_seconds[indices[valid]]
        was_seen = seen != _UNSEEN_SECONDS
        age_hours = np.zeros(seen.shape, dtype=np.float64)
        age_hours[was_seen] = np.maximum(
            0.0,
            (float(seconds) - seen[was_seen].astype(np.float64)) / 3600.0,
        )
        age[valid] = np.log1p(
            np.minimum(age_hours, self.max_age_hours)
        ).astype(np.float32)
        return age


def run_build_item_age(cfg: dict[str, Any]) -> None:
    recency_cfg = dict(cfg.get("posthoc_recency", {}))
    if not bool(recency_cfg.get("enabled", False)):
        raise ValueError("posthoc_recency.enabled must be true.")
    dataset_name = str(
        recency_cfg.get(
            "age_dataset_name",
            cfg.get("data", {}).get("dataset_name", ""),
        )
    )
    if not dataset_name:
        raise ValueError("posthoc_recency.age_dataset_name is required.")
    processed_root = Path(cfg["data"]["processed_root"]) / dataset_name
    maps = IdMaps.load(processed_root / "id_maps.json")
    raw_root = Path(cfg["data"]["raw_root"])
    behavior_paths = [
        raw_root / cfg["data"][name] / "behaviors.tsv"
        for name in ("train_dir", "dev_dir", "test_dir")
        if cfg["data"].get(name)
    ]
    if not behavior_paths:
        raise ValueError(
            "At least one of data.train_dir, dev_dir, or test_dir is required."
        )
    missing = [path for path in behavior_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing behavior file(s): "
            + ", ".join(path.as_posix() for path in missing)
        )
    index = ItemAgeIndex.build(
        behavior_paths,
        maps.news2idx,
        max_age_hours=float(recency_cfg.get("max_age_hours", 720.0)),
    )
    output_path = item_age_artifact_path(processed_root)
    index.save(output_path)
    print(f"Saved item-age index: {output_path}")
