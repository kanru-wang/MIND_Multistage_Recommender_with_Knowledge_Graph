from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"
BASE_DENSE_COLUMNS = ["history_len", "item_clicks_log1p"]
ITEM_TREND_DENSE_COLUMNS = ["item_age_log1p", "item_burst"]
ITEM_TREND_ARTIFACT = "item_trend_index.npz"
_EPOCH = datetime(1970, 1, 1)
_UNSEEN_SECONDS = np.iinfo(np.int64).max


def item_trend_config(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict(cfg.get("ranker", {}).get("item_trend", {}))
    dense_features = raw.get("dense_features", ITEM_TREND_DENSE_COLUMNS)
    if isinstance(dense_features, str):
        dense_features = [dense_features]
    dense_features = [str(name) for name in dense_features]
    unknown_dense_features = sorted(
        set(dense_features) - set(ITEM_TREND_DENSE_COLUMNS)
    )
    if unknown_dense_features:
        raise ValueError(
            "Unknown ranker.item_trend.dense_features: "
            + ", ".join(unknown_dense_features)
        )
    out = {
        "enabled": bool(raw.get("enabled", False)),
        "dense_features": dense_features,
        # Burst can be retained for a non-dense branch in a future experiment.
        # By default it is only loaded/computed when the dense model requests it.
        "compute_burst": bool(
            raw.get("compute_burst", "item_burst" in dense_features)
        ),
        "recent_hours": int(raw.get("recent_hours", 3)),
        "baseline_hours": int(raw.get("baseline_hours", 24)),
        "smoothing": float(raw.get("smoothing", 5.0)),
        "max_abs_burst": float(raw.get("max_abs_burst", 3.0)),
        "max_age_hours": float(raw.get("max_age_hours", 24.0 * 30.0)),
    }
    if out["recent_hours"] < 1:
        raise ValueError("ranker.item_trend.recent_hours must be at least 1.")
    if out["baseline_hours"] < 1:
        raise ValueError("ranker.item_trend.baseline_hours must be at least 1.")
    if out["smoothing"] <= 0.0:
        raise ValueError("ranker.item_trend.smoothing must be positive.")
    if out["max_abs_burst"] <= 0.0:
        raise ValueError("ranker.item_trend.max_abs_burst must be positive.")
    if out["max_age_hours"] <= 0.0:
        raise ValueError("ranker.item_trend.max_age_hours must be positive.")
    return out


def item_age_residual_config(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = dict(cfg.get("ranker", {}).get("item_age_residual", {}))
    trend_cfg = item_trend_config(cfg)
    out = {
        "enabled": bool(raw.get("enabled", False)),
        "max_age_hours": float(
            raw.get("max_age_hours", trend_cfg["max_age_hours"])
        ),
        "max_abs_logit_adjustment": float(
            raw.get("max_abs_logit_adjustment", 0.25)
        ),
    }
    if out["enabled"] and not trend_cfg["enabled"]:
        raise ValueError(
            "ranker.item_age_residual requires ranker.item_trend.enabled=true."
        )
    if out["enabled"] and trend_cfg["dense_features"]:
        raise ValueError(
            "ranker.item_age_residual must be isolated from the dense MLP; set "
            "ranker.item_trend.dense_features=[]."
        )
    if out["enabled"] and not np.isclose(
        out["max_age_hours"], trend_cfg["max_age_hours"]
    ):
        raise ValueError(
            "ranker.item_age_residual.max_age_hours must match "
            "ranker.item_trend.max_age_hours."
        )
    if out["max_age_hours"] <= 0.0:
        raise ValueError("ranker.item_age_residual.max_age_hours must be positive.")
    if out["max_abs_logit_adjustment"] <= 0.0:
        raise ValueError(
            "ranker.item_age_residual.max_abs_logit_adjustment must be positive."
        )
    return out


def ranker_dense_columns(cfg: dict[str, Any]) -> list[str]:
    columns = list(BASE_DENSE_COLUMNS)
    trend_cfg = item_trend_config(cfg)
    if trend_cfg["enabled"]:
        columns.extend(trend_cfg["dense_features"])
    return columns


def item_trend_artifact_path(proc_root: Path) -> Path:
    return proc_root / ITEM_TREND_ARTIFACT


def candidate_dense_matrix(
    cfg: dict[str, Any],
    history_len: float,
    item_clicks_log1p: np.ndarray,
    item_age_log1p: np.ndarray | None = None,
    item_burst: np.ndarray | None = None,
) -> np.ndarray:
    clicks = np.asarray(item_clicks_log1p, dtype=np.float32)
    columns = [np.full(clicks.shape, float(history_len), dtype=np.float32), clicks]
    trend_cfg = item_trend_config(cfg)
    if trend_cfg["enabled"]:
        for name in trend_cfg["dense_features"]:
            value = item_age_log1p if name == "item_age_log1p" else item_burst
            if value is None:
                raise ValueError(
                    f"Enabled dense item-trend feature is missing: {name}."
                )
            feature = np.asarray(value, dtype=np.float32)
            if feature.shape != clicks.shape:
                raise ValueError(
                    f"Dense item-trend feature {name} must match the candidate shape."
                )
            columns.append(feature)
    return np.stack(columns, axis=1)


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


def _iter_raw_time_and_candidates(
    behavior_paths: Iterable[Path],
) -> Iterable[tuple[int, list[str]]]:
    for behavior_path in behavior_paths:
        with Path(behavior_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                seconds = _parse_time_seconds(parts[2])
                if seconds is None:
                    continue
                yield seconds, parts[4].split()


def _raw_time_bounds(behavior_paths: Iterable[Path]) -> tuple[int, int]:
    min_hour: int | None = None
    max_hour: int | None = None
    for behavior_path in behavior_paths:
        with Path(behavior_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split("\t", 3)
                if len(parts) < 3:
                    continue
                seconds = _parse_time_seconds(parts[2])
                if seconds is None:
                    continue
                hour = seconds // 3600
                min_hour = hour if min_hour is None else min(min_hour, hour)
                max_hour = hour if max_hour is None else max(max_hour, hour)
    if min_hour is None or max_hour is None:
        raise ValueError("Could not find any valid MIND behavior timestamps.")
    return min_hour, max_hour


@dataclass
class ItemTrendIndex:
    """Hourly, past-only candidate-exposure statistics for article trend features."""

    item_prefix_counts: np.ndarray
    total_prefix_counts: np.ndarray
    first_seen_seconds: np.ndarray
    origin_hour: int
    recent_hours: int
    baseline_hours: int
    smoothing: float
    max_abs_burst: float
    max_age_hours: float
    use_burst: bool = True

    @classmethod
    def build(
        cls,
        behavior_paths: Iterable[Path],
        news2idx: dict[str, int],
        *,
        recent_hours: int = 3,
        baseline_hours: int = 24,
        smoothing: float = 5.0,
        max_abs_burst: float = 3.0,
        max_age_hours: float = 24.0 * 30.0,
        use_burst: bool = True,
        flush_candidates: int = 500_000,
    ) -> "ItemTrendIndex":
        paths = [Path(path) for path in behavior_paths]
        print(
            "Build item-trend index: scan timestamp bounds from "
            f"{len(paths)} behavior file(s)."
        )
        min_hour, max_hour = _raw_time_bounds(paths)

        n_news = max(news2idx.values(), default=0) + 1
        n_hours = max_hour - min_hour + 1
        # Column zero is the empty prefix. Exposure in hour h is accumulated in
        # column h+1 before the in-place cumulative sum.
        item_prefix = np.zeros((n_news, n_hours + 1), dtype=np.int32)
        total_by_hour = np.zeros(n_hours, dtype=np.int64)
        first_seen = np.full(n_news, _UNSEEN_SECONDS, dtype=np.int64)

        buffered_news: list[int] = []
        buffered_hours: list[int] = []
        buffered_seconds: list[int] = []
        lookup = news2idx.get
        print(
            "Build item-trend index: aggregate candidate exposures into "
            f"{n_hours} hourly bins."
        )

        def flush() -> None:
            if not buffered_news:
                return
            news_idx = np.asarray(buffered_news, dtype=np.int64)
            hour_idx = np.asarray(buffered_hours, dtype=np.int64)
            seconds_array = np.asarray(buffered_seconds, dtype=np.int64)
            np.add.at(item_prefix, (news_idx, hour_idx + 1), 1)
            np.minimum.at(first_seen, news_idx, seconds_array)
            buffered_news.clear()
            buffered_hours.clear()
            buffered_seconds.clear()

        for seconds, tokens in _iter_raw_time_and_candidates(paths):
            hour_idx = seconds // 3600 - min_hour
            total_by_hour[hour_idx] += len(tokens)
            indices = [
                int(idx)
                for token in tokens
                if (idx := lookup(_news_id_from_token(token))) is not None
                and int(idx) != 0
            ]
            buffered_news.extend(indices)
            buffered_hours.extend([hour_idx] * len(indices))
            buffered_seconds.extend([seconds] * len(indices))
            if len(buffered_news) >= flush_candidates:
                flush()
        flush()

        np.cumsum(item_prefix, axis=1, dtype=np.int32, out=item_prefix)
        total_prefix = np.zeros(n_hours + 1, dtype=np.int64)
        total_prefix[1:] = np.cumsum(total_by_hour, dtype=np.int64)
        print(
            "Build item-trend index: aggregated "
            f"{int(total_prefix[-1]):,} candidate appearances."
        )
        return cls(
            item_prefix_counts=item_prefix,
            total_prefix_counts=total_prefix,
            first_seen_seconds=first_seen,
            origin_hour=int(min_hour),
            recent_hours=int(recent_hours),
            baseline_hours=int(baseline_hours),
            smoothing=float(smoothing),
            max_abs_burst=float(max_abs_burst),
            max_age_hours=float(max_age_hours),
            use_burst=bool(use_burst),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            item_prefix_counts=self.item_prefix_counts,
            total_prefix_counts=self.total_prefix_counts,
            first_seen_seconds=self.first_seen_seconds,
            origin_hour=np.asarray(self.origin_hour, dtype=np.int64),
            recent_hours=np.asarray(self.recent_hours, dtype=np.int64),
            baseline_hours=np.asarray(self.baseline_hours, dtype=np.int64),
            smoothing=np.asarray(self.smoothing, dtype=np.float64),
            max_abs_burst=np.asarray(self.max_abs_burst, dtype=np.float64),
            max_age_hours=np.asarray(self.max_age_hours, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path, *, use_burst: bool = True) -> "ItemTrendIndex":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                # NPZ members are loaded lazily. Avoiding these two members saves
                # the 164 MiB expanded prefix matrix in age-only experiments.
                item_prefix_counts=(
                    data["item_prefix_counts"]
                    if use_burst
                    else np.empty((0, 0), dtype=np.int32)
                ),
                total_prefix_counts=(
                    data["total_prefix_counts"]
                    if use_burst
                    else np.empty(0, dtype=np.int64)
                ),
                first_seen_seconds=data["first_seen_seconds"],
                origin_hour=int(data["origin_hour"]),
                recent_hours=int(data["recent_hours"]),
                baseline_hours=int(data["baseline_hours"]),
                smoothing=float(data["smoothing"]),
                max_abs_burst=float(data["max_abs_burst"]),
                max_age_hours=float(data["max_age_hours"]),
                use_burst=bool(use_burst),
            )

    def features(
        self,
        news_indices: np.ndarray | list[int],
        time_value: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(news_indices, dtype=np.int64)
        age = np.zeros(indices.shape, dtype=np.float32)
        burst = np.zeros(indices.shape, dtype=np.float32)
        seconds = _parse_time_seconds(time_value)
        if seconds is None or indices.size == 0:
            return age, burst

        valid = (indices > 0) & (indices < len(self.first_seen_seconds))
        if not np.any(valid):
            return age, burst
        valid_indices = indices[valid]

        seen = self.first_seen_seconds[valid_indices]
        valid_seen = seen != _UNSEEN_SECONDS
        age_hours = np.zeros(valid_indices.shape, dtype=np.float64)
        age_hours[valid_seen] = np.maximum(
            0.0,
            (float(seconds) - seen[valid_seen].astype(np.float64)) / 3600.0,
        )
        age_values = np.log1p(np.minimum(age_hours, self.max_age_hours))
        age[valid] = age_values.astype(np.float32)

        if not self.use_burst:
            return age, burst

        current_hour = seconds // 3600 - self.origin_hour
        n_hours = self.item_prefix_counts.shape[1] - 1
        recent_end = int(np.clip(current_hour, 0, n_hours))
        recent_start = max(0, recent_end - self.recent_hours)
        baseline_end = recent_start
        baseline_start = max(0, baseline_end - self.baseline_hours)

        recent_total = int(
            self.total_prefix_counts[recent_end]
            - self.total_prefix_counts[recent_start]
        )
        baseline_total = int(
            self.total_prefix_counts[baseline_end]
            - self.total_prefix_counts[baseline_start]
        )
        if recent_total <= 0 or baseline_total <= 0:
            return age, burst

        recent_counts = (
            self.item_prefix_counts[valid_indices, recent_end]
            - self.item_prefix_counts[valid_indices, recent_start]
        ).astype(np.float64)
        baseline_counts = (
            self.item_prefix_counts[valid_indices, baseline_end]
            - self.item_prefix_counts[valid_indices, baseline_start]
        ).astype(np.float64)
        expected_recent = baseline_counts * (recent_total / baseline_total)
        burst_values = np.log2(
            (recent_counts + self.smoothing)
            / (expected_recent + self.smoothing)
        )
        burst_values = np.clip(
            burst_values,
            -self.max_abs_burst,
            self.max_abs_burst,
        )
        burst[valid] = burst_values.astype(np.float32)
        return age, burst


def build_item_trend_index(
    cfg: dict[str, Any],
    behavior_paths: Iterable[Path],
    news2idx: dict[str, int],
) -> ItemTrendIndex | None:
    trend_cfg = item_trend_config(cfg)
    if not trend_cfg["enabled"]:
        return None
    return ItemTrendIndex.build(
        behavior_paths=behavior_paths,
        news2idx=news2idx,
        recent_hours=trend_cfg["recent_hours"],
        baseline_hours=trend_cfg["baseline_hours"],
        smoothing=trend_cfg["smoothing"],
        max_abs_burst=trend_cfg["max_abs_burst"],
        max_age_hours=trend_cfg["max_age_hours"],
        use_burst=trend_cfg["compute_burst"],
    )


def load_item_trend_index(
    cfg: dict[str, Any],
    proc_root: Path,
) -> ItemTrendIndex | None:
    trend_cfg = item_trend_config(cfg)
    if not trend_cfg["enabled"]:
        return None
    path = item_trend_artifact_path(proc_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing item-trend artifact: {path}. Run preprocessing first."
        )
    index = ItemTrendIndex.load(
        path,
        use_burst=trend_cfg["compute_burst"],
    )
    expected = {
        "recent_hours": trend_cfg["recent_hours"],
        "baseline_hours": trend_cfg["baseline_hours"],
        "smoothing": trend_cfg["smoothing"],
        "max_abs_burst": trend_cfg["max_abs_burst"],
        "max_age_hours": trend_cfg["max_age_hours"],
    }
    actual = {name: getattr(index, name) for name in expected}
    mismatches = [
        name
        for name, expected_value in expected.items()
        if not np.isclose(float(actual[name]), float(expected_value))
    ]
    if mismatches:
        fields = ", ".join(mismatches)
        raise ValueError(
            f"Item-trend artifact does not match the config ({fields}). "
            "Run preprocessing again."
        )
    return index
