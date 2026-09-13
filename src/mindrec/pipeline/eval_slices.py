from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mindrec.utils import test_split_name, validation_split_name


def sanitize_slice_value(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return "unknown"
    text = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_") or "unknown"


def history_len_bucket(history_len: float) -> str:
    resolved = int(round(float(history_len)))
    if resolved <= 0:
        return "0"
    if resolved <= 4:
        return "1_4"
    if resolved <= 20:
        return "5_20"
    return "21_plus"


def popularity_bucket(click_count: int) -> str:
    if click_count <= 0:
        return "0"
    if click_count <= 4:
        return "1_4"
    if click_count <= 19:
        return "5_19"
    return "20_plus"


def resolve_eval_splits(cfg: dict[str, Any]) -> list[str]:
    raw_splits = cfg.get("eval", {}).get("report_splits", ["test"])
    resolved: list[str] = []
    for split in raw_splits:
        split_name = str(split)
        if split_name == "val":
            split_name = validation_split_name(cfg)
        elif split_name == "test":
            split_name = test_split_name(cfg)
        if split_name not in resolved:
            resolved.append(split_name)
    return resolved


def attach_time_periods(
    frame: pd.DataFrame,
    n_periods: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = frame.copy()
    out["time_period"] = "unknown"
    meta: list[dict[str, Any]] = []

    if n_periods <= 0 or "time" not in out.columns:
        return out, meta

    parsed = pd.to_datetime(
        out["time"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    valid_idx = np.flatnonzero(parsed.notna().to_numpy())
    if len(valid_idx) == 0:
        return out, meta

    order = np.argsort(
        parsed.iloc[valid_idx].to_numpy(dtype="datetime64[ns]"),
        kind="stable",
    )
    ordered_valid_idx = valid_idx[order]
    chunks = np.array_split(
        ordered_valid_idx,
        min(n_periods, len(ordered_valid_idx)),
    )

    for i, chunk in enumerate(chunks, start=1):
        if len(chunk) == 0:
            continue
        label = f"period_{i}_of_{len(chunks)}"
        out.iloc[chunk, out.columns.get_loc("time_period")] = label
        times = parsed.iloc[chunk]
        meta.append(
            {
                "name": label,
                "n_impressions": int(len(chunk)),
                "time_min": str(times.min()),
                "time_max": str(times.max()),
            }
        )
    return out, meta
