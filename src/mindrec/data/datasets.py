from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from mindrec.data.featurize import IdMaps


@dataclass
class PairSample:
    user_idx: int
    news_idx: int
    cat_idx: int
    subcat_idx: int
    hist_news_idx: np.ndarray
    dense: np.ndarray
    label: int
    is_cold_user: int
    is_new_item: int


class PairDataset(Dataset):
    def __init__(
        self,
        pairs: pd.DataFrame,
        dense_cols: list[str],
    ) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.dense_cols = dense_cols

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.pairs.iloc[idx]
        dense = np.array([float(r[c]) for c in self.dense_cols], dtype=np.float32)
        hist_news_idx = np.array(r["hist_news_idx"], dtype=np.int64)
        return {
            "user_idx": torch.tensor(int(r["user_idx"]), dtype=torch.long),
            "news_idx": torch.tensor(int(r["news_idx"]), dtype=torch.long),
            "cat_idx": torch.tensor(int(r["cat_idx"]), dtype=torch.long),
            "subcat_idx": torch.tensor(int(r["subcat_idx"]), dtype=torch.long),
            "hour_idx": torch.tensor(int(r.get("hour_idx", 0)), dtype=torch.long),
            "weekday_idx": torch.tensor(int(r.get("weekday_idx", 0)), dtype=torch.long),
            "hist_news_idx": torch.tensor(hist_news_idx, dtype=torch.long),
            "dense": torch.tensor(dense, dtype=torch.float32),
            "label": torch.tensor(float(r["label"]), dtype=torch.float32),
            "is_cold_user": torch.tensor(int(r["is_cold_user"]), dtype=torch.long),
            "is_new_item": torch.tensor(int(r["is_new_item"]), dtype=torch.long),
            "news_id": r["news_id"],
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for k in [
        "user_idx",
        "news_idx",
        "cat_idx",
        "subcat_idx",
        "hour_idx",
        "weekday_idx",
        "dense",
        "label",
        "is_cold_user",
        "is_new_item",
    ]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    max_hist_len = max(max(int(b["hist_news_idx"].numel()) for b in batch), 1)
    hist = torch.zeros((len(batch), max_hist_len), dtype=torch.long)
    hist_mask = torch.zeros((len(batch), max_hist_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        h = b["hist_news_idx"]
        if h.numel() <= 0:
            continue
        hist[i, : h.numel()] = h
        hist_mask[i, : h.numel()] = True
    out["hist_news_idx"] = hist
    out["hist_mask"] = hist_mask
    out["news_id"] = [b["news_id"] for b in batch]
    return out
