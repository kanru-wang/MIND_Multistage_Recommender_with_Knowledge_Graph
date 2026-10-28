from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class ImpressionBatchSampler(Sampler[list[int]]):
    """Shuffle complete sampled impressions, packing up to max_rows per batch."""

    def __init__(self, impression_idx, max_rows: int, seed: int):
        if max_rows < 1:
            raise ValueError("Grouped batch size must be positive.")
        codes, _ = pd.factorize(np.asarray(impression_idx), sort=False)
        if len(codes) == 0 or np.any(codes < 0):
            raise ValueError("Training requires nonempty, nonmissing impression IDs.")
        self.rows = np.argsort(codes, kind="stable")
        self.sizes = np.bincount(codes)
        self.offsets = np.concatenate(([0], np.cumsum(self.sizes)))
        if int(self.sizes.max()) > max_rows:
            raise ValueError("An impression exceeds ranker.batch_size; increase it to avoid splitting impressions.")
        self.max_rows = max_rows
        self.seed = seed
        self.epoch = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self.order = np.random.default_rng(self.seed + epoch).permutation(len(self.sizes))
        self.batches = []
        start = 0
        used = 0
        for end, size in enumerate(self.sizes[self.order]):
            if used + size > self.max_rows:
                self.batches.append((start, end))
                start = end
                used = 0
            used += int(size)
        self.batches.append((start, len(self.order)))

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        for start, end in self.batches:
            rows = []
            for group in self.order[start:end]:
                rows.extend(self.rows[self.offsets[group] : self.offsets[group + 1]].tolist())
            yield rows
