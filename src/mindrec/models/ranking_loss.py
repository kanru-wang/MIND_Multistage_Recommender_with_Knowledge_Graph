from __future__ import annotations

import torch
import torch.nn.functional as F


def impression_pairwise_loss(
    logits: torch.Tensor, labels: torch.Tensor, impression_idx: torch.Tensor,
) -> torch.Tensor:
    """Mean logistic positive-negative loss per impression, then per batch.

    Candidates are the existing sampled training rows, including negative
    multiplicities from multi-click sampling. Never compare across impressions.
    The batch sampler must retain every sampled row of each included impression.
    """
    _, inverse = torch.unique(impression_idx, return_inverse=True)
    positive = labels == 1
    negative = labels == 0
    pos_groups = inverse[positive]
    neg_groups = inverse[negative]
    same_impression = pos_groups[:, None] == neg_groups[None, :]
    pos_rows, neg_rows = same_impression.nonzero(as_tuple=True)
    # Gather valid pairs before subtracting scores. Computing cross-impression
    # differences first wastes softplus work and can produce inf * 0 -> NaN.
    losses = F.softplus(logits[negative][neg_rows] - logits[positive][pos_rows])
    pair_groups = pos_groups[pos_rows]
    # Allocate at most one slot per row, avoiding a GPU -> CPU scalar sync.
    sums = logits.new_zeros(logits.numel()).scatter_add(0, pair_groups, losses)
    counts = logits.new_zeros(logits.numel()).scatter_add(0, pair_groups, torch.ones_like(losses))
    valid = counts > 0
    return (sums / counts.clamp_min(1)).sum() / valid.sum().clamp_min(1)
