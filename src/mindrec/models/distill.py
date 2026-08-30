from __future__ import annotations

import torch
import torch.nn.functional as F


REPRESENTATION_DISTILLATION_TARGETS = frozenset({"full", "item_only"})


def _validate_representation_target(target: str) -> str:
    resolved = str(target)
    if resolved not in REPRESENTATION_DISTILLATION_TARGETS:
        choices = ", ".join(sorted(REPRESENTATION_DISTILLATION_TARGETS))
        raise ValueError(
            "ranker.distill.representation_target must be one of "
            f"{choices}; got {resolved!r}."
        )
    return resolved


def representation_distillation_dims(
    student_emb_dim: int,
    teacher_dim: int,
    target: str,
) -> tuple[int, int]:
    """Return projection input/output widths for a distillation target."""
    resolved = _validate_representation_target(target)
    if resolved == "item_only":
        return int(student_emb_dim), int(teacher_dim)
    return 3 * int(student_emb_dim), 2 * int(teacher_dim)


def select_representation_distillation_inputs(
    student_repr: torch.Tensor,
    teacher_user: torch.Tensor,
    teacher_item: torch.Tensor,
    *,
    student_emb_dim: int,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the student representation and matching teacher target."""
    resolved = _validate_representation_target(target)
    if resolved == "item_only":
        start = int(student_emb_dim)
        return student_repr[:, start : 2 * start], teacher_item
    return student_repr, torch.cat([teacher_user, teacher_item], dim=1)


def distillation_history_masks(
    has_history: torch.Tensor,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row masks for logit and representation distillation."""
    resolved = _validate_representation_target(target)
    history_mask = has_history.float()
    representation_mask = (
        torch.ones_like(history_mask)
        if resolved == "item_only"
        else history_mask
    )
    return history_mask, representation_mask


def logit_distill_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    t = float(temperature)
    ps = F.log_softmax(student_logits / t, dim=0)
    pt = F.softmax(teacher_logits / t, dim=0)
    return F.kl_div(ps, pt, reduction="batchmean") * (t * t)


def pairwise_logit_distill_bce(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    # For pairwise samples, distill teacher probability sigmoid(logit/T)
    t = float(temperature)
    target = torch.sigmoid(teacher_logits / t)
    return F.binary_cross_entropy_with_logits(student_logits, target)


def repr_distill_mse(
    student_repr: torch.Tensor, teacher_repr: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(student_repr, teacher_repr)
