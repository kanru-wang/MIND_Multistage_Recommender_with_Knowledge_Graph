from __future__ import annotations

import gc
import hashlib
import json
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir, load_config
from mindrec.pipeline.evaluate import _load_model
from mindrec.pipeline.ranker_scoring import (
    precompute_item_semantics,
    score_prepared_groups,
)
from mindrec.utils import log_device, resolve_device, save_json


@dataclass
class _MemberRuntime:
    name: str
    model: torch.nn.Module
    encoded_items: torch.Tensor
    item_semantics: torch.Tensor


def _validate_ranks(ranks: np.ndarray, source: str) -> None:
    if ranks.ndim != 1 or len(ranks) == 0:
        raise ValueError(f"{source} must contain a non-empty one-dimensional rank list.")
    expected = np.arange(1, len(ranks) + 1, dtype=np.int64)
    if not np.array_equal(np.sort(ranks), expected):
        raise ValueError(f"{source} ranks must be a permutation of 1..N.")


def _parse_prediction_line(raw: bytes, source: str) -> tuple[str, np.ndarray]:
    try:
        impression_id, encoded_ranks = raw.decode("utf-8").rstrip("\r\n").split(" ", 1)
        ranks = np.asarray(json.loads(encoded_ranks), dtype=np.int64)
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed prediction line in {source}.") from exc
    if not impression_id:
        raise ValueError(f"Missing impression ID in {source}.")
    _validate_ranks(ranks, source)
    return impression_id, ranks


def _weighted_borda_ranks(
    primary_ranks: np.ndarray,
    secondary_ranks: np.ndarray,
    primary_weight: float,
) -> np.ndarray:
    """Fuse two rank permutations; prefer the stronger primary model on ties."""
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("ensemble.selected_primary_weight must be between 0 and 1.")
    if len(primary_ranks) != len(secondary_ranks):
        raise ValueError("Ensemble members have different candidate counts.")
    _validate_ranks(primary_ranks, "primary prediction")
    _validate_ranks(secondary_ranks, "secondary prediction")

    costs = (
        primary_weight * primary_ranks.astype(np.float64)
        + (1.0 - primary_weight) * secondary_ranks.astype(np.float64)
    )
    candidate_position = np.arange(len(costs), dtype=np.int64)
    order = np.lexsort((candidate_position, primary_ranks, costs))
    fused = np.empty(len(costs), dtype=np.int32)
    fused[order] = np.arange(1, len(costs) + 1, dtype=np.int32)
    return fused


def _prediction_stream(stack: ExitStack, path: Path) -> BinaryIO:
    if not path.exists():
        raise FileNotFoundError(f"Missing ensemble member submission: {path}")
    archive = stack.enter_context(zipfile.ZipFile(path))
    if archive.namelist() != ["prediction.txt"]:
        raise ValueError(f"{path} must contain only a top-level prediction.txt.")
    return stack.enter_context(archive.open("prediction.txt"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_ensemble_submission(cfg: dict[str, Any]) -> None:
    ensemble_cfg = dict(cfg.get("ensemble", {}))
    members = list(ensemble_cfg.get("members", []))
    if len(members) != 2:
        raise ValueError("Rank ensembling currently requires exactly two members.")
    if not bool(ensemble_cfg.get("frozen", False)):
        raise ValueError("Set ensemble.frozen=true after selecting the validation weight.")

    weight = float(ensemble_cfg["selected_primary_weight"])
    primary_path = Path(members[0]["prediction_zip"])
    secondary_path = Path(members[1]["prediction_zip"])
    out_root = ensure_dir(Path("runs") / cfg["run_name"] / "submission")
    zip_path = out_root / "prediction.zip"
    meta_path = out_root / "submission_meta.json"
    expected_impressions = ensemble_cfg.get("expected_impressions")

    n_impressions = 0
    exact_members = 0
    same_member_top1 = 0
    changed_from_primary = 0
    prediction_digest = hashlib.sha256()

    with ExitStack() as stack:
        primary_stream = _prediction_stream(stack, primary_path)
        secondary_stream = _prediction_stream(stack, secondary_path)
        with zipfile.ZipFile(
            zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as output_archive:
            with output_archive.open("prediction.txt", mode="w") as output:
                pairs = zip_longest(primary_stream, secondary_stream)
                for primary_raw, secondary_raw in tqdm(
                    pairs,
                    total=int(expected_impressions) if expected_impressions else None,
                    desc="Ensemble hidden-test ranks",
                ):
                    if primary_raw is None or secondary_raw is None:
                        raise ValueError("Ensemble member submissions have different line counts.")
                    primary_id, primary_ranks = _parse_prediction_line(
                        primary_raw, str(primary_path)
                    )
                    secondary_id, secondary_ranks = _parse_prediction_line(
                        secondary_raw, str(secondary_path)
                    )
                    if primary_id != secondary_id:
                        raise ValueError(
                            "Ensemble member impression IDs are misaligned: "
                            f"{primary_id!r} != {secondary_id!r}."
                        )
                    fused = _weighted_borda_ranks(
                        primary_ranks,
                        secondary_ranks,
                        weight,
                    )
                    encoded = (
                        f"{primary_id} "
                        f"{json.dumps(fused.tolist(), separators=(',', ':'))}\n"
                    ).encode("utf-8")
                    output.write(encoded)
                    prediction_digest.update(encoded)

                    n_impressions += 1
                    exact_members += int(np.array_equal(primary_ranks, secondary_ranks))
                    same_member_top1 += int(
                        int(np.argmin(primary_ranks)) == int(np.argmin(secondary_ranks))
                    )
                    changed_from_primary += int(not np.array_equal(fused, primary_ranks))

    if expected_impressions is not None and n_impressions != int(expected_impressions):
        raise ValueError(
            f"Expected {int(expected_impressions)} impressions, found {n_impressions}."
        )

    save_json(
        meta_path,
        {
            "method": "weighted_borda_rank_fusion",
            "primary_tiebreaker": members[0]["name"],
            "selected_primary_weight": weight,
            "secondary_weight": 1.0 - weight,
            "selection_artifact": ensemble_cfg.get("selection_artifact"),
            "n_impressions": n_impressions,
            "members": [
                {
                    "name": member["name"],
                    "prediction_zip": str(path),
                    "sha256": _sha256(path),
                }
                for member, path in zip(members, [primary_path, secondary_path])
            ],
            "agreement": {
                "exact_rank_fraction": exact_members / n_impressions,
                "same_top1_fraction": same_member_top1 / n_impressions,
                "ensemble_changed_from_primary_fraction": (
                    changed_from_primary / n_impressions
                ),
            },
            "prediction_zip": str(zip_path),
            "prediction_zip_sha256": _sha256(zip_path),
            "prediction_content_sha256": prediction_digest.hexdigest(),
            "format": "MIND leaderboard prediction.txt: impression_id compact_json_ranks",
        },
    )


def _prepare_group(row: Any) -> tuple[dict[str, Any], np.ndarray]:
    labels = np.asarray(row.cand_label, dtype=np.int32)
    clicks = np.asarray(row.cand_item_clicks_log1p, dtype=np.float32)
    dense = np.stack(
        [np.full_like(clicks, float(row.history_len)), clicks],
        axis=1,
    )
    return (
        {
            "user_idx": int(row.user_idx),
            "hist_news_idx": [int(x) for x in row.hist_news_idx],
            "cand_news_idx": np.asarray(row.cand_news_idx, dtype=np.int64),
            "cand_cat_idx": np.asarray(row.cand_cat_idx, dtype=np.int64),
            "cand_subcat_idx": np.asarray(row.cand_subcat_idx, dtype=np.int64),
            "cand_is_new": np.asarray(row.cand_is_new_item, dtype=np.int64),
            "dense": dense,
        },
        labels,
    )


def _empty_metric_sums(n_weights: int) -> dict[str, np.ndarray]:
    return {
        name: np.zeros(n_weights, dtype=np.float64)
        for name in ("auc", "mrr", "ndcg@5", "ndcg@10")
    }


def _accumulate_rank_metrics(
    sums: dict[str, np.ndarray],
    weights: np.ndarray,
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    labels: np.ndarray,
) -> None:
    if labels.sum() <= 0:
        return
    primary_ranks = _scores_to_float_ranks(primary_scores)
    secondary_ranks = _scores_to_float_ranks(secondary_scores)
    costs = (
        weights[:, None] * primary_ranks[None, :]
        + (1.0 - weights[:, None]) * secondary_ranks[None, :]
    )
    candidate_position = np.arange(len(labels), dtype=np.int64)
    order = np.stack(
        [
            np.lexsort((candidate_position, primary_ranks, weight_costs))
            for weight_costs in costs
        ]
    )
    fused_ranks = np.empty_like(costs)
    fused_ranks[
        np.arange(len(weights))[:, None],
        order,
    ] = np.arange(1, len(labels) + 1, dtype=np.float64)[None, :]
    positives = labels == 1
    negatives = labels == 0
    if positives.any() and negatives.any():
        pos_ranks = fused_ranks[:, positives, None]
        neg_ranks = fused_ranks[:, None, negatives]
        sums["auc"] += (pos_ranks < neg_ranks).mean(axis=(1, 2))

    ordered_labels = np.take_along_axis(
        np.broadcast_to(labels, costs.shape), order, axis=1
    )
    positions = np.arange(1, len(labels) + 1, dtype=np.float64)
    sums["mrr"] += (ordered_labels / positions[None, :]).sum(axis=1) / labels.sum()
    for k in (5, 10):
        width = min(k, len(labels))
        discounts = 1.0 / np.log2(np.arange(2, width + 2, dtype=np.float64))
        dcg = (ordered_labels[:, :width] * discounts[None, :]).sum(axis=1)
        ideal_width = min(width, int(labels.sum()))
        ideal = float(discounts[:ideal_width].sum())
        sums[f"ndcg@{k}"] += dcg / ideal if ideal > 0 else 0.0


def _scores_to_float_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return ranks


def _score_split(
    proc_root: Path,
    split_name: str,
    runtimes: list[_MemberRuntime],
    weights: np.ndarray,
    batch_size: int,
    candidate_buffer_size: int,
    device: torch.device,
) -> dict[str, Any]:
    path = proc_root / f"{split_name}_impressions.parquet"
    impressions = pd.read_parquet(path)
    sums = _empty_metric_sums(len(weights))
    n_scored = 0
    n_skipped = 0
    pending_groups: list[dict[str, Any]] = []
    pending_labels: list[np.ndarray] = []
    pending_candidates = 0

    progress = tqdm(total=len(impressions), desc=f"Rank-ensemble evaluation ({split_name})")

    def flush() -> None:
        nonlocal n_scored, n_skipped, pending_candidates
        if not pending_groups:
            return
        member_scores = [
            score_prepared_groups(
                model=runtime.model,
                groups=pending_groups,
                encoded_items=runtime.encoded_items,
                item_semantics=runtime.item_semantics,
                batch_size=batch_size,
                device=device,
            )
            for runtime in runtimes
        ]
        for primary, secondary, labels in zip(
            member_scores[0], member_scores[1], pending_labels
        ):
            if labels.sum() <= 0:
                n_skipped += 1
                continue
            _accumulate_rank_metrics(sums, weights, primary, secondary, labels)
            n_scored += 1
        progress.update(len(pending_groups))
        pending_groups.clear()
        pending_labels.clear()
        pending_candidates = 0

    for row in impressions.itertuples(index=False):
        group, labels = _prepare_group(row)
        pending_groups.append(group)
        pending_labels.append(labels)
        pending_candidates += len(labels)
        if pending_candidates >= candidate_buffer_size:
            flush()
    flush()
    progress.close()

    metrics = []
    for index, weight in enumerate(weights):
        metrics.append(
            {
                "primary_weight": float(weight),
                "secondary_weight": float(1.0 - weight),
                **{
                    name: float(values[index] / n_scored) if n_scored else 0.0
                    for name, values in sums.items()
                },
            }
        )
    return {
        "split": split_name,
        "n_impressions": int(len(impressions)),
        "n_scored_impressions": n_scored,
        "n_skipped_impressions": n_skipped,
        "metrics": metrics,
    }


def _load_member_runtimes(
    cfg: dict[str, Any],
    proc_root: Path,
    device: torch.device,
    item_encoding_batch_size: int,
) -> list[_MemberRuntime]:
    runtimes = []
    for member in cfg["ensemble"]["members"]:
        member_cfg = load_config(member["config"])
        member_proc_root = (
            Path(member_cfg["data"]["processed_root"])
            / member_cfg["data"]["dataset_name"]
        )
        if member_proc_root.resolve() != proc_root.resolve():
            raise ValueError("All ensemble members must use the ensemble processed data.")
        model, item_base, _ = _load_model(
            member_cfg,
            proc_root,
            device,
            load_teacher_item=False,
        )
        encoded_items, item_semantics = precompute_item_semantics(
            model=model,
            item_base=item_base,
            device=device,
            batch_size=item_encoding_batch_size,
            description=f"Pre-encode {member['name']} items",
        )
        runtimes.append(
            _MemberRuntime(
                name=str(member["name"]),
                model=model,
                encoded_items=encoded_items,
                item_semantics=item_semantics,
            )
        )
        del item_base
        gc.collect()
    return runtimes


def run_ensemble_search(cfg: dict[str, Any]) -> None:
    ensemble_cfg = dict(cfg.get("ensemble", {}))
    members = list(ensemble_cfg.get("members", []))
    if len(members) != 2:
        raise ValueError("Rank ensembling currently requires exactly two members.")
    weights = np.asarray(ensemble_cfg.get("primary_weight_grid", []), dtype=np.float64)
    if len(weights) == 0 or np.any(weights < 0.0) or np.any(weights > 1.0):
        raise ValueError("ensemble.primary_weight_grid must contain values in [0, 1].")
    weights = np.unique(weights)
    if 0.0 not in weights or 1.0 not in weights:
        raise ValueError("The ensemble weight grid must include both member baselines 0 and 1.")

    proc_root = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    out_root = ensure_dir(Path("runs") / cfg["run_name"] / "ensemble")
    device = resolve_device(ensemble_cfg.get("device", "cuda"))
    log_device(device, "Rank ensemble")
    batch_size = int(ensemble_cfg.get("batch_size", 2048))
    candidate_buffer_size = int(ensemble_cfg.get("candidate_buffer_size", 65536))
    item_encoding_batch_size = int(
        ensemble_cfg.get("item_encoding_batch_size", max(batch_size, 8192))
    )
    if batch_size < 1 or candidate_buffer_size < batch_size:
        raise ValueError("Ensemble batch and candidate-buffer sizes are invalid.")

    runtimes = _load_member_runtimes(
        cfg,
        proc_root,
        device,
        item_encoding_batch_size,
    )
    tune_split = str(ensemble_cfg["tune_split"])
    report_split = str(ensemble_cfg["report_split"])
    if tune_split == report_split:
        raise ValueError("Ensemble tuning and reporting splits must differ.")

    tune = _score_split(
        proc_root,
        tune_split,
        runtimes,
        weights,
        batch_size,
        candidate_buffer_size,
        device,
    )
    selected = max(
        tune["metrics"],
        key=lambda row: (row["auc"], row["ndcg@10"], row["primary_weight"]),
    )
    report_weights = np.unique(
        np.asarray([0.0, selected["primary_weight"], 1.0], dtype=np.float64)
    )
    report = _score_split(
        proc_root,
        report_split,
        runtimes,
        report_weights,
        batch_size,
        candidate_buffer_size,
        device,
    )
    selected_report = next(
        row
        for row in report["metrics"]
        if row["primary_weight"] == selected["primary_weight"]
    )
    result = {
        "method": "weighted_borda_rank_fusion",
        "members": [
            {"name": member["name"], "config": member["config"]}
            for member in members
        ],
        "selection_metric": "mean_impression_auc",
        "tie_breakers": ["ndcg@10", "larger_primary_weight"],
        "tune": tune,
        "selected": selected,
        "report": report,
        "selected_report": selected_report,
        "device": str(device),
        "batch_size": batch_size,
        "candidate_buffer_size": candidate_buffer_size,
        "item_encoding_batch_size": item_encoding_batch_size,
    }
    save_json(out_root / "search.json", result)
