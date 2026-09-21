from __future__ import annotations

import gc
import hashlib
import json
import os
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir, load_config
from mindrec.data.featurize import IdMaps
from mindrec.data.item_age import ItemAgeIndex, item_age_artifact_path
from mindrec.data.recency_tiebreaker import apply_recency_tiebreaker, recency_tiebreaker_config
from mindrec.pipeline.evaluate import _load_model
from mindrec.pipeline.ranker_scoring import (
    precompute_item_semantics,
    score_prepared_groups,
)
from mindrec.pipeline.submission import _has_near_tied_scores, _score_reference_group, _scores_to_ranks
from mindrec.utils import log_device, resolve_device, save_json, teacher_artifact_root


@dataclass
class _MemberRuntime:
    name: str
    model: torch.nn.Module
    encoded_items: torch.Tensor
    item_semantics: torch.Tensor
    item_base: np.ndarray


@dataclass
class _RecencyRuntime:
    index: ItemAgeIndex
    news2idx: dict[str, int]
    alpha: float

    def ages(self, row: Any) -> np.ndarray:
        # The age index uses submission news indices, NOT temporal news indices.
        indices = [self.news2idx.get(str(nid), 0) for nid in row.cand_news_id]
        return self.index.ages(indices, row.time)


def _load_recency(cfg: dict[str, Any]) -> _RecencyRuntime | None:
    recency = recency_tiebreaker_config(cfg)
    if not recency["enabled"]:
        return None
    root = Path(cfg["data"]["processed_root"]) / recency["age_dataset_name"]
    return _RecencyRuntime(
        ItemAgeIndex.load(item_age_artifact_path(root),
                          expected_max_age_hours=recency["max_age_hours"]),
        IdMaps.load(root / "id_maps.json").news2idx,
        recency["alpha"],
    )


def _validate_ranks(ranks: np.ndarray, source: str) -> None:
    if ranks.ndim != 1 or len(ranks) == 0:
        raise ValueError(f"{source} must contain a non-empty one-dimensional rank list.")
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError(f"{source} ranks must be integers.")
    expected = np.arange(1, len(ranks) + 1, dtype=np.int64)
    if not np.array_equal(np.sort(ranks), expected):
        raise ValueError(f"{source} ranks must be a permutation of 1..N.")


def _parse_prediction_line(raw: bytes, source: str) -> tuple[str, np.ndarray]:
    try:
        impression_id, encoded_ranks = raw.decode("utf-8").rstrip("\r\n").split(" ", 1)
        values = json.loads(encoded_ranks)
        if not isinstance(values, list) or any(type(v) is not int for v in values):
            raise ValueError("Ranks must be a list of JSON integers.")
        ranks = np.asarray(values, dtype=np.int64)
    except (UnicodeDecodeError, ValueError, TypeError, OverflowError) as exc:
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
    if len(primary_ranks) != len(secondary_ranks):
        raise ValueError("Ensemble members have different candidate counts.")
    _validate_ranks(primary_ranks, "primary prediction")
    _validate_ranks(secondary_ranks, "secondary prediction")

    order = _borda_order(primary_ranks, secondary_ranks, primary_weight)
    fused = np.empty(len(order), dtype=np.int32)
    fused[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return fused


@lru_cache(maxsize=128)
def _weight_ratio(weight: float) -> tuple[int, int]:
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Primary weight must be finite and between 0 and 1.")
    # Interpret the declared decimal exactly; 0.6 becomes 3/5, not binary FP.
    ratio = Fraction(str(float(weight)))
    return ratio.numerator, ratio.denominator


def _borda_order(primary: np.ndarray, secondary: np.ndarray, weight: float) -> np.ndarray:
    """Shared kernel for validated permutations; no repeated permutation checks."""
    numerator, denominator = _weight_ratio(weight)
    # Python integers handle unusually precise decimal weights without overflow.
    dtype = np.int64 if denominator * len(primary) <= np.iinfo(np.int64).max else object
    costs = numerator * primary.astype(dtype) + (denominator - numerator) * secondary.astype(dtype)
    return np.lexsort((primary, costs))


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


def _selection_context(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hash the bytes used for inference, not unrelated training/reranking options."""
    ensemble_cfg = cfg["ensemble"]
    proc_root = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    recency = recency_tiebreaker_config(cfg)

    def artifact(path: Path) -> dict[str, str]:
        return {"path": path.as_posix(), "sha256": _sha256(path)}

    members = []
    for member in ensemble_cfg["members"]:
        member_cfg = load_config(member["config"])
        member_root = Path(member_cfg["data"]["processed_root"]) / member_cfg["data"]["dataset_name"]
        if member_root.resolve() != proc_root.resolve():
            raise ValueError("All ensemble members must use the ensemble processed data.")
        ranker_run = member_cfg.get("artifacts", {}).get("ranker_run_name", member_cfg["run_name"])
        embedding_name = ("item_ranker_base_emb.npy"
                          if member_cfg.get("knowledge_graph", {}).get("enabled", False)
                          else "item_base_emb.npy")
        prediction_path = Path(member["prediction_zip"])
        if recency["enabled"]:
            metadata = json.loads(prediction_path.with_name("submission_meta.json").read_text(encoding="utf-8"))
            recorded = metadata.get("posthoc_recency", {})
            for key in ("enabled", "alpha", "age_dataset_name"):
                if recorded.get(key) != recency[key]:
                    raise ValueError(f"{prediction_path}: recency {key} does not match validation.")
        # ZIP timestamps/compression do not affect the predictions we consume.
        content_digest = hashlib.sha256()
        with ExitStack() as stack:
            stream = _prediction_stream(stack, prediction_path)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                content_digest.update(block)
        members.append({
            "name": member["name"],
            "checkpoint": artifact(Path("runs") / ranker_run / "ranker" / "best.pt"),
            "item_embeddings": artifact(teacher_artifact_root(member_cfg) / embedding_name),
            "prediction_content_sha256": content_digest.hexdigest(),
        })
    data_artifacts = {
        name: artifact(proc_root / name)
        for name in ("id_maps.json", "news.parquet",
                     f"{ensemble_cfg['tune_split']}_impressions.parquet",
                     f"{ensemble_cfg['report_split']}_impressions.parquet")
    }
    if recency["enabled"]:
        age_root = Path(cfg["data"]["processed_root"]) / recency["age_dataset_name"]
        data_artifacts["age_index"] = artifact(item_age_artifact_path(age_root))
        data_artifacts["age_id_maps"] = artifact(age_root / "id_maps.json")
    return {
        "schema_version": 2,
        "scoring_protocol": "integer_borda_recency_reference_guard_v2",
        "data_artifacts": data_artifacts,
        "members": members,
        "primary_weight_grid": sorted(set(ensemble_cfg["primary_weight_grid"])),
        "tune_split": ensemble_cfg["tune_split"],
        "report_split": ensemble_cfg["report_split"],
        "method": "weighted_borda_rank_fusion",
        "rank_tiebreaker": "primary_rank",
        "recency": recency,
        "scoring": _scoring_settings(ensemble_cfg),
    }


def _scoring_settings(ensemble_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": int(ensemble_cfg.get("batch_size", 2048)),
        "candidate_buffer_size": int(ensemble_cfg.get("candidate_buffer_size", 65536)),
        "item_encoding_batch_size": int(ensemble_cfg.get("item_encoding_batch_size", 8192)),
        "exact_rank_guard_threshold": float(ensemble_cfg.get("exact_rank_guard_threshold", 1e-5)),
        "reference_batch_size": int(ensemble_cfg.get("reference_batch_size", 2048)),
    }


def _submission_weight(cfg: dict[str, Any]) -> float:
    ensemble_cfg = cfg["ensemble"]
    source = ensemble_cfg.get("selection_source", "config")
    if source == "search_artifact":
        path = Path(ensemble_cfg["selection_artifact"])
        if not path.is_file():
            raise ValueError(f"Run ensemble_search first; missing selection artifact: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("selection_context") != _selection_context(cfg):
            raise ValueError("Selection artifact does not match these members/configs/ZIPs; rerun ensemble_search.")
        if not result.get("selection_frozen") or "selected_report" not in result:
            raise ValueError("Selection artifact is incomplete; finish ensemble_search first.")
        weight = float(result["selected"]["primary_weight"])
    elif source == "config":
        if not bool(ensemble_cfg.get("frozen", False)):
            raise ValueError("Set ensemble.frozen=true after selecting the validation weight.")
        weight = float(ensemble_cfg["selected_primary_weight"])
    else:
        raise ValueError(f"Unknown ensemble.selection_source: {source!r}")
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("Selected primary weight must be finite and between 0 and 1.")
    return weight


def run_ensemble_submission(cfg: dict[str, Any]) -> None:
    ensemble_cfg = dict(cfg.get("ensemble", {}))
    members = list(ensemble_cfg.get("members", []))
    if len(members) != 2:
        raise ValueError("Rank ensembling currently requires exactly two members.")
    weight = _submission_weight(cfg)
    final_root = ensure_dir(Path("runs") / cfg["run_name"] / "submission")
    if any(Path(m["prediction_zip"]).resolve() == (final_root / "prediction.zip").resolve()
           for m in members):
        raise ValueError("Ensemble output must not overwrite a member ZIP.")
    # A failed parse/count/CRC/hash check only affects staging, never an old result.
    with TemporaryDirectory(prefix=".ensemble-", dir=final_root) as directory:
        staging = Path(directory)
        _write_ensemble_submission(cfg, weight, staging, final_root)
        _publish_submission(staging, final_root)


def _publish_submission(staging: Path, final_root: Path) -> None:
    """Publish validated files, restoring both previous files on an I/O failure."""
    names = ("submission_meta.json", "prediction.zip")
    backed_up: list[str] = []
    installed: list[str] = []
    try:
        # Remove the old success marker first, so an interrupted publish never
        # pairs the new ZIP with stale success metadata.
        for name in names:
            previous = final_root / name
            if previous.exists():
                os.replace(previous, staging / (name + ".previous"))
                backed_up.append(name)
        for name in reversed(names):
            os.replace(staging / name, final_root / name)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            (final_root / name).unlink()
        for name in reversed(backed_up):
            os.replace(staging / (name + ".previous"), final_root / name)
        raise


def _write_ensemble_submission(
    cfg: dict[str, Any], weight: float, out_root: Path, final_root: Path,
) -> None:
    ensemble_cfg = cfg["ensemble"]
    members = ensemble_cfg["members"]
    print(f"Ensemble weights: {members[0]['name']}={weight:g}, "
          f"{members[1]['name']}={1.0 - weight:g}", flush=True)
    primary_path = Path(members[0]["prediction_zip"])
    secondary_path = Path(members[1]["prediction_zip"])
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
                    if len(primary_ranks) != len(secondary_ranks):
                        raise ValueError("Ensemble members have different candidate counts.")
                    if primary_id != str(n_impressions + 1):
                        raise ValueError("Expected sequential MIND impression IDs starting at 1.")
                    order = _borda_order(primary_ranks, secondary_ranks, weight)
                    fused = np.empty(len(order), dtype=np.int32)
                    fused[order] = np.arange(1, len(order) + 1, dtype=np.int32)
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

    if n_impressions == 0:
        raise ValueError("Cannot publish an empty submission.")
    verified_digest = hashlib.sha256()
    with ExitStack() as stack:
        stream = _prediction_stream(stack, zip_path)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            verified_digest.update(block)
    if verified_digest.digest() != prediction_digest.digest():
        raise ValueError("Written prediction content hash does not match.")

    save_json(
        meta_path,
        {
            "method": "weighted_borda_rank_fusion",
            "scoring_protocol": "integer_borda_recency_reference_guard_v2",
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
            "prediction_zip": str(final_root / "prediction.zip"),
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
    if not np.isfinite(primary_scores).all() or not np.isfinite(secondary_scores).all():
        raise ValueError("Ensemble scoring produced non-finite logits.")
    primary_ranks = _scores_to_ranks(primary_scores)
    secondary_ranks = _scores_to_ranks(secondary_scores)
    order = np.stack([_borda_order(primary_ranks, secondary_ranks, float(w)) for w in weights])
    fused_ranks = np.empty_like(order)
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
        np.broadcast_to(labels, order.shape), order, axis=1
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


def _score_split(
    proc_root: Path,
    split_name: str,
    runtimes: list[_MemberRuntime],
    weights: np.ndarray,
    batch_size: int,
    candidate_buffer_size: int,
    device: torch.device,
    *,
    recency: _RecencyRuntime | None = None,
    exact_rank_guard_threshold: float = 1e-5,
    reference_batch_size: int = 2048,
) -> dict[str, Any]:
    path = proc_root / f"{split_name}_impressions.parquet"
    impressions = pd.read_parquet(path)
    if recency is not None and not {"time", "cand_news_id"}.issubset(impressions.columns):
        raise ValueError("Recency evaluation requires time and cand_news_id columns.")
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
        member_scores = []
        for runtime in runtimes:
            scores_by_group = score_prepared_groups(
                model=runtime.model,
                groups=pending_groups,
                encoded_items=runtime.encoded_items,
                item_semantics=runtime.item_semantics,
                batch_size=batch_size,
                device=device,
            )
            adjusted_scores = []
            for group, scores in zip(pending_groups, scores_by_group):
                if _has_near_tied_scores(scores, exact_rank_guard_threshold):
                    scores = _score_reference_group(
                        runtime.model, group, runtime.item_base, reference_batch_size, device,
                    )
                if recency is not None:
                    scores = apply_recency_tiebreaker(scores, group["recency_ages"], recency.alpha)
                adjusted_scores.append(scores)
            member_scores.append(adjusted_scores)
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
        if recency is not None:
            group["recency_ages"] = recency.ages(row)
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
        "recency_alpha": recency.alpha if recency is not None else 0.0,
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
                item_base=item_base,
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
    if (weights.ndim != 1 or len(weights) == 0 or not np.isfinite(weights).all()
            or np.any(weights < 0.0) or np.any(weights > 1.0)):
        raise ValueError("ensemble.primary_weight_grid must contain values in [0, 1].")
    weights = np.unique(weights)
    if 0.0 not in weights or 1.0 not in weights:
        raise ValueError("The ensemble weight grid must include both member baselines 0 and 1.")

    tune_split = str(ensemble_cfg["tune_split"])
    report_split = str(ensemble_cfg["report_split"])
    if tune_split == report_split:
        raise ValueError("Ensemble tuning and reporting splits must differ.")
    automatic_selection = ensemble_cfg.get("selection_source", "config") == "search_artifact"
    selection_context = _selection_context(cfg) if automatic_selection else None

    proc_root = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    out_root = ensure_dir(Path("runs") / cfg["run_name"] / "ensemble")
    device = resolve_device(ensemble_cfg.get("device", "cuda"))
    log_device(device, "Rank ensemble")
    scoring = _scoring_settings(ensemble_cfg)
    batch_size = scoring["batch_size"]
    candidate_buffer_size = scoring["candidate_buffer_size"]
    item_encoding_batch_size = scoring["item_encoding_batch_size"]
    if batch_size < 1 or candidate_buffer_size < batch_size:
        raise ValueError("Ensemble batch and candidate-buffer sizes are invalid.")
    if (item_encoding_batch_size < 1 or scoring["reference_batch_size"] < 1
            or not np.isfinite(scoring["exact_rank_guard_threshold"])
            or scoring["exact_rank_guard_threshold"] < 0):
        raise ValueError("Invalid ensemble item/reference scoring settings.")
    score_options = {
        "recency": _load_recency(cfg),
        "exact_rank_guard_threshold": scoring["exact_rank_guard_threshold"],
        "reference_batch_size": scoring["reference_batch_size"],
    }

    runtimes = _load_member_runtimes(
        cfg,
        proc_root,
        device,
        item_encoding_batch_size,
    )
    tune = _score_split(
        proc_root,
        tune_split,
        runtimes,
        weights,
        batch_size,
        candidate_buffer_size,
        device,
        **score_options,
    )
    selected = max(
        tune["metrics"],
        key=lambda row: (row["auc"], row["ndcg@10"], row["primary_weight"]),
    )
    if tune["n_scored_impressions"] == 0:
        raise ValueError("Cannot select an ensemble weight without labeled tuning impressions.")
    print(f"Selected MPNet weight on {tune_split}: {selected['primary_weight']:g} "
          f"(AUC={selected['auc']:.6f}); frozen before {report_split}.", flush=True)
    save_json(out_root / "selection.json", {
        "selected": selected,
        "selection_frozen": True,
        "selection_context": selection_context,
        "tune": tune,
    })
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
        **score_options,
    )
    if report["n_scored_impressions"] == 0:
        raise ValueError("Cannot complete search without labeled reporting impressions.")
    selected_report = next(
        row
        for row in report["metrics"]
        if row["primary_weight"] == selected["primary_weight"]
    )
    result = {
        "method": "weighted_borda_rank_fusion",
        "selection_frozen": True,
        "selection_context": selection_context,
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
    result_path = (
        Path(ensemble_cfg["selection_artifact"])
        if automatic_selection else out_root / "search.json"
    )
    save_json(result_path, result)
    print(f"Search saved to {result_path}; report AUC={selected_report['auc']:.6f}", flush=True)
