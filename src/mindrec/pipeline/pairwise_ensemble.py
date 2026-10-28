"""Inference-only addition of a pairwise student to the frozen two-model blend."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from mindrec.config import load_config
from mindrec.data.item_age import item_age_artifact_path
from mindrec.data.recency_tiebreaker import recency_tiebreaker_config
from mindrec.pipeline.ensemble import (
    _accumulate_order_metrics, _borda_order, _empty_metric_sums,
    _iter_scored_batches, _load_member_runtimes, _load_recency,
    _scoring_settings, _sha256, _weight_ratio,
)
from mindrec.pipeline.submission import _scores_to_ranks
from mindrec.utils import log_device, ranker_checkpoint_name, resolve_device, teacher_artifact_root


def _write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _context_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _three_order(primary, secondary, pairwise, primary_weight, pairwise_share, *, baseline_ranks=None):
    """Flat three-member Borda costs; resolve ties by the original blend order."""
    bn, bd = _weight_ratio(float(primary_weight))
    sn, sd = _weight_ratio(float(pairwise_share))
    if baseline_ranks is None:
        baseline_order = _borda_order(primary, secondary, float(primary_weight))
        baseline_ranks = np.empty(len(primary), dtype=np.int64)
        baseline_ranks[baseline_order] = np.arange(1, len(primary) + 1)
    dtype = np.int64 if bd * sd * len(primary) <= np.iinfo(np.int64).max else object
    costs = ((sd - sn) * bn * primary.astype(dtype)
             + (sd - sn) * (bd - bn) * secondary.astype(dtype)
             + sn * bd * pairwise.astype(dtype))
    return np.lexsort((baseline_ranks, costs))


def _prepare_context(cfg):
    experiment = cfg["pairwise_ensemble"]
    baseline_cfg = load_config(experiment["baseline_config"])
    source_path = Path(baseline_cfg["ensemble"]["selection_artifact"])
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if not source.get("selection_frozen") or "selected_report" not in source:
        raise ValueError("The existing ensemble selection must be complete and frozen.")
    saved = source["selection_context"]
    if saved["scoring_protocol"] != "integer_borda_recency_reference_guard_v2":
        raise ValueError("Unsupported baseline scoring protocol.")
    weight = float(source["selected"]["primary_weight"])
    _weight_ratio(weight)
    shares = sorted(set(float(x) for x in experiment["pairwise_share_grid"]))
    if shares != [0.0, 0.1, 0.25, 0.5]:
        raise ValueError("This experiment uses the prespecified shares [0, 0.1, 0.25, 0.5].")
    members = deepcopy(baseline_cfg["ensemble"]["members"])
    if len(members) != 2:
        raise ValueError("The frozen baseline must have exactly two members.")
    members.append(dict(experiment["member"]))
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    baseline_proc = Path(baseline_cfg["data"]["processed_root"]) / baseline_cfg["data"]["dataset_name"]
    if proc.resolve() != baseline_proc.resolve():
        raise ValueError("Baseline and experiment processed data must match.")
    scoring = _scoring_settings(cfg["ensemble"])
    if (scoring["batch_size"] < 1 or scoring["candidate_buffer_size"] < scoring["batch_size"]
            or scoring["item_encoding_batch_size"] < 1 or scoring["reference_batch_size"] < 1
            or not math.isfinite(scoring["exact_rank_guard_threshold"])
            or scoring["exact_rank_guard_threshold"] < 0):
        raise ValueError("Invalid ensemble scoring settings.")
    recency = recency_tiebreaker_config(cfg)
    if (not recency["enabled"] or recency != saved["recency"]
            or scoring != saved["scoring"]):
        raise ValueError("Recency and scoring settings must match the frozen baseline.")
    tune = cfg["ensemble"]["tune_split"]
    report = cfg["ensemble"]["report_split"]
    if tune == report or (tune, report) != (saved["tune_split"], saved["report_split"]):
        raise ValueError("Use the baseline's distinct tuning and reporting splits.")

    def artifact(path):
        return {"path": path.as_posix(), "sha256": _sha256(path)}

    data = {name: artifact(proc / name) for name in (
        "id_maps.json", "news.parquet", f"{tune}_impressions.parquet", f"{report}_impressions.parquet",
    )}
    age_root = Path(cfg["data"]["processed_root"]) / recency["age_dataset_name"]
    data["age_index"] = artifact(item_age_artifact_path(age_root))
    data["age_id_maps"] = artifact(age_root / "id_maps.json")
    if data != saved["data_artifacts"]:
        raise ValueError("Data or age artifacts differ from the frozen baseline selection.")
    recorded_members = []
    source_runs = {baseline_cfg["run_name"]}
    for index, member in enumerate(members):
        member_cfg = load_config(member["config"])
        member_proc = Path(member_cfg["data"]["processed_root"]) / member_cfg["data"]["dataset_name"]
        if member_proc.resolve() != proc.resolve():
            raise ValueError("All three members must use the same processed data.")
        run = str(member_cfg.get("artifacts", {}).get("ranker_run_name", member_cfg["run_name"]))
        source_runs.add(run)
        embedding = "item_ranker_base_emb.npy" if member_cfg.get("knowledge_graph", {}).get("enabled", False) else "item_base_emb.npy"
        recorded = {
            "name": member["name"],
            "checkpoint": artifact(Path("runs") / run / "ranker" / ranker_checkpoint_name(member_cfg)),
            "item_embeddings": artifact(teacher_artifact_root(member_cfg) / embedding),
        }
        if index < 2 and any(recorded[k] != saved["members"][index][k] for k in recorded):
            raise ValueError("A baseline member differs from the frozen ensemble selection.")
        recorded_members.append(recorded)
    if cfg["run_name"] in source_runs:
        raise ValueError("Use a new run_name to preserve source experiments.")
    implementation = {}
    package = Path(__file__).resolve().parents[1]
    for name in ("pipeline/pairwise_ensemble.py", "pipeline/ensemble.py", "pipeline/evaluate.py",
                 "pipeline/ranker_scoring.py", "pipeline/submission.py", "models/dlrm.py",
                 "data/item_age.py", "data/recency_tiebreaker.py"):
        implementation[name] = _sha256(package / name)
    context = {
        "schema_version": 1, "baseline_selection": artifact(source_path),
        "members": recorded_members, "data_artifacts": data,
        "primary_weight": weight, "pairwise_share_grid": shares,
        "tune_split": tune, "report_split": report,
        "tune_date": experiment["tune_date"], "report_date": experiment["report_date"],
        "recency": recency, "scoring": scoring, "implementation": implementation,
        "fusion": "flat_integer_three_member_borda", "rank_tiebreaker": "original_blend_rank",
        "share_selection_tiebreaker": "smaller_pairwise_share",
    }
    runtime_cfg = deepcopy(cfg)
    runtime_cfg["ensemble"]["members"] = members
    root = Path("runs") / cfg["run_name"] / "ensemble"
    if (root / "context.json").exists() and json.loads((root / "context.json").read_text()) != context:
        raise ValueError("Cached run inputs or implementation changed; use a new run_name.")
    return runtime_cfg, context, source, root


def _cache_paths(root, split):
    return root / f"{split}_ranks.npz", root / f"{split}_cache.json"


def _load_cache(root, split, context):
    path, metadata_path = _cache_paths(root, split)
    if not path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata["context_hash"] != _context_hash(context) or metadata["sha256"] != _sha256(path):
        raise ValueError(f"Stale or damaged rank cache: {path}")
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name] for name in ("offsets", "labels", "ranks")}
    offsets, labels, ranks = result["offsets"], result["labels"], result["ranks"]
    if (offsets.dtype != np.int64 or labels.dtype != np.int8
            or offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(labels)
            or np.any(np.diff(offsets) < 0) or labels.ndim != 1
            or ranks.shape != (3, len(labels)) or ranks.dtype != np.int32
            or not np.isin(labels, [0, 1]).all() or np.any(ranks < 1)):
        raise ValueError(f"Invalid rank-cache layout: {path}")
    print(f"Reusing verified rank cache: {path}", flush=True)
    return result


def _build_cache(cfg, root, split, expected_date, runtimes, context, device, recency):
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    impressions = pd.read_parquet(proc / f"{split}_impressions.parquet")
    dates = pd.to_datetime(impressions["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if dates.isna().any() or not (dates.dt.strftime("%Y-%m-%d") == expected_date).all():
        raise ValueError(f"{split} must contain only {expected_date} impressions.")
    lengths = np.fromiter((len(x) for x in impressions["cand_label"]), dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    ranks = np.empty((3, int(offsets[-1])), dtype=np.int32)
    labels = np.empty(int(offsets[-1]), dtype=np.int8)
    scoring = context["scoring"]
    row_index = 0
    for scores, batch_labels in _iter_scored_batches(
        impressions, runtimes, scoring["batch_size"], scoring["candidate_buffer_size"], device,
        recency=recency, exact_rank_guard_threshold=scoring["exact_rank_guard_threshold"],
        reference_batch_size=scoring["reference_batch_size"],
        description=f"Score three frozen models ({split})",
    ):
        for j, target in enumerate(batch_labels):
            start, end = offsets[row_index : row_index + 2]
            if not np.isin(target, [0, 1]).all():
                raise ValueError("Rank fusion requires binary click labels.")
            labels[start:end] = target
            for member in range(3):
                ranks[member, start:end] = _scores_to_ranks(scores[member][j])
            row_index += 1
    if row_index != len(impressions):
        raise ValueError("Scoring did not cover every impression.")
    path, metadata_path = _cache_paths(root, split)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez(stream, offsets=offsets, labels=labels, ranks=ranks)
    os.replace(temporary, path)
    _write_json(metadata_path, {"context_hash": _context_hash(context), "sha256": _sha256(path)})
    return {"offsets": offsets, "labels": labels, "ranks": ranks}


def _evaluate_cache(cache, shares, primary_weight, split):
    sums = _empty_metric_sums(len(shares))
    offsets = cache["offsets"]
    n_scored = 0
    for start, end in tqdm(zip(offsets[:-1], offsets[1:]), total=len(offsets) - 1,
                           desc=f"Evaluate frozen shares ({split})", unit="impr"):
        labels = cache["labels"][start:end]
        if labels.sum() <= 0:
            continue
        ranks = cache["ranks"][:, start:end]
        baseline_order = _borda_order(ranks[0], ranks[1], float(primary_weight))
        baseline_ranks = np.empty(len(labels), dtype=np.int64)
        baseline_ranks[baseline_order] = np.arange(1, len(labels) + 1)
        orders = np.stack([
            baseline_order if share == 0 else
            _three_order(*ranks, primary_weight, share, baseline_ranks=baseline_ranks)
            for share in shares
        ])
        _accumulate_order_metrics(sums, orders, labels)
        n_scored += 1
    if n_scored == 0:
        raise ValueError("Cannot select or report without clicked impressions.")
    return {
        "split": split, "n_impressions": len(offsets) - 1,
        "n_scored_impressions": n_scored, "n_candidates": len(cache["labels"]),
        "metrics": [{"pairwise_share": float(s),
                     "effective_weights": [(1-s)*primary_weight, (1-s)*(1-primary_weight), s],
                     **{k: float(v[i] / n_scored) for k, v in sums.items()}}
                    for i, s in enumerate(shares)],
    }


def _verify_baseline(result, source, role):
    baseline = next(x for x in result["metrics"] if x["pairwise_share"] == 0)
    expected = source["selected"] if role == "tune" else source["selected_report"]
    if result["n_scored_impressions"] != source[role]["n_scored_impressions"] or any(
        not math.isclose(baseline[k], expected[k], abs_tol=1e-6, rel_tol=0)
        for k in ("auc", "mrr", "ndcg@5", "ndcg@10")
    ):
        raise ValueError("The 0% share does not reproduce the saved baseline; do not interpret the blend comparison.")
    return baseline


def run_pairwise_ensemble_search(cfg, *, check_only=False):
    print("Verifying frozen checkpoints, embeddings, validation data and age artifacts...", flush=True)
    cfg, context, source, root = _prepare_context(cfg)
    if check_only:
        print("Preflight passed: three frozen models, matching baseline inputs, fixed shares. No inference launched.")
        return
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "context.json", context)
    device = None
    runtimes = None
    recency = None

    def cache_for(split, date):
        nonlocal device, runtimes, recency
        cache = _load_cache(root, split, context)
        if cache is not None:
            return cache
        if runtimes is None:
            device = resolve_device(cfg["ensemble"].get("device", "cuda"))
            log_device(device, "Frozen rank ensemble")
            runtimes = _load_member_runtimes(cfg, Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"],
                                            device, context["scoring"]["item_encoding_batch_size"])
            recency = _load_recency(cfg)
        return _build_cache(cfg, root, split, date, runtimes, context, device, recency)

    tune = _evaluate_cache(cache_for(context["tune_split"], context["tune_date"]),
                           context["pairwise_share_grid"], context["primary_weight"], context["tune_split"])
    tune_baseline = _verify_baseline(tune, source, "tune")
    selected = max(tune["metrics"], key=lambda x: (x["auc"], -x["pairwise_share"]))
    selection = {"selection_frozen": True, "context_hash": _context_hash(context),
                 "selection_metric": "mean_impression_auc", "selected": selected, "tune": tune}
    _write_json(root / "selection.json", selection)
    share = selected["pairwise_share"]
    print(f"Selected pairwise share {share:g} on Nov 14; frozen before Nov 15 reporting.", flush=True)
    report = _evaluate_cache(cache_for(context["report_split"], context["report_date"]),
                             sorted({0.0, share}), context["primary_weight"], context["report_split"])
    report_baseline = _verify_baseline(report, source, "report")
    selected_report = next(x for x in report["metrics"] if x["pairwise_share"] == share)
    deltas = {k: selected_report[k] - report_baseline[k] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")}
    result = {**selection, "selection_context": context, "report": report,
              "baseline_report": report_baseline, "selected_report": selected_report,
              "report_delta": deltas, "tune_auc_gain": selected["auc"] - tune_baseline["auc"],
              "improved_auc_on_both_days": selected["auc"] > tune_baseline["auc"] and deltas["auc"] > 0,
              "note": "Reused development validation; no training, hidden-test scoring, or submission."
              }
    _write_json(root / "search.json", result)
    print(f"Report AUC={selected_report['auc']:.6f}; delta vs baseline={deltas['auc']:+.6f}. Saved {root / 'search.json'}", flush=True)
