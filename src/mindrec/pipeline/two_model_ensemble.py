"""Select weights for exactly two mixed-loss students; no final-fit training."""
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from mindrec.config import load_config
from mindrec.data.item_age import item_age_artifact_path
from mindrec.data.recency_tiebreaker import recency_tiebreaker_config
from mindrec.pipeline.ensemble import (
    _load_member_runtimes, _load_recency, _score_split, _scoring_settings, _sha256,
)
from mindrec.pipeline.pairwise_ensemble import _context_hash, _write_json
from mindrec.utils import load_json as _read
from mindrec.utils import log_device, ranker_checkpoint_name, resolve_device, teacher_artifact_root


def _artifact(path):
    path = Path(path)
    return {"path": path.as_posix(), "sha256": _sha256(path)}


def _member_context(member, proc):
    cfg = load_config(member["config"])
    member_proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    ranker = cfg["ranker"]
    if (member_proc.resolve() != proc.resolve() or cfg["data"].get("mode") != "temporal_tune"
            or not ranker.get("grouped_batches") or ranker.get("pairwise_weight") != .25
            or ranker.get("checkpoint_selection", {}).get("metric") != "mean_impression_auc"
            or ranker_checkpoint_name(cfg) != "best.pt"):
        raise ValueError("Use temporal, impression-AUC-selected BCE + 0.25 pairwise students on the same data.")
    run = cfg.get("artifacts", {}).get("ranker_run_name", cfg["run_name"])
    root = Path("runs") / run / "ranker"
    if not (root / "train_summary.json").is_file():
        raise FileNotFoundError(f"Complete the mixed-loss student training first: {root}")
    summary = _read(root / "train_summary.json")
    if (_read(root / "training_config.json") != cfg or not summary.get("has_validation")
            or summary["stopped_epoch"] != ranker["epochs"] or summary["stop_reason"] != "max_epochs"
            or summary.get("selection_mode") != "mean_impression_auc"):
        raise ValueError(f"Incomplete or mismatched temporal training: {root}")
    selected = summary["best_by_metric"]["mean_impression_auc"]
    checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    checkpoint_artifact = _artifact(root / "best.pt")
    if (checkpoint["cfg"] != cfg or checkpoint["epoch"] != selected["epoch"]
            or checkpoint["selection_mode"] != "mean_impression_auc"
            or not math.isclose(checkpoint["val_auc"], selected["score"], abs_tol=1e-12)
            or checkpoint_artifact["sha256"] != _sha256(root / "best_mean_impression_auc.pt")):
        raise ValueError(f"The primary checkpoint must match the selected mean impression AUC checkpoint: {root}")
    embedding = "item_ranker_base_emb.npy" if cfg.get("knowledge_graph", {}).get("enabled", False) else "item_base_emb.npy"
    return {
        "name": member["name"], "config": cfg,
        "checkpoint": checkpoint_artifact,
        "training_summary": _artifact(root / "train_summary.json"),
        "item_embeddings": _artifact(teacher_artifact_root(cfg) / embedding),
    }


def _prepare(cfg):
    ensemble = cfg["ensemble"]
    settings = cfg["two_model_ensemble"]
    if len(ensemble.get("members", [])) != 2:
        raise ValueError("This experiment requires exactly two student models.")
    grid = np.asarray(ensemble["primary_weight_grid"], dtype=np.float64)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or np.any(grid < 0) or np.any(grid > 1) or 0 not in grid or 1 not in grid):
        raise ValueError("Use finite ensemble weights in [0, 1], including both single-model endpoints.")
    grid = np.unique(grid)
    baseline_cfg = load_config(settings["baseline_config"])
    baseline_path = Path(baseline_cfg["ensemble"]["selection_artifact"])
    baseline = _read(baseline_path)
    if not baseline.get("selection_frozen") or "selected_report" not in baseline:
        raise ValueError("The reference champion must have a complete frozen temporal report.")
    saved = baseline["selection_context"]
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    scoring = _scoring_settings(ensemble)
    recency = recency_tiebreaker_config(cfg)
    tune, report = ensemble["tune_split"], ensemble["report_split"]
    if ((tune, report) != (saved["tune_split"], saved["report_split"]) or tune == report
            or settings["tune_date"] != "2019-11-14" or settings["report_date"] != "2019-11-15"
            or scoring != saved["scoring"] or recency != saved["recency"]
            or saved["scoring_protocol"] != "integer_borda_recency_reference_guard_v2"):
        raise ValueError("Use the champion's scoring, recency and distinct Nov 14 / Nov 15 splits.")
    data = {name: _artifact(proc / name) for name in (
        "id_maps.json", "news.parquet", f"{tune}_impressions.parquet", f"{report}_impressions.parquet",
    )}
    age_root = Path(cfg["data"]["processed_root"]) / recency["age_dataset_name"]
    data.update(age_index=_artifact(item_age_artifact_path(age_root)), age_id_maps=_artifact(age_root / "id_maps.json"))
    if data != saved["data_artifacts"]:
        raise ValueError("Data or age inputs differ from the champion's temporal report.")
    for split, date in ((tune, settings["tune_date"]), (report, settings["report_date"])):
        times = pd.read_parquet(proc / f"{split}_impressions.parquet", columns=["time"])["time"]
        dates = pd.to_datetime(times, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if dates.empty or dates.isna().any() or not (dates.dt.strftime("%Y-%m-%d") == date).all():
            raise ValueError(f"{split} must contain only {date} impressions.")
    members = [_member_context(m, proc) for m in ensemble["members"]]
    backbones = [m["config"].get("teacher", {}).get("model_name") for m in members]
    if backbones != ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"]:
        raise ValueError("Place the MPNet student first and the MiniLM student second so primary_weight always means MPNet weight.")
    if members[0]["checkpoint"]["path"] == members[1]["checkpoint"]["path"]:
        raise ValueError("Select two distinct student checkpoints.")
    protected = {baseline_cfg["run_name"]}
    for member in members:
        protected.update((member["config"]["run_name"], Path(member["checkpoint"]["path"]).parts[1],
                          Path(member["item_embeddings"]["path"]).parts[1]))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["run_name"]) or cfg["run_name"] in protected:
        raise ValueError("Use an isolated new ensemble run_name to protect source models and the champion.")
    package = Path(__file__).resolve().parents[1]
    implementation = {name: _sha256(package / name) for name in (
        "pipeline/two_model_ensemble.py", "pipeline/ensemble.py", "pipeline/submission.py",
        "pipeline/evaluate.py", "pipeline/ranker_scoring.py", "models/dlrm.py",
        "data/recency_tiebreaker.py", "data/item_age.py", "pipeline/pairwise_ensemble.py",
        "utils.py", "config.py",
    )}
    context = {"config": cfg, "baseline_selection": _artifact(baseline_path), "members": members,
               "data_artifacts": data, "scoring": scoring, "recency": recency,
               "weights": grid.tolist(), "implementation": implementation,
               "fusion": "integer_two_member_borda", "rank_tiebreaker": "primary_rank",
               "selection_tiebreakers": ["ndcg@10", "larger_primary_weight"]}
    root = Path("runs") / cfg["run_name"] / "ensemble"
    if (root / "context.json").exists():
        if _read(root / "context.json") != context:
            raise ValueError("Inputs or inference code changed; use a new ensemble run_name.")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Unowned ensemble output exists; use a new run_name.")
    return context, baseline, root


def _load_stage(path, context):
    if not path.is_file():
        return None
    result = _read(path)
    if result.get("context_hash") != _context_hash(context) or result.get("result_hash") != _context_hash(result["result"]):
        raise ValueError(f"Stale or damaged completed inference stage: {path}")
    print(f"Reusing completed inference: {path.stem}", flush=True)
    return result["result"]


def run_two_model_ensemble_search(cfg, *, check_only=False):
    print("Verifying the two mixed-loss checkpoints and champion comparison inputs...", flush=True)
    context, baseline, root = _prepare(cfg)
    if check_only:
        print("Preflight passed: exactly two mixed-loss students, matching temporal inputs. No inference launched.")
        return
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "context.json", context)
    ensemble = cfg["ensemble"]
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    runtimes = None
    recency = None

    def score(role, weights):
        nonlocal runtimes, recency
        split = ensemble[f"{role}_split"]
        path = root / f"{role}_metrics.json"
        result = _load_stage(path, context)
        if result is None:
            device = resolve_device(ensemble.get("device", "cuda"))
            if runtimes is None:
                log_device(device, "Two mixed-loss students")
                runtimes = _load_member_runtimes(cfg, proc, device, context["scoring"]["item_encoding_batch_size"])
                recency = _load_recency(cfg)
            settings = context["scoring"]
            result = _score_split(proc, split, runtimes, np.asarray(weights), settings["batch_size"],
                                  settings["candidate_buffer_size"], device, recency=recency,
                                  exact_rank_guard_threshold=settings["exact_rank_guard_threshold"],
                                  reference_batch_size=settings["reference_batch_size"])
        if (result["split"] != split or result["n_scored_impressions"] != baseline[role]["n_scored_impressions"]
                or result["n_impressions"] != baseline[role]["n_impressions"]
                or [row["primary_weight"] for row in result["metrics"]] != list(weights)
                or not all(math.isfinite(row[k]) for row in result["metrics"] for k in ("auc", "mrr", "ndcg@5", "ndcg@10"))):
            raise ValueError("Inference results must match the expected split, weights, counts and finite metrics.")
        _write_json(path, {"context_hash": _context_hash(context), "result_hash": _context_hash(result), "result": result})
        return result

    tune = score("tune", context["weights"])
    selected = max(tune["metrics"], key=lambda row: (row["auc"], row["ndcg@10"], row["primary_weight"]))
    selection = {"selection_frozen": True, "selection_metric": "mean_impression_auc", "context_hash": _context_hash(context),
                 "selected": selected, "tune": tune}
    _write_json(root / "selection.json", selection)
    print(f"Frozen MPNet weight {selected['primary_weight']:g} on Nov 14; reporting Nov 15 next.", flush=True)
    report = score("report", sorted({0., selected["primary_weight"], 1.}))
    selected_report = next(row for row in report["metrics"] if row["primary_weight"] == selected["primary_weight"])
    deltas = {k: selected_report[k] - baseline["selected_report"][k] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")}
    tune_gain = selected["auc"] - baseline["selected"]["auc"]
    _write_json(root / "search.json", {**selection, "selection_context": context, "report": report,
        "selected_report": selected_report, "baseline_tune": baseline["selected"],
        "baseline_report": baseline["selected_report"], "report_delta": deltas, "tune_auc_gain": tune_gain,
        "improved_auc_on_both_days": tune_gain > 0 and deltas["auc"] > 0,
        "note": "Exactly two mixed-loss students. Reused development validation; no final fit, hidden-test inference or submission."})
    print(f"Two-model report AUC={selected_report['auc']:.6f}; delta vs champion temporal blend={deltas['auc']:+.6f}. Saved {root / 'search.json'}", flush=True)
