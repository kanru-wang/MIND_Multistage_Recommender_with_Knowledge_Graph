"""Compare 0.5 with retained 0.25 students, then evaluate fixed 75/25 fusion."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from tqdm import tqdm

from run_ranker_loss_comparison import completed_summary, preflight
from mindrec.config import load_config
from mindrec.pipeline.ensemble import _load_member_runtimes, _load_recency, _score_split, _sha256
from mindrec.pipeline.pairwise_ensemble import _context_hash, _write_json
from mindrec.pipeline.two_model_ensemble import _load_stage, _prepare
from mindrec.utils import load_json, log_device, resolve_device


BASELINE_CONFIG = "configs/mind_large_ensemble_two_mixed_loss.yaml"
CONFIGS = {
    "mpnet": "configs/mind_large_temporal_mpnet_p1_pairwise_w050.yaml",
    "minilm": "configs/mind_large_temporal_minilm_pairwise_w050.yaml",
}
ROOT = Path("runs/mind_large_pairwise_w050_comparison_v1")


def validate_change(baseline, candidate):
    left, right = deepcopy(baseline), deepcopy(candidate)
    if left["ranker"]["pairwise_weight"] != .25 or right["ranker"]["pairwise_weight"] != .5:
        raise ValueError("This comparison requires weights 0.25 and 0.5.")
    for cfg in (left, right):
        ranker = cfg["ranker"]
        selection = ranker["checkpoint_selection"]
        if (not ranker["grouped_batches"] or ranker["epochs"] != 8
                or ranker["early_stopping"]["enabled"] or ranker["calibration"]["enabled"]
                or not selection["track_both"] or not selection["preserve_training_rng"]
                or selection["metric"] != "mean_impression_auc"):
            raise ValueError("Keep the eight-epoch grouped-batch, dual-selector protocol.")
    if left.pop("run_name") == right.pop("run_name"):
        raise ValueError("Use a new run_name for weight 0.5.")
    left["ranker"].pop("pairwise_weight")
    right["ranker"].pop("pairwise_weight")
    if left != right:
        raise ValueError("Only run_name and pairwise_weight may differ from the completed 0.25 run.")
    name = candidate["run_name"]
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Use a simple isolated run_name.")


def checked_evaluation(cfg):
    summary = completed_summary(cfg)
    root = Path("runs") / cfg["run_name"]
    checkpoint = torch.load(root / "ranker/best.pt", map_location="cpu", weights_only=False)
    selected = summary["best_by_metric"]["mean_impression_auc"]
    if (checkpoint["cfg"] != cfg or checkpoint["epoch"] != selected["epoch"]
            or checkpoint["selection_mode"] != "mean_impression_auc"
            or not math.isfinite(checkpoint["val_auc"])
            or not math.isclose(checkpoint["val_auc"], selected["score"], rel_tol=0, abs_tol=1e-12)
            or _sha256(root / "ranker/best.pt") != _sha256(root / "ranker/best_mean_impression_auc.pt")):
        raise ValueError("Primary checkpoint must match the selected mean-impression checkpoint.")
    report = load_json(root / "eval/ranker_eval_val.json")
    if (report["checkpoint_name"] != "best.pt" or not math.isfinite(report["ranking"]["auc"])
            or not math.isclose(report["ranking"]["auc"], selected["score"], rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Evaluation does not match the selected checkpoint.")
    return summary, report


def validate_baseline_rows(baseline):
    for role, selected_key in (("tune", "selected"), ("report", "selected_report")):
        rows = [row for row in baseline[role]["metrics"] if row["primary_weight"] == .75]
        if len(rows) != 1 or rows[0] != baseline[selected_key]:
            raise ValueError("Baseline selected metrics must match its completed 75/25 inference rows.")


def validate_output_roots(candidates):
    outputs = [ROOT.resolve(), *[(Path("runs") / c["run_name"]).resolve() for c in candidates.values()]]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Comparison and both candidate students require distinct output directories.")


def prepare():
    baseline_cfg = load_config(BASELINE_CONFIG)
    context, _, root = _prepare(baseline_cfg)
    baseline = load_json(root / "search.json")
    if (baseline["context_hash"] != _context_hash(context) or baseline["selection_context"] != context
            or not baseline["selection_frozen"] or baseline["selected"]["primary_weight"] != .75):
        raise ValueError("Require the completed matching 0.25/0.25 ensemble selected at 75/25.")
    for role in ("tune", "report"):
        if _load_stage(root / f"{role}_metrics.json", context) != baseline[role]:
            raise ValueError("Baseline report must match completed inference stages.")
    validate_baseline_rows(baseline)
    candidates = {name: load_config(path) for name, path in CONFIGS.items()}
    validate_output_roots(candidates)
    originals = {}
    inputs = [root / "search.json"]
    for name, member in zip(CONFIGS, baseline_cfg["ensemble"]["members"]):
        original = load_config(member["config"])
        validate_change(original, candidates[name])
        preflight(candidates[name])
        checked_evaluation(original)
        originals[name] = original
        inputs.append(Path("runs") / original["run_name"] / "eval/ranker_eval_val.json")
    # Include training inputs as well as the temporal inference inputs already
    # fingerprinted by the retained baseline context.
    proc = Path(baseline_cfg["data"]["processed_root"]) / baseline_cfg["data"]["dataset_name"]
    inputs.extend(proc / name for name in ("train_behaviors.parquet", "item_click_counts.json", "val_pairs.parquet", "val_impressions.parquet"))
    for member in context["members"]:
        teacher = Path(member["item_embeddings"]["path"]).parent
        inputs.extend(teacher / name for name in ("model.pt", "item_teacher_emb.npy"))
    inputs.extend([Path(__file__), Path(__file__).with_name("run_ranker_loss_comparison.py")])
    package = Path(__file__).resolve().parents[1] / "src/mindrec"
    inputs.extend(sorted(package.rglob("*.py")))
    experiment = {"baseline_context": context, "configs": candidates, "fixed_weights": [.75, .25],
                  "inputs": {p.as_posix(): _sha256(p) for p in tqdm(sorted(set(inputs)), desc="Fingerprint comparison inputs", unit="file")}}
    ownership = ROOT / "context.json"
    if ownership.exists():
        if load_json(ownership) != experiment:
            raise ValueError("Inputs or code changed; choose a new ROOT and new candidate run names.")
    elif ROOT.exists() or any((Path("runs") / c["run_name"]).exists() for c in candidates.values()):
        raise ValueError("Preserving existing unowned experiment outputs; use new output names.")
    return baseline_cfg, baseline, originals, candidates, experiment


def compare_students(original, candidate):
    a, b = checked_evaluation(original), checked_evaluation(candidate)
    if any(a[1][k] != b[1][k] for k in ("n_impressions", "n_scored_pairs")):
        raise ValueError("Student evaluation counts differ.")
    return {"baseline_run": original["run_name"], "candidate_run": candidate["run_name"],
            "baseline_best_epoch": a[0]["best_epoch"], "candidate_best_epoch": b[0]["best_epoch"],
            "baseline_ranking": a[1]["ranking"], "candidate_ranking": b[1]["ranking"],
            "delta": {k: v - a[1]["ranking"][k] for k, v in b[1]["ranking"].items()},
            "slice_auc_delta": {k: v["auc"] - a[1]["slices"][k]["auc"]
                                for k, v in b[1].get("slices", {}).items() if k in a[1].get("slices", {})}}


def train_student(name, cfg, experiment):
    root = Path("runs") / cfg["run_name"]
    if (root / "ranker/train_summary.json").exists():
        completed_summary(cfg)
        print(f"Reusing completed training: {name}", flush=True)
    else:
        print(f"Training {name} at weight 0.5; incomplete fits restart at epoch 1.", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "mindrec.cli", "train_ranker", "--config", CONFIGS[name]], check=True)
    marker = ROOT / f"{name}_evaluation.json"
    state = {"experiment": _context_hash(experiment), "checkpoint": _sha256(root / "ranker/best.pt")}
    if marker.exists():
        saved = load_json(marker)
        path = root / "eval/ranker_eval_val.json"
        if saved != {**state, "evaluation": _sha256(path)}:
            raise ValueError("Completed candidate evaluation changed; preserve it and use new output names.")
    else:
        subprocess.run([sys.executable, "-u", "-m", "mindrec.cli", "evaluate", "--config", CONFIGS[name]], check=True)
        checked_evaluation(cfg)
        _write_json(marker, {**state, "evaluation": _sha256(root / "eval/ranker_eval_val.json")})


def evaluate_ensemble(baseline_cfg, baseline, candidates, experiment):
    cfg = deepcopy(baseline_cfg)
    for member, name in zip(cfg["ensemble"]["members"], CONFIGS):
        checked_evaluation(candidates[name])
        member.update(name=f"{name}_bce_pairwise_w050", config=CONFIGS[name])
    checkpoint_hashes = {name: _sha256(Path("runs") / c["run_name"] / "ranker/best.pt") for name, c in candidates.items()}
    context = {"experiment": experiment, "checkpoints": checkpoint_hashes}
    # This is a prespecified comparison, never a new weight search.
    _write_json(ROOT / "fixed_weights.json", {"primary_weight": .75, "secondary_weight": .25,
                                             "context_hash": _context_hash(context), "selected_before_inference": True})
    runtimes = None
    recency = None
    results = {}
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    settings = experiment["baseline_context"]["scoring"]
    for role in ("tune", "report"):
        path = ROOT / f"{role}_metrics.json"
        result = _load_stage(path, context)
        if result is None:
            device = resolve_device(cfg["ensemble"]["device"])
            if runtimes is None:
                log_device(device, "Fixed 75/25 weight-0.5 ensemble")
                runtimes = _load_member_runtimes(cfg, proc, device, settings["item_encoding_batch_size"])
                recency = _load_recency(cfg)
            result = _score_split(proc, cfg["ensemble"][f"{role}_split"], runtimes, np.array([.75]),
                                  settings["batch_size"], settings["candidate_buffer_size"], device, recency=recency,
                                  exact_rank_guard_threshold=settings["exact_rank_guard_threshold"],
                                  reference_batch_size=settings["reference_batch_size"])
        if (result["n_impressions"] != baseline[role]["n_impressions"]
                or result["n_scored_impressions"] != baseline[role]["n_scored_impressions"]
                or result["split"] != cfg["ensemble"][f"{role}_split"]
                or [r["primary_weight"] for r in result["metrics"]] != [.75]
                or not all(math.isfinite(result["metrics"][0][k]) for k in ("auc", "mrr", "ndcg@5", "ndcg@10"))):
            raise ValueError("Fixed-weight inference must match baseline splits, counts and finite metrics.")
        _write_json(path, {"context_hash": _context_hash(context), "result_hash": _context_hash(result), "result": result})
        reference = baseline["selected"] if role == "tune" else baseline["selected_report"]
        results[role] = {"candidate": result["metrics"][0], "baseline": reference,
                         "delta": {k: result["metrics"][0][k] - reference[k] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")}}
    _write_json(ROOT / "ensemble_comparison.json", {"weights": [.75, .25], "context": context, **results,
        "improved_auc_on_both_days": all(r["delta"]["auc"] > 0 for r in results.values()),
        "note": "Reused temporal validation. Baseline is the 0.25/0.25 pair associated with user-reported Large Test AUC 0.6990. No new Large Test score."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "ensemble"], default="all")
    parser.add_argument("--backbone", choices=["both", *CONFIGS], default="both", help="Applies to the training stage; ensemble always uses both students")
    parser.add_argument("--check", action="store_true", help="Read-only inputs and baseline checks; no training or inference")
    args = parser.parse_args()
    if args.stage == "all" and args.backbone != "both":
        parser.error("Use --stage train with a single --backbone; the ensemble stage requires both completed students.")
    baseline_cfg, baseline, originals, candidates, experiment = prepare()
    if args.stage == "ensemble":
        for cfg in candidates.values():
            checked_evaluation(cfg)
    if args.check:
        print("Preflight passed. Both candidate configs differ only in run name and 0.5 loss coefficient. No training or inference launched.")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(ROOT / "context.json", experiment)
    if args.stage in ("train", "all"):
        for name in CONFIGS if args.backbone == "both" else [args.backbone]:
            train_student(name, candidates[name], experiment)
            _write_json(ROOT / f"{name}_comparison.json", compare_students(originals[name], candidates[name]))
    if args.stage in ("ensemble", "all"):
        evaluate_ensemble(baseline_cfg, baseline, candidates, experiment)
    print(f"Results saved under {ROOT}. No maximum-data fits or hidden-test submission launched.")


if __name__ == "__main__":
    main()
