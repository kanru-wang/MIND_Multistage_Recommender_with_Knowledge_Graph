"""Compare coefficient 0.75 against completed 0.5 students at fixed 75/25."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from tqdm import tqdm

# Historical runners are retained byte-for-byte because completed experiments
# fingerprint them. Reuse their coefficient-independent checks here.
from run_pairwise_weight_comparison import checked_evaluation, compare_students
from run_ranker_loss_comparison import completed_summary, preflight
from mindrec.config import load_config
from mindrec.pipeline.ensemble import _load_member_runtimes, _load_recency, _score_split, _sha256
from mindrec.pipeline.pairwise_ensemble import _context_hash, _write_json
from mindrec.pipeline.two_model_ensemble import _load_stage, _prepare
from mindrec.pipeline.two_model_final_fit import _verify_selection_stages, _weight050_evidence
from mindrec.utils import load_json, log_device, resolve_device


BASELINE_CONFIG = "configs/mind_large_ensemble_two_mixed_loss.yaml"
BASELINE_ROOT = Path("runs/mind_large_pairwise_w050_comparison_v1")
CONFIGS = {
    "mpnet": "configs/mind_large_temporal_mpnet_p1_pairwise_w075.yaml",
    "minilm": "configs/mind_large_temporal_minilm_pairwise_w075.yaml",
}
ROOT = Path("runs/mind_large_pairwise_w075_comparison_v1")


def validate_change(original, candidate):
    a, b = deepcopy(original), deepcopy(candidate)
    if a["ranker"]["pairwise_weight"] != .5 or b["ranker"]["pairwise_weight"] != .75:
        raise ValueError("Require the completed 0.5 student and a coefficient-0.75 candidate.")
    for cfg in (a, b):
        ranker = cfg["ranker"]
        selection = ranker["checkpoint_selection"]
        if (ranker["epochs"] != 8 or not ranker["grouped_batches"]
                or ranker["early_stopping"]["enabled"] or ranker["calibration"]["enabled"]
                or selection["metric"] != "mean_impression_auc" or not selection["track_both"]
                or not selection["preserve_training_rng"]):
            raise ValueError("Preserve the eight-epoch grouped-batch, dual-selector protocol.")
        cfg["ranker"].pop("pairwise_weight")
    if a.pop("run_name") == b.pop("run_name") or a != b:
        raise ValueError("Only a fresh run name and pairwise coefficient may change.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate["run_name"]):
        raise ValueError("Use a simple isolated candidate run name.")


def prepare():
    cfg = load_config(BASELINE_CONFIG)
    old_context, _, old_root = _prepare(cfg)
    old = load_json(old_root / "search.json")
    if (not old["selection_frozen"] or old["context_hash"] != _context_hash(old_context)
            or old["selection_context"] != old_context or old["selected"]["primary_weight"] != .75):
        raise ValueError("The original frozen two-model selection no longer matches its inputs.")
    _verify_selection_stages(old, old_context, old_root)
    context050, report_path, evidence_inputs = _weight050_evidence(
        {"weight_comparison_root": str(BASELINE_ROOT)}, old_context, old)
    report050 = load_json(report_path)
    baseline = {"selected": report050["tune"]["candidate"], "selected_report": report050["report"]["candidate"]}
    for role in ("tune", "report"):
        # The evidence verifier has already matched the 0.5 stage counts to
        # these original split counts; don't load the same stage files again.
        baseline[role] = {key: old[role][key] for key in ("n_impressions", "n_scored_impressions")}
    originals, candidates = {}, {name: load_config(path) for name, path in CONFIGS.items()}
    inputs = [report_path, *evidence_inputs]
    for name, member in zip(CONFIGS, context050["members"]):
        original = member["config"]
        validate_change(original, candidates[name])
        preflight(candidates[name])
        checked_evaluation(original)
        originals[name] = original
        inputs.append(Path("runs") / original["run_name"] / "eval/ranker_eval_val.json")
    outputs = [ROOT.resolve(), *[(Path("runs") / c["run_name"]).resolve() for c in candidates.values()]]
    if len(set(outputs)) != 3:
        raise ValueError("Comparison and both candidate students need distinct output directories.")
    # The 0.5 evidence verifier has already checked historical inputs. Snapshot
    # their current bytes plus this runner for reproducible stage reuse.
    inputs.extend(Path(p) for p in report050["context"]["experiment"]["inputs"])
    inputs.append(Path(__file__))
    experiment = {"baseline_context": context050, "baseline_report": report050,
                  "configs": candidates, "fixed_weights": [.75, .25],
                  "baseline_pairwise_weight": .5, "candidate_pairwise_weight": .75,
                  "inputs": {p.as_posix(): _sha256(p) for p in tqdm(sorted(set(inputs)), desc="Fingerprint 0.75 comparison inputs", unit="file")}}
    ownership = ROOT / "context.json"
    if ownership.exists():
        if load_json(ownership) != experiment:
            raise ValueError("Inputs or code changed; use new comparison and candidate output names.")
    elif ROOT.exists() or any((Path("runs") / c["run_name"]).exists() for c in candidates.values()):
        raise ValueError("Preserving unowned outputs; use new comparison and candidate output names.")
    for name, candidate in candidates.items():
        check_existing_student(name, candidate, experiment)
    return cfg, baseline, originals, candidates, experiment


def check_existing_student(name, cfg, experiment):
    """Check both resumable fits before any long-running stage can start."""
    root = Path("runs") / cfg["run_name"]
    marker = ROOT / f"{name}_evaluation.json"
    summary = root / "ranker/train_summary.json"
    saved_config = root / "ranker/training_config.json"
    if saved_config.exists() and load_json(saved_config) != cfg:
        raise ValueError("Saved candidate configuration changed; preserve it and use new output names.")
    if marker.exists() and not summary.exists():
        raise ValueError("A sealed evaluation requires completed training; refusing to overwrite the checkpoint.")
    if marker.exists():
        checked_evaluation(cfg)
        expected = {"experiment": _context_hash(experiment),
                    "checkpoint": _sha256(root / "ranker/best.pt"),
                    "evaluation": _sha256(root / "eval/ranker_eval_val.json")}
        if load_json(marker) != expected:
            raise ValueError("Completed candidate evaluation changed; preserve it and use new output names.")
    elif summary.exists():
        completed_summary(cfg)
    return summary.exists()


def train_student(name, cfg, experiment):
    root = Path("runs") / cfg["run_name"]
    if check_existing_student(name, cfg, experiment):
        print(f"Reusing completed coefficient-0.75 training: {name}", flush=True)
    else:
        print(f"Train {name}: BCE + 0.75 * pairwise + distillation; partial fits restart at epoch 1.", flush=True)
        subprocess.run([sys.executable, "-u", "-m", "mindrec.cli", "train_ranker", "--config", CONFIGS[name]], check=True)
        completed_summary(cfg)
    marker = ROOT / f"{name}_evaluation.json"
    path = root / "eval/ranker_eval_val.json"
    if not marker.exists():
        state = {"experiment": _context_hash(experiment), "checkpoint": _sha256(root / "ranker/best.pt")}
        subprocess.run([sys.executable, "-u", "-m", "mindrec.cli", "evaluate", "--config", CONFIGS[name]], check=True)
        checked_evaluation(cfg)
        _write_json(marker, {**state, "evaluation": _sha256(path)})


def evaluate_ensemble(baseline_cfg, baseline, candidates, experiment):
    cfg = deepcopy(baseline_cfg)
    for member, name in zip(cfg["ensemble"]["members"], CONFIGS):
        checked_evaluation(candidates[name])
        member.update(name=f"{name}_bce_pairwise_w075", config=CONFIGS[name])
    context = {"experiment": experiment,
               "checkpoints": {name: _sha256(Path("runs") / c["run_name"] / "ranker/best.pt") for name, c in candidates.items()}}
    fixed_path = ROOT / "fixed_weights.json"
    fixed = {"primary_weight": .75, "secondary_weight": .25,
             "selected_before_inference": True, "context_hash": _context_hash(context)}
    if fixed_path.exists():
        if load_json(fixed_path) != fixed:
            raise ValueError("Stale or damaged fixed-weight record; preserve it and use new output names.")
    else:
        _write_json(fixed_path, fixed)
    runtimes = recency = None
    results = {}
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    settings = experiment["baseline_context"]["scoring"]
    for role in ("tune", "report"):
        path = ROOT / f"{role}_metrics.json"
        result = _load_stage(path, context)
        if result is None:
            device = resolve_device(cfg["ensemble"]["device"])
            if runtimes is None:
                log_device(device, "Fixed 75/25 coefficient-0.75 ensemble")
                runtimes = _load_member_runtimes(cfg, proc, device, settings["item_encoding_batch_size"])
                recency = _load_recency(cfg)
            result = _score_split(proc, cfg["ensemble"][f"{role}_split"], runtimes, np.array([.75]),
                                  settings["batch_size"], settings["candidate_buffer_size"], device, recency=recency,
                                  exact_rank_guard_threshold=settings["exact_rank_guard_threshold"],
                                  reference_batch_size=settings["reference_batch_size"])
        if (result["split"] != cfg["ensemble"][f"{role}_split"]
                or result["n_impressions"] != baseline[role]["n_impressions"]
                or result["n_scored_impressions"] != baseline[role]["n_scored_impressions"]
                or [(r["primary_weight"], r["secondary_weight"]) for r in result["metrics"]] != [(.75, .25)]
                or not all(math.isfinite(result["metrics"][0][k]) for k in ("auc", "mrr", "ndcg@5", "ndcg@10"))):
            raise ValueError("Fixed-weight inference must match the baseline splits, counts and 75/25 weights.")
        if not path.exists():
            _write_json(path, {"context_hash": _context_hash(context), "result_hash": _context_hash(result), "result": result})
        reference = baseline["selected"] if role == "tune" else baseline["selected_report"]
        results[role] = {"candidate": result["metrics"][0], "baseline": reference,
                         "delta": {k: result["metrics"][0][k] - reference[k] for k in ("auc", "mrr", "ndcg@5", "ndcg@10")}}
    _write_json(ROOT / "ensemble_comparison.json", {"weights": [.75, .25], "context": context, **results,
        "improved_auc_on_both_days": all(r["delta"]["auc"] > 0 for r in results.values()),
        "note": "Coefficient 0.75 retains supervised BCE and distillation. Reused temporal validation against coefficient 0.5; the preserved Large Test champion is 0.7011. No new Large Test result."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "train", "ensemble"], default="all")
    parser.add_argument("--backbone", choices=["both", *CONFIGS], default="both")
    parser.add_argument("--check", action="store_true", help="Read-only verification; no training or inference")
    args = parser.parse_args()
    if args.stage != "train" and args.backbone != "both":
        parser.error("Use --stage train for a single backbone; the ensemble always requires both students.")
    cfg, baseline, originals, candidates, experiment = prepare()
    if args.check:
        if args.stage == "ensemble":
            for candidate in candidates.values():
                checked_evaluation(candidate)
        print("Preflight passed: two coefficient-0.75 fits versus retained coefficient-0.5 comparators. No training or inference launched.")
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(ROOT / "context.json", experiment)
    if args.stage in ("all", "train"):
        for name in CONFIGS if args.backbone == "both" else [args.backbone]:
            train_student(name, candidates[name], experiment)
            _write_json(ROOT / f"{name}_comparison.json", compare_students(originals[name], candidates[name]))
    if args.stage in ("all", "ensemble"):
        evaluate_ensemble(cfg, baseline, candidates, experiment)
    print(f"Results: {ROOT}. No maximum-data fit, hidden-test prediction or submission launched.")


if __name__ == "__main__":
    main()
