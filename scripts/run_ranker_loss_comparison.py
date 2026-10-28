"""Matched temporal BCE / BCE+pairwise experiment; invoke from the repo root."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

from mindrec.config import load_config
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.utils import load_json, save_json, teacher_artifact_root


CONFIGS = {
    "control": "configs/mind_large_temporal_mpnet_p1_dual_bce.yaml",
    "pairwise": "configs/mind_large_temporal_mpnet_p1_dual_pairwise.yaml",
}
REPORT = Path("runs/mind_large_temporal_mpnet_p1_loss_comparison_v1/comparison.json")
MINILM_CONFIGS = {
    "control": "configs/mind_large_temporal_minilm_dual_bce.yaml",
    "pairwise": "configs/mind_large_temporal_minilm_dual_pairwise.yaml",
}
MINILM_REPORT = Path("runs/mind_large_temporal_minilm_loss_comparison_v1/comparison.json")


def validate_matched_configs(configs):
    normalized = []
    for arm, cfg in configs.items():
        ranker = cfg["ranker"]
        selection = ranker.get("checkpoint_selection", {})
        if (not ranker.get("grouped_batches") or not selection.get("track_both")
                or not selection.get("preserve_training_rng")
                or selection.get("metric") != "mean_impression_auc"
                or ranker.get("early_stopping", {}).get("enabled", True)
                or int(ranker["epochs"]) < 1):
            raise ValueError("Comparison requires grouped batches, dual selectors, RNG preservation and a fixed epoch budget.")
        weight = float(ranker.get("pairwise_weight", 0))
        if not math.isfinite(weight) or (arm == "control" and weight != 0) or (arm == "pairwise" and weight <= 0):
            raise ValueError("Control must use zero pairwise weight and treatment a finite positive weight.")
        copy = deepcopy(cfg)
        copy.pop("run_name")
        copy["ranker"].pop("pairwise_weight", None)
        normalized.append(copy)
    if normalized[0] != normalized[1] or configs["control"]["run_name"] == configs["pairwise"]["run_name"]:
        raise ValueError("Arms must differ only in run_name and pairwise_weight.")


def completed_summary(cfg):
    root = Path("runs") / cfg["run_name"] / "ranker"
    if not (root / "training_config.json").exists():
        raise ValueError(f"Saved training configuration is missing: {root}")
    if load_json(root / "training_config.json") != cfg:
        raise ValueError(f"Current config differs from the saved training configuration: {root}")
    summary = load_json(root / "train_summary.json")
    if summary["stopped_epoch"] != cfg["ranker"]["epochs"] or summary["stop_reason"] != "max_epochs":
        raise ValueError(f"Run did not complete the configured fixed epoch budget: {root}")
    for metric in ["mean_impression_auc", "sampled_pair_auc"]:
        if not (root / f"best_{metric}.pt").is_file() or metric not in summary["best_by_metric"]:
            raise ValueError(f"Missing dual-selector checkpoint or metadata: {root}")
    return summary


def preflight(cfg):
    proc = Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]
    required = [proc / name for name in [
        "id_maps.json", "news.parquet", "train_behaviors.parquet",
        "item_click_counts.json", "val_impressions.parquet", "val_pairs.parquet",
    ]]
    required += [teacher_artifact_root(cfg) / name for name in [
        "model.pt", "item_ranker_base_emb.npy", "item_teacher_emb.npy",
    ]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(missing))
    print(f"Inputs ready: {cfg['run_name']}; epochs={cfg['ranker']['epochs']}; "
          f"pairwise_weight={cfg['ranker']['pairwise_weight']}")


def write_comparison(config_paths=None, report_path=None):
    configs = {arm: load_config(path) for arm, path in (CONFIGS if config_paths is None else config_paths).items()}
    report_path = REPORT if report_path is None else report_path
    validate_matched_configs(configs)
    results = {}
    for arm, cfg in configs.items():
        root = Path("runs") / cfg["run_name"]
        summary_path = root / "ranker/train_summary.json"
        if not summary_path.exists():
            continue
        summary = completed_summary(cfg)
        eval_path = root / "eval/ranker_eval_val.json"
        evaluation = load_json(eval_path) if eval_path.exists() else None
        selected = summary["best_by_metric"]
        if evaluation is not None and (
            evaluation.get("checkpoint_name") != "best.pt"
            or not math.isclose(evaluation["ranking"]["auc"], selected["mean_impression_auc"]["score"], abs_tol=1e-6)
        ):
            raise ValueError(f"Evaluation does not match the selected primary checkpoint: {root}")
        results[arm] = {
            "run_name": cfg["run_name"],
            "best_by_metric": selected,
            "selector_mean_auc_gain": selected["mean_impression_auc"]["metrics"]["mean_impression_auc"] - selected["sampled_pair_auc"]["metrics"]["mean_impression_auc"],
            "stopped_epoch": summary["stopped_epoch"],
            "ranking": evaluation["ranking"] if evaluation is not None else None,
            "n_impressions": evaluation["n_impressions"] if evaluation is not None else None,
            "n_scored_pairs": evaluation["n_scored_pairs"] if evaluation is not None else None,
        }
    report = {"arms": results, "note": "Temporal validation comparison; no hidden-test inference or submission."}
    if len(results) == 2:
        a, b = results["control"], results["pairwise"]
        if a["ranking"] is not None and b["ranking"] is not None:
            if (a["n_impressions"], a["n_scored_pairs"]) != (b["n_impressions"], b["n_scored_pairs"]):
                raise ValueError("Evaluation candidate counts differ between comparison arms.")
            report["pairwise_minus_control"] = {
                key: b["ranking"][key] - value for key, value in a["ranking"].items()
            }
    save_json(report_path, report)
    print(f"Comparison written to {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["mpnet", "minilm"], default="mpnet", help="Student/teacher backbone; defaults to the original MPNet experiment")
    parser.add_argument("--arm", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--check", action="store_true", help="Read-only input preflight; no training")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate already completed arms")
    args = parser.parse_args()
    config_paths = MINILM_CONFIGS if args.backbone == "minilm" else CONFIGS
    report_path = MINILM_REPORT if args.backbone == "minilm" else REPORT
    arms = list(config_paths) if args.arm == "all" else [args.arm]
    all_configs = {arm: load_config(path) for arm, path in config_paths.items()}
    validate_matched_configs(all_configs)
    configs = [all_configs[arm] for arm in arms]
    for cfg in configs:
        preflight(cfg)
        root = Path("runs") / cfg["run_name"] / "ranker"
        if not args.check:
            if args.evaluate_only:
                if not (root / "train_summary.json").exists():
                    raise FileNotFoundError(f"Training is not complete: {root}")
                completed_summary(cfg)
            elif root.exists():
                raise FileExistsError(
                    f"Preserving existing output {root}. Use --evaluate-only, "
                    "--arm for the other arm, or a new run_name."
                )
    if args.check:
        print("Preflight passed. No training or evaluation launched.")
        return
    for cfg in configs:
        if not args.evaluate_only:
            run_train_ranker(cfg)
        run_evaluate(cfg)
        write_comparison(config_paths, report_path)


if __name__ == "__main__":
    main()
