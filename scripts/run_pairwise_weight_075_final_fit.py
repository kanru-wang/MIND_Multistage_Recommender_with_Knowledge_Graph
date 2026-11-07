"""Two fixed-epoch coefficient-0.75 students and a separate 75/25 submission."""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path
import re
import subprocess
import sys

import torch
from tqdm import tqdm

import run_pairwise_weight_075_comparison as temporal
from mindrec.config import load_config
from mindrec.data.item_age import item_age_artifact_path
from mindrec.data.recency_tiebreaker import recency_tiebreaker_config
from mindrec.pipeline.ensemble import _scoring_settings, _sha256
from mindrec.pipeline.pairwise_ensemble import _context_hash, _write_json
from mindrec.pipeline.two_model_ensemble import _load_stage
from mindrec.pipeline.two_model_final_fit import (
    _completed, _content_hash, _fuse, _seal, _verify_member_metadata, _verify_training,
)
from mindrec.utils import load_json, teacher_artifact_root


CONFIG = "configs/mind_large_two_model_final_fit_w075.yaml"


def temporal_evidence(settings):
    """Bind promotion to the exact completed 0.75 experiment and both dates."""
    _, baseline, _, candidates, experiment = temporal.prepare()
    root = Path(settings["weight_comparison_root"])
    if root.resolve() != temporal.ROOT.resolve():
        raise ValueError("Use the completed coefficient-0.75 comparison directory.")
    path = root / "ensemble_comparison.json"
    report = load_json(path)
    context = {"experiment": experiment, "checkpoints": {
        name: _sha256(Path("runs") / c["run_name"] / "ranker/best.pt")
        for name, c in candidates.items()}}
    fixed = {"primary_weight": .75, "secondary_weight": .25,
             "selected_before_inference": True, "context_hash": _context_hash(context)}
    if (report["context"] != context or report["weights"] != [.75, .25]
            or not report["improved_auc_on_both_days"]
            or load_json(root / "fixed_weights.json") != fixed):
        raise ValueError("Require completed coefficient-0.75 gains on both dates at frozen 75/25.")
    for role in ("tune", "report"):
        stage = _load_stage(root / f"{role}_metrics.json", context)
        reference = baseline["selected" if role == "tune" else "selected_report"]
        if (stage is None or len(stage["metrics"]) != 1
                or stage["metrics"][0] != report[role]["candidate"]
                or stage["metrics"][0]["primary_weight"] != .75
                or stage["metrics"][0]["secondary_weight"] != .25
                or stage["split"] != experiment["baseline_context"]["config"]["ensemble"][f"{role}_split"]
                or any(stage[k] != baseline[role][k] for k in ("n_impressions", "n_scored_impressions"))
                or report[role]["baseline"] != reference
                or not all(math.isfinite(stage["metrics"][0][k]) and math.isclose(
                    report[role]["delta"][k], stage["metrics"][0][k] - reference[k], rel_tol=0, abs_tol=1e-12)
                    for k in ("auc", "mrr", "ndcg@5", "ndcg@10"))
                or report[role]["delta"]["auc"] <= 0):
            raise ValueError("0.75 report must match both completed inference stages and improve AUC.")
    inputs = [root / name for name in ("context.json", "ensemble_comparison.json", "fixed_weights.json",
              "tune_metrics.json", "report_metrics.json", "mpnet_evaluation.json", "minilm_evaluation.json")]
    return candidates, experiment["baseline_context"], context, inputs


def validate_student(final, reference, selected):
    expected = deepcopy(reference)
    expected["run_name"] = final["run_name"]
    expected["ranker"]["pairwise_weight"] = .75
    if (final != expected or reference["ranker"]["pairwise_weight"] != .5
            or final["data"]["mode"] != "leaderboard_submission"
            or final["ranker"]["epochs"] != 2 or not final["ranker"]["grouped_batches"]
            or final["ranker"]["early_stopping"]["enabled"]
            or final["ranker"]["calibration"]["enabled"]
            or "ranker_run_name" in final["artifacts"] or final["submission"].get("save_scores", False)):
        raise ValueError("Change only the final 0.5 student's run name and coefficient; retain two fixed epochs.")
    for key in ("batch_size", "lr", "weight_decay", "dlrm", "distill", "hard_negative_sampling", "pairwise_weight", "grouped_batches"):
        value, target = deepcopy(final["ranker"][key]), deepcopy(selected["ranker"][key])
        if key == "distill":
            value.setdefault("representation_target", "full")
            target.setdefault("representation_target", "full")
        if value != target:
            raise ValueError(f"Final student differs from the selected temporal settings: {key}")


def prepare(config_path=CONFIG):
    cfg = load_config(config_path)
    settings = cfg["final_fit"]
    if settings["pairwise_weight"] != .75 or settings["expected_impressions"] != 2370727 or len(settings["students"]) != 2:
        raise ValueError("Require exactly two coefficient-0.75 students and 2,370,727 test impressions.")
    candidates, scoring_context, evidence, inputs = temporal_evidence(settings)
    inputs.extend([Path(config_path), Path(__file__)])
    configs = []
    protected = {settings["preserved_champion_run"], temporal.ROOT.name}
    for (name, selected), member in zip(candidates.items(), settings["students"]):
        if member["name"] != f"{name}_bce_pairwise_w075":
            raise ValueError("Use MPNet first and MiniLM second, both with coefficient 0.75.")
        final = load_config(member["config"])
        reference_path = Path("runs") / member["reference_ranker_run"] / "ranker/best.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)["cfg"]
        if reference["run_name"] != member["reference_ranker_run"]:
            raise ValueError("Reference checkpoint must belong to the retained coefficient-0.5 final student.")
        validate_student(final, reference, selected)
        if (recency_tiebreaker_config(final) != scoring_context["recency"]
                or _scoring_settings(final["submission"]) != scoring_context["scoring"]):
            raise ValueError("Preserve the temporal ensemble's recency and scoring settings.")
        proc = Path(final["data"]["processed_root"]) / final["data"]["dataset_name"]
        meta = load_json(proc / "preprocess_meta.json")
        if (meta["mode"] != "leaderboard_submission" or meta["n_submission_impressions"] != 2370727
                or meta["n_train_impressions"] != meta["n_original_train_impressions"] + meta["n_original_dev_impressions"]):
            raise ValueError("Require existing maximum-data Train+Dev preprocessing.")
        teacher = teacher_artifact_root(final)
        teacher_meta = load_json(teacher / "meta.json")
        if teacher_meta["selection_mode"] != "fixed_epoch" or teacher_meta["stopped_epoch"] != 4:
            raise ValueError("Reuse the completed four-epoch final teacher.")
        inputs.extend([Path(member["config"]), reference_path,
                       *[proc / n for n in ("preprocess_meta.json", "id_maps.json", "news.parquet", "train_behaviors.parquet", "item_click_counts.json")],
                       item_age_artifact_path(proc),
                       Path(final["data"]["raw_root"]) / final["data"]["test_dir"] / "behaviors.tsv",
                       *[teacher / n for n in ("meta.json", "model.pt", "item_ranker_base_emb.npy", "item_teacher_emb.npy")]])
        protected.update((reference["run_name"], selected["run_name"], final["artifacts"]["teacher_run_name"]))
        configs.append(final)
    if configs[0]["data"] != configs[1]["data"]:
        raise ValueError("Both students must use identical final-fit data and candidate order.")
    names = [cfg["run_name"], *[c["run_name"] for c in configs]]
    if len(set(names)) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) or n in protected for n in names):
        raise ValueError("Use three distinct isolated output names to protect source artifacts.")
    root = Path("runs") / cfg["run_name"]
    context = {"config": cfg, "students": configs, "weights": [.75, .25], "temporal_evidence": evidence,
               "selection_artifact": (temporal.ROOT / "ensemble_comparison.json").as_posix(),
               "selection_sha256": _sha256(temporal.ROOT / "ensemble_comparison.json"),
               "inputs": {p.as_posix(): _sha256(p) for p in tqdm(sorted(set(inputs)), desc="Fingerprint 0.75 final-fit inputs", unit="file")}}
    ownership = root / "workflow/context.json"
    if ownership.exists():
        if load_json(ownership) != context:
            raise ValueError("Inputs or code changed; use new final-fit output names.")
    elif any((Path("runs") / n).exists() for n in names):
        raise ValueError("Unowned final-fit output exists; preserve it and use new output names.")
    fusion = root / "workflow/fusion.json"
    for member, final in zip(settings["students"], configs):
        saved = Path("runs") / final["run_name"] / "ranker/training_config.json"
        if saved.exists() and load_json(saved) != final:
            raise ValueError("Existing final training configuration changed.")
        train, score = [root / "workflow" / f"{member['name']}_{s}.json" for s in ("train", "score")]
        if score.exists() and not train.exists():
            raise ValueError("A completed scoring stage requires its training completion record.")
        if fusion.exists() and not score.exists():
            raise ValueError("Completed fusion requires both student scoring completion records.")
        for marker in (train, score):
            _completed(marker, context)
    _completed(fusion, context)
    return cfg, context, configs, root


def run(config_path=CONFIG, *, check_only=False):
    cfg, context, configs, root = prepare(config_path)
    if check_only:
        print("Preflight passed: two fixed two-epoch coefficient-0.75 fits and 75/25 fusion. No training, scoring or writes.")
        return
    workflow = root / "workflow"
    workflow.mkdir(parents=True, exist_ok=True)
    _write_json(workflow / "context.json", context)
    for member, student in zip(cfg["final_fit"]["students"], configs):
        student_root = Path("runs") / student["run_name"]
        for command, stage in (("train_ranker", "train"), ("write_submission", "score")):
            marker = workflow / f"{member['name']}_{stage}.json"
            if _completed(marker, context):
                continue
            print(f"Starting {member['name']}: {command}. Incomplete stages restart from the beginning.", flush=True)
            subprocess.run([sys.executable, "-u", "-m", "mindrec.cli", command, "--config", member["config"]], check=True)
            if stage == "train":
                outputs = _verify_training(student, student_root)
            else:
                meta = student_root / "submission/submission_meta.json"
                _verify_member_metadata(load_json(meta), {"recency": recency_tiebreaker_config(student),
                                                         "scoring": _scoring_settings(student["submission"])}, 2370727)
                outputs = [meta, student_root / "submission/prediction.zip"]
                _content_hash(outputs[-1])
            _seal(marker, context, outputs)
    if not _completed(workflow / "fusion.json", context):
        _seal(workflow / "fusion.json", context, _fuse(cfg, configs, context, root))
    print(f"Ready for manual competition upload: {root / 'submission/prediction.zip'}\nCompare Large Test AUC with the preserved {cfg['final_fit']['preserved_champion_auc']:.4f} champion.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--check", action="store_true", help="Read-only verification; no training or scoring")
    args = parser.parse_args()
    run(args.config, check_only=args.check)


if __name__ == "__main__":
    main()
