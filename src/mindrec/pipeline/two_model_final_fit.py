"""Fixed two-epoch fits and frozen 75/25 submission for two mixed-loss students."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from tqdm import tqdm

from mindrec.config import load_config
from mindrec.data.item_age import item_age_artifact_path
from mindrec.data.recency_tiebreaker import recency_tiebreaker_config
from mindrec.pipeline.ensemble import (_prediction_stream, _publish_submission, _scoring_settings,
                                      _sha256, _write_ensemble_submission)
from mindrec.pipeline.pairwise_ensemble import _context_hash, _write_json
from mindrec.pipeline.two_model_ensemble import _load_stage, _prepare
from mindrec.utils import load_json as _read, teacher_artifact_root


def _verify_selection_stages(report, context, root):
    """Bind the final-fit decision to the completed, checksummed search stages."""
    tune = _load_stage(root / "tune_metrics.json", context)
    reporting = _load_stage(root / "report_metrics.json", context)
    if tune is None or reporting is None or tune != report["tune"] or reporting != report["report"]:
        raise ValueError("Final-fit selection must match both completed temporal inference stages.")
    best = max(tune["metrics"], key=lambda row: (row["auc"], row["ndcg@10"], row["primary_weight"]))
    selected_report = next((row for row in reporting["metrics"]
                            if row["primary_weight"] == best["primary_weight"]), None)
    if best != report["selected"] or selected_report != report["selected_report"]:
        raise ValueError("Final-fit selection or reporting metrics differ from the frozen search decision.")


def _prepare_final_fit(cfg):
    settings = cfg["final_fit"]
    temporal_cfg = load_config(settings["temporal_config"])
    temporal_context, _, temporal_root = _prepare(temporal_cfg)
    report_path = temporal_root / "search.json"
    report, selection = _read(report_path), _read(temporal_root / "selection.json")
    if (not report.get("selection_frozen") or not selection.get("selection_frozen")
            or not report.get("improved_auc_on_both_days")
            or report["selection_context"] != temporal_context
            or report["context_hash"] != _context_hash(temporal_context)
            or selection["context_hash"] != report["context_hash"]
            or selection["selected"] != report["selected"]
            or report["selected"]["primary_weight"] != .75
            or report["selected"]["secondary_weight"] != .25):
        raise ValueError("Require the completed matching two-model selection frozen at 75/25.")
    _verify_selection_stages(report, temporal_context, temporal_root)
    students = settings["students"]
    if len(students) != 2 or settings["expected_impressions"] != 2370727:
        raise ValueError("Require exactly two students and 2,370,727 hidden-test impressions.")
    inputs = [report_path, temporal_root / "selection.json"]
    configs, roots = [], []
    protected = {temporal_cfg["run_name"]}
    for student, temporal in zip(students, temporal_context["members"]):
        final = load_config(student["config"])
        reference_path = Path("runs") / student["reference_ranker_run"] / "ranker/best.pt"
        reference = torch.load(reference_path, map_location="cpu", weights_only=False)["cfg"]
        expected = deepcopy(reference["ranker"])
        expected.update(grouped_batches=True, pairwise_weight=.25)
        if (student["name"] != temporal["name"] or final["ranker"] != expected
                or final["ranker"]["epochs"] != 2
                or final["ranker"]["early_stopping"]["enabled"]
                or final["ranker"]["calibration"]["enabled"]
                or final["data"] != reference["data"] or final["data"]["mode"] != "leaderboard_submission"
                or final["knowledge_graph"] != reference["knowledge_graph"]
                or final["teacher"] != reference["teacher"]
                or final["artifacts"].get("teacher_run_name") != reference["artifacts"]["teacher_run_name"]
                or "ranker_run_name" in final["artifacts"]):
            raise ValueError("Preserve each final teacher and two-epoch reference settings, adding only the mixed loss and grouped batches.")
        for key in ("batch_size", "lr", "weight_decay", "dlrm", "distill", "hard_negative_sampling", "pairwise_weight", "grouped_batches"):
            value, target = deepcopy(final["ranker"][key]), deepcopy(temporal["config"]["ranker"][key])
            if key == "distill":
                value.setdefault("representation_target", "full")
                target.setdefault("representation_target", "full")
            if value != target:
                raise ValueError(f"Final student differs from selected temporal settings: {key}")
        if (recency_tiebreaker_config(final) != temporal_context["recency"]
                or _scoring_settings(final["submission"]) != temporal_context["scoring"]
                or final["submission"].get("save_scores", False)):
            raise ValueError("Use frozen recency and scoring settings for both students.")
        proc = Path(final["data"]["processed_root"]) / final["data"]["dataset_name"]
        meta = _read(proc / "preprocess_meta.json")
        if (meta["mode"] != "leaderboard_submission" or meta["n_submission_impressions"] != 2370727
                or meta["n_train_impressions"] != meta["n_original_train_impressions"] + meta["n_original_dev_impressions"]):
            raise ValueError("Require existing Train+Dev final-fit preprocessing.")
        teacher = teacher_artifact_root(final)
        teacher_meta = _read(teacher / "meta.json")
        if teacher_meta["selection_mode"] != "fixed_epoch" or teacher_meta["stopped_epoch"] != 4:
            raise ValueError("Require the retained completed four-epoch final teacher.")
        inputs.extend([Path(student["config"]), reference_path,
                       *[proc / name for name in ("preprocess_meta.json", "id_maps.json", "news.parquet", "train_behaviors.parquet", "item_click_counts.json")],
                       item_age_artifact_path(proc),
                       Path(final["data"]["raw_root"]) / final["data"]["test_dir"] / "behaviors.tsv",
                       *[teacher / name for name in ("meta.json", "model.pt", "item_ranker_base_emb.npy", "item_teacher_emb.npy")]])
        roots.append(Path("runs") / final["run_name"])
        configs.append(final)
        protected.update((student["reference_ranker_run"], final["artifacts"]["teacher_run_name"], temporal["config"]["run_name"]))
    if configs[0]["data"] != configs[1]["data"]:
        raise ValueError("Both students must use identical final-fit data and candidate order.")
    root = Path("runs") / cfg["run_name"]
    all_roots = [root, *roots]
    names = [cfg["run_name"], *[c["run_name"] for c in configs]]
    if len(set(all_roots)) != 3 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in protected for name in names):
        raise ValueError("Use three distinct isolated output run names to protect source artifacts.")
    package = Path(__file__).resolve().parents[1]
    inputs.extend(sorted(package.rglob("*.py")))
    inputs = sorted(set(inputs))
    context = {"config": cfg, "students": configs, "weights": [.75, .25],
               "selection_artifact": report_path.as_posix(), "selection_sha256": _sha256(report_path),
               "inputs": {p.as_posix(): _sha256(p) for p in tqdm(inputs, desc="Fingerprint final-fit inputs", unit="file")}}
    ownership = root / "workflow/context.json"
    if ownership.exists():
        if _read(ownership) != context:
            raise ValueError("Inputs or code changed; use new final-fit output run names.")
    elif any(p.exists() for p in all_roots):
        raise ValueError("Unowned final-fit output exists; use new run names to preserve it.")
    return context, configs, root


def _content_hash(path):
    digest = hashlib.sha256()
    with ExitStack() as stack:
        stream = _prediction_stream(stack, path)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_member_metadata(metadata, context, count):
    recency = context["recency"]
    if (metadata["n_impressions"] != count or metadata["split_name"] != "submission_test"
            or any(metadata["posthoc_recency"].get(k) != recency[k] for k in ("enabled", "alpha", "age_dataset_name"))
            or _scoring_settings(metadata) != context["scoring"]):
        raise ValueError("Member predictions must use the frozen hidden-test scoring and recency settings.")


def _completed(marker, context):
    if not marker.exists():
        return False
    saved = _read(marker)
    if saved["context_hash"] != _context_hash(context) or any(
            not Path(p).is_file() or _sha256(Path(p)) != digest for p, digest in saved["outputs"].items()):
        raise ValueError(f"Completed stage was changed or damaged: {marker}")
    print(f"Reusing verified stage: {marker.stem}", flush=True)
    return True


def _seal(marker, context, paths):
    _write_json(marker, {"context_hash": _context_hash(context),
                         "outputs": {p.as_posix(): _sha256(p) for p in paths}})


def _verify_training(cfg, root):
    art = root / "ranker"
    summary = _read(art / "train_summary.json")
    checkpoint = torch.load(art / "best.pt", map_location="cpu", weights_only=False)
    if (summary["has_validation"] or summary["selection_mode"] != "fixed_epoch"
            or summary["stopped_epoch"] != 2 or summary["best_epoch"] != 2
            or summary["stop_reason"] != "max_epochs" or not summary["grouped_batches"]
            or summary["pairwise_weight"] != .25 or checkpoint["epoch"] != 2
            or checkpoint["selection_mode"] != "fixed_epoch" or checkpoint["cfg"] != cfg
            or _read(art / "training_config.json") != cfg):
        raise ValueError("Final pairwise training did not complete the prescribed two-epoch fit.")
    return [art / name for name in ("best.pt", "train_summary.json", "training_config.json", "epochs.json")]


def _fuse(cfg, configs, context, root):
    members = [{"name": s["name"], "prediction_zip": (Path("runs") / c["run_name"] / "submission/prediction.zip").as_posix()}
               for s, c in zip(cfg["final_fit"]["students"], configs)]
    output = root / "submission"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".two-model-", dir=output) as directory:
        staging = Path(directory)
        fusion_cfg = {"ensemble": {"members": members, "expected_impressions": 2370727,
                                   "selection_artifact": context["selection_artifact"]}}
        _write_ensemble_submission(fusion_cfg, .75, staging, output)
        meta_path = staging / "submission_meta.json"
        metadata = _read(meta_path)
        metadata.update(context_hash=_context_hash(context), selection_sha256=context["selection_sha256"],
                        effective_weights=[.75, .25], leaderboard_status="awaiting_manual_upload_and_score",
                        preserved_champion_auc=.6972)
        _write_json(meta_path, metadata)
        _publish_submission(staging, output)
    return [output / "prediction.zip", output / "submission_meta.json"]


def run_two_model_final_fit(cfg, *, check_only=False):
    print("Checking the frozen 75/25 selection, final teachers and Train+Dev inputs...", flush=True)
    context, configs, root = _prepare_final_fit(cfg)
    if check_only:
        print("Preflight passed. Two fixed two-epoch student fits; no training or scoring launched.")
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
                _verify_member_metadata(_read(meta), {"recency": recency_tiebreaker_config(student),
                                                     "scoring": _scoring_settings(student["submission"])}, 2370727)
                outputs = [meta, student_root / "submission/prediction.zip"]
                _content_hash(outputs[-1])
            _seal(marker, context, outputs)
    if not _completed(workflow / "fusion.json", context):
        _seal(workflow / "fusion.json", context, _fuse(cfg, configs, context, root))
    print(f"Ready for manual competition upload: {root / 'submission/prediction.zip'}\nRecord the platform AUC against the preserved 0.6972 champion.", flush=True)


