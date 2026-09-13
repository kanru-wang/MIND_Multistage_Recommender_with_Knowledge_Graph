from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.pipeline.sweep_utils import best_result, float_slug
from mindrec.utils import load_json, save_json, teacher_artifact_root


def _variant_run_name(base_run_name: str, lr: float) -> str:
    return f"{base_run_name}_ranker_lr_{float_slug(lr)}"


def _read_json_if_exists(path: Path) -> Any | None:
    return load_json(path) if path.exists() else None


@lru_cache(maxsize=None)
def _sha256_file_state(path_text: str, size: int, mtime_ns: int) -> str:
    del mtime_ns
    if not Path(path_text).is_file():
        raise FileNotFoundError(path_text)
    digest = hashlib.sha256()
    with (
        open(path_text, "rb") as handle,
        tqdm(
            total=size,
            desc=f"Fingerprint {Path(path_text).name}",
            unit="B",
            unit_scale=True,
            disable=size < 32 * 1024 * 1024,
        ) as progress,
    ):
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return _sha256_file_state(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _artifact_signature(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    exists = path.is_file()
    return {
        "path": str(path.resolve()),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "mtime_ns": path.stat().st_mtime_ns if exists else None,
        "sha256": _sha256_file(path) if exists and content_hash else None,
    }


def _ranker_training_provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a stable identity for a ranker sweep variant and its inputs."""
    ranker_cfg = deepcopy(cfg["ranker"])
    ranker_cfg.pop("lr_sweep", None)
    data_cfg = deepcopy(cfg["data"])
    proc_root = Path(data_cfg["processed_root"]) / str(data_cfg["dataset_name"])
    teacher_root = teacher_artifact_root(cfg)
    ranker_base_name = (
        "item_ranker_base_emb.npy"
        if bool(cfg.get("knowledge_graph", {}).get("enabled", False))
        else "item_base_emb.npy"
    )
    implementation_paths = [
        Path(__file__).with_name("ranker_train.py"),
        Path(__file__).with_name("hard_negative_sampling.py"),
        Path(__file__).parents[1] / "models" / "dlrm.py",
        Path(__file__).parents[1] / "models" / "distill.py",
        Path(__file__).parents[1] / "data" / "datasets.py",
    ]
    payload = {
        "schema_version": 1,
        "settings": {
            "ranker": ranker_cfg,
            "data": data_cfg,
            "knowledge_graph": deepcopy(cfg.get("knowledge_graph", {})),
            "eval": deepcopy(cfg.get("eval", {})),
        },
        "data_artifacts": {
            name: _artifact_signature(proc_root / name, content_hash=True)
            for name in (
                "preprocess_meta.json",
                "id_maps.json",
                "news.parquet",
                "train_pairs.parquet",
                "train_behaviors.parquet",
                "val_pairs.parquet",
                "val_behaviors.parquet",
                "val_impressions.parquet",
                "item_click_counts.json",
            )
        },
        "teacher_artifacts": {
            "root": str(teacher_root.resolve()),
            "meta": _artifact_signature(teacher_root / "meta.json", content_hash=True),
            "ranker_item_base": _artifact_signature(
                teacher_root / ranker_base_name
            ),
            "item_teacher": _artifact_signature(
                teacher_root / "item_teacher_emb.npy"
            ),
            "user_teacher": _artifact_signature(
                teacher_root / "user_teacher_emb.npy"
            ),
        },
        "implementation": {
            str(path.relative_to(Path(__file__).parents[2])): _sha256_file(path)
            for path in implementation_paths
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "payload": payload,
    }


def _completed_training_status(run_name: str, fingerprint: str) -> str:
    ranker_root = Path("runs") / run_name / "ranker"
    summary_path = ranker_root / "train_summary.json"
    checkpoint_path = ranker_root / "best.pt"
    completion_marker = ranker_root / "calibration_stats.json"
    if (
        not summary_path.exists()
        or not checkpoint_path.exists()
        or not completion_marker.exists()
    ):
        return "incomplete" if ranker_root.exists() and any(ranker_root.iterdir()) else "missing"
    summary = load_json(summary_path)
    if (
        summary.get("provenance_fingerprint") == fingerprint
        and summary.get("best_epoch") is not None
        and summary.get("stop_reason") is not None
    ):
        return "compatible"
    return "incompatible"


def _result_for_run(
    run_name: str,
    lr: float,
    evaluate: bool,
    *,
    status: str,
    provenance_fingerprint: str,
) -> dict[str, Any]:
    runs_root = Path("runs") / run_name
    train_summary = _read_json_if_exists(runs_root / "ranker" / "train_summary.json")
    epochs = _read_json_if_exists(runs_root / "ranker" / "epochs.json")
    eval_result = (
        _read_json_if_exists(runs_root / "eval" / "ranker_eval_val.json")
        if evaluate
        else None
    )
    return {
        "status": status,
        "lr": float(lr),
        "run_name": run_name,
        "ranker_train_summary_path": str(runs_root / "ranker" / "train_summary.json"),
        "ranker_epochs_path": str(runs_root / "ranker" / "epochs.json"),
        "eval_path": str(runs_root / "eval" / "ranker_eval_val.json") if evaluate else None,
        "best_epoch": None if train_summary is None else train_summary.get("best_epoch"),
        "best_val_auc": None if train_summary is None else train_summary.get("best_val_auc"),
        "stopped_epoch": None if train_summary is None else train_summary.get("stopped_epoch"),
        "ranking": None if eval_result is None else eval_result.get("ranking"),
        "n_eval_impressions": None if eval_result is None else eval_result.get("n_impressions"),
        "epochs": epochs,
        "provenance_fingerprint": provenance_fingerprint,
    }


def _best_result(results: list[dict[str, Any]], metric_path: list[str]) -> dict[str, Any] | None:
    return best_result(
        results,
        metric_path,
        eligible_statuses={"completed", "reused"},
    )


def run_train_ranker_lr_sweep(cfg: dict[str, Any]) -> None:
    sweep_cfg = dict(cfg.get("ranker", {}).get("lr_sweep", {}))
    lrs: list[float] = []
    for raw_lr in sweep_cfg.get("lrs", [1.0e-3, 3.0e-4, 1.0e-4]):
        lr = float(raw_lr)
        if not math.isfinite(lr) or lr <= 0.0:
            raise ValueError("ranker.lr_sweep.lrs must be finite and positive.")
        if lr not in lrs:
            lrs.append(lr)
    if not lrs:
        raise ValueError("ranker.lr_sweep.lrs must contain at least one learning rate.")

    base_run_name = str(cfg["run_name"])
    teacher_run_name = str(sweep_cfg.get("teacher_run_name", base_run_name))
    evaluate = bool(sweep_cfg.get("evaluate", True))
    reuse_completed = bool(sweep_cfg.get("reuse_completed", True))
    summary_root = ensure_dir(Path("runs") / base_run_name / "tuning" / "ranker_lr_sweep")
    summary_path = summary_root / "sweep.json"

    results: list[dict[str, Any]] = []
    for lr in lrs:
        variant_cfg = deepcopy(cfg)
        run_name = _variant_run_name(base_run_name, lr)
        variant_cfg["run_name"] = run_name
        variant_cfg["ranker"]["lr"] = float(lr)
        variant_cfg.setdefault("artifacts", {})["teacher_run_name"] = teacher_run_name
        provenance = _ranker_training_provenance(variant_cfg)
        variant_cfg["_runtime_provenance"] = {
            "ranker_training_fingerprint": provenance["fingerprint"],
            "ranker_training_payload": provenance["payload"],
        }
        existing = _completed_training_status(run_name, provenance["fingerprint"])
        if existing in {"incompatible", "incomplete"}:
            raise ValueError(
                f"Existing ranker sweep artifacts for {run_name} are {existing} and "
                "will not be overwritten. Use a new base run_name or archive the "
                "existing variant explicitly."
            )
        if existing == "compatible" and not reuse_completed:
            raise ValueError(
                f"Compatible completed artifacts already exist for {run_name}; "
                "set ranker.lr_sweep.reuse_completed: true or use a new run_name."
            )
        status = "reused" if existing == "compatible" else "completed"
        if existing == "compatible":
            print(f"Reusing compatible completed ranker {run_name}.", flush=True)
        else:
            print(f"Training ranker sweep variant {run_name}.", flush=True)
            run_train_ranker(variant_cfg)
            if _completed_training_status(run_name, provenance["fingerprint"]) != "compatible":
                raise RuntimeError(
                    f"Ranker {run_name} did not produce a compatible completed artifact."
                )
        if evaluate:
            eval_path = Path("runs") / run_name / "eval" / "ranker_eval_val.json"
            if status != "reused" or not eval_path.exists():
                run_evaluate(variant_cfg)

        results.append(
            _result_for_run(
                run_name=run_name,
                lr=lr,
                evaluate=evaluate,
                status=status,
                provenance_fingerprint=provenance["fingerprint"],
            )
        )
        save_json(
            summary_path,
            {
                "base_run_name": base_run_name,
                "teacher_run_name": teacher_run_name,
                "lrs": lrs,
                "evaluate": evaluate,
                "reuse_completed": reuse_completed,
                "results": results,
                "best_by_ranker_val_auc": _best_result(results, ["best_val_auc"]),
                "best_by_eval_auc": _best_result(results, ["ranking", "auc"]),
                "best_by_eval_ndcg10": _best_result(results, ["ranking", "ndcg@10"]),
            },
        )

    print(f"Wrote ranker LR sweep summary to {summary_path}")
