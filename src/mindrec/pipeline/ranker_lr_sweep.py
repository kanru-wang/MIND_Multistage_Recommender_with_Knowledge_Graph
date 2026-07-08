from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mindrec.config import ensure_dir
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.utils import load_json, save_json


def _lr_slug(lr: float) -> str:
    mantissa, exponent_text = f"{lr:.12e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    exponent = int(exponent_text)
    exponent_sign = "m" if exponent < 0 else ""
    return f"{mantissa}e{exponent_sign}{abs(exponent):02d}"


def _variant_run_name(base_run_name: str, lr: float) -> str:
    return f"{base_run_name}_ranker_lr_{_lr_slug(lr)}"


def _read_json_if_exists(path: Path) -> Any | None:
    return load_json(path) if path.exists() else None


def _result_for_run(run_name: str, lr: float, evaluate: bool) -> dict[str, Any]:
    runs_root = Path("runs") / run_name
    train_summary = _read_json_if_exists(runs_root / "ranker" / "train_summary.json")
    epochs = _read_json_if_exists(runs_root / "ranker" / "epochs.json")
    eval_result = (
        _read_json_if_exists(runs_root / "eval" / "ranker_eval_val.json")
        if evaluate
        else None
    )
    return {
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
    }


def _best_result(results: list[dict[str, Any]], metric_path: list[str]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_value = float("-inf")
    for result in results:
        current: Any = result
        for key in metric_path:
            if current is None:
                break
            current = current.get(key)
        if current is None:
            continue
        value = float(current)
        if value > best_value:
            best = result
            best_value = value
    return best


def run_train_ranker_lr_sweep(cfg: dict[str, Any]) -> None:
    sweep_cfg = dict(cfg.get("ranker", {}).get("lr_sweep", {}))
    lrs = [float(lr) for lr in sweep_cfg.get("lrs", [1.0e-3, 3.0e-4, 1.0e-4])]
    if not lrs:
        raise ValueError("ranker.lr_sweep.lrs must contain at least one learning rate.")

    base_run_name = str(cfg["run_name"])
    teacher_run_name = str(sweep_cfg.get("teacher_run_name", base_run_name))
    evaluate = bool(sweep_cfg.get("evaluate", True))
    summary_root = ensure_dir(Path("runs") / base_run_name / "tuning" / "ranker_lr_sweep")
    summary_path = summary_root / "sweep.json"

    results: list[dict[str, Any]] = []
    for lr in lrs:
        variant_cfg = deepcopy(cfg)
        run_name = _variant_run_name(base_run_name, lr)
        variant_cfg["run_name"] = run_name
        variant_cfg["ranker"]["lr"] = float(lr)
        variant_cfg.setdefault("artifacts", {})["teacher_run_name"] = teacher_run_name

        run_train_ranker(variant_cfg)
        if evaluate:
            run_evaluate(variant_cfg)

        results.append(_result_for_run(run_name=run_name, lr=lr, evaluate=evaluate))
        save_json(
            summary_path,
            {
                "base_run_name": base_run_name,
                "teacher_run_name": teacher_run_name,
                "lrs": lrs,
                "evaluate": evaluate,
                "results": results,
                "best_by_ranker_val_auc": _best_result(results, ["best_val_auc"]),
                "best_by_eval_auc": _best_result(results, ["ranking", "auc"]),
                "best_by_eval_ndcg10": _best_result(results, ["ranking", "ndcg@10"]),
            },
        )

    print(f"Wrote ranker LR sweep summary to {summary_path}")
