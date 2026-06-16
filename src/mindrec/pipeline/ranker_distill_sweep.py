from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from mindrec.config import ensure_dir
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.utils import save_json, test_split_name, validation_split_name


def _value_label(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def distill_sweep_grid(cfg: dict[str, Any]) -> list[tuple[float, float]]:
    sweep_cfg = dict(cfg.get("ranker", {}).get("distill_sweep", {}))
    lambda_logits = [
        float(x) for x in sweep_cfg.get("lambda_logits", [0.2, 0.4, 0.7])
    ]
    temperatures = [
        float(x) for x in sweep_cfg.get("temperatures", [0.5, 1.0, 2.0])
    ]
    if not lambda_logits:
        raise ValueError("ranker.distill_sweep.lambda_logits must not be empty")
    if not temperatures:
        raise ValueError("ranker.distill_sweep.temperatures must not be empty")
    if any(value < 0.0 for value in lambda_logits):
        raise ValueError("Final-logit distillation weights must be non-negative")
    if any(value <= 0.0 for value in temperatures):
        raise ValueError("Distillation temperatures must be positive")
    return list(product(lambda_logits, temperatures))


def select_best_distill_result(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select a best result from an empty distillation sweep")
    return max(rows, key=lambda row: float(row["validation"]["ndcg@10"]))


def run_ranker_distill_sweep(cfg: dict[str, Any]) -> None:
    grid = distill_sweep_grid(cfg)
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    sweep_root = ensure_dir(runs_root / "tuning" / "distill_final_logit")
    val_split = validation_split_name(cfg)
    test_split = test_split_name(cfg)
    lambda_repr = float(cfg["ranker"]["distill"].get("lambda_repr", 0.05))
    rows: list[dict[str, Any]] = []

    for lambda_logit, temperature in grid:
        trial_cfg = deepcopy(cfg)
        distill_cfg = trial_cfg.setdefault("ranker", {}).setdefault("distill", {})
        distill_cfg["lambda_logit"] = lambda_logit
        distill_cfg["temperature"] = temperature
        distill_cfg["lambda_repr"] = lambda_repr
        trial_cfg.setdefault("eval", {})["report_splits"] = [val_split, test_split]

        trial_name = (
            f"lambda_{_value_label(lambda_logit)}"
            f"__temp_{_value_label(temperature)}"
        )
        trial_root = ensure_dir(sweep_root / trial_name)
        ranker_root = ensure_dir(trial_root / "ranker")
        eval_root = ensure_dir(trial_root / "eval")

        run_train_ranker(trial_cfg, ranker_art_root=ranker_root)
        reports = run_evaluate(
            trial_cfg,
            ranker_art_root=ranker_root,
            eval_out_root=eval_root,
        )
        row = {
            "lambda_logit": lambda_logit,
            "temperature": temperature,
            "lambda_repr": lambda_repr,
            "ranker_art_root": str(ranker_root),
            "eval_out_root": str(eval_root),
            "validation": reports[val_split]["ranking"],
            "test": reports[test_split]["ranking"],
        }
        rows.append(row)
        save_json(
            sweep_root / "sweep.json",
            {
                "selection_metric": "validation.ndcg@10",
                "results": rows,
            },
        )

    best = select_best_distill_result(rows)
    save_json(
        sweep_root / "sweep.json",
        {
            "selection_metric": "validation.ndcg@10",
            "best": best,
            "results": rows,
        },
    )
