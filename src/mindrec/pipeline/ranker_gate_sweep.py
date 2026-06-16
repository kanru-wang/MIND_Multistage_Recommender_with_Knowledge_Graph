from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from mindrec.config import ensure_dir
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.utils import save_json, test_split_name, validation_split_name


def _gate_label(gate: float) -> str:
    return f"{gate:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def run_ranker_kg_gate_sweep(cfg: dict[str, Any]) -> None:
    sweep_cfg = dict(cfg.get("ranker", {}).get("kg_gate_sweep", {}))
    gates = [float(x) for x in sweep_cfg.get("values", [0.0, 0.025, 0.05, 0.1, 0.15])]
    if not gates:
        raise ValueError("ranker.kg_gate_sweep.values must contain at least one gate")
    if any(gate < 0.0 or gate > 1.0 for gate in gates):
        raise ValueError("Fixed KG gates must be between 0 and 1")

    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    sweep_root = ensure_dir(runs_root / "tuning" / "kg_gate_fixed")
    val_split = validation_split_name(cfg)
    test_split = test_split_name(cfg)
    rows: list[dict[str, Any]] = []

    for gate in gates:
        gate_cfg = deepcopy(cfg)
        gate_cfg.setdefault("ranker", {}).setdefault("dlrm", {})
        gate_cfg["ranker"]["dlrm"]["kg_gate_init"] = gate
        gate_cfg["ranker"]["dlrm"]["kg_gate_trainable"] = False
        gate_cfg.setdefault("eval", {})["report_splits"] = [val_split, test_split]

        gate_root = ensure_dir(sweep_root / f"gate_{_gate_label(gate)}")
        ranker_root = ensure_dir(gate_root / "ranker")
        eval_root = ensure_dir(gate_root / "eval")

        run_train_ranker(gate_cfg, ranker_art_root=ranker_root)
        reports = run_evaluate(
            gate_cfg,
            ranker_art_root=ranker_root,
            eval_out_root=eval_root,
        )
        val_ranking = reports[val_split]["ranking"]
        test_ranking = reports[test_split]["ranking"]
        row = {
            "kg_gate": gate,
            "kg_gate_trainable": False,
            "ranker_art_root": str(ranker_root),
            "eval_out_root": str(eval_root),
            "validation": val_ranking,
            "test": test_ranking,
        }
        rows.append(row)
        save_json(sweep_root / "sweep.json", {"results": rows})

    best = max(rows, key=lambda row: float(row["validation"]["ndcg@10"]))
    save_json(
        sweep_root / "sweep.json",
        {
            "selection_metric": "validation.ndcg@10",
            "best": best,
            "results": rows,
        },
    )
