from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mindrec.config import ensure_dir, load_config
from mindrec.pipeline.sweep_utils import best_result, float_slug
from mindrec.pipeline.text_encoder_adapt import (
    run_adapt_text_encoder,
    text_adaptation_provenance,
)
from mindrec.utils import load_json, save_json


def _variant_run_name(
    base_run_name: str,
    lr: float,
    temperature: float,
    max_updates: int,
    validation_interval: int,
) -> str:
    return (
        f"{base_run_name}_ta_lr_{float_slug(lr)}_t_{float_slug(temperature)}"
        f"_u{max_updates}_v{validation_interval}"
    )


def _unique_positive_floats(values: list[Any], field: str) -> list[float]:
    out: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} values must be finite and greater than zero.")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"{field} must contain at least one value.")
    return out


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    return best_result(
        results,
        ["best_val_impression_auc"],
        eligible_statuses={"completed", "reused"},
    )


def _result_for_run(
    run_name: str,
    lr: float,
    temperature: float,
    stage: str,
    status: str,
) -> dict[str, Any]:
    root = Path("runs") / run_name / "text_encoder"
    meta_path = root / "meta.json"
    history_path = root / "validation_history.json"
    meta = load_json(meta_path)
    history = load_json(history_path)
    return {
        "stage": stage,
        "status": status,
        "lr": float(lr),
        "temperature": float(temperature),
        "run_name": run_name,
        "meta_path": str(meta_path),
        "validation_history_path": str(history_path),
        "model_path": str(root / "model"),
        "best_update": meta.get("best_update"),
        "best_val_impression_auc": meta.get("best_val_impression_auc"),
        "completed_optimizer_updates": meta.get("completed_optimizer_updates"),
        "stop_reason": meta.get("stop_reason"),
        "provenance_fingerprint": meta.get("provenance_fingerprint"),
        "validation_history": history,
    }


def _completed_variant_status(
    run_name: str,
    *,
    expected_provenance_fingerprint: str,
) -> str:
    """Return missing, compatible, or incompatible for an existing variant."""
    root = Path("runs") / run_name / "text_encoder"
    meta_path = root / "meta.json"
    model_root = root / "model"
    model_path = model_root / "modules.json"
    history_path = root / "validation_history.json"
    has_weights = model_root.exists() and any(
        path.is_file()
        for pattern in ("*.safetensors", "*.bin")
        for path in model_root.rglob(pattern)
    )
    if (
        not meta_path.exists()
        or not model_path.exists()
        or not has_weights
        or not history_path.exists()
    ):
        return "missing"

    meta = load_json(meta_path)
    compatible = (
        meta.get("provenance_fingerprint") == expected_provenance_fingerprint
        and bool(meta.get("early_stopping_enabled", False))
        and meta.get("best_update") is not None
        and meta.get("best_val_impression_auc") is not None
    )
    return "compatible" if compatible else "incompatible"


def _variant_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    lr: float,
    temperature: float,
    max_updates: int,
    validation_interval: int,
    patience: int,
    min_delta: float,
) -> dict[str, Any]:
    variant_cfg = deepcopy(cfg)
    variant_cfg["run_name"] = run_name
    variant_adaptation = variant_cfg["teacher"]["text_adaptation"]
    variant_adaptation["lr"] = float(lr)
    variant_adaptation["temperature"] = float(temperature)
    variant_adaptation["max_optimizer_updates"] = max_updates
    variant_adaptation["early_stopping"] = {
        "enabled": True,
        "validation_interval_updates": validation_interval,
        "patience": patience,
        "min_delta": min_delta,
    }
    variant_adaptation.pop("sweep", None)
    return variant_cfg


def _summary_payload(
    *,
    base_run_name: str,
    strategy: str,
    lrs: list[float],
    temperatures: list[float],
    baseline_temperature: float,
    max_updates: int,
    validation_interval: int,
    patience: int,
    min_delta: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    planned_variant_count = _planned_variant_count(
        strategy, lrs, temperatures, baseline_temperature
    )
    completed_variant_count = len(
        {
            (float(result["lr"]), float(result["temperature"]))
            for result in results
            if result.get("status") in {"completed", "reused"}
        }
    )
    return {
        "base_run_name": base_run_name,
        "strategy": strategy,
        "lrs": lrs,
        "temperatures": temperatures,
        "baseline_temperature": baseline_temperature,
        "max_optimizer_updates": max_updates,
        "validation_interval_updates": validation_interval,
        "early_stopping_patience": patience,
        "early_stopping_min_delta": min_delta,
        "selection_metric": "history_mean_candidate_cosine_impression_auc",
        "planned_variant_count": planned_variant_count,
        "completed_variant_count": completed_variant_count,
        "sweep_complete": completed_variant_count >= planned_variant_count,
        "results": results,
        "best_by_val_impression_auc": _best_result(results),
    }


def _planned_variant_count(
    strategy: str,
    lrs: list[float],
    temperatures: list[float],
    baseline_temperature: float,
) -> int:
    if strategy == "grid":
        return len(lrs) * len(temperatures)
    return len(lrs) + sum(
        temperature != baseline_temperature for temperature in temperatures
    )


def run_adapt_text_encoder_sweep(cfg: dict[str, Any]) -> None:
    adaptation = dict(cfg.get("teacher", {}).get("text_adaptation", {}))
    if not adaptation.get("enabled", False):
        raise ValueError("teacher.text_adaptation.enabled must be true")
    if adaptation.get("initial_model_from_run") is not None:
        raise ValueError(
            "The priority-1 sweep must start from teacher.model_name; remove "
            "teacher.text_adaptation.initial_model_from_run."
        )

    sweep = dict(adaptation.get("sweep", {}))
    strategy = str(sweep.get("strategy", "staged")).strip().lower()
    if strategy not in {"staged", "grid"}:
        raise ValueError("teacher.text_adaptation.sweep.strategy must be staged or grid.")
    if strategy not in {"staged", "grid"}:
        raise ValueError("teacher.text_adaptation.sweep.strategy must be staged or grid.")

    lrs = _unique_positive_floats(
        list(sweep.get("lrs", [1.0e-5, 2.0e-5, 3.0e-5])),
        "teacher.text_adaptation.sweep.lrs",
    )
    temperatures = _unique_positive_floats(
        list(sweep.get("temperatures", [0.03, 0.05, 0.08])),
        "teacher.text_adaptation.sweep.temperatures",
    )
    baseline_temperature = float(
        sweep.get("baseline_temperature", adaptation.get("temperature", 0.05))
    )
    if not math.isfinite(baseline_temperature) or baseline_temperature <= 0.0:
        raise ValueError(
            "teacher.text_adaptation.sweep.baseline_temperature must be finite "
            "and greater than zero."
        )

    max_updates = int(sweep.get("max_optimizer_updates", 12_000))
    validation_interval = int(sweep.get("validation_interval_updates", 500))
    patience = int(sweep.get("patience", 6))
    min_delta = float(sweep.get("min_delta", 1.0e-4))
    if max_updates < 1 or validation_interval < 1 or patience < 1:
        raise ValueError(
            "Sweep max updates, validation interval, and patience must be at least 1."
        )
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError("Sweep min_delta must be finite and non-negative.")

    reuse_completed = bool(sweep.get("reuse_completed", True))
    base_run_name = str(cfg["run_name"])
    summary_root = ensure_dir(
        Path("runs") / base_run_name / "tuning" / "text_encoder_priority1_sweep"
    )
    summary_path = summary_root / "sweep.json"
    results: list[dict[str, Any]] = []
    planned_variants = _planned_variant_count(
        strategy, lrs, temperatures, baseline_temperature
    )
    print(
        f"Starting {strategy} text-encoder sweep with {planned_variants} planned "
        f"variants; validating every {validation_interval:,} updates, up to "
        f"{max_updates:,} updates per variant.",
        flush=True,
    )

    def save_summary() -> None:
        save_json(
            summary_path,
            _summary_payload(
                base_run_name=base_run_name,
                strategy=strategy,
                lrs=lrs,
                temperatures=temperatures,
                baseline_temperature=baseline_temperature,
                max_updates=max_updates,
                validation_interval=validation_interval,
                patience=patience,
                min_delta=min_delta,
                results=results,
            ),
        )

    def run_variant(lr: float, temperature: float, stage: str) -> None:
        run_name = _variant_run_name(
            base_run_name,
            lr,
            temperature,
            max_updates,
            validation_interval,
        )
        variant_cfg = _variant_config(
            cfg,
            run_name=run_name,
            lr=lr,
            temperature=temperature,
            max_updates=max_updates,
            validation_interval=validation_interval,
            patience=patience,
            min_delta=min_delta,
        )
        print(
            f"\nSweep variant {len(results) + 1}/{planned_variants} "
            f"[{stage}]: lr={lr:.3g}, temperature={temperature:.3g}",
            flush=True,
        )
        print("Checking data, configuration, and code provenance...", flush=True)
        expected_provenance = text_adaptation_provenance(variant_cfg)
        existing = _completed_variant_status(
            run_name,
            expected_provenance_fingerprint=expected_provenance["fingerprint"],
        )
        if existing == "incompatible":
            raise ValueError(
                f"Existing sweep artifacts for {run_name} are incompatible with "
                "the requested settings. Use a new base run_name."
            )
        if existing == "compatible" and reuse_completed:
            print(f"Reusing compatible completed run {run_name}.", flush=True)
            results.append(
                _result_for_run(run_name, lr, temperature, stage, "reused")
            )
            save_summary()
            return

        print(f"Training new run {run_name}.", flush=True)
        try:
            run_adapt_text_encoder(variant_cfg)
        except Exception as exc:
            results.append(
                {
                    "stage": stage,
                    "status": "failed",
                    "lr": float(lr),
                    "temperature": float(temperature),
                    "run_name": run_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            save_summary()
            raise
        results.append(
            _result_for_run(run_name, lr, temperature, stage, "completed")
        )
        completed = results[-1]
        print(
            f"Completed {run_name}: best update={completed['best_update']:,}, "
            f"validation AUC={completed['best_val_impression_auc']:.6f}.",
            flush=True,
        )
        save_summary()

    save_summary()
    if strategy == "grid":
        for lr in lrs:
            for temperature in temperatures:
                run_variant(lr, temperature, "grid")
    else:
        for lr in lrs:
            run_variant(lr, baseline_temperature, "learning_rate")
        best_lr_result = _best_result(results)
        if best_lr_result is None:
            raise RuntimeError("The learning-rate stage produced no selectable result.")
        best_lr = float(best_lr_result["lr"])
        completed_pairs = {
            (float(result["lr"]), float(result["temperature"]))
            for result in results
            if result.get("status") in {"completed", "reused"}
        }
        for temperature in temperatures:
            if (best_lr, temperature) not in completed_pairs:
                run_variant(best_lr, temperature, "temperature")

    save_summary()
    best = _best_result(results)
    if best is None:
        raise RuntimeError("The text-encoder sweep produced no selectable result.")
    print(
        "Text-encoder priority-1 sweep complete: "
        f"lr={best['lr']:.3g}, temperature={best['temperature']:.3g}, "
        f"update={best['best_update']:,}, "
        f"val AUC={best['best_val_impression_auc']:.6f}."
    )
    print(f"Wrote text-encoder sweep summary to {summary_path}")


def _save_yaml(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _promotion_setting(
    promotion: dict[str, Any],
    key: str,
    default: str,
) -> str:
    value = str(promotion.get(key, default)).strip()
    if not value:
        raise ValueError(f"teacher.text_adaptation.sweep.promotion.{key} cannot be empty")
    return value


def run_promote_text_encoder_sweep(cfg: dict[str, Any]) -> None:
    """Materialize Phase 2/3 configs that route through the selected encoder."""
    base_run_name = str(cfg["run_name"])
    summary_root = (
        Path("runs")
        / base_run_name
        / "tuning"
        / "text_encoder_priority1_sweep"
    )
    summary_path = summary_root / "sweep.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Sweep summary not found at {summary_path}. Run "
            "adapt_text_encoder_sweep first."
        )
    summary = load_json(summary_path)
    results = list(summary.get("results", []))
    winner = _best_result(results)
    if winner is None:
        raise ValueError(f"No completed, selectable sweep result in {summary_path}")

    sweep = dict(cfg["teacher"]["text_adaptation"].get("sweep", {}))
    strategy = str(sweep.get("strategy", "staged")).strip().lower()
    lrs = _unique_positive_floats(list(sweep.get("lrs", [])), "sweep.lrs")
    temperatures = _unique_positive_floats(
        list(sweep.get("temperatures", [])), "sweep.temperatures"
    )
    baseline_temperature = float(
        sweep.get(
            "baseline_temperature",
            cfg["teacher"]["text_adaptation"].get("temperature", 0.05),
        )
    )
    planned_variant_count = _planned_variant_count(
        strategy, lrs, temperatures, baseline_temperature
    )
    completed_variant_count = len(
        {
            (float(result["lr"]), float(result["temperature"]))
            for result in results
            if result.get("status") in {"completed", "reused"}
        }
    )
    sweep_complete = completed_variant_count >= planned_variant_count
    promotion = dict(sweep.get("promotion", {}))
    allow_partial = bool(promotion.get("allow_partial", False))
    selection_note = str(promotion.get("selection_note", "")).strip()
    if not sweep_complete and not allow_partial:
        raise ValueError(
            f"Sweep is incomplete ({completed_variant_count}/{planned_variant_count} "
            "variants). Resume adapt_text_encoder_sweep, or explicitly set "
            "teacher.text_adaptation.sweep.promotion.allow_partial: true and add "
            "a non-empty selection_note documenting the decision."
        )
    if not sweep_complete and not selection_note:
        raise ValueError(
            "Partial sweep promotion requires a non-empty "
            "teacher.text_adaptation.sweep.promotion.selection_note."
        )

    max_updates = int(sweep.get("max_optimizer_updates", 12_000))
    validation_interval = int(sweep.get("validation_interval_updates", 500))
    patience = int(sweep.get("patience", 6))
    min_delta = float(sweep.get("min_delta", 1.0e-4))
    winner_run_name = str(winner["run_name"])
    winner_cfg = _variant_config(
        cfg,
        run_name=winner_run_name,
        lr=float(winner["lr"]),
        temperature=float(winner["temperature"]),
        max_updates=max_updates,
        validation_interval=validation_interval,
        patience=patience,
        min_delta=min_delta,
    )
    expected_provenance = text_adaptation_provenance(winner_cfg)
    winner_meta_path = Path("runs") / winner_run_name / "text_encoder" / "meta.json"
    artifact_status = _completed_variant_status(
        winner_run_name,
        expected_provenance_fingerprint=expected_provenance["fingerprint"],
    )
    if artifact_status != "compatible":
        raise ValueError(
            f"The selected sweep artifact {winner_run_name!r} is {artifact_status}; "
            "its model, metadata, validation history, and provenance must all be "
            "present and compatible before promotion."
        )
    winner_meta = load_json(winner_meta_path)
    if winner_meta.get("provenance_fingerprint") != expected_provenance["fingerprint"]:
        raise ValueError(
            "The selected sweep artifact no longer matches the current data, "
            "configuration, or implementation. Rerun the sweep with a new run_name."
        )
    best_update = int(winner_meta["best_update"])
    if (
        int(winner.get("best_update", -1)) != best_update
        or not math.isclose(
            float(winner.get("best_val_impression_auc", float("nan"))),
            float(winner_meta["best_val_impression_auc"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "The selected sweep result disagrees with its artifact metadata. "
            "Do not promote until the sweep summary is repaired or regenerated."
        )

    phase2_run = _promotion_setting(
        promotion,
        "phase2_run_name",
        f"{base_run_name}_selected",
    )
    phase3_encoder_run = _promotion_setting(
        promotion,
        "phase3_encoder_run_name",
        f"{base_run_name}_submission_text_continue",
    )
    phase3_teacher_run = _promotion_setting(
        promotion,
        "phase3_teacher_run_name",
        f"{base_run_name}_submission_teacher",
    )
    phase3_ranker_run = _promotion_setting(
        promotion,
        "phase3_ranker_run_name",
        f"{base_run_name}_submission_ranker",
    )
    phase3_output_run = _promotion_setting(
        promotion,
        "phase3_output_run_name",
        f"{base_run_name}_submission_output",
    )
    continuation_updates = int(promotion.get("continuation_updates", 2_000))
    if continuation_updates < 1:
        raise ValueError("Sweep promotion continuation_updates must be at least 1")

    phase2_cfg = deepcopy(winner_cfg)
    phase2_cfg["run_name"] = phase2_run
    phase2_cfg.setdefault("artifacts", {})[
        "text_encoder_run_name"
    ] = winner_run_name

    continuation_cfg = load_config(
        _promotion_setting(
            promotion,
            "submission_continuation_config",
            "configs/mind_large_submission_mpnet_text_continue.yaml",
        )
    )
    selected_adaptation = deepcopy(winner_cfg["teacher"]["text_adaptation"])
    continuation_cfg["teacher"]["text_adaptation"].update(selected_adaptation)
    continuation_cfg["teacher"]["text_adaptation"].update(
        {
            "initial_model_from_run": winner_run_name,
            "expected_initial_update": best_update,
            "training_split": "val",
            "max_optimizer_updates": continuation_updates,
            "early_stopping": {"enabled": False},
        }
    )
    continuation_cfg["run_name"] = phase3_encoder_run
    continuation_cfg.get("artifacts", {}).pop("text_encoder_run_name", None)

    phase3_teacher_cfg = load_config(
        _promotion_setting(
            promotion,
            "submission_teacher_config",
            "configs/mind_large_submission_mpnet.yaml",
        )
    )
    phase3_teacher_cfg["run_name"] = phase3_teacher_run
    phase3_teacher_cfg.setdefault("artifacts", {})[
        "text_encoder_run_name"
    ] = phase3_encoder_run

    phase3_ranker_cfg = load_config(
        _promotion_setting(
            promotion,
            "submission_ranker_config",
            "configs/mind_large_submission_mpnet_candidate_attention.yaml",
        )
    )
    phase3_ranker_cfg["run_name"] = phase3_ranker_run
    phase3_ranker_cfg.setdefault("artifacts", {}).update(
        {
            "text_encoder_run_name": phase3_encoder_run,
            "teacher_run_name": phase3_teacher_run,
        }
    )

    phase3_output_cfg = load_config(
        _promotion_setting(
            promotion,
            "submission_output_config",
            "configs/mind_large_submission_mpnet_candidate_attention_recency_alpha_002.yaml",
        )
    )
    phase3_output_cfg["run_name"] = phase3_output_run
    phase3_output_cfg.setdefault("artifacts", {}).update(
        {
            "text_encoder_run_name": phase3_encoder_run,
            "teacher_run_name": phase3_teacher_run,
            "ranker_run_name": phase3_ranker_run,
        }
    )

    config_paths = {
        "phase2": summary_root / "phase2_selected.yaml",
        "phase3_continuation": summary_root / "phase3_continuation.yaml",
        "phase3_teacher": summary_root / "phase3_teacher.yaml",
        "phase3_ranker": summary_root / "phase3_ranker.yaml",
        "phase3_output": summary_root / "phase3_output.yaml",
    }
    for key, generated_cfg in {
        "phase2": phase2_cfg,
        "phase3_continuation": continuation_cfg,
        "phase3_teacher": phase3_teacher_cfg,
        "phase3_ranker": phase3_ranker_cfg,
        "phase3_output": phase3_output_cfg,
    }.items():
        _save_yaml(config_paths[key], generated_cfg)

    commands = [
        f"python -m mindrec.cli train_teacher --config {config_paths['phase2']}",
        f"python -m mindrec.cli train_ranker --config {config_paths['phase2']}",
        f"python -m mindrec.cli evaluate --config {config_paths['phase2']}",
        f"python -m mindrec.cli adapt_text_encoder --config {config_paths['phase3_continuation']}",
        f"python -m mindrec.cli train_teacher --config {config_paths['phase3_teacher']}",
        f"python -m mindrec.cli train_ranker --config {config_paths['phase3_ranker']}",
        f"python -m mindrec.cli build_item_age --config {config_paths['phase3_output']}",
        f"python -m mindrec.cli write_submission --config {config_paths['phase3_output']}",
    ]
    manifest_path = summary_root / "promotion.json"
    save_json(
        manifest_path,
        {
            "source_sweep": str(summary_path),
            "winner": winner,
            "winner_provenance_fingerprint": expected_provenance["fingerprint"],
            "planned_variant_count": planned_variant_count,
            "completed_variant_count": completed_variant_count,
            "sweep_complete": sweep_complete,
            "partial_promotion_authorized": not sweep_complete and allow_partial,
            "selection_note": selection_note or None,
            "phase2_text_encoder_run_name": winner_run_name,
            "phase3_initial_update": best_update,
            "phase3_continuation_updates": continuation_updates,
            "generated_configs": {
                key: str(path) for key, path in config_paths.items()
            },
            "commands": commands,
        },
    )
    print(
        f"Promoted text-encoder sweep winner {winner_run_name} at update "
        f"{best_update:,}."
    )
    print(f"Wrote promotion manifest to {manifest_path}")
