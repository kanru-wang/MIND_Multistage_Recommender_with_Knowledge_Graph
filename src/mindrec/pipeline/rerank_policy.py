"""Shared reranker policy, guardrails, and frozen-selection validation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from mindrec.rerank.greedy import DEFAULT_OBJECTIVE_WEIGHTS, validate_rerank_config
from mindrec.utils import teacher_artifact_root

SELECTION_METHOD = "max_ndcg_subject_to_guardrails"
DEFAULT_GUARDRAILS = {
    "max_ndcg_drop_ratio": 0.02,
    "min_new_item_exposure_gain": 0.0,
    "min_category_coverage_gain": 0.25,
    "min_fairness_kl_pool_improvement": 0.03,
}


def resolve_guardrails(search_cfg: dict[str, Any]) -> dict[str, float]:
    obsolete = {"utility_scales", "utility_coefficients"}.intersection(search_cfg)
    if obsolete:
        raise ValueError(
            "Approach 1 selects by nDCG subject to guardrails; remove obsolete "
            "rerank.search fields: " + ", ".join(sorted(obsolete))
        )
    configured = dict(search_cfg.get("relative_guardrails", {}))
    unknown = configured.keys() - DEFAULT_GUARDRAILS.keys()
    if unknown:
        raise ValueError("Unknown reranker guardrails: " + ", ".join(sorted(unknown)))
    guardrails = {
        key: float(configured.get(key, default))
        for key, default in DEFAULT_GUARDRAILS.items()
    }
    invalid = [
        key for key, value in guardrails.items()
        if not np.isfinite(value) or value < 0
    ]
    invalid += [
        key for key in ("max_ndcg_drop_ratio", "min_new_item_exposure_gain")
        if guardrails[key] > 1
    ]
    if invalid:
        raise ValueError("Invalid reranker guardrails: " + ", ".join(invalid))
    return guardrails


def resolve_policy(rr_cfg: dict[str, Any]) -> dict[str, Any]:
    """Materialize defaults once so search, evaluation, and provenance agree."""
    policy = {
        "k_out": int(rr_cfg.get("k_out", 0)),
        "pool_size": int(rr_cfg.get("pool_size", 0)),
        "position_bias": str(rr_cfg.get("position_bias", "log")),
        "novelty_sim": str(rr_cfg.get("novelty_sim", "teacher_cosine")),
        "relevance_normalization": str(rr_cfg.get("relevance_normalization", "none")),
        **{
            name: float(rr_cfg.get(name, value))
            for name, value in DEFAULT_OBJECTIVE_WEIGHTS.items()
        },
        "coverage": {
            "category_bonus": 1.0,
            "entity_bonus": 0.3,
            "max_new_entities_per_item": 3,
            **rr_cfg.get("coverage", {}),
        },
        "fairness": {
            "enabled": False,
            "category_target": "catalog",
            "new_item_floor": 0.0,
            "penalty_weight": 0.0,
            **rr_cfg.get("fairness", {}),
        },
    }
    policy["fairness"]["position_bias"] = policy["position_bias"]
    validate_rerank_config(policy)
    return policy


def selection_context(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": resolve_policy(cfg["rerank"]),
        "guardrails": resolve_guardrails(cfg["rerank"].get("search", {})),
        "ranker_run_name": str(cfg.get("artifacts", {}).get("ranker_run_name", cfg["run_name"])),
        "teacher_artifact_root": teacher_artifact_root(cfg).as_posix(),
        "processed_root": (Path(cfg["data"]["processed_root"]) / cfg["data"]["dataset_name"]).as_posix(),
        "knowledge_graph_enabled": bool(cfg.get("knowledge_graph", {}).get("enabled", False)),
    }


def _same_settings(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_settings(left[k], right[k]) for k in left)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
    return left == right


def validate_selection_artifact(
    cfg: dict[str, Any], search_split: str, reporting_split: str
) -> None:
    path = Path(cfg["rerank"]["selection"]["search_artifact"])
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read reranker selection artifact {path}: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("selection_method") != SELECTION_METHOD:
        raise RuntimeError("Selection artifact is not an Approach 1 search; run rerank_search again.")
    selected = artifact.get("best_feasible")
    if artifact.get("selection_status") != "selected" or not isinstance(selected, dict):
        raise RuntimeError("Selection artifact has no feasible policy to freeze.")
    if artifact.get("validation_scope"):
        raise RuntimeError("A diagnostic/smoke artifact cannot be used for frozen selection.")
    if artifact.get("search_split") != search_split or artifact.get("reporting_split") != reporting_split:
        raise RuntimeError("Selection artifact tuning/reporting splits do not match the config.")
    try:
        expected = deepcopy(artifact["selection_context"])
        policy = expected["policy"]
        for name in ("relevance", "novelty", "coverage"):
            policy[f"{name}_weight"] = selected["weights"][name]
        policy["novelty_sim"] = selected["novelty_sim"]
        policy["fairness"].update(selected["fairness"])
        valid = _constraint_check(artifact["baseline"], selected, artifact["product_constraint"])["feasible"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Incomplete selection artifact; run rerank_search again.") from exc
    if not valid:
        raise RuntimeError("The selected artifact policy fails its guardrails.")
    if not _same_settings(expected, selection_context(cfg)):
        raise RuntimeError(
            "Frozen reranker settings, guardrails, or model/data sources differ from "
            "the selected search artifact. Copy best_feasible and preserve the search definitions."
        )


def _make_constraint(
    baseline: dict[str, float], search_cfg: dict[str, Any]
) -> dict[str, Any]:
    relative_guardrails = resolve_guardrails(search_cfg)
    return {
        "relative_guardrails": relative_guardrails,
        "baseline_metrics": {
            "ndcg@k": baseline["ndcg@k"],
            "new_item_exposure_frac": baseline["new_item_exposure_frac"],
            "category_coverage": baseline["category_coverage"],
            "fairness_kl_pool": baseline["fairness_kl_pool"],
            "fairness_kl_full": baseline["fairness_kl_full"],
        },
    }


def _constraint_check(
    baseline: dict[str, float], metrics: dict[str, Any], constraint: dict[str, Any]
) -> dict[str, Any]:
    relative = constraint["relative_guardrails"]
    ndcg_drop_ratio = max(
        0.0, (baseline["ndcg@k"] - metrics["ndcg@k"]) / max(baseline["ndcg@k"], 1e-12)
    )
    ndcg_drop_pct = 100.0 * ndcg_drop_ratio
    new_gain = metrics["new_item_exposure_frac"] - baseline["new_item_exposure_frac"]
    cov_gain = metrics["category_coverage"] - baseline["category_coverage"]
    fair_kl_pool_delta = metrics["fairness_kl_pool"] - baseline["fairness_kl_pool"]
    fair_kl_full_delta = metrics["fairness_kl_full"] - baseline["fairness_kl_full"]
    fair_kl_pool_improvement = baseline["fairness_kl_pool"] - metrics["fairness_kl_pool"]
    # Only absorb floating-point roundoff at an exact boundary.
    tolerance = 1e-12
    checks = {
        "max_ndcg_drop_ratio": (
            ndcg_drop_ratio <= relative["max_ndcg_drop_ratio"] + tolerance
        ),
        "min_new_item_exposure_gain": (
            new_gain + tolerance >= relative["min_new_item_exposure_gain"]
        ),
        "min_category_coverage_gain": (
            cov_gain + tolerance >= relative["min_category_coverage_gain"]
        ),
        "min_fairness_kl_pool_improvement": (
            fair_kl_pool_improvement + tolerance
            >= relative["min_fairness_kl_pool_improvement"]
        ),
    }
    finite = all(np.isfinite(value) for value in (
        *baseline.values(), metrics["ndcg@k"], metrics["new_item_exposure_frac"],
        metrics["category_coverage"], metrics["fairness_kl_pool"],
    ))
    failed = [name for name, passed in checks.items() if not passed]
    if not finite:
        failed.append("non_finite_metrics")
    feasible = not failed
    return {
        "feasible": bool(feasible),
        "failed_guardrails": failed,
        "ndcg_drop_pct": float(ndcg_drop_pct),
        "new_item_exposure_gain": float(new_gain),
        "category_coverage_gain": float(cov_gain),
        "fairness_kl_pool_delta": float(fair_kl_pool_delta),
        "fairness_kl_pool_improvement": float(fair_kl_pool_improvement),
        "fairness_kl_full_delta": float(fair_kl_full_delta),
        "absolute_metrics": {
            "ndcg@k": float(metrics["ndcg@k"]),
            "new_item_exposure_frac": float(metrics["new_item_exposure_frac"]),
            "category_coverage": float(metrics["category_coverage"]),
            "fairness_kl_pool": float(metrics["fairness_kl_pool"]),
        },
    }


