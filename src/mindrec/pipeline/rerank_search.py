from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mindrec.config import ensure_dir
from mindrec.pipeline.rerank_policy import (
    _constraint_check, _make_constraint, resolve_guardrails, resolve_policy,
    selection_context, SELECTION_METHOD,
)
from mindrec.pipeline.rerank_metrics import (
    ScoredRerankImpression,
    evaluate_baseline,
    evaluate_candidate,
)
from mindrec.pipeline.rerank_scoring import (
    load_rerank_scoring_assets,
    iter_scored_impressions,
    resolve_rerank_protocol,
)
from mindrec.rerank.greedy import build_news_meta
from mindrec.utils import (
    impression_artifact_path,
    log_device,
    resolve_device as resolve_torch_device,
    save_json,
)


def _resolve_device(cfg: dict[str, Any]) -> torch.device:
    return resolve_torch_device(cfg["ranker"].get("device", "cuda"))


def _score_impressions(
    cfg: dict[str, Any],
    proc_root: Path,
    device: torch.device,
    split_name: str,
) -> tuple[list[ScoredRerankImpression], dict[str, Any]]:
    assets = load_rerank_scoring_assets(cfg, proc_root, device)
    impr = pd.read_parquet(impression_artifact_path(proc_root, split_name))

    scored = list(iter_scored_impressions(impr, assets, device))

    return scored, {
        "teacher_item": assets.teacher_item,
        "scoring": assets.metadata,
    }


def _candidate_key(item: dict[str, Any]) -> tuple[Any, ...]:
    def _norm(x: float) -> float:
        return round(float(x), 8)

    return (
        _norm(item["weights"]["relevance"]),
        _norm(item["weights"]["coverage"]),
        _norm(item["fairness"]["penalty_weight"]),
    )


def _current_config_candidate(
    rr_cfg: dict[str, Any], fairness_base: dict[str, Any]
) -> dict[str, Any]:
    rr_cfg = resolve_policy(rr_cfg)
    return {
        "weights": {
            "relevance": rr_cfg["relevance_weight"],
            "coverage": rr_cfg["coverage_weight"],
        },
        "fairness": {
            "penalty_weight": float(fairness_base.get("penalty_weight", 0.0)),
            "category_target": fairness_base.get("category_target", "catalog"),
        },
    }


def _attach_objective_views(
    baseline: dict[str, float],
    metrics: dict[str, Any],
    constraint: dict[str, Any],
) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["constraint"] = _constraint_check(baseline, metrics, constraint)

    ndcg_delta_ratio = (metrics["ndcg@k"] - baseline["ndcg@k"]) / max(
        baseline["ndcg@k"], 1e-12
    )
    new_gain = (
        metrics["new_item_exposure_frac"] - baseline["new_item_exposure_frac"]
    )
    cov_gain = metrics["category_coverage"] - baseline["category_coverage"]
    fair_pool_delta = metrics["fairness_kl_pool"] - baseline["fairness_kl_pool"]
    metrics["objective_view"] = {
        "deltas_vs_baseline": {
            "ndcg@k_ratio": float(ndcg_delta_ratio),
            "new_item_exposure_gain": float(new_gain),
            "category_coverage_gain": float(cov_gain),
            "fairness_kl_pool_delta": float(fair_pool_delta),
            "fairness_kl_full_delta": float(
                metrics["fairness_kl_full"] - baseline["fairness_kl_full"]
            ),
        },
    }
    return metrics


def _sort_by_ndcg(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer nDCG; exact ties use a deterministic parameter order, not utility."""
    return sorted(results, key=lambda r: (
        -r["ndcg@k"] if np.isfinite(r["ndcg@k"]) else float("inf"),
        _candidate_key(r),
    ))


def _sort_feasible_first(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        _sort_by_ndcg(results),
        key=lambda r: not r["constraint"]["feasible"],
    )


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    not_worse = (
        a["ndcg@k"] >= b["ndcg@k"]
        and a["new_item_exposure_frac"] >= b["new_item_exposure_frac"]
        and a["category_coverage"] >= b["category_coverage"]
        and a["fairness_kl_pool"] <= b["fairness_kl_pool"]
    )
    strictly_better = (
        a["ndcg@k"] > b["ndcg@k"]
        or a["new_item_exposure_frac"] > b["new_item_exposure_frac"]
        or a["category_coverage"] > b["category_coverage"]
        or a["fairness_kl_pool"] < b["fairness_kl_pool"]
    )
    return not_worse and strictly_better


def _pareto_frontier(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for candidate in results:
        if any(_dominates(other, candidate) for other in results if other is not candidate):
            continue
        frontier.append(candidate)
    return _sort_feasible_first(frontier)


def _format_frontier_row(idx: int, item: dict[str, Any]) -> str:
    weights = item["weights"]
    fairness = item["fairness"]
    constraint = item["constraint"]
    feasible = "Y" if bool(constraint["feasible"]) else "N"
    return (
        f"| {idx} | {feasible} | {item['ndcg@k']:.6f} | "
        f"{item['new_item_exposure_frac']:.6f} | {item['category_coverage']:.6f} | "
        f"{item['fairness_kl_pool']:.6f} | {item['ild']:.6f} | "
        f"{weights['relevance']:.3f} | "
        f"{weights['coverage']:.3f} | {fairness['penalty_weight']:.3f} |"
    )


def _write_pareto_frontier_md(out_root: Path, out: dict[str, Any]) -> None:
    baseline = out["baseline"]
    guardrails = out["product_constraint"]["relative_guardrails"]
    best_feasible = out.get("best_feasible")
    frontier = out.get("pareto_frontier", [])

    lines = [
        "# Pareto Frontier Summary",
        "",
        f"Source: `{(out_root / 'rerank_search.json').as_posix()}`",
        "",
        "Selection: highest nDCG among full-tuning candidates passing every guardrail.",
        "Exact nDCG ties use deterministic parameter order. Pareto points are diagnostic only.",
        "ILD measures teacher-cosine diversity of completed lists; it does not affect scoring or selection.",
        "",
        (
            "Baseline: "
            f"nDCG@k={baseline['ndcg@k']:.6f}, "
            f"new_item_exposure_frac={baseline['new_item_exposure_frac']:.6f}, "
            f"category_coverage={baseline['category_coverage']:.6f}, "
            f"fairness_kl_pool={baseline['fairness_kl_pool']:.6f}"
        ),
        "",
        (
            "Guardrails: "
            f"max_ndcg_drop_ratio={guardrails['max_ndcg_drop_ratio']}, "
            f"min_new_item_exposure_gain={guardrails['min_new_item_exposure_gain']:.2f}, "
            f"min_category_coverage_gain={guardrails['min_category_coverage_gain']:.2f}, "
            f"min_fairness_kl_pool_improvement={guardrails['min_fairness_kl_pool_improvement']:.2f}"
        ),
        "",
    ]

    if best_feasible is not None:
        lines.extend(
            [
                (
                    "Best feasible: "
                    f"nDCG@k={best_feasible['ndcg@k']:.6f}, "
                    f"new_item_exposure_frac={best_feasible['new_item_exposure_frac']:.6f}, "
                    f"category_coverage={best_feasible['category_coverage']:.6f}, "
                    f"fairness_kl_pool={best_feasible['fairness_kl_pool']:.6f}, "
                    f"fairness_penalty={best_feasible['fairness']['penalty_weight']:.3f}"
                ),
                "",
            ]
        )
    else:
        lines.extend(["Best feasible: none. No policy selected; inspect range diagnostics and candidate-pool opportunity before revising the grid. Keep guardrails fixed during range discovery.", ""])

    lines.extend(
        [
            "| # | Feasible | nDCG@k | New Item Exposure | Category Coverage | Fairness KL | Intra-List Diversity | Relevance Weight | Coverage Weight | Fairness Penalty |",
            "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        _format_frontier_row(idx, item) for idx, item in enumerate(frontier, start=1)
    )
    lines.append("")

    diagnostics = out.get("grid_diagnostics")
    if diagnostics:
        lines.extend(["## Search range diagnostics", "", diagnostics["scope"], "",
                      f"Sample feasible: {diagnostics['n_sample_feasible']}; full-tuning feasible: {diagnostics['n_full_feasible']}.",
                      f"Full grid evaluated: {diagnostics['full_grid_evaluated']}.",
                      f"Sample/full winner match: {diagnostics['sample_and_full_winner_match']}.", "",
                      "| Parameter | Value | Feasible / tested | Best feasible nDCG | Max category gain | Max KL improvement |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        def number(x):
            return "n/a" if x is None else f"{x:.6f}"
        for axis, profile in diagnostics["parameter_profiles"].items():
            for row in profile:
                lines.append(f"| {axis} | {row['value']:g} | {row['n_feasible']} / {row['n_tested']} | {number(row['best_feasible_ndcg'])} | {number(row['max_category_gain'])} | {number(row['max_kl_improvement'])} |")
        lines.extend(["", "Each maximum may come from a different setting; profile gains do not guarantee joint feasibility.", "",
                      "Failed requirements in profile data: " + str(diagnostics["failed_guardrail_counts"]),
                      "Maximum gains within relevance budget in profile data: " + str(diagnostics["max_gains_within_ndcg_budget"]), ""])
        for flag in diagnostics["range_flags"]:
            lines.append(f"- {flag['parameter']}: {flag['reason']}. {flag['message']} Candidate extension: {flag.get('candidate_extension', 'n/a')}.")
        if not diagnostics["range_flags"]:
            lines.append("No boundary or missing-control flags; this is not proof that the range is globally optimal.")
        opportunity = diagnostics.get("pool_opportunity", {})
        lines.extend(["", "Pool opportunity: " + str(opportunity), "",
                      "Suggested local values: " + str(diagnostics["suggested_local_values"]), "",
                      diagnostics["next_step"], ""])
    (out_root / "pareto_frontier.md").write_text("\n".join(lines), encoding="utf-8")


def _resolve_search_settings(search_cfg: dict[str, Any]) -> dict[str, Any]:
    resolve_guardrails(search_cfg)
    settings = {
        "seed": int(search_cfg.get("seed", 13)),
        "sample_size": int(search_cfg.get("sample_size", 5000)),
        "shortlist_size": int(search_cfg.get("shortlist_size", 20)),
    }
    for name, default in (
        ("coverage_weights", [0.0, 0.025, 0.05, 0.10]),
        ("fairness_penalties", [0.0, 0.025, 0.05, 0.10, 0.20]),
    ):
        values = search_cfg.get(name, default)
        if not isinstance(values, list) or not values:
            raise ValueError(f"rerank.search.{name} must be a non-empty YAML list.")
        values = [float(value) for value in values]
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(f"rerank.search.{name} must contain finite non-negative values.")
        if name == "coverage_weights" and any(value >= 1.0 for value in values):
            raise ValueError("rerank.search.coverage_weights must be less than 1; relevance is 1 - coverage.")
        settings[name] = sorted(set(values))
    for name in ("sample_size", "shortlist_size"):
        if settings[name] < 1:
            raise ValueError(f"rerank.search.{name} must be at least 1.")
    return settings


def _build_shortlist(
    results: list[dict[str, Any]],
    size: int,
) -> tuple[list[dict[str, Any]], set[tuple[Any, ...]]]:
    """Use two thirds feasible-by-nDCG; reserve the rest for uncertain rejects."""
    if size < 1:
        raise ValueError("Shortlist size must be at least 1.")
    ordered = []
    unique_keys = set()
    for item in _sort_by_ndcg(results):
        key = _candidate_key(item)
        if key not in unique_keys:
            ordered.append(item)
            unique_keys.add(key)
    feasible = [r for r in ordered if r["constraint"]["feasible"]]
    infeasible = [r for r in ordered if not r["constraint"]["feasible"]]
    reserve = size // 3
    priority = feasible[:size - reserve] + infeasible[:reserve] + feasible + infeasible
    shortlist = []
    seen = set()
    for item in priority:
        key = _candidate_key(item)
        if key not in seen:
            shortlist.append(item)
            seen.add(key)
        if len(shortlist) >= size:
            break
    return shortlist, seen


def _build_search_space(
    coverage_weights: list[float],
    fairness_penalties: list[float],
    fairness_enabled: bool = True,
) -> list[tuple[float, float, float]]:
    """Build effective (relevance, coverage, KL penalty) combinations."""
    return list(dict.fromkeys(
        (1.0 - coverage, coverage, penalty)
        for coverage in coverage_weights
        for penalty in (fairness_penalties if fairness_enabled else [0.0])
    ))


def _grid_diagnostics(sample_results, full_results, settings):
    """Describe explored ranges; suggestions never change eligibility or the grid."""
    axes = {
        "coverage_weight": sorted(set(settings["coverage_weights"])),
        "penalty_weight": sorted(set(settings["fairness_penalties"])),
    }
    def value(row, axis):
        return row["fairness"][axis] if axis == "penalty_weight" else row["weights"][axis.removesuffix("_weight")]
    sample_feasible = [r for r in _sort_by_ndcg(sample_results) if r["constraint"]["feasible"]]
    full_feasible = [r for r in _sort_by_ndcg(full_results) if r["constraint"]["feasible"]]
    sample_best = sample_feasible[0] if sample_feasible else None
    full_best = full_feasible[0] if full_feasible else None
    full_grid_evaluated = bool(sample_results) and {
        _candidate_key(r) for r in sample_results
    }.issubset({_candidate_key(r) for r in full_results})
    # Never mix sample trends with full-tuning evidence. A partial full run
    # describes only its evaluated candidates, including empty profile cells.
    profile_results = full_results if full_results else sample_results
    profile_source = "full_tuning" if full_results else "screening_sample"
    profile_best = full_best if full_results else sample_best
    if full_results:
        scope = (
            "Profiles and boundary flags use full-tuning results for the entire grid."
            if full_grid_evaluated else
            "Profiles and boundary flags use only fully evaluated candidates; untested grid points are not evidence of failure."
        )
    else:
        scope = "No full-tuning results: profiles and boundary flags use the screening sample only."
    scope += " Profiles optimize other parameters and are not isolated causal effects. Pool-opportunity statistics remain sample-based."
    profiles, flags, refinement = {}, [], {}
    for axis, values in axes.items():
        profiles[axis] = []
        for x in values:
            rows = [r for r in profile_results if value(r, axis) == x]
            eligible = [r for r in rows if r["constraint"]["feasible"]]
            profiles[axis].append({
                "value": x, "n_tested": len(rows), "n_feasible": len(eligible),
                "best_feasible_ndcg": max((r["ndcg@k"] for r in eligible), default=None),
                "max_category_gain": max((r["constraint"]["category_coverage_gain"] for r in rows), default=None),
                "max_kl_improvement": max((r["constraint"]["fairness_kl_pool_improvement"] for r in rows), default=None),
                "min_ndcg_drop_pct": min((r["constraint"]["ndcg_drop_pct"] for r in rows), default=None),
            })
        if len(values) == 1:
            flags.append({"parameter": axis, "reason": "fixed_range", "message": "Only one value tested; this run cannot assess its range."})
        else:
            if profile_best is not None and value(profile_best, axis) == values[-1]:
                next_value = 2.0 * values[-1]
                if axis != "penalty_weight":
                    next_value = min(next_value, 1.0 - 1e-6)
                flags.append({
                    "parameter": axis, "reason": "winner_at_upper_boundary",
                    "candidate_extension": next_value if next_value > values[-1] else None,
                    "message": "Inspect the profile trend before extending; a boundary winner alone is not evidence that larger values improve results.",
                })
            if values[0] > 0:
                flags.append({"parameter": axis, "reason": "missing_zero_control", "candidate_extension": 0.0,
                              "message": "Include zero to measure whether this component is needed."})
        if full_best is not None:
            center = value(full_best, axis)
            candidates = {center}
            lower = [x for x in values if x < center]
            upper = [x for x in values if x > center]
            if lower: candidates.add(round((max(lower) + center) / 2, 8))
            if upper: candidates.add(round((min(upper) + center) / 2, 8))
            refinement[axis] = sorted(candidates)
    failures = {}
    for row in profile_results:
        for name in row["constraint"]["failed_guardrails"]:
            failures[name] = failures.get(name, 0) + 1
    within_budget = [r for r in profile_results if not {"max_ndcg_drop_ratio", "non_finite_metrics"}.intersection(r["constraint"]["failed_guardrails"])]
    return {
        "scope": scope,
        "profile_source": profile_source,
        "n_sample_feasible": len(sample_feasible), "n_full_feasible": len(full_feasible),
        "sample_and_full_winner_match": (_candidate_key(sample_best) == _candidate_key(full_best)) if sample_best is not None and full_best is not None else None,
        "full_grid_evaluated": full_grid_evaluated,
        "failed_guardrail_counts": failures,
        "max_gains_within_ndcg_budget": {
            metric: max((r["constraint"][metric] for r in within_budget), default=None)
            for metric in ("category_coverage_gain", "fairness_kl_pool_improvement", "new_item_exposure_gain")
        },
        "parameter_profiles": profiles, "range_flags": flags,
        "suggested_local_values": refinement,
        "next_step": (
            "No full-tuning feasible setting. Inspect failed requirements, profiles, and pool opportunity; expand only supported ranges. Keep guardrails fixed."
            if full_best is None else
            "Review boundary flags and sample/full disagreement, then consider one local grid around suggested values. Include the current full-tuning winner and fully evaluate that local grid."
        ),
    }


def _pool_opportunities(rows, news_meta, k_out, pool_size, position_bias, baseline):
    from mindrec.utils import position_bias_weights
    category_limits, new_limits, gaps, kl_spans = [], [], [], []
    for row in rows:
        pool = np.argsort(-row.scores, kind="stable")[:pool_size]
        length = min(k_out, len(pool))
        cats = {news_meta[row.cand_news_id[i]].cat_idx for i in pool if row.cand_news_id[i] in news_meta}
        cats.discard(0)
        counts = [sum(news_meta.get(row.cand_news_id[i]) is not None and news_meta[row.cand_news_id[i]].cat_idx == cat for i in pool) for cat in cats]
        kl_spans.append(float(np.log(max(counts) / min(counts))) if counts else 0.0)
        scores = row.scores[pool]
        if len(scores) > 1:
            normalized = (scores - scores[-1]) / max(float(scores[0] - scores[-1]), 1e-12)
            gaps.extend((-np.diff(normalized)).tolist())
        category_limits.append(min(length, len(cats)))
        n_new = min(length, sum(int(row.cand_is_new[i]) == 1 for i in pool))
        weights = position_bias_weights(length, mode=position_bias)
        new_limits.append(float(weights[:n_new].sum() / (weights.sum() + 1e-12)))
    return {
        "scope": "Optimistic per-list bounds in the screening sample; ignore relevance and other simultaneous constraints.",
        "adjacent_normalized_relevance_gaps": {str(q): float(np.quantile(gaps, q)) if gaps else 0.0 for q in (0.5, 0.9, 0.99)},
        "first_position_kl_span": {str(q): float(np.quantile(kl_spans, q)) if kl_spans else 0.0 for q in (0.5, 0.9, 0.99)},
        "category_coverage_gain_upper_bound": float(np.mean(category_limits) - baseline["category_coverage"]),
        "new_item_exposure_gain_upper_bound": float(np.mean(new_limits) - baseline["new_item_exposure_frac"]),
    }


def run_rerank_search(cfg: dict[str, Any]) -> None:
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    rr_cfg = cfg["rerank"]
    out_root = ensure_dir(runs_root / rr_cfg.get("output_subdir", "rerank"))
    policy = resolve_policy(rr_cfg)
    protocol = resolve_rerank_protocol(cfg)
    search_cfg = dict(rr_cfg.get("search", {}))
    search_settings = _resolve_search_settings(search_cfg)

    device = _resolve_device(cfg)
    log_device(device, "Rerank search")
    news = pd.read_parquet(proc_root / "news.parquet")
    news_meta = build_news_meta(news)
    search_split = protocol.search_split
    scored_impressions, assets = _score_impressions(
        cfg, proc_root, device, split_name=search_split
    )
    if not scored_impressions:
        raise RuntimeError(
            f"No labeled impressions with positive clicks were found in {search_split!r}."
        )
    teacher_item = assets["teacher_item"]

    k_out = policy["k_out"]
    pool_size = policy["pool_size"]
    position_bias = policy["position_bias"]
    coverage_cfg = dict(policy["coverage"])
    fairness_base = dict(policy["fairness"])
    relevance_normalization = policy["relevance_normalization"]

    baseline = evaluate_baseline(
        scored_impressions=scored_impressions,
        teacher_item=teacher_item,
        news_meta=news_meta,
        k_out=k_out,
        pool_size=pool_size,
        position_bias=position_bias,
        category_target=fairness_base.get("category_target", "catalog"),
    )
    constraint = _make_constraint(baseline, search_cfg)
    seed = search_settings["seed"]
    search_sample_size = search_settings["sample_size"]
    if len(scored_impressions) > search_sample_size:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(
            rng.choice(len(scored_impressions), size=search_sample_size, replace=False)
        )
        scored_search = [scored_impressions[int(i)] for i in sample_idx.tolist()]
    else:
        scored_search = scored_impressions

    coverage_weights = search_settings["coverage_weights"]
    fairness_penalties = search_settings["fairness_penalties"]

    sample_baseline = baseline if scored_search is scored_impressions else evaluate_baseline(
        scored_impressions=scored_search,
        teacher_item=teacher_item,
        news_meta=news_meta,
        k_out=k_out,
        pool_size=pool_size,
        position_bias=position_bias,
        category_target=fairness_base.get("category_target", "catalog"),
    )

    sample_results = []
    search_space = _build_search_space(
        coverage_weights=coverage_weights,
        fairness_penalties=fairness_penalties,
        fairness_enabled=bool(fairness_base["enabled"]),
    )

    for (
        relevance_weight,
        coverage_weight,
        penalty_weight,
    ) in tqdm(search_space, desc="Search rerank grid"):
        fairness_cfg = dict(fairness_base)
        fairness_cfg["penalty_weight"] = penalty_weight

        metrics = evaluate_candidate(
            scored_impressions=scored_search,
            teacher_item=teacher_item,
            news_meta=news_meta,
            k_out=k_out,
            pool_size=pool_size,
            position_bias=position_bias,
            coverage_cfg=coverage_cfg,
            fairness_cfg=fairness_cfg,
            relevance_weight=relevance_weight,
            coverage_weight=coverage_weight,
            relevance_normalization=relevance_normalization,
        )
        sample_results.append(
            _attach_objective_views(sample_baseline, metrics, constraint)
        )

    sample_results = _sort_feasible_first(sample_results)
    sample_pareto = _pareto_frontier(sample_results)

    shortlist_size = search_settings["shortlist_size"]
    shortlist, seen = _build_shortlist(sample_results, shortlist_size)

    current_candidate = _current_config_candidate(
        rr_cfg=rr_cfg,
        fairness_base=fairness_base,
    )
    current_key = _candidate_key(current_candidate)
    if current_key not in seen:
        shortlist.append(current_candidate)
        seen.add(current_key)

    results = []
    # Small datasets may already have been fully evaluated during screening.
    full_cache = {_candidate_key(item): item for item in sample_results} if scored_search is scored_impressions else {}
    reused_full_results = 0
    for item in tqdm(shortlist, desc=f"Evaluate shortlist on full {search_split}"):
        key = _candidate_key(item)
        if key in full_cache:
            results.append(full_cache[key])
            reused_full_results += 1
            continue
        fairness_cfg = dict(fairness_base)
        fairness_cfg["penalty_weight"] = item["fairness"]["penalty_weight"]
        metrics = evaluate_candidate(
            scored_impressions=scored_impressions,
            teacher_item=teacher_item,
            news_meta=news_meta,
            k_out=k_out,
            pool_size=pool_size,
            position_bias=position_bias,
            coverage_cfg=coverage_cfg,
            fairness_cfg=fairness_cfg,
            relevance_weight=item["weights"]["relevance"],
            coverage_weight=item["weights"]["coverage"],
            relevance_normalization=relevance_normalization,
        )
        results.append(_attach_objective_views(baseline, metrics, constraint))

    feasible = [r for r in results if r["constraint"]["feasible"]]
    feasible = _sort_feasible_first(feasible)
    results = _sort_feasible_first(results)
    pareto_frontier = _pareto_frontier(results)

    diagnostics = _grid_diagnostics(sample_results, results, search_settings)
    diagnostics["pool_opportunity"] = _pool_opportunities(
        scored_search, news_meta, k_out, pool_size, position_bias, sample_baseline
    )
    out = {
        "schema_version": 5,
        "grid_diagnostics": diagnostics,
        "selection_method": SELECTION_METHOD,
        "selection_context": selection_context(cfg),
        "selection_status": "selected" if feasible else "no_feasible_policy",
        "tie_breaker": "candidate_parameter_order",
        "coverage": coverage_cfg,
        "k_out": k_out,
        "pool_size": pool_size,
        "position_bias": position_bias,
        "relevance_normalization": relevance_normalization,
        "search_split": search_split,
        "reporting_split": protocol.reporting_split,
        "selection": protocol.selection,
        "scoring": assets["scoring"],
        "search_configuration": search_settings,
        "baseline": baseline,
        "sample_baseline": sample_baseline,
        "product_constraint": constraint,
        "search_sample_size": len(scored_search),
        "search_seed": seed,
        "n_candidates_screened": len(sample_results),
        "n_impressions_evaluated": len(scored_impressions),
        "n_candidates_evaluated_full": len(results),
        "n_full_results_reused_from_screen": reused_full_results,
        "n_shortlisted_full_eval": len(shortlist),
        "n_feasible": len(feasible),
        "best_feasible": feasible[0] if feasible else None,
        "pareto_frontier": pareto_frontier,
        "pareto_frontier_sample": sample_pareto,
        # top_10_sample: Best settings on the sampled search subset, ranked by
        # feasibility first and then nDCG.
        # top_10: Best settings after reevaluating the shortlisted candidates on
        # the full validation split, ranked by feasibility first and then nDCG.
        "top_10": results[:10],
        "top_10_sample": sample_results[:10],
        "results": results,
        "sample_results": sample_results,
    }
    save_json(out_root / "rerank_search.json", out)
    _write_pareto_frontier_md(out_root, out)
    if not feasible:
        print(
            "No feasible reranking policy found. See failed_guardrails in "
            "rerank_search.json; do not freeze a candidate."
        )
    else:
        print(
            f"Selected highest feasible nDCG: {feasible[0]['ndcg@k']:.6f}. "
            "Review and freeze before reporting."
        )
