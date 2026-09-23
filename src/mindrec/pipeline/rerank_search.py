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
        item["novelty_sim"],
        _norm(item["weights"]["relevance"]),
        _norm(item["weights"]["novelty"]),
        _norm(item["weights"]["coverage"]),
        _norm(item["fairness"]["penalty_weight"]),
        _norm(item["fairness"]["new_item_floor"]),
    )


def _current_config_candidate(
    rr_cfg: dict[str, Any], fairness_base: dict[str, Any], novelty_sim: str
) -> dict[str, Any]:
    rr_cfg = resolve_policy(rr_cfg)
    return {
        "weights": {
            "relevance": rr_cfg["relevance_weight"],
            "novelty": rr_cfg["novelty_weight"],
            "coverage": rr_cfg["coverage_weight"],
        },
        "fairness": {
            "penalty_weight": float(fairness_base.get("penalty_weight", 0.0)),
            "new_item_floor": float(fairness_base.get("new_item_floor", 0.0)),
            "category_target": fairness_base.get("category_target", "catalog"),
        },
        "novelty_sim": novelty_sim,
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
        f"{weights['relevance']:.3f} | {weights['novelty']:.3f} | "
        f"{weights['coverage']:.3f} | {fairness['penalty_weight']:.3f} | "
        f"{fairness['new_item_floor']:.3f} |"
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
                    f"fairness_penalty={best_feasible['fairness']['penalty_weight']:.2f}, "
                    f"new_item_floor={best_feasible['fairness']['new_item_floor']:.2f}"
                ),
                "",
            ]
        )
    else:
        lines.extend(["Best feasible: none. No policy selected; revise the grid or requirements on tuning data.", ""])

    lines.extend(
        [
            "| # | Feasible | nDCG@k | New Item Exposure | Category Coverage | Fairness KL | Intra-List Diversity | Relevance Weight | Novelty Weight | Coverage Weight | Fairness Penalty | New Item Floor |",
            "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(
        _format_frontier_row(idx, item) for idx, item in enumerate(frontier, start=1)
    )
    lines.append("")

    (out_root / "pareto_frontier.md").write_text("\n".join(lines), encoding="utf-8")


def _resolve_search_settings(search_cfg: dict[str, Any]) -> dict[str, Any]:
    resolve_guardrails(search_cfg)
    raw_novelty_sims = search_cfg.get("novelty_sims", ["teacher_cosine"])
    if isinstance(raw_novelty_sims, str):
        raise ValueError("rerank.search.novelty_sims must be a YAML list.")
    raw_weight_pairs = search_cfg.get(
        "weight_pairs",
        [
            [0.0, 0.0], [0.025, 0.0], [0.0, 0.025], [0.025, 0.025],
            [0.05, 0.0], [0.0, 0.05], [0.05, 0.025], [0.025, 0.05],
            [0.05, 0.05], [0.025, 0.075], [0.05, 0.075], [0.05, 0.10],
        ],
    )
    weight_pairs = []
    for index, pair in enumerate(raw_weight_pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"rerank.search.weight_pairs[{index}] must be "
                "[novelty_weight, coverage_weight]."
            )
        weight_pairs.append([float(pair[0]), float(pair[1])])

    settings = {
        "seed": int(search_cfg.get("seed", 13)),
        "sample_size": int(search_cfg.get("sample_size", 5000)),
        "shortlist_size": int(search_cfg.get("shortlist_size", 15)),
        "novelty_sims": [str(value) for value in raw_novelty_sims],
        "weight_pairs": weight_pairs,
        "fairness_penalties": [
            float(value)
            for value in search_cfg.get(
                "fairness_penalties", [0.0, 0.05, 0.10, 0.20, 0.30]
            )
        ],
        "new_item_floors": [
            float(value)
            for value in search_cfg.get(
                "new_item_floors", [0.0, 0.20, 0.30]
            )
        ],
    }
    if settings["sample_size"] < 1:
        raise ValueError("rerank.search.sample_size must be at least 1.")
    if settings["shortlist_size"] < 1:
        raise ValueError("rerank.search.shortlist_size must be at least 1.")
    if not settings["novelty_sims"]:
        raise ValueError("rerank.search.novelty_sims cannot be empty.")
    allowed_novelty = {"teacher_cosine", "category", "entity_jaccard"}
    if any(value not in allowed_novelty for value in settings["novelty_sims"]):
        raise ValueError(
            "rerank.search.novelty_sims contains an unsupported similarity."
        )
    if not settings["weight_pairs"]:
        raise ValueError("rerank.search.weight_pairs cannot be empty.")
    for novelty_weight, coverage_weight in settings["weight_pairs"]:
        if (
            not np.isfinite(novelty_weight)
            or not np.isfinite(coverage_weight)
            or novelty_weight < 0.0
            or coverage_weight < 0.0
            or novelty_weight + coverage_weight >= 1.0
        ):
            raise ValueError(
                "Each rerank.search.weight_pairs entry must contain non-negative "
                "novelty and coverage weights whose sum is less than 1."
            )
    if not settings["fairness_penalties"] or any(
        not np.isfinite(value) or value < 0.0
        for value in settings["fairness_penalties"]
    ):
        raise ValueError(
            "rerank.search.fairness_penalties must contain finite non-negative values."
        )
    if not settings["new_item_floors"] or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in settings["new_item_floors"]
    ):
        raise ValueError(
            "rerank.search.new_item_floors must contain values between 0 and 1."
        )
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
    novelty_sims: list[str],
    weight_pairs: list[list[float]],
    fairness_penalties: list[float],
    new_item_floors: list[float],
    fairness_enabled: bool = True,
) -> list[tuple[str, float, float, float, float, float]]:
    """Build effective policies, omitting settings that rank identically."""

    search_space: list[tuple[str, float, float, float, float, float]] = []
    seen = set()
    for novelty_sim in novelty_sims:
        for novelty_weight, coverage_weight in weight_pairs:
            relevance_weight = 1.0 - novelty_weight - coverage_weight
            if relevance_weight <= 0.0:
                continue
            for penalty_weight in (fairness_penalties if fairness_enabled else [0.0]):
                # With no fairness penalty the new-item floor cannot affect a
                # score, so evaluate one canonical floor instead of duplicates.
                effective_floors = (
                    [0.0] if penalty_weight == 0.0 else new_item_floors
                )
                for new_item_floor in effective_floors:
                    candidate = (
                        novelty_sim if novelty_weight > 0 else novelty_sims[0],
                        relevance_weight, novelty_weight, coverage_weight,
                        penalty_weight, new_item_floor,
                    )
                    if candidate not in seen:
                        search_space.append(candidate)
                        seen.add(candidate)
    return search_space


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

    novelty_sims = search_settings["novelty_sims"]
    weight_pairs = search_settings["weight_pairs"]
    fairness_penalties = search_settings["fairness_penalties"]
    new_item_floors = search_settings["new_item_floors"]

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
        novelty_sims=novelty_sims,
        weight_pairs=weight_pairs,
        fairness_penalties=fairness_penalties,
        new_item_floors=new_item_floors,
        fairness_enabled=bool(fairness_base["enabled"]),
    )

    for (
        novelty_sim,
        relevance_weight,
        novelty_weight,
        coverage_weight,
        penalty_weight,
        new_item_floor,
    ) in tqdm(search_space, desc="Search rerank grid"):
        fairness_cfg = dict(fairness_base)
        fairness_cfg["penalty_weight"] = penalty_weight
        fairness_cfg["new_item_floor"] = new_item_floor

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
            novelty_weight=novelty_weight,
            coverage_weight=coverage_weight,
            novelty_sim=novelty_sim,
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
        novelty_sim=policy["novelty_sim"],
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
        fairness_cfg["new_item_floor"] = item["fairness"]["new_item_floor"]
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
            novelty_weight=item["weights"]["novelty"],
            coverage_weight=item["weights"]["coverage"],
            novelty_sim=item["novelty_sim"],
            relevance_normalization=relevance_normalization,
        )
        results.append(_attach_objective_views(baseline, metrics, constraint))

    feasible = [r for r in results if r["constraint"]["feasible"]]
    feasible = _sort_feasible_first(feasible)
    results = _sort_feasible_first(results)
    pareto_frontier = _pareto_frontier(results)

    out = {
        "schema_version": 2,
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
        # Retained for backward compatibility with historical search JSON.
        "eval_split": search_split,
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
