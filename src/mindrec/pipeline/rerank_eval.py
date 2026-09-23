from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mindrec.config import ensure_dir
from mindrec.pipeline.rerank_metrics import (
    MetricAccumulator,
    baseline_metrics_for_impression,
    candidate_metrics_for_impression,
)
from mindrec.pipeline.rerank_scoring import (
    load_rerank_scoring_assets,
    iter_scored_impressions,
    resolve_rerank_protocol,
)
from mindrec.pipeline.rerank_policy import (
    _constraint_check, _make_constraint, resolve_guardrails, resolve_policy, SELECTION_METHOD,
)
from mindrec.rerank.greedy import build_news_meta
from mindrec.utils import (
    impression_artifact_path,
    log_device,
    resolve_device,
    save_json,
)


def _metric_deltas(
    baseline: dict[str, float], reranked: dict[str, float]
) -> dict[str, float]:
    delta = {
        key: float(reranked[key] - baseline[key])
        for key in baseline
        if key in reranked
    }
    base_ndcg = float(baseline.get("ndcg@k", 0.0))
    delta["ndcg_drop_ratio"] = float(
        max(0.0, (base_ndcg - float(reranked.get("ndcg@k", 0.0))))
        / max(base_ndcg, 1e-12)
    )
    return delta


def _write_rerank_report(out_root: Path, out: dict[str, Any]) -> None:
    baseline = out["baseline"]
    reranked = out["reranked"]
    delta = out["delta"]
    reporting_note = out.get("selection", {}).get("reporting_note")
    lines = [
        "# Reranker Evaluation",
        "",
        f"Split: `{out['eval_split']}`  ",
        f"Evaluated impressions: {out['n_impressions_evaluated']}  ",
        f"Top-K / pool: {out['k_out']} / {out['pool_size']}",
        "",
        *([f"Evaluation context: {reporting_note}", ""] if reporting_note else []),
        "| Metric | Baseline | Reranked | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in (
        "ndcg@k",
        "recall@k",
        "ild",
        "category_coverage",
        "category_entropy",
        "fairness_kl_pool",
        "fairness_kl_full",
        "fairness_gini",
        "new_item_exposure_frac",
    ):
        lines.append(
            f"| {key} | {baseline[key]:.6f} | {reranked[key]:.6f} | "
            f"{delta[key]:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"Relative nDCG drop: {100.0 * delta['ndcg_drop_ratio']:.3f}%",
            f"Guardrails passed: {out['constraint']['feasible']}",
            "Failed guardrails: "
            + (", ".join(out["constraint"]["failed_guardrails"]) or "none"),
            "",
            (
                "Lower is better for fairness KL and fairness Gini; higher is "
                "better for the other reported metrics."
            ),
            "",
        ]
    )
    (out_root / "rerank_eval.md").write_text("\n".join(lines), encoding="utf-8")


def run_rerank_eval(cfg: dict[str, Any]) -> None:
    rr_cfg = cfg["rerank"]
    policy = resolve_policy(rr_cfg)
    resolve_guardrails(rr_cfg.get("search", {}))
    protocol = resolve_rerank_protocol(cfg, require_frozen=True)
    ds = cfg["data"]["dataset_name"]
    proc_root = Path(cfg["data"]["processed_root"]) / ds
    runs_root = ensure_dir(Path("runs") / cfg["run_name"])
    out_root = ensure_dir(runs_root / cfg["rerank"].get("output_subdir", "rerank"))

    device = resolve_device(cfg["ranker"].get("device", "cuda"))
    log_device(device, "Rerank eval")

    news = pd.read_parquet(proc_root / "news.parquet")
    news_meta = build_news_meta(news)

    eval_split = protocol.reporting_split
    impr = pd.read_parquet(impression_artifact_path(proc_root, eval_split))

    scoring_assets = load_rerank_scoring_assets(cfg, proc_root, device)
    teacher_item = scoring_assets.teacher_item
    k_out = policy["k_out"]
    pool_size = policy["pool_size"]
    pos_mode = policy["position_bias"]

    rel_w = policy["relevance_weight"]
    nov_w = policy["novelty_weight"]
    cov_w = policy["coverage_weight"]
    novelty_sim = policy["novelty_sim"]
    relevance_normalization = policy["relevance_normalization"]
    coverage_cfg = dict(policy["coverage"])
    fairness_cfg = dict(policy["fairness"])

    baseline_accumulator = MetricAccumulator()
    reranked_accumulator = MetricAccumulator()

    for scored in iter_scored_impressions(impr, scoring_assets, device):
        baseline_accumulator.add(
            baseline_metrics_for_impression(
                row=scored,
                teacher_item=teacher_item,
                news_meta=news_meta,
                k_out=k_out,
                pool_size=pool_size,
                position_bias=pos_mode,
                category_target=str(
                    fairness_cfg.get("category_target", "catalog")
                ),
            )
        )
        reranked_accumulator.add(
            candidate_metrics_for_impression(
                row=scored,
                teacher_item=teacher_item,
                news_meta=news_meta,
                k_out=k_out,
                pool_size=pool_size,
                position_bias=pos_mode,
                coverage_cfg=coverage_cfg,
                fairness_cfg=fairness_cfg,
                relevance_weight=rel_w,
                novelty_weight=nov_w,
                coverage_weight=cov_w,
                novelty_sim=novelty_sim,
                relevance_normalization=relevance_normalization,
            )
        )

    if baseline_accumulator.count == 0:
        raise RuntimeError(
            f"No labeled impressions with positive clicks were found in {eval_split!r}."
        )

    baseline = baseline_accumulator.mean()
    reranked = reranked_accumulator.mean()
    constraint = _make_constraint(baseline, rr_cfg.get("search", {}))
    out = {
        "selection_method": SELECTION_METHOD,
        "product_constraint": constraint,
        "constraint": _constraint_check(baseline, reranked, constraint),
        "k_out": k_out,
        "pool_size": pool_size,
        "n_impressions_evaluated": baseline_accumulator.count,
        "baseline": baseline,
        "reranked": reranked,
        "delta": _metric_deltas(baseline, reranked),
        "weights": {
            "relevance": rel_w,
            "novelty": nov_w,
            "coverage": cov_w,
        },
        "novelty_sim": novelty_sim,
        "relevance_normalization": relevance_normalization,
        "coverage": coverage_cfg,
        "fairness": fairness_cfg,
        "position_bias": pos_mode,
        "eval_split": eval_split,
        "scoring": scoring_assets.metadata,
        "selection": protocol.selection,
    }
    save_json(out_root / "rerank_eval.json", out)
    _write_rerank_report(out_root, out)
