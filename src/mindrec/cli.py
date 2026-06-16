from __future__ import annotations

import argparse
from pathlib import Path

from mindrec.config import load_config
from mindrec.pipeline.build_ranker_kg import run_build_ranker_kg
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.preprocess import run_preprocess
from mindrec.pipeline.ranker_distill_sweep import run_ranker_distill_sweep
from mindrec.pipeline.ranker_gate_sweep import run_ranker_kg_gate_sweep
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.pipeline.rerank_eval import run_rerank_eval
from mindrec.pipeline.rerank_search import run_rerank_search
from mindrec.pipeline.retrieval import (
    run_build_index,
    run_eval_retrieval,
    run_eval_retrieval_sweep,
)
from mindrec.pipeline.teacher_train import run_train_teacher


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, type=str, help="Path to YAML config")


def main() -> None:
    parser = argparse.ArgumentParser(prog="mindrec")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("preprocess", help="Parse raw MIND TSV into processed parquet")
    _add_config_arg(p)

    p = sub.add_parser(
        "train_teacher", help="Train/compute teacher embeddings (item+user)"
    )
    _add_config_arg(p)

    p = sub.add_parser("build_index", help="Build Faiss ANN index for retrieval")
    _add_config_arg(p)

    p = sub.add_parser(
        "eval_retrieval",
        help="Evaluate retrieval recall@K on splits (val and test) from eval.report_splits",
    )
    _add_config_arg(p)
    p = sub.add_parser(
        "eval_retrieval_sweep",
        help="Sweep hybrid retrieval settings on the validation split",
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "train_ranker", help="Train DLRM student ranker with distillation"
    )
    _add_config_arg(p)
    p = sub.add_parser(
        "build_ranker_kg",
        help="Rebuild ranker-only KG item features without retraining the text teacher",
    )
    _add_config_arg(p)
    p = sub.add_parser(
        "ranker_kg_gate_sweep",
        help="Train and evaluate fixed KG-gate rankers using the validation split for selection",
    )
    _add_config_arg(p)
    p = sub.add_parser(
        "ranker_distill_sweep",
        help="Tune final-logit distillation using validation nDCG@10 for selection",
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "evaluate",
        help="Evaluate ranker on splits (val and test) from eval.report_splits (many metrics)",
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "rerank_eval", help="Evaluate diversity+coverage+fairness reranker, on test data only"
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "rerank_search", help="Search reranker hyperparameters on val data under product constraints"
    )
    _add_config_arg(p)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.cmd == "preprocess":
        run_preprocess(cfg)
        return
    if args.cmd == "train_teacher":
        run_train_teacher(cfg)
        return
    if args.cmd == "build_index":
        run_build_index(cfg)
        return
    if args.cmd == "eval_retrieval":
        run_eval_retrieval(cfg)
        return
    if args.cmd == "eval_retrieval_sweep":
        run_eval_retrieval_sweep(cfg)
        return
    if args.cmd == "train_ranker":
        run_train_ranker(cfg)
        return
    if args.cmd == "build_ranker_kg":
        run_build_ranker_kg(cfg)
        return
    if args.cmd == "ranker_kg_gate_sweep":
        run_ranker_kg_gate_sweep(cfg)
        return
    if args.cmd == "ranker_distill_sweep":
        run_ranker_distill_sweep(cfg)
        return
    if args.cmd == "evaluate":
        run_evaluate(cfg)
        return
    if args.cmd == "rerank_eval":
        run_rerank_eval(cfg)
        return
    if args.cmd == "rerank_search":
        run_rerank_search(cfg)
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
