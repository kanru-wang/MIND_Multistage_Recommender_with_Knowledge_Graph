from __future__ import annotations

import argparse
from pathlib import Path

from mindrec.config import load_config
from mindrec.data.item_age import run_build_item_age
from mindrec.data.taxonomy_support import run_build_taxonomy_support
from mindrec.pipeline.evaluate import run_evaluate
from mindrec.pipeline.preprocess import run_preprocess
from mindrec.pipeline.ranker_train import run_train_ranker
from mindrec.pipeline.ranker_lr_sweep import run_train_ranker_lr_sweep
from mindrec.pipeline.rerank_eval import run_rerank_eval
from mindrec.pipeline.rerank_search import run_rerank_search
from mindrec.pipeline.retrieval import (
    run_build_index,
    run_eval_retrieval,
    run_eval_retrieval_sweep,
)
from mindrec.pipeline.submission import run_write_submission
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
        "train_ranker_lr_sweep",
        help="Train ranker variants over ranker.lr_sweep.lrs and summarize val metrics",
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

    p = sub.add_parser(
        "write_submission",
        help="Score the hidden MIND test split and write leaderboard prediction.txt",
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "build_item_age",
        help="Build the first-seen article-age index used by alpha=0.02",
    )
    _add_config_arg(p)

    p = sub.add_parser(
        "build_taxonomy_support",
        help="Bind exact trained category/subcategory IDs to a ranker checkpoint",
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
    if args.cmd == "train_ranker_lr_sweep":
        run_train_ranker_lr_sweep(cfg)
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
    if args.cmd == "write_submission":
        run_write_submission(cfg)
        return
    if args.cmd == "build_item_age":
        run_build_item_age(cfg)
        return
    if args.cmd == "build_taxonomy_support":
        run_build_taxonomy_support(cfg)
        return

    raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
