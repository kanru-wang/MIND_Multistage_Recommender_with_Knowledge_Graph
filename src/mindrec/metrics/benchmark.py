from __future__ import annotations

from typing import Any


OFFICIAL_MIND_METRIC_KEYS = {
    "AUC": "auc",
    "MRR": "mrr",
    "nDCG@5": "ndcg@5",
    "nDCG@10": "ndcg@10",
}


MIND_BENCHMARK_REFERENCES: list[dict[str, Any]] = [
    {
        "name": "MIND official leaderboard top entry",
        "dataset": "MIND-large's hidden test dataset",
        "source": "https://msnews.github.io/",
        "retrieved_on": "2026-06-03",
        "metrics": {
            "AUC": 0.7316,
            "MRR": 0.3768,
            "nDCG@5": 0.4162,
            "nDCG@10": 0.4722,
        },
        "note": "Official leaderboard values are from the MIND-large's hidden test dataset and are not directly comparable to local MIND-small dev dataset (which is split into our validation/test).",
    },
    {
        "name": "CAUM, SIGIR 2022",
        "dataset": "MIND (paper does not explicitly state whether used small or large dataset)",
        "source": "https://www.atailab.cn/seminar2022Spring/pdf/2022_SIGIR_News%20Recommendation%20with%20Candidate-aware%20User%20Modeling.pdf",
        "metrics": {
            "AUC": 0.7004,
            "MRR": 0.3471,
            "nDCG@5": 0.3789,
            "nDCG@10": 0.4357,
        },
        "note": "Paper does not explicitly identify the dataset, but most likely used MIND-large's hidden test dataset.",
    },
]


def official_mind_benchmark_view(ranking: dict[str, float]) -> dict[str, Any]:
    model_metrics = {
        official_name: float(ranking[local_name])
        for official_name, local_name in OFFICIAL_MIND_METRIC_KEYS.items()
        if local_name in ranking
    }
    return {
        "metrics": model_metrics,
        "references": MIND_BENCHMARK_REFERENCES,
        "note": "These references are not directly comparable to this repo's time-split MIND-small dev evaluation.",
    }
