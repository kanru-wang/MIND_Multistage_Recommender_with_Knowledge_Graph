# Pareto Frontier Summary

Source: `runs/mind_small_demo/eval/rerank_search.json`

Baseline: nDCG@k=0.405304, new_item_exposure_frac=0.565136, category_coverage=5.413288, fairness_kl_pool=0.406672

Guardrails: max_ndcg_drop_ratio=0.021, min_new_item_exposure_gain=0.00, min_category_coverage_gain=0.25, min_fairness_kl_pool_improvement=0.04

Best feasible: nDCG@k=0.397311, new_item_exposure_frac=0.583991, category_coverage=6.199484, fairness_kl_pool=0.291011, fairness_penalty=0.30, new_item_floor=0.20

Best scalar utility: nDCG@k=0.394854, new_item_exposure_frac=0.585410, category_coverage=6.326601, fairness_kl_pool=0.278173, fairness_penalty=0.30, new_item_floor=0.20

| # | Feasible | nDCG@k | New Item Exposure | Category Coverage | Fairness KL | Intra-List Diversity | Relevance Weight | Novelty Weight | Coverage Weight | Fairness Penalty | New Item Floor | Utility |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Y | 0.397311 | 0.583991 | 6.199484 | 0.291011 | 0.400985 | 0.89 | 0.05 | 0.06 | 0.30 | 0.20 | 12.985003 |
| 2 | Y | 0.397352 | 0.582354 | 6.199313 | 0.291242 | 0.400981 | 0.89 | 0.05 | 0.06 | 0.30 | 0.17 | 12.974937 |
| 3 | Y | 0.397870 | 0.579667 | 6.167581 | 0.299381 | 0.400814 | 0.89 | 0.05 | 0.06 | 0.25 | 0.17 | 12.483081 |
| 4 | Y | 0.398087 | 0.582313 | 6.136686 | 0.296333 | 0.401570 | 0.88 | 0.07 | 0.05 | 0.30 | 0.17 | 12.415467 |
| 5 | Y | 0.398309 | 0.578140 | 6.135421 | 0.307658 | 0.400659 | 0.89 | 0.05 | 0.06 | 0.20 | 0.20 | 11.983306 |
| 6 | N | 0.394854 | 0.585410 | 6.326601 | 0.278173 | 0.401080 | 0.88 | 0.05 | 0.07 | 0.30 | 0.20 | 14.205597 |
| 7 | N | 0.394874 | 0.583704 | 6.326361 | 0.278380 | 0.401070 | 0.88 | 0.05 | 0.07 | 0.30 | 0.17 | 14.195738 |
| 8 | N | 0.394908 | 0.582155 | 6.326361 | 0.278595 | 0.401061 | 0.88 | 0.05 | 0.07 | 0.30 | 0.15 | 14.187244 |
| 9 | N | 0.395543 | 0.582374 | 6.296492 | 0.286113 | 0.400912 | 0.88 | 0.05 | 0.07 | 0.25 | 0.20 | 13.732479 |
| 10 | N | 0.396152 | 0.579373 | 6.266486 | 0.294340 | 0.400741 | 0.88 | 0.05 | 0.07 | 0.20 | 0.20 | 13.248437 |
| 11 | N | 0.396253 | 0.578239 | 6.266332 | 0.294496 | 0.400732 | 0.88 | 0.05 | 0.07 | 0.20 | 0.17 | 13.242106 |
| 12 | N | 0.396387 | 0.577127 | 6.266110 | 0.294651 | 0.400725 | 0.88 | 0.05 | 0.07 | 0.20 | 0.15 | 13.235736 |
