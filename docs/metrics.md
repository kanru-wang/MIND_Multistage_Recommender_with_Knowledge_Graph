# Metrics in this repo

## Ranking quality (per impression)
- **AUC**: discrimination between clicked vs non-clicked candidates within impressions.
- **MRR**: reciprocal rank of the first clicked item.
- **nDCG@K**: position-aware gain for clicked items; normalized by ideal DCG.
- **Recall@K**: fraction of clicked items captured in top-K.
- **MAP@K**: precision integrated over ranks up to K.

### MIND submission evaluation

The official MIND evaluator reads `prediction.txt` lines as `impression_id [rank,...]`, where rank `1` is the highest-scored candidate. Local MIND metrics report AUC, MRR, nDCG@5, and nDCG@10 using the same per-impression ranking definitions as the official evaluator; leaderboard rank is primarily by AUC.

See the [submission reference](submission.md) for the recency adjustment applied before ranks are written.

## Calibration (global, over all scored pairs)
- **Brier score**: mean squared error of predicted probability vs label.
- **ECE (Expected Calibration Error)**: bins predictions; compares bin accuracy vs confidence.

## Diversity (list-level, top-K)
- **ILD** (intra-list diversity): 1 - average pairwise similarity (teacher cosine) within the list.
- **Category coverage@K**: number of unique categories in top-K.
- **Category entropy@K**: entropy of category distribution in top-K (higher = more spread).

## Exposure fairness (list-level, top-K)
Position-weighted exposure uses a bias curve v(pos) (log or linear).
- **KL disparity**: compare exposure distribution vs target distribution (catalog or uniform).
- **Gini**: inequality of exposure allocation across the union of exposed categories and categories present in the pool target, including target categories that received zero exposure.
- **New-item exposure fraction**: how much position-weighted exposure is allocated to items tagged as new/rare.
- **fairness_kl_pool**: KL divergence between top-K exposure and the reranker's top-`pool_size` candidate mix.
- **fairness_kl_full**: KL divergence between top-K exposure and the full impression candidate mix.

Target definition note:
- `catalog` target means the empirical category mix of the impression candidate pool.
- `uniform` target means equal mass across the categories present in that impression candidate pool.
- The target is not derived from the selected top-K list, and `catalog` here is not a global full-corpus category prior.
- Unknown category `0` is excluded from the target but included in observed exposure. Both greedy selection and evaluation use `1e-12` smoothing in KL, so unknown-category exposure is penalized consistently. It earns no coverage bonus.
- When both `fairness_kl_pool` and `fairness_kl_full` are reported, the first uses the reranker's accessible pool as reference and the second uses the broader full candidate set as reference.

## Notes
- `p` is the actual position-weighted exposure distribution of the selected top-K list across groups such as categories.
- `q` is the target group distribution used for comparison.
    - Higher-ranked items contribute more to `p` because position weights decay with rank.
    - `fairness_kl_pool` compares `p` to the reranker's top-`pool_size` candidate mix.
    - `fairness_kl_full` compares `p` to the full impression candidate mix.
    - Lower KL means the observed exposure pattern is closer to the target pattern.

- Semantic diversity is reported as teacher-cosine ILD on completed lists. It has no term in the greedy score, no guardrail, and no influence on selection or tie-breaking.
- Recommended reranker configs min-max normalize ranker logits within the accessible candidate pool before combining relevance with coverage and KL. This changes score scale, not the baseline relevance order.
- Coverage rewards adding new information to the list. In the current code this means bonus for a previously unseen category and bonus for previously unseen entities.
- New-item exposure fraction is the fraction of total position-weighted exposure assigned to items flagged as new/rare (by train-click-count thresholds). It has no dedicated term in the greedy score; the aggregate search guardrail requires its mean not to decrease. ILD remains diagnostic and has no acceptance guardrail.
