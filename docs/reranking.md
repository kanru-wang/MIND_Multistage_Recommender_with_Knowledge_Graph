# Optional reranking workflow

The reranker is a post-ranking learning exercise. It does not train a model and is not called by `write_submission`. It consumes a frozen ranker checkpoint, considers that ranker's top `pool_size` candidates, and greedily constructs a top-`k_out` list that trades relevance against novelty, coverage, and exposure fairness.

## Completed Large MPNet result

The MPNet candidate-attention experiment is complete. Its frozen policy is
`relevance=0.85`, `novelty=0.05`, `coverage=0.10`, fairness penalty `0.10`, and
new-item floor `0.30`. Selection used 528 policies on November 14; the fixed
policy was then reported once on November 15.

| Metric | Nov 14 tuning delta | Nov 15 reporting delta |
| --- | ---: | ---: |
| Relative nDCG@10 drop | 1.911% | 2.352% |
| Recall@10 | -0.013548 | -0.015779 |
| ILD | +0.019496 | +0.012007 |
| Category coverage@10 | +1.015309 | +0.976867 |
| Category entropy@10 | +0.196082 | +0.185228 |
| Fairness KL, pool | -0.118901 | -0.119241 |
| Fairness KL, full | -0.125949 | -0.125054 |
| Fairness Gini | -0.069443 | -0.064852 |
| New-item exposure | +0.021555 | +0.006301 |

The diversity and fairness effects transferred with the same direction and
similar magnitude. The reporting-day nDCG drop exceeded the 2.1% tuning
guardrail by 0.252 percentage points. New-item exposure had less room to grow
because its reporting-day baseline was already 89.17%, versus 71.47% on the
tuning day. This is retained as the honest final learning result; the policy is
not retuned on November 15.

Artifacts:

- `runs/mind_large_temporal_mpnet_candidate_attention_v1/eval/rerank_search.json`
- `runs/mind_large_temporal_mpnet_candidate_attention_v1/eval/pareto_frontier.md`
- `runs/mind_large_temporal_mpnet_candidate_attention_v1/eval/rerank_eval.json`
- `runs/mind_large_temporal_mpnet_candidate_attention_v1/eval/rerank_eval.md`

## Recommended run order

Use `configs/mind_small.yaml` for a clean validation/test workflow. Complete preprocessing, teacher training, and ranker training first, then run:

```powershell
python -m mindrec.cli rerank_search --config configs/mind_small.yaml
```

Inspect `runs/mind_small_demo/eval/rerank_search.json` and `pareto_frontier.md`. Select an operating point using validation only, copy its `weights` and `fairness` values into the config, and run the fixed report once:

```powershell
python -m mindrec.cli rerank_eval --config configs/mind_small.yaml
```

That writes `rerank_eval.json` and `rerank_eval.md` on the configured `rerank.eval_split` (`test` by default).

For MIND Large, the temporal configs preserve the historical combined `val`
artifact for upstream model comparisons and additionally materialize:

- `rerank_tune`: 431,517 impressions from November 14, 2019, used by
  `rerank_search` and every iteration on priorities, guardrails, and search ranges.
- `rerank_test`: 376,471 impressions from November 15, 2019, reserved for the
  one-time `rerank_eval` report after selection is frozen.

Fresh preprocessing creates both artifacts automatically. For the existing
processed Large Temporal Val, backfill them without rebuilding pairs, maps, or
embeddings. The inherited MPNet candidate-attention config then routes search
and evaluation to the correct days:

```powershell
python -m mindrec.cli prepare_rerank_holdout --config configs/mind_large_temporal_mpnet.yaml
python -m mindrec.cli rerank_search --config configs/mind_large_temporal_mpnet.yaml
```

Shared Large temporal configs start unfrozen. The MPNet config now contains the
completed frozen selection and provenance. Do not rerun its search/evaluation
workflow as if November 15 were still unused; create a distinct experiment if
you want to study a different priority policy.

The November 15 result is independent of the new reranker-selection process,
but it is not a fully untouched end-to-end model holdout: historical upstream
ranker development already examined the combined November 14-15 validation set.

## Iterating on priorities without consuming a report split

For a future experiment with an unused reporting split, treat its guardrails and
utility coefficients as provisional starting values. On its tuning split only:

1. Decide which outcomes are hard guardrails (normally relevance retention) and
   which are preferences used to rank feasible candidates.
2. Run the broader search grid and inspect `best_feasible`,
   `best_scalar_utility`, and the Pareto frontier. Qualitative list review is a
   separate analysis step; the current CLI emits aggregate search metrics, not
   per-impression representative lists or reranker metric slices.
3. Adjust guardrails, utility coefficients, search ranges, or metric definitions
   and repeat until the choice is stable. The Nov-14 rows can be subdivided
   internally for exploratory and confirmation passes if desired.
4. Freeze the complete decision rule and selected parameters before looking at
   the reporting split.

For the completed MPNet experiment, November 15 is now consumed. If its result
is used to change the reranker, the new outcome must be described as follow-up
analysis rather than an independent final report.

## Objective

At each output position, every unselected pool candidate receives:

```text
relevance_weight * relevance
+ novelty_weight * novelty
+ coverage_weight * coverage
- fairness.penalty_weight * fairness_penalty
```

The checked-in learning configs use `relevance_normalization: minmax`, which maps ranker logits within each impression's accessible pool to `[0, 1]`. The baseline ranker order is unchanged, while objective weights remain comparable across checkpoints with different raw-logit scales. `none` exists only for reproducing legacy raw-logit experiments.

- `teacher_cosine` novelty penalizes the maximum cosine similarity to an already selected article.
- Category or entity-Jaccard novelty can be searched by adding them to `search.novelty_sims`.
- Coverage rewards a new category and a capped count of newly covered entities.
- Fairness compares position-weighted category exposure with either the empirical pool mix (`catalog`) or a uniform distribution over pool categories.
- `new_item_floor` is a soft prefix penalty target. It does not guarantee a minimum final exposure; use measured search guardrails when a gain is required.

## Configuration reference

This is an illustrative generic template, not the frozen MPNet selection.
The exact completed policy is in `configs/mind_large_temporal_mpnet.yaml` and is
summarized above.

```yaml
rerank:
  k_out: 10
  pool_size: 50
  eval_split: "test"             # optional; defaults to test
  selection:
    require_frozen_for_eval: false
    require_distinct_splits: false
    require_provenance_for_eval: false
    source_split: "val"
    reporting_split: "test"
    search_artifact: null
    decision_note: null
    frozen: true
  position_bias: "log"           # log|linear
  relevance_normalization: "minmax"  # minmax|none
  relevance_weight: 0.875
  novelty_weight: 0.05
  coverage_weight: 0.075
  novelty_sim: "teacher_cosine"   # teacher_cosine|category|entity_jaccard
  scoring:
    batch_size: 2048
    item_encoding_batch_size: 8192
    impression_batch_size: 128
  coverage:
    max_new_entities_per_item: 3
    category_bonus: 1.0
    entity_bonus: 0.3
  fairness:
    enabled: true
    category_target: "catalog"    # catalog|uniform
    new_item_floor: 0.20
    penalty_weight: 0.25
  search:
    split: "val"                 # optional; defaults to val
    seed: 13
    sample_size: 500
    shortlist_size: 10
    novelty_sims: ["teacher_cosine"]
    weight_pairs:                 # [novelty, coverage]; relevance is the remainder
      - [0.05, 0.05]
      - [0.05, 0.075]
    fairness_penalties: [0.20, 0.25, 0.30]
    new_item_floors: [0.15, 0.175, 0.20]
    relative_guardrails:
      max_ndcg_drop_ratio: 0.021
      min_new_item_exposure_gain: 0.00
      min_category_coverage_gain: 0.25
      min_fairness_kl_pool_improvement: 0.04
    utility_scales:              # optional; each must be finite and > 0
      ndcg_drop_ratio: 0.021
      new_item_exposure_gain: 0.01
      category_coverage_gain: 0.25
      fairness_kl_pool_improvement: 0.04
    utility_coefficients:
      ndcg_retention_units: 4.0
      new_item_exposure_gain_units: 0.5
      category_coverage_gain_units: 1.0
      fairness_kl_pool_improvement_units: 1.0
```

Search scores all configured grid points on a deterministic sample, then full-validates a bounded round-robin shortlist drawn from feasible-first, scalar-utility, and Pareto rankings. The currently configured point is also full-evaluated when it is not already shortlisted.

Every utility term uses a meaningful positive scale. Relevance retention is
`1 - relative_ndcg_drop / ndcg_drop_ratio_scale`, so it falls from one unit at
the baseline to zero at the configured drop scale. Each improvement is divided
by its own scale. If a scale is omitted, its positive guardrail is used; when
that guardrail is zero, the scale defaults to `1.0`. For that reason, configure
an explicit new-item scale (for example, `0.01` for a one percentage-point gain)
when its minimum guardrail is zero.

Coverage and category-fairness are correlated. Giving both very large
coefficients can double-count diversity, so the Large temporal starting policy
weights each at `1.0`, below the `4.0` relevance coefficient.

## Reading the output

- `best_feasible` is the highest-utility point satisfying every guardrail.
- `best_scalar_utility` ignores feasibility and can reveal a more aggressive trade-off.
- `pareto_frontier` contains nondominated full-validation points.
- Negative `delta.fairness_kl_*` or `delta.fairness_gini` is an improvement.
- Positive deltas are improvements for nDCG, recall, ILD, coverage, entropy, and new-item exposure.
- `delta.ndcg_drop_ratio` reports the relative relevance cost directly.

The reports record the scoring mode and checkpoint history-pooling mode. Current reranker scoring uses the shared cached item/history semantics path, including candidate-aware history attention when enabled by the frozen ranker checkpoint.
