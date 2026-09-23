# Reranking

The reranker chooses the highest-scoring remaining article from the frozen
ranker's top-50 pool until it has ten articles, or exhausts the pool. Offline
search chooses its fixed parameters by **highest mean nDCG@10 subject to
aggregate guardrails**. It uses one ranker checkpoint, without ensembling, and
is separate from `write_submission`.

The [README reranking section](../README.md#36-reranking-search-offline-use-fixed-weights-in-production)
explains the score, search variables, and serving procedure.

## Score and metric definitions

For article `i` and selected prefix `S`:

```text
score(i | S) = wR * R(i) + wN * N(i, S) + wC * C(i, S) - lambda * P(i, S; f)
wR = 1 - wN - wC
```

`R` is min-max-normalized ranker relevance in the pool. `N` is negative maximum
teacher cosine similarity to already selected articles (zero for an empty
prefix). `C` is `1.0 * new_category + 0.3 * min(new_entity_count, 3)`; unknown
categories do not earn a category bonus. The entity set is updated after each
selection.

`P` evaluates the prospective prefix including `i`:

```text
P = 0.5 * KL(p || q) + 0.5 * sum_c |p[c] - q[c]| + 2 * max(0, f - e)
```

With `position_bias: log`, rank `r` has weight `1 / log2(r + 1)`.
`p` is category exposure normalized by total prefix exposure, `q` is the
pool's category-frequency distribution, and `e` is the fraction of exposure
assigned to new items. The setting `category_target: catalog` refers to the
accessible candidate pool, rather than the complete catalog. `f` is a soft
penalty target; it is not a quota or the aggregate exposure-gain guardrail.

The new-item term `2 * max(0, f - e)` is twice the prospective prefix's
shortfall below its soft new-item exposure target. For `f=0.40` and `e=0.25`,
it contributes `0.30` to the penalty, reducing the total score by `0.015` when
`lambda=0.05`. At or above the floor it contributes zero. The constant `2`
sets its strength relative to category mismatch; it is a technical coefficient,
not an extra business threshold. Setting `f=0` disables only this term and
leaves the category penalty active.

The reported metrics are computed on completed lists, then averaged equally
over labeled impressions with positive clicks. `fairness_kl_pool` uses the
accessible pool as its reference; `fairness_kl_full` uses the original impression's
candidate set. Guardrails use the pool metric. Category coverage counts distinct
known categories, and new-item exposure is a position-weighted fraction.
These topic-exposure metrics do not measure demographic fairness.

## Search and selection

Use `configs/mind_large_temporal_mpnet.yaml`. It fixes the checkpoint, scoring
definitions, pool size, list length, and requirements before each search:

| Requirement relative to the original ranker | Value |
| --- | ---: |
| Maximum relative nDCG loss | 0.02 |
| Minimum new-item exposure gain | 0.00 |
| Minimum mean category-coverage gain | 0.25 |
| Minimum mean pool-KL reduction | 0.03 |

The 18-point grid combines novelty weights `[0, 0.025, 0.05]`, coverage weight
`0.025`, penalties `[0.05, 0.075, 0.10]`, and soft floors `[0.30, 0.40]`.
Relevance is derived from novelty and coverage. A deterministic 5,000-impression
screen uses seed 13, and `shortlist_size: 18` ensures that every setting receives
full-tuning evaluation. No sample rejection excludes a setting from this grid.
Other configs can use the broader shared grid defaults.

Compare each complete-list metric against the baseline on the same impressions.
Reject a setting if any guardrail fails, then choose the highest-nDCG remaining
setting. Exact nDCG ties use lexicographic parameter order: similarity,
relevance, novelty, coverage, penalty, floor. Secondary gains do not compensate
for lower nDCG once all requirements are satisfied. Only floating-point roundoff
(`1e-12`) is tolerated at a constraint boundary.

`best_feasible` is the selection. If no setting qualifies, it is null and
`selection_status` is `no_feasible_policy`; do not freeze an infeasible fallback.
`results`, `sample_results`, and `failed_guardrails` explain the comparison.
The Pareto frontier is diagnostic and does not select the winner.

## Choosing score definitions and a search grid

Guardrails describe desired outcomes in interpretable units. Coverage bonuses,
entity caps, and local weights describe how the algorithm attempts to obtain
those outcomes, so business stakeholders need not specify them directly.

- A category bonus of 1.0 establishes a convenient score unit.
- An entity bonus of 0.3 makes one newly covered entity worth 30% of one newly
  covered category within the coverage component; this ratio is a modeling
  choice, not a measured product preference.
- The cap of three limits the reward for articles with many entity annotations.
- The coverage weight controls the overall influence of that component. It is
  a search parameter even though the current compact grid holds it at 0.025.

At the current weight, a new category adds 0.025 to the greedy score, each
eligible new entity adds 0.0075, and the maximum combined coverage contribution
is 0.0475. These figures are easier to calibrate against predicted-relevance
score gaps than asking stakeholders to choose raw reranker coefficients.
Keep score definitions fixed while searching weights. Changing the entity
ratio or cap is a separate comparison requiring reevaluation; tuning every
bonus and its overall multiplier simultaneously adds overlapping controls.

To choose an initial grid:

1. Inspect component values and relevance-score gaps on a deterministic tuning
   sample. Choose weights capable of changing close article decisions without
   immediately overwhelming relevance; normalized relevance alone does not
   normalize the other terms.
2. Include zero-weight controls and explore a coarse range. Possible starting
   values are `[0, 0.025, 0.05, 0.10]` for novelty and coverage and
   `[0, 0.05, 0.10, 0.20]` for the penalty. They are hypotheses, not universal
   optimal ranges. Require `wN + wC < 1`. If retaining the new-item term, include
   `f=0` as an ablation alongside plausible positive floors.
3. Inspect which guardrails fail and how often lists change. If all settings
   fail or rankings barely change, diagnose the component scale and candidate
   pool before expanding weights. No weight can add an unavailable category.
4. If the best eligible points lie on an explored boundary and the metric
   trend supports it, extend that range. Refine promising regions with smaller
   steps, then compare finalists on full tuning data. A boundary winner alone
   does not prove that extending the range will improve the result.
5. Freeze the selected score definitions and parameters before reporting.
   Even a broad finite grid identifies only its best measured eligible setting,
   not a proven global optimum.

For a simpler score, compare a version with `f=0` against the current version
while retaining the aggregate new-item exposure guardrail. The current exposure
requirement is non-regression, so a dedicated prefix target need not be retained
unless it helps meet that requirement. Removing it changes the algorithm and
requires a new search/evaluation; the current published results still use
`f=0.40` and the term above.

**Offline and serving must use the same list builder.** For each trial setting,
offline search constructs complete top-10 lists using only information available
to the reranker at serving time. It then uses click labels to measure nDCG@10.
The highest-nDCG eligible setting is copied to production unchanged, including
its fixed bonuses, normalization, pool size, and list length. Offline selection
does not produce a second set of production weights.

## Selected setting and results

The selected weights are relevance **0.95**, novelty **0.025**, coverage
**0.025**, fairness penalty **0.05**, and soft floor **0.40**. Sixteen of the
18 settings qualify on all 431,517 November 14 tuning impressions.

| Metric | Tuning baseline | Tuning reranked | Reporting baseline | Reporting reranked |
| --- | ---: | ---: | ---: | ---: |
| nDCG@10 | 0.424166344 | 0.423388827 | 0.439456710 | 0.438176210 |
| New-item exposure | 0.714744781 | 0.721941456 | 0.891706784 | 0.893975385 |
| Mean categories/list | 5.286097651 | 5.558770570 | 4.955560986 | 5.209904614 |
| Pool KL | 0.417462133 | 0.380625059 | 0.442186680 | 0.408160281 |

The tuning nDCG loss is 0.183305%; the reporting loss is 0.291383%. All four
requirements pass on both splits. Reporting covers 376,471 November 15
impressions. November 15 was reused while refining requirements; these are
follow-up results, not an independent test. Passing these aggregate checks
does not guarantee every list or a future traffic window will pass.

This section is the config's selection decision note: choose the highest
full-tuning nDCG satisfying the stated requirements, copy that setting, then
freeze it before running its reporting evaluation.

## Reproduce the process

Prepare the temporal data, MPNet teacher embeddings, and ranker checkpoint
using the main modeling workflow first. The search neither trains the ranker
nor depends on another reranking experiment's output. Keep these artifacts
fixed when comparing reranker settings. If the day-specific views are missing:

```powershell
python -m mindrec.cli prepare_rerank_holdout --config configs/mind_large_temporal_mpnet.yaml
```

Run the complete reranking workflow:

```powershell
python -m mindrec.cli rerank_search --config configs/mind_large_temporal_mpnet.yaml
python -m mindrec.cli rerank_eval --config configs/mind_large_temporal_mpnet.yaml
```

The current config contains the selected weights, `frozen: true`, and the
provenance path. Search rebuilds `rerank_search.json` from the ranker scores;
it does not read or require the previous search artifact. Because all 18 grid
points receive full evaluation, the screen cannot change the selected full-grid
winner. With unchanged prepared data, checkpoint, scoring definitions, and
compatible numerical environment, the selected setting and metrics should
reproduce within numerical tolerance. Retraining the upstream models is a
separate experiment and need not produce identical checkpoints.

If search selects another setting after changing any input, copy its three
weights, fairness penalty, floor, category target, and novelty similarity into
the config. Record the search artifact and decision note, then freeze before
evaluation. Evaluation refuses a frozen config that does not match its artifact.
The provenance validator compares model/data paths and score definitions; it
does not hash checkpoint or data contents. Do not replace files at those paths
between search and reporting.

All current outputs are in
`runs/mind_large_temporal_mpnet_candidate_attention_v1/rerank/`:

- `rerank_search.json`: full and sampled metrics, requirements, and selected setting.
- `pareto_frontier.md`: a readable diagnostic comparison.
- `rerank_eval.json` and `rerank_eval.md`: frozen reporting metrics and guardrail checks.
- `list_review.json`: a descriptive tuning sample and selected before/after lists.

Search and evaluation overwrite their respective outputs when rerun. The
existing full-tuning measurements were retained when consolidating the results;
selection was recalculated against the current guardrails. No previous run is
needed to regenerate these outputs. `list_review.json` is an optional inspection
artifact; the search and evaluation commands do not generate it or require it.

## Example list and serving behavior

In tuning impression `1703589`, the selected setting reduced lifestyle articles
from six to four, increased category coverage from four to six, and moved a
clicked news article from rank six to four. Sports and TV articles entered from
baseline ranks twelve and eleven. nDCG increased by 0.045661 for that list.
In another inspected impression (`155352`), adding a category displaced a
clicked article at rank ten. Aggregate success does not imply improvement for
every user.

The serving algorithm uses the selected fixed weights, chooses the highest
score at each step, updates the list state, and repeats. It cannot calculate
actual nDCG before observing relevance labels. Prefix diversity and exposure
are score features; aggregate guardrails apply to completed lists across
impressions, not to each added article. Enforcing a hard per-list requirement
would require a separate mechanism. Production monitoring and rollback services
are not implemented in this repository.
