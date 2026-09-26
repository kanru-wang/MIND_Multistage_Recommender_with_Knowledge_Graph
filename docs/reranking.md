# Reranking

Offline search selects the **highest mean nDCG@10 among settings meeting all aggregate guardrails**. Each setting builds actual top-10 lists from one frozen ranker's top-50 pools, without ensembling. The winning parameters and the same greedy algorithm are used at serving time. This is separate from competition submission generation.

## Score and metric definitions

For article `i` and selected prefix `S`:

```text
score(i | S) = wR * R(i) + wC * C(i, S) - lambda * KL(p || q)
wR = 1 - wC
```

- `R`: min-max-normalized ranker relevance in the accessible pool.
- `C`: `1.0 * new_category + 0.3 * min(new_entity_count, 3)`. Unknown categories earn no category bonus. Covered categories and entities update after selection.
- `p`: position-weighted category exposure of the prospective prefix including `i`; `q`: the accessible pool's category-frequency distribution.

With `position_bias: log`, position `r` has weight `1 / log2(r + 1)`. `category_target: catalog` means the accessible pool, not the entire catalog. The KL coefficient inside the penalty is one. There is no L1 term, new-item shortfall term, or soft new-item floor. New-item exposure remains an aggregate acceptance requirement; it does not directly enter this greedy score.

Unknown category `0` earns no coverage bonus and is excluded from the target distribution. Its exposure is included in both selection and evaluation KL, using the same `1e-12` smoothing for zero target probability. This penalizes unknown-category exposure consistently rather than treating it as free exposure.

Semantic novelty is removed from the score and search. Teacher-cosine ILD (`1 - mean pairwise similarity` on the completed list) remains diagnostic only, with no guardrail or selection tie-breaker. Teacher embeddings are still used for this metric and the upstream model, but the greedy builder needs no embeddings or similarity matrix.

Completed-list metrics are averaged equally over labeled impressions with positive clicks. Category coverage counts distinct known categories; new-item exposure is a position-weighted fraction. `fairness_kl_pool` uses the accessible pool as reference; `fairness_kl_full` uses the original impression candidate set. Guardrails use the pool metric. These are topic-exposure measures, not demographic fairness measures.

## Search and selection

The checkpoint, teacher embeddings, pool size, list length, score definitions, and requirements are fixed before search. Business decisions concern outcomes; the technical search determines `wC` and `lambda`, deriving `wR = 1 - wC`. Configure only `coverage_weight` and the KL penalty weight. The resolved policy and result artifacts include derived relevance for transparency; YAML must not contain `relevance_weight`.

| Requirement relative to the original ranker | Value |
| --- | ---: |
| Maximum relative nDCG@10 loss | 0.02 |
| Minimum new-item exposure gain | 0.00 |
| Minimum mean category-coverage gain | 0.25 |
| Minimum mean pool-KL reduction | 0.03 |

Zero exposure gain requires non-regression; it is not disabled. Coverage and KL gains are absolute differences, not percentages. No scalar utility scales or coefficients are needed.

The existing `configs/mind_large_temporal_mpnet.yaml` now runs a 15-setting local refinement:

```yaml
coverage_weights: [0.025, 0.03, 0.035, 0.04, 0.05]
fairness_penalties: [0.00, 0.025, 0.05]
shortlist_size: 15
```

1. Score November 14 (`rerank_tune`) with the frozen ranker. Its top ten form the baseline. Keep scored pools fixed across trials.
2. Build complete lists for all 15 settings on a deterministic 5,000-impression sample, seed 13. Labels are used only afterward to evaluate the lists.
3. With `shortlist_size: 15`, every grid combination proceeds to full tuning evaluation. Include the configured starting setting as well if it is outside the grid. The current MPNet starting setting is already in the grid.
4. Evaluate all combinations on the same full tuning impressions. Reject any setting failing a guardrail, then select the highest-nDCG eligible setting.
5. Inspect range diagnostics and refine as below. Freeze after the final full-tuning comparison, then report on November 15.

`best_feasible` is the selection. With no eligible finalist it is null and `selection_status` is `no_feasible_policy`; never freeze an infeasible fallback. Exact nDCG ties use lexicographic parameter order: relevance, coverage, penalty. Only floating-point roundoff (`1e-12`) is tolerated at guardrail boundaries. The Pareto frontier is diagnostic, not the selector. The current grid is fully evaluated, so the result is its highest-nDCG eligible setting. If expanding the grid beyond `shortlist_size`, increase that budget to keep full evaluation. With a smaller budget the search reserves one third of shortlist slots for sample rejects, but may miss the full-grid optimum.

## Choosing score definitions and a search grid

The category bonus of 1.0 sets a score unit. Entity bonus 0.3 and cap three are fixed modeling choices limiting annotation-heavy articles' reward. Search controls their combined influence through `wC`. Searching both these bonuses and their multiplier would introduce overlapping controls.

Use this process in the same config, keeping guardrails fixed:

1. **Start broad with zero controls.** For a new range discovery, the shared defaults test coverage `[0, 0.025, 0.05, 0.10]` and KL penalties `[0, 0.025, 0.05, 0.10, 0.20]`, including ranker-only ordering. These are starting hypotheses, not universal ranges. The current local grid narrows the promising region after that comparison; it retains zero KL but omits zero coverage, which failed the coverage requirement throughout the tested broad grid. Require nonnegative weights and `0 <= wC < 1`.
2. **Check scales and available candidates.** Open `pareto_frontier.md` or `rerank_search.json`'s `grid_diagnostics`. Compare reported adjacent normalized relevance gaps times `wR` with coverage changes (at most `1.9 * wC`) and `lambda * KL_difference`. At `wC=0.025`, a new category contributes 0.025 and the maximum category-plus-entity bonus is 0.0475. Reported first-position KL spans are a scale clue, not a bound for every prefix. Relevance normalization does not normalize KL or coverage.
3. **Diagnose failures before widening.** The report provides guardrail-failure counts, feasible counts by parameter value, and best feasible full-tuning nDCG. Profiles and boundary flags use full-tuning results whenever available. For a partially evaluated grid, untested points remain empty; the report explicitly labels this scope. Only a run with no full-tuning results falls back to sample profiles. Pool-opportunity statistics still use the sample. Optimistic pool-based coverage and exposure ceilings show whether candidates offer enough opportunity. Larger weights cannot create missing categories. These ceilings ignore relevance and joint constraints. Separate maximum gains may come from different settings and do not prove joint feasibility. Persistent new-item exposure failures may mean this score cannot meet that requirement; do not silently relax it or increase unrelated weights forever.
4. **Expand supported ranges.** An upper-boundary winner is flagged with a candidate extension, normally twice the upper value. A penalty winner at 0.20 suggests testing 0.40 if the profile remains promising within the nDCG budget. Preserve earlier promising values and zero controls. A boundary winner alone does not prove larger values help. With no feasible winner, inspect failure/profile trends to choose a direction; the tool does not invent a winner or relax thresholds.
5. **Refine around the full-tuning winner.** `suggested_local_values` includes the winner and midpoints toward adjacent tested values. A winner at 0.05 between 0.025 and 0.10 gives `[0.0375, 0.05, 0.075]`. Copy the suggested axes into the same config; set `shortlist_size` to at least their Cartesian-product size (at most nine for three values per axis). Retain the winning combination and evaluate every local combination on full tuning data. Retain useful zero-control comparisons and increase the shortlist accordingly.
6. **Check stability and stop deliberately.** Sample/full winner disagreement means the sample alone is unreliable; use full-tuning results. If only a shortlist was fully evaluated, increase the sample or full-evaluation budget before trusting the region. `full_grid_evaluated` confirms whether all tested combinations received full evaluation. Stop when a full local comparison yields no material nDCG improvement and no supported boundary expansion remains. This is a practical stopping rule, not proof of a global optimum. Keep the final grid and selected parameters in the config for reproduction.

Profiles take the best results across other parameter choices; they are not isolated causal effects. Suggestions do not change selection or edit the config automatically. ILD does not influence eligibility, selection, or range flags.

A **zero-weight control** disables one component, such as `wC=0` to remove coverage or `lambda=0` to remove KL. Keep the other independent weight fixed; `wR=1-wC` is still recalculated. Setting both to zero recovers the ranker's ordering, a useful baseline even if it fails improvement guardrails.

A grid need not contain a failing setting. Compare eligible settings by nDCG and investigate promising boundaries instead of expanding solely to force a failure. If new-item exposure repeatedly fails, inspect candidate availability and consider a direct new-item bonus as a separate experiment; the current score has none.

## Current status and reproduction

The final search evaluated all 15 combinations on 431,517 November 14 tuning impressions; 11 passed every guardrail. The frozen selection is coverage weight **0.035**, KL penalty **0**, and derived relevance weight **0.965**. Both the screening sample and full-tuning evaluation selected this setting.

| Metric | Tuning baseline | Frozen selection | Change |
| --- | ---: | ---: | --- |
| nDCG@10 | 0.424166344 | 0.423647826 | 0.122244% relative loss |
| New-item exposure | 0.714744781 | 0.716921479 | +0.217670 percentage points |
| Categories/list | 5.286097651 | 5.585281692 | +0.299184 |
| Pool KL | 0.417462133 | 0.387194205 | Reduction 0.030268 |
| Teacher-cosine ILD | 0.573309446 | 0.576908985 | +0.003600; diagnostic only |

**Selection decision:** choose the highest full-tuning nDCG among settings meeting the predetermined guardrails on tuning data. This setting met all four tuning requirements, so the selected KL penalty is zero. Its tuning KL reduction exceeded the threshold by only 0.000268. This section records the original selection decision; the reporting outcome below does not change that decision retroactively.

**Reporting outcome: three of four guardrails pass.** The frozen setting was applied to all 376,471 November 15 reporting impressions. The artifact matches the frozen parameters and score definitions; its deltas and constraint checks are consistent. Pool-KL reduction misses the required 0.03 by 0.001264.

| Metric | Reporting baseline | Reranked | Change | Requirement | Result |
| --- | ---: | ---: | --- | --- | --- |
| nDCG@10 | 0.439456710 | 0.438302290 | 0.262693% relative loss | At most 2% loss | Pass |
| New-item exposure | 0.891706784 | 0.892422982 | +0.071620 percentage points | No decline | Pass |
| Categories/list | 4.955560986 | 5.236411304 | +0.280850 | At least +0.25 | Pass |
| Pool KL | 0.442186680 | 0.413450382 | Reduction 0.028736 | At least 0.03 reduction | **Fail** |
| Teacher-cosine ILD | 0.590849660 | 0.592672516 | +0.001823 | Diagnostic only | No guardrail |

Recall@10 also decreased from 0.697696 to 0.696025. Category entropy increased and Gini decreased; the generated report records all diagnostic metrics. Full-candidate KL reduction is 0.030125, but the acceptance requirement uses **pool KL**, so it cannot substitute for the failed pool metric.

The run is technically valid, but the selected setting has not met every reporting requirement. Do not describe it as passing all guardrails or approved for production under these requirements. `frozen: true` preserves the evaluated selection for reproduction; it is not a production-approval flag. The narrow tuning KL margin did not carry over to reporting, illustrating why tuning feasibility alone is insufficient.

The measured results, thresholds, and frozen parameters remain unchanged. Do not lower the threshold or choose another weight using this reporting result and present it as the original successful evaluation. Any further development should define its selection rule on tuning data and identify a new reporting period for an independent assessment. November 15 was already reused and is a follow-up report, as stated in the config.

Prepare temporal data, MPNet teacher embeddings, and the ranker checkpoint using the main modeling workflow. Search does not train or ensemble models. If the day-specific views are missing, run `prepare_rerank_holdout` first.

```powershell
python -m mindrec.cli rerank_search --config configs/mind_large_temporal_mpnet.yaml
```

The current config contains the selected coverage and KL weights, the search artifact path, this decision note, and `frozen: true`. To reproduce the completed reporting evaluation, run:

```powershell
python -m mindrec.cli rerank_eval --config configs/mind_large_temporal_mpnet.yaml
```

Evaluation rejects mismatched selections and obsolete scoring versions. Provenance records model/data paths and score definitions, not content hashes; do not replace these files between search and reporting. Unchanged inputs and a compatible numerical environment should reproduce results within numerical tolerance. Retraining upstream models need not produce identical checkpoints. November 15 was reused while refining requirements; its report is a follow-up evaluation, not an independent test. Do not tune weights against it.

Outputs share `runs/mind_large_temporal_mpnet_candidate_attention_v1/rerank/`:

- `rerank_search.json`: screened/full metrics, selection, and range diagnostics.
- `pareto_frontier.md`: readable candidate comparison and grid diagnostics.
- `rerank_eval.json` and `rerank_eval.md`: frozen reporting results.

Commands overwrite their respective files. When changing the formula or selection, invalidate previous reporting results before publishing new ones. Search artifacts identify the data with `search_split` and `reporting_split`; the evaluation artifact's `eval_split` refers only to the reporting split.

See [Serving behavior in the README](../README.md#serving-behavior) for list construction, aggregate monitoring, and fallback behavior.
