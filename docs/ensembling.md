# Ensembling: MPNet and MiniLM

Ensembling is an optional submission-level extension, separate from the
single-model architecture and training workflow in the [README](../README.md).
It combines predictions from independently trained MPNet and MiniLM models
without retraining either model.

The best reported ensemble achieves **Large Test AUC 0.6972**, using
**75% improved MPNet + 25% selected candidate-attention MiniLM**. This is
`+0.0012` over both the best single MPNet and the original ensemble.

## Results at a glance

| Model or ensemble | Large Test AUC | Role |
| --- | ---: | --- |
| Selected candidate-attention MiniLM | 0.6869 | MiniLM member in both ensembles |
| Original MPNet, text-encoder `lr=2e-5` | 0.6948 | Original MPNet member |
| Original MPNet + MiniLM, 75/25 | 0.6960 | First ensemble |
| Improved MPNet, text-encoder `lr=1e-5` (`31ab682`) | 0.6960 | Best reported single model |
| Improved MPNet + MiniLM, 75/25 | **0.6972** | Best reported ensemble |

Large Test scores are user-reported competition-platform results; hidden test
labels are unavailable locally. Each ensemble submission contains 2,370,727
impressions. The two `0.6960` scores belong to distinct submissions.
The [experiment registry](experiment_registry.md#current-large-submission)
retains the wider experiment history.

## Method and validation protocol

The retained member submission ZIPs contain ranks rather than raw logits.
The implemented method is weighted Borda rank fusion: for each candidate,
compute `w * MPNet_rank + (1 - w) * MiniLM_rank` and sort in ascending order.
Rank `1` is best. MPNet resolves exact fusion ties.

Search selects `w` by mean impression AUC on Nov 14 (`rerank_tune`) from a
0.00--1.00 grid with step 0.05. The selection is recorded and frozen before
reporting on Nov 15 (`rerank_test`). Both days were already used during
upstream model development, so these are reused diagnostic splits, not a new
independent holdout. Local temporal AUC must not be confused with Large Test AUC.

Both experiments selected 75/25, but the improved-model experiment searched
independently rather than inheriting the original weight.

## Current ensemble: improved MPNet + MiniLM

### Members and provenance

Commit `31ab682` on `feature/lr_sweep_2` promoted MPNet with text-encoder
`lr=1e-5`, selected at update 9,000. Its temporal run is
`mind_large_temporal_mpnet_p1_promoted`, with full temporal AUC `0.691440`.
For maximum-data training, the encoder continues for 2,000 updates; its
single-model submission achieved Large Test AUC `0.6960`.

The MiniLM member is the selected schedule from the candidate-aware attention
over clicked history experiment, associated with commit
`1d7fc9a2286f7ca3105541f230018750b6c82460`. Its Large Test AUC is `0.6869`.

| Member | Temporal inference config | Maximum-data submission ZIP |
| --- | --- | --- |
| Improved MPNet | `configs/promoted/mind_large_mpnet_p1/phase2_selected.yaml` | `runs/mind_large_submission_mpnet_p1_output/submission/prediction.zip` |
| Selected MiniLM | `configs/mind_large_temporal_candidate_attention.yaml` | `runs/mind_large_submission_text_adapt_candidate_attention_low_lr_2ep_recency_alpha_002_v1/submission/prediction.zip` |

The promoted temporal configs were adapted from `31ab682` for inference;
they do not import that branch's complete training workflow. The
`scripts/run_mpnet_backbone.ps1` runner in this branch reproduces the original
`lr=2e-5` MPNet, not the improved model.

### Completed validation results

| Split | Impressions | MiniLM AUC | Improved MPNet AUC | 75/25 ensemble AUC | Ensemble delta vs MPNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nov 14 weight selection | 431,517 | 0.675268 | 0.697527 | **0.698596** | +0.001069 |
| Nov 15 frozen-weight report | 376,471 | 0.667200 | 0.684624 | **0.686049** | +0.001425 |

These results include recency `alpha=0.02`. On Nov 15, relative to improved
MPNet alone, MRR decreased by `0.000665` and nDCG@5 by `0.000137`, while
nDCG@10 increased by `0.000131`. This is an AUC-targeted competition ensemble,
not an improvement on every ranking metric.

The completed hidden-test ensemble achieved **Large Test AUC 0.6972**.
Its score and verified ZIP checksum are retained in
[leaderboard_result.json](../runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/ensemble/leaderboard_result.json).
The validation search is recorded in
[search.json](../runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/ensemble/search.json).

### Generate the submission

Run all commands from the repository root, using the existing Python virtual
environment. The member ZIPs, temporal checkpoints, embeddings, processed
validation data, and age artifacts must be available for source verification.

If the member ZIPs and completed search are unchanged, only this command is
needed; it performs rank fusion without retraining:

```powershell
.\.venv\Scripts\python.exe -m mindrec.cli ensemble_submission --config configs/mind_large_ensemble_mpnet_p1_minilm.yaml
```

Output:

```text
runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/submission/prediction.zip
```

To replay preparation and selection, use the commands below. Skip the first
if the MiniLM ZIP already exists. It regenerates predictions from the retained
checkpoint, not model training. The improved MPNet ZIP must already be present.

```powershell
.\.venv\Scripts\python.exe -m mindrec.cli write_submission --config configs/mind_large_submission_candidate_attention_recency_alpha_002.yaml
.\.venv\Scripts\python.exe -m mindrec.cli ensemble_search --config configs/mind_large_ensemble_mpnet_p1_minilm.yaml
.\.venv\Scripts\python.exe -m mindrec.cli ensemble_submission --config configs/mind_large_ensemble_mpnet_p1_minilm.yaml
```

Search scores the two temporal models, records the selected weight, reports
Nov 15, and saves `runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/ensemble/search.json`.
Submission generation reads that selection and fuses the maximum-data ZIPs.
No manual weight edit is needed.

The config intentionally has `selection_source: search_artifact`,
`selected_primary_weight: null`, and `frozen: false`. In this mode, the completed
search artifact supplies the weight and records `selection_frozen: true`.
Those YAML values do not mean selection is pending.

Rerunning search replaces the saved selection. Preserve the completed run and
leaderboard result when testing new settings: use a separate run name and a
matching selection-artifact path.

## Historical ensemble: original MPNet + MiniLM

The first ensemble combined the original `lr=2e-5` MPNet with the same selected
MiniLM, using a frozen 75/25 weight. Its Large Test AUC was **0.6960**.

| Split | MiniLM AUC | Original MPNet AUC | 75/25 ensemble AUC | Ensemble delta vs MPNet |
| --- | ---: | ---: | ---: | ---: |
| Nov 14 weight selection | 0.675224 | 0.693301 | **0.694412** | +0.001111 |
| Nov 15 frozen-weight report | 0.667431 | 0.683811 | **0.684632** | +0.000820 |

These historical validation results did **not** include recency, although the
hidden-test member ZIPs did. Current searches apply recency before ranking
and use exact integer Borda arithmetic, so they do not reproduce this table.
Do not compare the historical and current tables as matched-protocol results.

The original blend reduced Nov 15 MRR by `0.000807`, nDCG@5 by `0.000783`,
and nDCG@10 by `0.000284` relative to its MPNet member.

To regenerate this historical submission from the original member ZIPs with
its already-frozen weight:

```powershell
.\.venv\Scripts\python.exe -m mindrec.cli ensemble_submission --config configs/mind_large_ensemble_mpnet_minilm.yaml
```

Output: `runs/mind_large_ensemble_mpnet_minilm_rank_v1/submission/prediction.zip`.
The completed artifact passed rank-permutation, candidate-count,
member-alignment, ZIP-integrity, and content-hash checks.

## Consistency and artifact safeguards

- Current temporal selection applies
  `zscore(logit) + 0.02 * freshness_percentile` to each member before ranking,
  matching the adjustment already included in maximum-data member ZIPs.
  Fusion does not apply recency a second time.
- Age lookup maps article IDs to the age index's own news indices. Temporal
  scoring uses the submission path's near-tie reference-scoring guard.
- Validation and submission share exact integer Borda sorting and the MPNet
  tiebreaker, avoiding floating-point differences at exact fusion ties.
- Search fingerprints cover temporal checkpoints, embeddings, processed news
  and ID maps, validation files, age index/maps, scoring settings, and member
  prediction contents. Changes to these inputs invalidate the selection;
  unrelated training/reranker settings and ZIP timestamps/compression do not.
- Search artifacts from the older scoring protocol are incompatible and must
  be regenerated for current search-based submission generation.
- The writer validates impression IDs, candidate counts, rank permutations,
  ZIP CRC, and prediction-content hash before publishing staged output. Failed
  generation preserves previous outputs.

This guide records completed rank-fusion experiments. Alternative fusion
methods and future tuning proposals are not part of the reported `0.6972`
result.
