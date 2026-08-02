# Experiment Registry

This registry names the split protocol behind each major result set. Use it before comparing metrics: runs are only comparable when their validation protocol is the same.

## Result Sets

| Label | Protocol | Reference | Config | Processed data | Run artifacts | Primary eval file |
| --- | --- | --- | --- | --- | --- | --- |
| Small dev split | Train on all `MINDsmall_train`; split `MINDsmall_dev` by time into Small Val and Small Test. | Commit `ad1fd9b6af3c28da7c5add090a93eb8a045d5bf3` | `configs/mind_small.yaml` | `data/processed/MINDsmall` | `runs/mind_small_demo` | `runs/mind_small_demo/eval/ranker_eval_test.json` |
| Large dev only | Train on all `MINDlarge_train`; validate on official `MINDlarge_dev` only, Nov 15. | Commit `b83a9308123fa196cf6be627762f01845545b123` | `configs/mind_large_tune.yaml` | `data/processed/MINDlarge_tune` | `runs/mind_large_tune` | `runs/mind_large_tune/eval/ranker_eval_val.json` |
| Large temporal baseline | Train on `MINDlarge_train` before Nov 14; validate on Nov 14 tail from `MINDlarge_train` plus all `MINDlarge_dev`; use random ranker negatives. | Current repo | `configs/mind_large_temporal_baseline.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_tune` | `runs/mind_large_temporal_tune/eval/ranker_eval_val.json` |
| Large temporal hard-negative v4 | Use the Large temporal split and baseline teacher; mine hard negatives only for cold users with usable history. | Current repo | `configs/mind_large_temporal_tune.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_hard_neg_v4` | `runs/mind_large_temporal_hard_neg_v4/eval/ranker_eval_val.json` |
| Large temporal current-code single-interest control | Reproduce hard-negative v4 under the multi-interest branch's current code with one mean-pooled user vector. | Branch `feature/multi_interest_user_representation` | `configs/mind_large_temporal_single_interest_current.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_single_interest_current` | `runs/mind_large_temporal_single_interest_current/eval/ranker_eval_val.json` |
| Large temporal four-interest, disagreement `0.01` | Replace the single pooled user vector with four candidate-conditioned poly-attention interests and squared-cosine disagreement regularization. | Branch `feature/multi_interest_user_representation` | `configs/mind_large_temporal_multi_interest.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_multi_interest` | `runs/mind_large_temporal_multi_interest/eval/ranker_eval_val.json` |
| Large temporal four-interest, no disagreement | Use the same four-interest model with `disagreement_weight: 0.0`. | Branch `feature/multi_interest_user_representation` | `configs/mind_large_temporal_multi_interest_no_disagreement.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_multi_interest_no_disagreement` | `runs/mind_large_temporal_multi_interest_no_disagreement/eval/ranker_eval_val.json` |
| Small temporal | Train on `MINDsmall_train` before Nov 14; validate on Nov 14 tail from `MINDsmall_train` plus all `MINDsmall_dev`. | Current repo | `configs/mind_small_temporal_tune.yaml` | `data/processed/MINDsmall_temporal_tune` | `runs/mind_small_temporal_tune` | `runs/mind_small_temporal_tune/eval/ranker_eval_val.json` |

## Latest Completed Large Temporal Baseline

| Validation impressions | Teacher best epoch | Ranker best epoch | AUC | MRR | nDCG@5 | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 807,988 | 4 | 1 | 0.646597 | 0.304210 | 0.330920 | 0.393785 |

This run trains on `MINDlarge_train` impressions before Nov 14 and evaluates
the full validation window formed from the Nov 14 tail of `MINDlarge_train`
plus all of `MINDlarge_dev` on Nov 15. The teacher stopped at epoch 6 and
selected epoch 4; the ranker stopped at epoch 3 and selected epoch 1. These are
full impression-ranking metrics from `ranker_eval_val.json`; the ranker's
sampled-pair early-stopping AUC is a different quantity.

## Latest Completed Large Temporal Hard-Negative V4

This result uses the same 807,988 validation impressions as the baseline.

| AUC | MRR | nDCG@5 | nDCG@10 |
| ---: | ---: | ---: | ---: |
| 0.645785 | 0.305195 | 0.332726 | 0.394743 |

V4 uses `configs/mind_large_temporal_tune.yaml`. Its zero-history groups use
four random negatives; cold users with usable history use one teacher-hard
plus three random negatives.

## Multi-Interest User Representation Experiment

Completed on branch `feature/multi_interest_user_representation` on 2026-08-02.
The experiment tested whether replacing the student's single mean-pooled
history vector with four candidate-conditioned interest vectors improves the
Large temporal ranker. Four learned poly-attention queries summarize the
clicked history. Candidate-interest cosine similarity and temperature-`0.1`
softmax weighting produce the normalized `user_sem` consumed by the unchanged
DLRM interaction, fusion, and distillation paths.

All comparable runs reused:

- processed data `data/processed/MINDlarge_temporal_tune`;
- teacher artifacts `runs/mind_large_temporal_tune/teacher`;
- validation split of 807,988 impressions and 31,485,663 scored candidates;
- seed 13, hard-negative v4 sampling, ranker LR `1.0e-4`, weight decay
  `3.0e-5`, and validation-AUC early stopping;
- the same DLRM, distillation, KG, and dense-feature settings apart from the
  explicitly named multi-interest fields.

### Overall results

| Variant | Interests | Disagreement weight | Best pair-val AUC | Best epoch | Full AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current-code control | 1 | 0.0 | **0.657270** | 5 | **0.654162** | **0.304436** | 0.331108 | **0.394604** |
| Four-interest | 4 | 0.01 | 0.649636 | 3 | 0.644329 | 0.298329 | 0.326131 | 0.388352 |
| Four-interest, no disagreement | 4 | 0.0 | 0.653072 | 3 | 0.648444 | 0.303849 | **0.332026** | 0.393941 |

The disagreement penalty harmed every overall ranking metric. Removing it
recovered most of the nDCG/MRR loss, but the four-interest model still lost
`0.005719` AUC, `0.000587` MRR, and `0.000663` nDCG@10 relative to the matched
current-code control. Its `0.000917` nDCG@5 gain was insufficient to offset
the other official-metric regressions.

### Diagnostic slices: four interests without disagreement versus control

| Slice | Impressions | AUC delta | MRR delta | nDCG@5 delta | nDCG@10 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold user | 109,447 | +0.002054 | +0.003594 | +0.004974 | +0.005318 |
| History length 0 | 22,663 | +0.008342 | +0.006931 | +0.006224 | +0.008240 |
| History length 1-4 | 86,784 | +0.000412 | +0.002722 | +0.004648 | +0.004555 |
| History length 5-20 | 309,696 | -0.004701 | -0.000300 | +0.001547 | +0.000122 |
| History length 21+ | 388,845 | -0.008716 | -0.001992 | -0.000726 | -0.002972 |
| Impression has clicked new item | 686,516 | -0.000859 | +0.004647 | +0.006964 | +0.005029 |
| Impression has clicked warm item | 203,658 | -0.026818 | -0.021869 | -0.025327 | -0.025296 |

The original hypothesis predicted the clearest benefit for longer histories,
but the largest history-dependent regression occurred for length 21+. The
apparent zero-history gain cannot be caused by interest selection because the
semantic history vector is forced to zero for those users. Query initialization
also consumed RNG state before later shared layers were initialized, so some
slice differences include a shared-initialization confound.

### Provenance correction and decision

The older `runs/mind_large_temporal_hard_neg_v4` evaluation used the same
dataset, split, counts, and embedded configuration, but its checkpoint predates
later source changes and does not contain the current taxonomy-support buffers.
A fresh single-interest control was therefore trained with current code. It
reproduced the historical full metrics to within `0.000037` AUC and
`0.000005` nDCG@10, empirically validating the old record while providing the
proper matched control for this experiment.

Decision: do not promote the four-interest architecture, do not use the
disagreement penalty, and do not run the proposed two-interest follow-up. Keep
the current single-interest representation for the final submission path. The
multi-interest implementation, configs, and result JSONs are retained on the
feature branch as a documented negative result.

## Additional Current Metrics

Values below were read from the listed local evaluation artifacts on 2026-07-03.
They describe completed runs, not untrained working-tree configuration changes.

| Label | Eval split | Impressions | Teacher best epoch | Ranker best epoch | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small dev split | test | 14,631 | 2 | 5 | 0.6619 | 0.3589 | 0.3463 | 0.4050 |
| Large dev only | val | 376,471 | 2 | 1 | 0.6686 | 0.3313 | 0.3659 | 0.4252 |
| Large temporal hard-negative v3 | val | 807,988 | 4 (reused) | 1 | 0.6498 | 0.3082 | 0.3344 | 0.3969 |
| Small temporal | val | 103,422 | 1 | 3 | 0.6297 | 0.2995 | 0.3263 | 0.3867 |

The hard-negative v3 run applied 1 teacher-hard plus 3 random negatives to all cold
users, including zero-history groups.

## Failed Trend Experiments

| Experiment | Result | Decision |
| --- | ---: | --- |
| Item exposure burst + age features | 0.6518 Large Test | Removed |
| Learned age-residual branch | 0.6679 Large Test | Removed |
| Recency `alpha=0.03` | 0.6723 Large Test | Removed; keep `0.02` |
| Semantic-neighborhood burst | Temporal tuning selected zero | Removed |
| Content trend propensity | 0.6722 Large Test | Removed |
| Conservative content tiebreaker grids | Guardrails selected zero | Removed |

## Current Large Submission

| Label | Fit data | Teacher epochs | Ranker epochs | Test impressions | Submission artifact |
| --- | --- | ---: | ---: | ---: | --- |
| Hard-negative v4 | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 2,370,727 | `runs/mind_large_submission_hard_neg_v4/submission/prediction.zip` |

The hidden test labels are unavailable locally, so this row records generation metadata
rather than ranking metrics.

## Sanity Rules

- For temporal configs, `preprocess_meta.json:n_validation_eval_impressions` must equal `ranker_eval_val.json:n_impressions`.
- The older Small dev split is a historical demo result, not the current architecture-search baseline.
- Use Small temporal for fast architecture experiments.
- Use `configs/mind_large_temporal_baseline.yaml` to reproduce the random-negative
  baseline and `configs/mind_large_temporal_tune.yaml` to reproduce hard-negative v4.
- Use `configs/mind_large_submission.yaml` only after choosing fixed settings; it trains on `MINDlarge_train + MINDlarge_dev` and writes hidden-test submission ranks.

## Known Metadata Note

`data/processed/MINDlarge_temporal_tune/preprocess_meta.json` may show the older internal mode name `leaderboard_temporal_tune` if it was generated before the config was renamed to `temporal_tune`. The directory name, split counts, and eval file identify the intended Large temporal protocol. Rerunning preprocess with the current config refreshes the mode string.
