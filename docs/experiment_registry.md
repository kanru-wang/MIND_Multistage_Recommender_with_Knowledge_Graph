# Experiment Registry

This registry names the split protocol behind each major result set. Use it before comparing metrics: runs are only comparable when their validation protocol is the same.

## Result Sets

| Label | Protocol | Reference | Config | Processed data | Run artifacts | Primary eval file |
| --- | --- | --- | --- | --- | --- | --- |
| Small dev split | Train on all `MINDsmall_train`; split `MINDsmall_dev` by time into Small Val and Small Test. | Commit `ad1fd9b6af3c28da7c5add090a93eb8a045d5bf3` | `configs/mind_small.yaml` | `data/processed/MINDsmall` | `runs/mind_small_demo` | `runs/mind_small_demo/eval/ranker_eval_test.json` |
| Large dev only | Train on all `MINDlarge_train`; validate on official `MINDlarge_dev` only, Nov 15. | Commit `b83a9308123fa196cf6be627762f01845545b123` | `configs/mind_large_tune.yaml` | `data/processed/MINDlarge_tune` | `runs/mind_large_tune` | `runs/mind_large_tune/eval/ranker_eval_val.json` |
| Large temporal baseline | Train on `MINDlarge_train` before Nov 14; validate on Nov 14 tail from `MINDlarge_train` plus all `MINDlarge_dev`; use random ranker negatives. | Current repo | `configs/mind_large_temporal_baseline.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_tune` | `runs/mind_large_temporal_tune/eval/ranker_eval_val.json` |
| Large temporal hard-negative v4 | Use the Large temporal split and baseline teacher; mine hard negatives only for cold users with usable history. | Current repo | `configs/mind_large_temporal_tune.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_hard_neg_v4` | `runs/mind_large_temporal_hard_neg_v4/eval/ranker_eval_val.json` |
| Large temporal text-adapt v1 | Adapt MiniLM on Large Temporal Train, select its update count on Large Temporal Val, and train the temporal teacher/ranker with the selected encoder. | Current repo | `configs/mind_large_temporal_text_adapt.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_text_adapt_v1` | `runs/mind_large_temporal_text_adapt_v1/eval/ranker_eval_val.json` |
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
| 0.654199 | 0.304433 | 0.331107 | 0.394609 |

V4 uses `configs/mind_large_temporal_tune.yaml`. Its zero-history groups use
four random negatives; cold users with usable history use one teacher-hard
plus three random negatives.

The previously recorded `0.645785` AUC belongs to the separate
`mind_large_temporal_hard_neg_v4_ranker_lr_3em04` run, not the canonical
`mind_large_temporal_hard_neg_v4` artifact.

## Completed MiniLM Text Adaptation V1

Phase 1 adapted `sentence-transformers/all-MiniLM-L6-v2` on Large Temporal
Train and evaluated the raw history-mean/candidate-cosine objective every 1,000
optimizer updates on 785,325 evaluable Large Temporal Val impressions. The
base encoder scored `0.623156` AUC. Update 6,000 was selected at `0.670009`
AUC; update 7,000 reached `0.670094`, but its `+0.000084` change did not meet
the configured `1e-4` minimum improvement. Early stopping fired at update
9,000.

Phase 2 froze the selected update-6,000 encoder and trained the temporal
teacher/ranker. These metrics use the full 807,988-impression Large Temporal
Val evaluation and are directly comparable with the canonical frozen-MiniLM
hard-negative v4 run:

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Frozen MiniLM hard-negative v4 | 0.654199 | 0.304433 | 0.331107 | 0.394609 |
| Adapted MiniLM v1 | 0.664328 | 0.311520 | 0.341145 | 0.403632 |
| Absolute gain | +0.010130 | +0.007087 | +0.010038 | +0.009023 |

Phase 3 loaded the selected update-6,000 encoder and continued adaptation for
exactly 2,000 updates on Large Temporal Val with a fresh optimizer. The final
encoder therefore records 8,000 cumulative staged updates. The maximum-data
teacher and ranker were then fitted on `MINDlarge_train + MINDlarge_dev` with
four teacher epochs and one ranker epoch. No labeled local validation split was
retained for this final fit.

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

| Label | Text encoder | Fit data | Teacher epochs | Ranker epochs | Recency alpha | Large Test AUC | Test impressions | Submission artifact |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Frozen-MiniLM hard-negative v4 | Frozen `all-MiniLM-L6-v2` | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6724 | 2,370,727 | `runs/mind_large_submission_recency_alpha_002_v1/submission/prediction.zip` |
| Text-adapt v1 | Phase 1 update 6,000 + 2,000 Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | **0.6848** | 2,370,727 | `runs/mind_large_submission_text_adapt_recency_alpha_002_v1/submission/prediction.zip` |

The adapted-text submission improved Large Test AUC from `0.6724` to `0.6848`,
an absolute gain of `+0.0124`. The Large Test scores were returned by the
competition platform; hidden test labels remain unavailable locally. The
adapted score was reported on 2026-08-07. Local artifact validation confirmed
2,370,727 unique impression IDs, complete candidate-rank permutations, and a
valid submission ZIP.

## Sanity Rules

- For temporal configs, `preprocess_meta.json:n_validation_eval_impressions` must equal `ranker_eval_val.json:n_impressions`.
- The older Small dev split is a historical demo result, not the current architecture-search baseline.
- Use Small temporal for fast architecture experiments.
- Use `configs/mind_large_temporal_baseline.yaml` to reproduce the random-negative
  baseline and `configs/mind_large_temporal_tune.yaml` to reproduce hard-negative v4.
- Use `configs/mind_large_submission.yaml` only after choosing fixed settings; it trains on `MINDlarge_train + MINDlarge_dev` and writes hidden-test submission ranks.

## Known Metadata Note

`data/processed/MINDlarge_temporal_tune/preprocess_meta.json` may show the older internal mode name `leaderboard_temporal_tune` if it was generated before the config was renamed to `temporal_tune`. The directory name, split counts, and eval file identify the intended Large temporal protocol. Rerunning preprocess with the current config refreshes the mode string.
