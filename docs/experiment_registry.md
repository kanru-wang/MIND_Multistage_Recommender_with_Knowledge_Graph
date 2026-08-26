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
| Large temporal candidate-attention v1 | Reuse the text-adapt v1 teacher and replace only student mean history pooling with candidate-aware attention. | Current repo; completed | `configs/mind_large_temporal_candidate_attention.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_text_adapt_candidate_attention_v1` | `runs/mind_large_temporal_text_adapt_candidate_attention_v1/eval/ranker_eval_val.json` |
| Rejected large temporal candidate-attention item-only distillation v1 | Reuse candidate-attention v1 and change representation distillation from the full user/item target to `item_sem -> teacher_item`, including zero-history rows; teacher-logit distillation and all configured weights remain fixed. | Current repo; completed/rejected | `configs/mind_large_temporal_candidate_attention_item_only_distill.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_text_adapt_candidate_attention_item_only_distill_v1` | `runs/mind_large_temporal_text_adapt_candidate_attention_item_only_distill_v1/eval/ranker_eval_val.json` |
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

### Rejected Phase 3 replay experiment

A follow-up Phase 3 experiment tested whether replaying older temporal-training
examples would reduce forgetting during the 2,000-update continuation. It used
the same Phase 1 update-6,000 checkpoint and an exact 20-sample mixture cycle:

| Continuation source | Fraction | Samples |
| --- | ---: | ---: |
| Large Temporal Val (Nov 14-15) | 85% | 108,800 |
| Large Temporal Train, Nov 13 | 10% | 12,800 |
| Large Temporal Train, Nov 12 | 5% | 6,400 |
| **Total** | **100%** | **128,000** |

The maximum-data teacher/ranker and post-hoc recency setting remained fixed at
four teacher epochs, one ranker epoch, and `alpha=0.02`. Large Test AUC fell to
`0.6800`, which is `-0.0048` versus the pure-Temporal-Val Phase 3 result of
`0.6848` (but still `+0.0076` versus the frozen-MiniLM baseline).

The replay data had already been seen during Phase 1, while it displaced 15% of
the newer Nov 14-15 continuation samples. The result therefore provides no
evidence that this run needed protection from catastrophic forgetting and is
consistent with older replay weakening adaptation to the later target period.
The replay implementation and configuration were discarded; it is a rejected
result, not a supported current pipeline. Historical local artifacts, if kept,
are under `runs/mind_large_submission_text_adapt_replay_85_10_5_v1` and the
submitted ZIP is under
`runs/mind_large_submission_text_adapt_replay_85_10_5_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 Nov 15 overweight experiment

A second follow-up kept the Phase 1 update-6,000 checkpoint, 2,000 continuation
updates, learning rate, hard-negative policy, and downstream pipeline fixed. It
used only Large Temporal Val but changed its clicked-positive sample mix with
an exact 20-sample cycle:

| Continuation date | Fraction | Samples |
| --- | ---: | ---: |
| Nov 14 | 45% | 57,600 |
| Nov 15 | 55% | 70,400 |
| **Total** | **100%** | **128,000** |

The maximum-data teacher/ranker again used four teacher epochs and one ranker
epoch, followed by post-hoc recency `alpha=0.02`. Large Test AUC was `0.6796`,
which is `-0.0052` versus the natural Large Temporal Val mixture at `0.6848`
(but still `+0.0072` versus the frozen-MiniLM baseline).

The earlier observation that adaptation improved Nov 15 evaluation more than
Nov 14 did not imply that Nov 15 examples deserved greater training weight.
Overweighting Nov 15 displaced Nov 14 topical/source diversity and did not
generalize to the later hidden test period. Together with the older-data replay
result, this provides evidence against further manual date-mixture sweeps; keep
the natural shuffled Large Temporal Val distribution. The implementation and
configuration were discarded. Historical local artifacts, if kept, are under
`runs/mind_large_submission_text_adapt_nov15_weighted_v1`, and the submitted
ZIP is under
`runs/mind_large_submission_text_adapt_nov15_weighted_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 2,500-update experiment

A third follow-up changed only the natural Large Temporal Val continuation
length. It started again from the Phase 1 update-6,000 checkpoint, retained the
globally shuffled unweighted validation distribution and `lr=2e-5`, and trained
for 2,500 updates (160,000 samples) instead of 2,000. The resulting encoder
recorded 8,500 cumulative staged updates. The maximum-data teacher/ranker and
post-hoc recency settings remained fixed at four teacher epochs, one ranker
epoch, and `alpha=0.02`.

Large Test AUC was `0.6842`, which is `-0.0006` versus the then-current
2,000-update text-adapt v1 result at `0.6848` (and `+0.0118` versus the
frozen-MiniLM baseline). Downstream
training losses improved slightly despite the hidden-test regression: teacher
loss moved from `6.139574` to `6.138398`, and ranker loss from `0.834028` to
`0.833814`. This is consistent with mild overfitting or overshooting during the
additional 500 constant-learning-rate updates. Keep 2,000 Phase 3 updates; do
not extend the continuation to 2,500 at `lr=2e-5`.

The implementation and configuration were discarded. Historical local
artifacts, if kept, are under
`runs/mind_large_submission_text_adapt_updates_2500_v1`, and the submitted ZIP
is under
`runs/mind_large_submission_text_adapt_updates_2500_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 teacher-guided hard-negative experiment

A fourth follow-up changed only Phase 3 adaptation hard-negative scoring. It
loaded the selected Phase 2 teacher at epoch 4 and used its frozen projected
item vectors plus transformer/attention history representation to score each
sampled negative pool. The Phase 1 update-6,000 source, natural shuffled Large
Temporal Val, 2,000 updates, `lr=2e-5`, cold-user-only policy, hard fraction
`0.25`, pool size `20`, consistency guard, and downstream pipeline remained
fixed. MiniLM still optimized the original history-mean/clicked-article
contrastive objective.

Large Test AUC was `0.6835`, which is `-0.0013` versus the then-current
snapshot-mined text-adapt v1 result at `0.6848` (and `+0.0111` versus the
frozen-MiniLM baseline). The
teacher changed hard-negative identity rather than quantity: both runs mined
11,809 groups, while the teacher-guided run selected 10,632 hard negatives
versus 10,635 for the reference run. Its consistency guard rejected 63,876 of
177,471 scored pool negatives because the teacher placed them above the
clicked positive.

The likely problem was objective/representation mismatch. The teacher judged
difficulty with learned attention and projected item vectors, but the MiniLM
adaptation loss represented history with a mean of raw MiniLM vectors. A
teacher-hard negative was therefore not necessarily a useful contrastive
target for the representation MiniLM was actually trained to produce.
Downstream losses were essentially unchanged (teacher `6.139544` versus
`6.139574`; ranker `0.834134` versus `0.834028`), providing no evidence of an
operational failure. Keep the original update-6,000 MiniLM snapshot scorer.

The implementation and configuration were discarded. Historical local
artifacts, if kept, are under
`runs/mind_large_submission_text_adapt_teacher_guided_v1`, and the submitted
ZIP is under
`runs/mind_large_submission_text_adapt_teacher_guided_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 learning-rate 1.5e-5 experiment

A fifth follow-up changed only the Phase 3 MiniLM learning rate from `2e-5`
to `1.5e-5`. It retained the Phase 1 update-6,000 checkpoint, natural shuffled
Large Temporal Val data, 2,000 continuation updates (128,000 samples), original
snapshot-mined hard negatives, maximum-data teacher/ranker fit, and post-hoc
recency `alpha=0.02`.

Large Test AUC was `0.6797`, which is `-0.0051` versus the then-current
`lr=2e-5` text-adapt v1 result at `0.6848` (but still `+0.0073` versus the
frozen-MiniLM baseline). The run
used the intended data and artifact routing, and its sample and negative-policy
counts matched the reference run. Its teacher loss was marginally lower (`6.139317`
versus `6.139574`), while ranker loss was slightly worse (`0.834304` versus
`0.834028`); neither suggests an operational failure. Compared with the
reference submission, only 283,858 of 2,370,727 impressions (11.97%) retained
an identical full ranking, and 1,646,229 (69.44%) retained the same top-ranked
candidate, confirming that the learning-rate change materially propagated
through the downstream pipeline.

With a fixed 2,000-update continuation, `1.5e-5` reduces the nominal Phase 3
update scale by 25%. The hidden-test regression is therefore more consistent
with insufficient adaptation to the recent Nov 14-15 distribution than with
successful overfitting control. Together with the rejected 2,500-update run,
the evidence supports keeping `lr=2e-5` and 2,000 Phase 3 updates rather than
continuing a constant-learning-rate or update-count sweep.

The implementation and configuration were discarded. Historical local
artifacts, if kept, are under
`runs/mind_large_submission_text_adapt_lr_1p5em05_v1`, and the submitted ZIP
is under
`runs/mind_large_submission_text_adapt_lr_1p5em05_recency_alpha_002_v1/submission/prediction.zip`.

## Rejected Candidate-Attention Follow-ups

All temporal results below use the same 807,988-impression validation protocol
and retain the 4-head, zero-dropout direct candidate-attention run at
`0.671593` AUC as the reference. Most earlier experimental implementations and
configs were discarded; the item-only distillation config is retained so its
negative result remains reproducible.

| Experiment | Temporal AUC | Delta vs direct | Decision |
| --- | ---: | ---: | --- |
| Impression-AUC checkpoint selection | 0.671593 | 0.000000 | Neutral: selected the same epoch and tensor-identical checkpoint; no model promotion |
| Gated residual between mean and direct attention | 0.669548 | -0.002044 | Rejected; also reduced MRR by 0.002766 and both nDCG metrics by about 0.0028 |
| 7 heads, dropout 0.00 | 0.672090 | +0.000497 | Rejected; gain was below 0.001 and MRR/nDCG regressed by more than 0.002 |
| 4 heads, dropout 0.05 | 0.668253 | -0.003339 | Rejected |
| 4 heads, dropout 0.10 | 0.669441 | -0.002151 | Rejected |
| 1 head, dropout 0.00 | 0.670408 | -0.001185 | Rejected |
| 2 heads, dropout 0.00 | 0.670967 | -0.000625 | Rejected |
| Four learned history interests with candidate conditioning | 0.669808 | -0.001784 | Rejected; MRR fell 0.003929, nDCG@5 fell 0.005053, and nDCG@10 fell 0.004460 |
| Learned reverse-position embeddings (`0` = newest click) | 0.672423 | +0.000830 | Mixed/rejected: AUC improved by less than 0.001, while MRR fell 0.002318, nDCG@5 fell 0.002771, and nDCG@10 fell 0.002278 |
| Item-only representation distillation | 0.666388 | -0.005205 | Rejected; MRR fell 0.004828, nDCG@5 fell 0.006241, and nDCG@10 fell 0.005455 |

The item-only representation-distillation run completed on 2026-08-27. It
kept candidate-aware scoring, teacher-logit distillation, the teacher and data
artifacts, optimizer schedule, and all configured loss weights unchanged. It
replaced the joint projection from `[user_sem, item_sem, sem_fused]` to
`[teacher_user, teacher_item]` with `item_sem -> teacher_item`; item
representation loss also remained active for zero-history rows. The selected
sampled-pair checkpoint moved from epoch 3 and `0.674491` AUC to epoch 2 and
`0.671052` AUC, and the full-impression result regressed across every primary
ranking metric:

| Metric | Direct candidate attention | Item-only distillation | Delta |
| --- | ---: | ---: | ---: |
| AUC | 0.671593 | 0.666388 | -0.005205 |
| MRR | 0.321535 | 0.316708 | -0.004828 |
| nDCG@5 | 0.352780 | 0.346539 | -0.006241 |
| nDCG@10 | 0.414064 | 0.408609 | -0.005455 |
| Recall@5 | 0.489773 | 0.482717 | -0.007056 |
| Recall@10 | 0.664185 | 0.659521 | -0.004665 |

The intended cold slices did not recover: AUC changed by `-0.003285` for empty
history, `-0.003851` for cold users, and `-0.002040` for impressions whose
clicked item was new. Larger losses on clicked warm items (`-0.016775` AUC),
popularity 20+ (`-0.016911`), and history length 21+ (`-0.006257`) indicate
that removing user/fused representation supervision discarded useful signal.
Item-only distillation still exceeded the older mean-pooling text-adapt model
by `+0.002060` AUC, so candidate-aware pooling remains beneficial; the failed
component is the item-only objective. This variant is not promoted to
maximum-data training or submission, and the verified full-distillation
candidate-attention workflow remains the active champion.

The reverse-position run selected epoch 3 and learned a substantial newest-click
signal, but its gains were not stable across history lengths. Relative to direct
attention, AUC changed by `-0.004674` for empty history, `-0.003251` for history
length 1--4, `+0.002937` for length 5--20, and only `+0.000384` for length 21+.
Its sampled-pair validation AUC also fell from `0.674491` to `0.672318`.
Because the small overall impression-AUC gain came with broad top-ranking and
short/cold-history regressions, neither the model nor a further recency-logit
bias was promoted. The implementation and config were discarded; historical
artifacts, if kept, are under
`runs/mind_large_temporal_text_adapt_candidate_attention_reverse_position_v1`.

The head/dropout comparison is preserved in
`runs/mind_large_temporal_text_adapt_candidate_attention_impression_auc_v1/tuning/candidate_attention_sweep/sweep.json`.
The gated and four-interest evaluations remain under
`runs/mind_large_temporal_text_adapt_gated_candidate_attention_v1` and
`runs/mind_large_temporal_text_adapt_multi_interest_candidate_attention_v1`.
No variant was promoted to maximum-data training. Separately, the original
one-epoch candidate-attention maximum-data schedule tied text-adapt v1 at
`0.6848` Large Test AUC; the selected two-epoch lower-learning-rate schedule
superseded it at `0.6869`.

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
| Text-adapt v1 | Phase 1 update 6,000 + 2,000 Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6848 | 2,370,727 | `runs/mind_large_submission_text_adapt_recency_alpha_002_v1/submission/prediction.zip` |
| Candidate attention, original schedule | Same as text-adapt v1; candidate-aware student history pooling | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6848 | 2,370,727 | `runs/mind_large_submission_text_adapt_candidate_attention_recency_alpha_002_v1/submission/prediction.zip` |
| Candidate attention, selected schedule | Same as text-adapt v1; candidate-aware student history pooling | `MINDlarge_train + MINDlarge_dev` | 4 | 2 | 0.02 | **0.6869** | 2,370,727 | `runs/mind_large_submission_text_adapt_candidate_attention_low_lr_2ep_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected attentive multi-view v1 | Separate title/abstract encoding with a jointly trained attentive gate | `MINDlarge_train + MINDlarge_dev` | 1 | 3 | 0.02 | 0.6746 | 2,370,727 | `runs/mind_large_submission_text_adapt_multiview_low_lr_3ep_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected replay 85/10/5 | Phase 1 update 6,000 + 2,000 replay-mixture updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6800 | 2,370,727 | `runs/mind_large_submission_text_adapt_replay_85_10_5_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected Nov 15 overweight 45/55 | Phase 1 update 6,000 + 2,000 date-weighted updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6796 | 2,370,727 | `runs/mind_large_submission_text_adapt_nov15_weighted_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected 2,500 updates | Phase 1 update 6,000 + 2,500 Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6842 | 2,370,727 | `runs/mind_large_submission_text_adapt_updates_2500_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected teacher-guided negatives | Phase 1 update 6,000 + 2,000 teacher-mined Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6835 | 2,370,727 | `runs/mind_large_submission_text_adapt_teacher_guided_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected Phase 3 `lr=1.5e-5` | Phase 1 update 6,000 + 2,000 lower-learning-rate Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6797 | 2,370,727 | `runs/mind_large_submission_text_adapt_lr_1p5em05_recency_alpha_002_v1/submission/prediction.zip` |

The adapted-text submission first improved Large Test AUC from `0.6724` to
`0.6848` (`+0.0124`). Candidate-aware pooling then improved matched temporal
validation AUC from `0.664328` to `0.671593`. Its original one-epoch
maximum-data schedule remained at `0.6848`, while transferring the selected
`lr=1e-4`, `weight_decay=3e-5` schedule for two maximum-data epochs reached
**`0.6869`**. The new champion gains `+0.0021` over text-adapt v1 and `+0.0145`
over the frozen-MiniLM baseline.

The attentive multi-view promotion is rejected. Its encouraging temporal AUC
of `0.675783` did not transfer: Large Test AUC was `0.6746`, which is `-0.0123`
versus the candidate-attention champion and only `+0.0022` over frozen MiniLM.
The submission was structurally valid. Post-result analysis found that its
one-epoch maximum-data teacher remained less mature than the champion's
four-epoch teacher, while the hidden ranking over-promoted sports and weather,
two categories that had already regressed in temporal slices. The retained
artifacts are historical negative evidence and must not be treated as the
active submission workflow.

The follow-up temporal-only diagnostic used a now-removed experimental config.
It trained the title/abstract gate during epoch 1, froze it, then continued the
teacher through fixed epoch 4 before validation-selected ranker training.

That redesigned temporal schedule is also rejected. The implementation behaved
as configured: its saved gate and attention arrays are tensor-identical to the
original epoch-1 gate, the gate was frozen for epochs 2--4, and the fixed
teacher checkpoint records epoch 4. Against the matched candidate-attention
reference, results were:

| Metric | Candidate-attention reference | Gate-1/teacher-4 v2 | Delta |
| --- | ---: | ---: | ---: |
| Teacher Recall@200 | 0.041199 | 0.034746 | -0.006453 |
| Full-impression AUC | 0.671593 | 0.672267 | +0.000675 |
| MRR | 0.321535 | 0.320110 | -0.001426 |
| nDCG@5 | 0.352780 | 0.351188 | -0.001592 |
| nDCG@10 | 0.414064 | 0.412651 | -0.001413 |

The AUC gain remained below the required `+0.001`, teacher Recall@200 fell by
`15.7%`, and all three top-ranking metrics regressed. Although zero-popularity,
period-4, and sports slices recovered, the errors shifted to new items
(`-0.000868` AUC), popularity 1--4 (`-0.009515`), period 1 (`-0.008519`), and
weather (`-0.047077`). This is distribution instability rather than a robust
improvement. The pure attentive multi-view branch is closed: do not promote
this schedule to maximum-data training and do not proceed to cross-field
contrastive adaptation.

Large Test scores were returned by the competition platform; hidden test
labels remain unavailable locally. The candidate-attention champion was
reported on 2026-08-16. Local artifact validation confirmed 2,370,727
impressions and a valid submission ZIP. The replay, Nov 15 overweight,
2,500-update, teacher-guided, and learning-rate `1.5e-5` rows are retained only
as negative experimental evidence.

## Sanity Rules

- For temporal configs, `preprocess_meta.json:n_validation_eval_impressions` must equal `ranker_eval_val.json:n_impressions`.
- The older Small dev split is a historical demo result, not the current architecture-search baseline.
- Use Small temporal for fast architecture experiments.
- Use `configs/mind_large_temporal_baseline.yaml` to reproduce the random-negative
  baseline and `configs/mind_large_temporal_tune.yaml` to reproduce hard-negative v4.
- Use `configs/mind_large_submission.yaml` only after choosing fixed settings; it trains on `MINDlarge_train + MINDlarge_dev` and writes hidden-test submission ranks.
- Use `configs/mind_large_submission_candidate_attention.yaml` and its recency
  companion to reproduce the current `0.6869` champion.

## Known Metadata Note

`data/processed/MINDlarge_temporal_tune/preprocess_meta.json` may show the older internal mode name `leaderboard_temporal_tune` if it was generated before the config was renamed to `temporal_tune`. The directory name, split counts, and eval file identify the intended Large temporal protocol. Rerunning preprocess with the current config refreshes the mode string.
