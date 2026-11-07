# Experiment Registry

This registry names the split protocol behind each major result set. Use it before comparing metrics: runs are only comparable when their validation protocol is the same.

## Best reported Large Test result (2026-11-07)

**Best reported Large Test AUC: `0.7025`**, reported by the user from the
competition platform on 2026-11-07 (Australia/Sydney). This is an absolute
**`+0.0014`** over coefficient 0.5's `0.7011`, **`+0.0035`** over coefficient
0.25's `0.6990`, and **`+0.0053`** over the previous BCE-student ensemble's
`0.6972`. This score is user-reported; hidden test labels remain unavailable
locally.

The submission combines exactly **75% MPNet / 25% MiniLM**, with both students
using **BCE + 0.75 * impression-normalized pairwise loss + teacher distillation**.
Both completed **two fixed epochs on Train+Dev**, reusing their retained final
four-epoch teachers and embeddings. Recency `alpha=0.02` is applied once per
member before exact integer weighted-Borda fusion; MPNet rank breaks exact ties.
The ensemble weights and final-fit epoch budget were unchanged from coefficient
0.5, and no additional weight search was performed.

- Runner: `scripts/run_pairwise_weight_075_final_fit.py`.
- Workflow config: `configs/mind_large_two_model_final_fit_w075.yaml`.
- Student configs: `configs/mind_large_submission_mpnet_p1_pairwise_w075.yaml`
  and `configs/mind_large_submission_minilm_pairwise_w075.yaml`.
- Submission ZIP: `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1/submission/prediction.zip`.
- Verified ZIP SHA256: `82fb56c6e5d181978935205725b4e18feede96a58a019e757547c6328d8c46dd`.
- Metadata: `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1/submission/submission_meta.json`.
- Temporal comparison: `runs/mind_large_pairwise_w075_comparison_v1/ensemble_comparison.json`.
- Test impressions: **2,370,727**. The ZIP and metadata hashes match the
  completed fusion record; both saved student summaries confirm the two-epoch
  fixed-fit schedule and coefficient 0.75.

| Reused temporal split | Coefficient 0.5 ensemble AUC | Coefficient 0.75 ensemble AUC | Delta |
| --- | ---: | ---: | ---: |
| November 14 | 0.702605 | 0.703738 | +0.001133 |
| November 15 | 0.690140 | 0.690959 | +0.000819 |

Individual temporal AUC gained `+0.001002` for MPNet but lost `0.001513` for
MiniLM; selected epochs remained 3 and 4. Ensemble MRR/nDCG declined on
November 14 and improved on November 15. These dates remain reused development
diagnostics; the separate platform score above establishes improvement for
this submitted ensemble, without proving improvement in every metric or member.
Preserve the 0.7011, 0.6990 and 0.6972 submissions and records. Slides are unchanged.

## Asymmetric-coefficient follow-up (reported 2026-11-07)

The user reported **Large Test AUC `0.7021`** for **MPNet coefficient 0.75 /
MiniLM coefficient 0.5**, at fixed **75% MPNet / 25% MiniLM** ensemble weights.
This is **`-0.0004`** versus the **0.7025** champion, which uses coefficient
0.75 for both students. It remains +0.0010 above the older coefficient-0.5
ensemble, but does not replace the current champion. The score is user-reported
from the competition platform on 2026-11-07 (Australia/Sydney); hidden test
labels remain unavailable locally.

The candidate reused existing two-epoch Train+Dev prediction ZIPs from MPNet
0.75 and MiniLM 0.5. No new training, candidate scoring or ensemble weight
search occurred. Existing recency alpha 0.02 was retained in each member's
ranks, without being applied again during exact integer weighted-Borda fusion.
MPNet rank breaks exact ties.

- Former runner (removed): `scripts/run_asymmetric_pairwise_submission.py`.
- Former temporal evidence (removed): `runs/mind_large_ensemble_mpnet_w075_minilm_w050_v1/ensemble/comparison.json`.
- Former submission ZIP (removed): `runs/mind_large_ensemble_mpnet_w075_minilm_w050_final_75_25_v1/submission/prediction.zip`.
- Verified ZIP SHA256: `c320b314fb85386ffe79f6b1ad3c0a5ee2f22583ad4c8172eec94a226ef952ce`.
- Former metadata (removed): the same submission directory's `submission_meta.json`.
- Test impressions: **2,370,727**. Before cleanup, actual ZIP and metadata
  checksums matched the completed fusion record; metadata confirmed both coefficients, 75/25 weights,
  reused source predictions and no additional recency adjustment.

Temporal ensemble AUC improved by +0.000052074 on November 14 and +0.000453407
on November 15 versus 0.75/0.75, with MRR/nDCG gains on both dates. Those gains
did not transfer to this Large Test submission. Both dates remain reused
development diagnostics; no statistical significance or general claim that
asymmetric coefficients are worse is established by this single comparison.
**Decision: do not promote this candidate; retain coefficient 0.75 for both
students and the 0.7025 champion.** Retain this registry entry as negative
evidence. Slides are unchanged.

At the user's request, the dedicated asymmetric experiment files were removed
on 2026-11-07 after recording the result and verifying the submitted ZIP's
identity. Cleanup removed:

- Runners: `scripts/run_asymmetric_pairwise_ensemble.py` and
  `scripts/run_asymmetric_pairwise_submission.py`.
- Tests: `tests/test_asymmetric_pairwise_ensemble.py` and
  `tests/test_asymmetric_pairwise_submission.py`, plus their ignore exceptions
  and the runners'/tests' cached bytecode. Task-specific temporary test outputs
  under `.pytest_tmp/asymmetric*` were also removed after validation.
- Guides: `docs/asymmetric_pairwise_ensemble.md` and
  `docs/asymmetric_pairwise_submission.md`; this entry consolidates their results.
- Run directories: `runs/mind_large_ensemble_mpnet_w075_minilm_w050_v1/` and
  `runs/mind_large_ensemble_mpnet_w075_minilm_w050_final_75_25_v1/`, including
  the failed candidate's combined submission ZIP and completion records.

The source MPNet-0.75 and MiniLM-0.5 student checkpoints and prediction ZIPs
are retained: they belong to successful earlier workflows and were only reused
by this experiment. The coefficient-0.75/0.75 champion, 0.5/0.5 and 0.25/0.25
comparators, shared training/fusion code, and all slide-related files are
preserved. Retained workflows have no dependency on the removed files.

## Previous champion: coefficient 0.5 (reported 2026-11-01)

**Previous champion Large Test AUC: `0.7011`**, reported by the user from the
competition platform on 2026-11-01 (Australia/Sydney). This is an absolute
**`+0.0021`** over coefficient 0.25's `0.6990`, and **`+0.0039`** over the
previous BCE-student ensemble's `0.6972`. This score is user-reported;
hidden test labels remain unavailable locally.

The candidate has exactly **75% MPNet / 25% MiniLM**, with both students using
**BCE + 0.5 * impression-normalized pairwise loss + teacher distillation**.
Both fit Train+Dev for **two fixed epochs**, reusing their retained final
four-epoch teachers and embeddings. Recency `alpha=0.02` is applied once per
member before exact integer weighted-Borda fusion; MPNet rank breaks exact
ties. The coefficient is not a fraction of the total gradient or pure
pairwise supervision: BCE and teacher distillation remain active.

- Workflow config: `configs/mind_large_two_model_final_fit_w050.yaml`.
- Student configs: `configs/mind_large_submission_mpnet_p1_pairwise_w050.yaml`
  and `configs/mind_large_submission_minilm_pairwise_w050.yaml`.
- Submission ZIP: `runs/mind_large_ensemble_two_mixed_loss_w050_final_75_25_v1/submission/prediction.zip`.
- Verified ZIP SHA256: `947e8a4d1d39aed43ffcd4d26d05556e84e4bd1437e1029ac6e19873463f5ac7`.
- Metadata: `runs/mind_large_ensemble_two_mixed_loss_w050_final_75_25_v1/submission/submission_meta.json`.
- Temporal comparison: `runs/mind_large_pairwise_w050_comparison_v1/ensemble_comparison.json`.
- Test impressions: **2,370,727**. Weights remained fixed at 75/25; no new
  ensemble weight search was performed.

| Reused temporal split | Coefficient 0.25 ensemble AUC | Coefficient 0.5 ensemble AUC | Delta |
| --- | ---: | ---: | ---: |
| November 14 | 0.699384 | 0.702605 | +0.003221 |
| November 15 | 0.687125 | 0.690140 | +0.003015 |

The individual temporal students gained +0.003579 AUC (MPNet) and +0.002792
(MiniLM), with selected epochs 3 and 4 respectively. November 15 ensemble
MRR declined 0.001894, nDCG@5 declined 0.003076, and nDCG@10 declined
0.002342. Individual cold-user, clicked-new-item and zero-prior-click AUC
slices also regressed. Record this as an AUC-targeted improvement, not an
improvement across all ranking metrics. The November dates remain reused
development diagnostics; the 0.7011 above is the separate Large Test result.
Preserve the 0.6990 and 0.6972 submissions and records. Slides are unchanged.

## Previous champion: coefficient 0.25 (reported 2026-10-29)

**Previous champion Large Test AUC: `0.6990`**, reported by the user from the
competition platform on 2026-10-29 (Australia/Sydney). This is an absolute
**`+0.0018`** over the preserved previous champion (`0.6972`). Hidden test
labels remain unavailable locally; the score is user-reported, not a local
AUC computation.

The submission combines exactly **75% mixed-loss MPNet student / 25% mixed-loss
MiniLM student**, using integer weighted-Borda rank fusion with MPNet rank
breaking exact fusion ties. Each student uses **BCE + 0.25 * impression-normalized
pairwise loss + teacher distillation**, fits Train+Dev for two fixed epochs,
and reuses its retained four-epoch final teacher and embeddings. Recency
`alpha=0.02` is applied once to each member before rank fusion.

- Workflow config: `configs/mind_large_two_model_final_fit.yaml`.
- Student configs: `configs/mind_large_submission_mpnet_p1_pairwise.yaml` and
  `configs/mind_large_submission_minilm_pairwise.yaml`.
- Submission ZIP: `runs/mind_large_ensemble_two_mixed_loss_final_75_25_v1/submission/prediction.zip`.
- ZIP SHA256 recorded in the verified completion marker: `390116e085d61dfabf9fe9d22cbc6cb2107eb8eacab417d18e8731664b5e1bac`.
- Metadata: `runs/mind_large_ensemble_two_mixed_loss_final_75_25_v1/submission/submission_meta.json`.
- Temporal selection: `runs/mind_large_ensemble_two_mixed_loss_v1/ensemble/search.json`.

The temporal search selected 75/25 on November 14, improving AUC from
`0.698596` to `0.699384` (`+0.000788`). With weights frozen, November 15 AUC
improved from `0.686049` to `0.687125` (`+0.001076`). Those dates remain reused
development diagnostics; the platform score above is the separate Large Test
result. This establishes improvement for the submitted end-to-end candidate,
not a causal attribution of the entire Large Test gain to the loss alone.
The previous `0.6972` ZIP and its result record remain preserved. Slides have
not been updated for this result.

## Configuration background

The coefficient-1.0 and mixed 1.0/0.5 variants are closed and rejected.
Their results and cleanup record are consolidated under
[rejected pairwise-weight variants](#rejected-pairwise-weight-variants).

The completed bounded comparison (agreed 2026-11-03) tests **coefficient 0.75 on
both students** against the completed 0.5 comparators. Only the coefficient
and isolated output names change; the eight-epoch temporal budget, full-candidate
mean impression AUC checkpoint selection, distillation, recency and fixed 75/25
ensemble weights remain unchanged. Use
`scripts/run_pairwise_weight_075_comparison.py`; see
[the coefficient-0.75 comparison](pairwise_weight_075_comparison.md).
Verified on 2026-11-05: both students completed eight epochs, selecting epochs
3 (MPNet) and 4 (MiniLM). MPNet individual AUC gained `+0.001002`; MiniLM lost
`0.001513`. The fixed 75/25 ensemble gained `+0.001133` AUC on November 14
(`0.703738` versus `0.702605`) and `+0.000819` on November 15 (`0.690959`
versus `0.690140`). MRR/nDCG declined on November 14 and improved on November 15.
Recorded checkpoints, stage hashes, fixed weights, counts and report deltas
passed verification. This meets the agreed promotion criterion.
The isolated [coefficient-0.75 final-fit workflow](pairwise_weight_075_final_fit.md)
is implemented in `scripts/run_pairwise_weight_075_final_fit.py`, using
`configs/mind_large_two_model_final_fit_w075.yaml`: both students reuse their
retained final teachers and fit Train+Dev for two fixed epochs, with a new
75/25 submission ZIP. Both final fits and submission generation completed;
the user reported Large Test AUC **0.7025** on 2026-11-07, an absolute
**+0.0014** over the preserved coefficient-0.5 champion. See the latest result
above for the verified submission identity.
The uniform-coefficient sweep is closed. The completed inference-only
**MPNet 0.75 / MiniLM 0.5** diagnostic kept 75/25 ensemble weights and
compared against the champion's **0.75/0.75 temporal ensemble**. It reused
existing checkpoints without training or weight search; only MiniLM changed.
The comparison passed verification: November 14 AUC improved
from `0.703737815` to `0.703789889` (`+0.000052074`), and November 15 from
`0.690959457` to `0.691412864` (`+0.000453407`). MRR and both nDCG metrics
also improved on both dates. The November 14 gain is small, and both dates
remain reused diagnostics. Individual student AUC alone did not determine
this decision.
The submission-only follow-up fused the existing two-epoch maximum-data
MPNet-0.75 and MiniLM-0.5 prediction ZIPs at 75/25, without training, candidate
scoring or reapplying recency. The user reported **Large Test AUC 0.7021** on
2026-11-07, **-0.0004** versus 0.7025. The asymmetric candidate is not promoted;
retain coefficient 0.75 for both students. See the follow-up record above for
the verified artifact identity and cleanup record. The dedicated asymmetric
code and artifacts were removed at the user's request; the source students
and successful submissions remain intact. This does not restore the rejected
coefficient-1.0 variants.
The popularity-feature ablation is deferred. The Large Test champion remains
the coefficient-0.75 ensemble at **0.7025**. Slides remain unchanged.

The completed temporal experiment increased the pairwise coefficient
from 0.25 to 0.5 on both MPNet and MiniLM, holding other training settings
fixed and reusing the completed 0.25 fits as comparators. It evaluates the two
new students at fixed 75/25 ensemble weights without a new weight search.
Use `scripts/run_pairwise_weight_comparison.py`; see
[the weight-comparison workflow](pairwise_weight_comparison.md).
Temporal fits and evaluation completed: MPNet AUC improved by +0.003579 and
MiniLM by +0.002792. The fixed 75/25 ensemble gained +0.003221 AUC on Nov 14
and +0.003015 on Nov 15, with regressions in top-ranking and cold/new-item
slices. The isolated final-fit candidate is implemented in
`configs/mind_large_two_model_final_fit_w050.yaml`: both students use the
retained final teachers and two fixed epochs, with a separate 75/25 ZIP.
Maximum-data execution completed; the user reported Large Test AUC **0.7011**
on 2026-11-01, establishing the best reported result at that stage. Slides are unchanged.

`configs/mind_large_temporal_baseline.yaml` defines the temporal split and shared defaults inherited by the current MPNet temporal config. It is a configuration dependency, not a separate prerequisite run. Running it directly reproduces the random-negative baseline listed below.

`mind_large_temporal_tune.yaml`, `mind_large_tune.yaml`, and the MiniLM submission configs remain available to reproduce other baselines; they are not commands in the current MPNet training path. Their protocols and results are recorded in this registry. The configs directly used by `scripts/run_mpnet_backbone.ps1` are listed in the [README training workflow](../README.md#34-phase-3-train-on-all-labeled-data-and-build-the-leaderboard-submission).

## Rejected pairwise-weight variants

**Closed; dedicated files removed on 2026-11-07 at the user's request.**
These experiments tested BCE + lambda * impression-normalized pairwise loss +
unchanged teacher distillation. Lambda 1.0 retained BCE; it was not pure
pairwise supervision. The matched comparator used lambda 0.5 for both students.
Architecture, teachers, embeddings, optimizer, seed, negative mining, grouped
sampled-impression batches and full-candidate mean impression AUC checkpoint
selection were fixed. Early stopping and calibration were disabled.

Both coefficient-1.0 students completed all eight temporal epochs and were
evaluated on the same **807,988 impressions / 31,485,663 candidates**:

| Student | Coefficient 0.5 AUC | Coefficient 1.0 AUC | Delta | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| MPNet | 0.695934653 | 0.696621897 | +0.000687244 | 3 for both |
| MiniLM | 0.679036448 | 0.679151188 | +0.000114740 | 4 for both |

All seven other overall student ranking metrics (MRR, nDCG@5/10,
Recall@5/10 and MAP@5/10) regressed for both students. Compared with 0.5,
clicked-new-item AUC fell by 0.002568 for MPNet and 0.006179 for MiniLM;
clicked-popularity-1--4 AUC fell by 0.008202 and 0.007694; latest-quarter AUC
fell by 0.001491 and 0.000993 respectively. MPNet cold-user AUC improved
0.004036, while MiniLM cold-user AUC declined 0.000228. These are descriptive
single-seed results, not causal proof of a popularity-bias mechanism.

The first ensemble used coefficient 1.0 on both students. The follow-up reused
MPNet 1.0 with the retained MiniLM 0.5 checkpoint, without additional training.
Both kept **75% MPNet / 25% MiniLM**, with no new weight search, recency alpha
0.02 applied once before ranking, the same reference-scoring guard and exact
integer weighted-Borda fusion, and MPNet rank breaking exact ties.

| Ensemble loss coefficients (MPNet / MiniLM) | November 14 AUC | Delta vs 0.5/0.5 | November 15 AUC | Delta vs 0.5/0.5 |
| --- | ---: | ---: | ---: | ---: |
| 0.5 / 0.5 comparator | 0.702604875 | -- | 0.690139988 | -- |
| 1.0 / 1.0, rejected | 0.704719190 | +0.002114315 | 0.689447560 | -0.000692428 |
| 1.0 / 0.5, rejected | 0.704572561 | +0.001967686 | 0.689544740 | -0.000595248 |

Both split stages completed (431,517 impressions on November 14; 376,471 on
November 15). MRR and both nDCG metrics declined on both dates for both variants.
November 15 deltas against the 0.5/0.5 comparator were:

| Variant | MRR delta | nDCG@5 delta | nDCG@10 delta |
| --- | ---: | ---: | ---: |
| 1.0 / 1.0 | -0.001627561 | -0.002060504 | -0.001814484 |
| 1.0 / 0.5 | -0.001145028 | -0.001492805 | -0.001239751 |

**Decision:** neither variant met the agreed requirement to improve ensemble
AUC on both dates. Neither received a maximum-data final fit or a Large Test
submission; no hidden-test score is claimed. The mixed follow-up shows that
changing only MPNet still produced the date-dependent reversal; it does not
establish that MiniLM caused the original regression. Both dates were already
reused during development, so these are validation diagnostics. Coefficient
0.75 later passed the two-date check and reached the user-reported **0.7025**
Large Test AUC; retain its workflow and the 0.5/0.25 comparators.

Cleanup consolidated the two guides here and removed their dedicated files:

- Configs: `configs/mind_large_temporal_mpnet_p1_pairwise_w100.yaml` and
  `configs/mind_large_temporal_minilm_pairwise_w100.yaml`.
- Runners: `scripts/run_pairwise_weight_100_comparison.py` and
  `scripts/run_mixed_pairwise_ensemble.py`, plus their two dedicated test files
  and cached bytecode.
- Guides: `docs/pairwise_weight_100_comparison.md` and
  `docs/mixed_pairwise_ensemble.md`.
- Run directories: `runs/mind_large_temporal_mpnet_p1_pairwise_w100_v1/`,
  `runs/mind_large_temporal_minilm_pairwise_w100_v1/`,
  `runs/mind_large_pairwise_w100_comparison_v1/`, and
  `runs/mind_large_ensemble_mpnet_w100_minilm_w050_v1/`.

The two deleted ensemble reports had SHA256 values
`40cb3411bfa09f47f12e71b361f8cc5a4502f59cbd809c185e75c2d3323f91e9`
(1.0/1.0) and
`4c973747efa09a382494e14b4256025b0e09d7969fcfb0058fecd1fdc558b75a`
(1.0/0.5). These hashes identify the source reports used to consolidate the
record above; the files themselves were removed. Retained workflows have no
dependency on the removed files. The 0.7025 champion, all earlier successful
submissions, shared training/fusion code and all slide-related files are preserved.

## Result Sets

| Label | Protocol | Reference | Config | Processed data | Run artifacts | Primary eval file |
| --- | --- | --- | --- | --- | --- | --- |
| Small dev split | Train on all `MINDsmall_train`; split `MINDsmall_dev` by time into Small Val and Small Test. | Commit `ad1fd9b6af3c28da7c5add090a93eb8a045d5bf3` | `configs/mind_small.yaml` | `data/processed/MINDsmall` | `runs/mind_small_demo` | `runs/mind_small_demo/eval/ranker_eval_test.json` |
| Large dev only | Train on all `MINDlarge_train`; validate on official `MINDlarge_dev` only, Nov 15. | Commit `b83a9308123fa196cf6be627762f01845545b123` | `configs/mind_large_tune.yaml` | `data/processed/MINDlarge_tune` | `runs/mind_large_tune` | `runs/mind_large_tune/eval/ranker_eval_val.json` |
| Large temporal baseline | Train on `MINDlarge_train` before Nov 14; validate on Nov 14 tail from `MINDlarge_train` plus all `MINDlarge_dev`; use random ranker negatives. | Current repo | `configs/mind_large_temporal_baseline.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_tune` | `runs/mind_large_temporal_tune/eval/ranker_eval_val.json` |
| Large temporal hard-negative v4 | Use the Large temporal split and baseline teacher; mine hard negatives only for cold users with usable history. | Current repo | `configs/mind_large_temporal_tune.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_hard_neg_v4` | `runs/mind_large_temporal_hard_neg_v4/eval/ranker_eval_val.json` |
| Large temporal text-adapt v1 | Adapt MiniLM on Large Temporal Train, select its update count on Large Temporal Val, and train the temporal teacher/ranker with the selected encoder. | Current repo | `configs/mind_large_temporal_text_adapt.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_text_adapt_v1` | `runs/mind_large_temporal_text_adapt_v1/eval/ranker_eval_val.json` |
| Large temporal candidate-attention v1 | Reuse the text-adapt v1 teacher and replace only student mean history pooling with candidate-aware attention. | Current repo; completed | `configs/mind_large_temporal_candidate_attention.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_text_adapt_candidate_attention_v1` | `runs/mind_large_temporal_text_adapt_candidate_attention_v1/eval/ranker_eval_val.json` |
| Large temporal MPNet backbone v1 | Replace MiniLM with adapted `all-mpnet-base-v2` in the selected candidate-attention pipeline; keep the temporal split, objective, negative policy, and downstream settings fixed. | Completed; promoted to Phase 3 | `configs/mind_large_temporal_mpnet.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_mpnet_candidate_attention_v1` | `runs/mind_large_temporal_mpnet_candidate_attention_v1/eval/ranker_eval_val.json` |
| Large temporal MPNet reranker | Highest full-tuning nDCG subject to aggregate guardrails on Nov 14; frozen Nov-15 follow-up. | Evaluated; 3/4 reporting guardrails pass (pool KL fails) | `configs/mind_large_temporal_mpnet.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_mpnet_candidate_attention_v1/rerank` | `runs/mind_large_temporal_mpnet_candidate_attention_v1/rerank/rerank_eval.json` |
| Original Large submission MPNet candidate attention + recency | Continue selected MPNet on Large Temporal Val, then fixed four-epoch teacher, fixed two-epoch candidate-attention ranker, and recency `alpha=0.02`. | Completed historical baseline, Large Test AUC `0.6948` | `configs/mind_large_submission_mpnet_candidate_attention_recency_alpha_002.yaml` | `data/processed/MINDlarge_submission` | `runs/mind_large_submission_mpnet_candidate_attention_low_lr_2ep_recency_alpha_002_v1` | `runs/mind_large_submission_mpnet_candidate_attention_low_lr_2ep_recency_alpha_002_v1/submission/prediction.zip` |
| Large MPNet + MiniLM rank ensemble | Select a weighted-Borda MPNet weight on Nov 14, freeze it, report on Nov 15, then fuse the two recency-adjusted hidden-test rank ZIPs. | Completed; user-reported Large Test AUC `0.6960` | `configs/mind_large_ensemble_mpnet_minilm.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_ensemble_mpnet_minilm_rank_v1` | `runs/mind_large_ensemble_mpnet_minilm_rank_v1/ensemble/search.json` |
| Improved MPNet (lr=1e-5) + MiniLM rank ensemble | Retune the Borda weight using the promoted temporal MPNet from `31ab682`, then fuse its maximum-data ZIP with the same selected MiniLM. | Completed; previous champion Large Test AUC `0.6972`; selected 75/25 | `configs/mind_large_ensemble_mpnet_p1_minilm.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1` | `runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/ensemble/search.json` |
| Two mixed-loss students, coefficient 0.75, frozen 75/25 (best reported) | Increase both pairwise coefficients from 0.5 to 0.75; retain 75/25 weights and two-epoch Train+Dev fits. | Completed; user-reported Large Test AUC `0.7025` on 2026-11-07 | `configs/mind_large_two_model_final_fit_w075.yaml` | `data/processed/MINDlarge_submission` | `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1` | `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1/submission/prediction.zip` |
| Asymmetric coefficients, MPNet 0.75 / MiniLM 0.5 (not promoted; cleaned up) | Reuse existing final predictions at fixed 75/25; no training or extra recency adjustment. | Completed; user-reported Large Test AUC `0.7021` on 2026-11-07, -0.0004 vs champion | Dedicated runner removed; see follow-up record above | Existing `MINDlarge_submission` source member predictions retained | Dedicated run directory removed after documentation | Combined ZIP removed; verified identity recorded above |
| Two mixed-loss students, coefficient 0.5, frozen 75/25 (previous champion) | Increase both pairwise coefficients from 0.25 to 0.5; retain 75/25 weights and two-epoch Train+Dev fits. | Completed; user-reported Large Test AUC `0.7011` on 2026-11-01 | `configs/mind_large_two_model_final_fit_w050.yaml` | `data/processed/MINDlarge_submission` | `runs/mind_large_ensemble_two_mixed_loss_w050_final_75_25_v1` | `runs/mind_large_ensemble_two_mixed_loss_w050_final_75_25_v1/submission/prediction.zip` |
| Two mixed-loss students, coefficient 0.25, frozen 75/25 (previous champion) | Matched temporal loss comparisons; select weight on Nov 14 and freeze before Nov 15; refit both students on Train+Dev for two epochs. | Completed; user-reported Large Test AUC `0.6990` on 2026-10-29 | `configs/mind_large_two_model_final_fit.yaml` | `data/processed/MINDlarge_submission` | `runs/mind_large_ensemble_two_mixed_loss_final_75_25_v1` | `runs/mind_large_ensemble_two_mixed_loss_final_75_25_v1/submission/prediction.zip` |
| Rejected Large temporal MPNet student width 96 | Reuse the selected MPNet encoder and 384-dimensional teacher; change only the learned student width from 56 to 96. | Completed/rejected on 2026-08-31; Phase 3 not run | Removed after rejection | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_mpnet_candidate_attention_student_width_96_v1` | `runs/mind_large_temporal_mpnet_candidate_attention_student_width_96_v1/eval/ranker_eval_val.json` |
| Rejected large temporal candidate-attention item-only distillation v1 | Reuse candidate-attention v1 and change representation distillation from the full user/item target to `item_sem -> teacher_item`, including zero-history rows; teacher-logit distillation and all configured weights remain fixed. | Current repo; completed/rejected | `configs/mind_large_temporal_candidate_attention_item_only_distill.yaml` | `data/processed/MINDlarge_temporal_tune` | `runs/mind_large_temporal_text_adapt_candidate_attention_item_only_distill_v1` | `runs/mind_large_temporal_text_adapt_candidate_attention_item_only_distill_v1/eval/ranker_eval_val.json` |
| Small temporal | Train on `MINDsmall_train` before Nov 14; validate on Nov 14 tail from `MINDsmall_train` plus all `MINDsmall_dev`. | Current repo | `configs/mind_small_temporal_tune.yaml` | `data/processed/MINDsmall_temporal_tune` | `runs/mind_small_temporal_tune` | `runs/mind_small_temporal_tune/eval/ranker_eval_val.json` |

## Closed experimental branches

The following experiments were rejected or superseded. Their unused configs,
duplicate setup documents and orphaned tests were removed during cleanup on
2026-10-21. Their runners/model extensions had already been removed. Saved
checkpoints, comparison files and metric artifacts under `runs/` are retained.
These are historical results, not instructions to rerun the closed branches.

### Removed configuration index

Every YAML removed in the cleanup is accounted for below. The two prepared
full-retraining configs were not run; they must not be interpreted as failed
full-training experiments. Only the associated continuation pilots supplied
the evidence for closing that branch.

| Removed file under `configs/` | Status and retained evidence |
| --- | --- |
| `inbatch_multi_positive_pilot.yaml` | Completed pilot; not promoted. Results and limitations are recorded below under in-batch multi-positive/masking. |
| `inbatch_positive_mask_pilot.yaml` | Completed pilot and expanded evaluation; not promoted because ranking regressed. Results are recorded below. |
| `mind_large_temporal_mpnet_inbatch_positive_control.yaml` | Prepared full-retraining control, not run. Removed after the pilot branch was closed; no full-training result is claimed. |
| `mind_large_temporal_mpnet_inbatch_positive_mask.yaml` | Prepared full-retraining treatment, not run. Removed after the pilot branch was closed; no full-training result is claimed. |
| `mind_large_temporal_mpnet_relation_weight_0p3.yaml` | Completed and rejected: AUC 0.691009 versus retained 0.691440. See higher KG relation weights below. |
| `mind_large_temporal_mpnet_relation_weight_1p0.yaml` | Completed and rejected: AUC 0.690835 versus retained 0.691440. See higher KG relation weights below. |
| `mind_large_temporal_mpnet_no_dense_output.yaml` | Completed and rejected: AUC 0.685379 versus its matched control's 0.688880. See direct dense-output ablation below. |
| `mind_large_temporal_mpnet_p1_impression_auc.yaml` | Completed standalone selector run, not promoted: AUC 0.689509 versus 0.691440. Superseded by the matched dual-selector experiment described in the next section. |

### In-batch multi-positive and known-positive masking

Both studies used matched short continuations from the selected improved
MPNet encoder/teacher, not full student retraining. Encoder budgets were 64
updates and teacher budgets 200 updates, on 20,000 sampled training impressions.
The expanded comparison used 48,623 eligible impressions from a fixed 50,000
validation sample (seed 14).

| Variant | Teacher AUC change vs matched control | Teacher nDCG@10 change | Catalogue Recall@200 change |
| --- | ---: | ---: | ---: |
| Multi-positive teacher only | +0.000287 | +0.000037 | -0.000051 |
| Multi-positive encoder only | -0.000404 | -0.000325 | -0.000106 |
| Multi-positive encoder and teacher | +0.000114 | -0.000034 | -0.000030 |
| Known-positive masking in both stages | -0.000485 | -0.000599 | -0.000144 |

The multi-positive control teacher had AUC 0.703622, nDCG@10 0.434263 and
Recall@200 0.040892. Teacher-only AUC's paired 95% interval was
[+0.000025, +0.000563], but retrieval recall showed no clear benefit.
Standalone encoder multi-positive loss reduced AUC by 0.000410.
For masking in both stages, AUC's interval was [-0.000757, -0.000214]
and nDCG@10's was [-0.000953, -0.000229]. These intervals describe impression
variation for one training seed, not seed uncertainty, and do not adjust for
user clustering or multiple comparisons. Validation has been reused.

Decision: stop both branches; retain the original encoder/teacher and the
selected production checkpoints. Neither experiment establishes improved
student ranking or retrieval recall. The source/artifact integrity checks and
full comparison details remain in:

- `runs/mind_large_temporal_inbatch_multi_positive_pilot_v1/`
- `runs/mind_large_temporal_inbatch_positive_mask_pilot_v2/`
- `runs/mind_large_temporal_inbatch_positive_mask_pilot_v2/evaluation_50000_seed_14/`

### Higher KG relation weights

The student-only tests reused SHA-256-identical frozen teachers, text embeddings
and teacher item embeddings. They changed `knowledge_graph.relation_weight`
while retaining the entity/neighbor/global KG weights, graph, temporal split,
sampling and student architecture. Every run evaluated 807,988 impressions and
31,485,663 candidates; training sampling statistics and taxonomy masks matched.

| Relation weight | Mean impression AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| 0.1, retained improved-MPNet baseline | 0.691440 | 0.337835 | 0.372175 | 0.432364 |
| 0.3 | 0.691009 | 0.337409 | 0.371868 | 0.431915 |
| 1.0 | 0.690835 | 0.337246 | 0.371487 | 0.431601 |

All three selected epoch 4 and stopped after epoch 6. Neither stronger weight
met the +0.001 promotion threshold; both shifted quality toward warm/popular
items while cold-user and new/rare-item slices regressed. This is descriptive
single-seed evidence, not proof of the mechanism.
Decision: retain 0.1 and stop this scalar-weight sweep. Results and preparation
manifests remain in `runs/mind_large_temporal_mpnet_p1_relation_weight_0p3_v1/`
and `runs/mind_large_temporal_mpnet_p1_relation_weight_1p0_v1/`.

### Direct dense-output ablation

The removed variant omitted the bottom MLP's direct output from the scorer,
reducing its first-layer input from 532 to 476 values. Dense embeddings,
all eight feature vectors, 28 interactions, semantic fusion and distillation
remained active. This was not a removal of popularity information.

| Metric | Original MPNet control | Without direct dense output | Change |
| --- | ---: | ---: | ---: |
| Mean impression AUC | 0.688880 | 0.685379 | -0.003501 |
| MRR | 0.336397 | 0.333980 | -0.002417 |
| nDCG@5 | 0.371215 | 0.367857 | -0.003358 |
| nDCG@10 | 0.431291 | 0.428079 | -0.003212 |

Both evaluations used the same 807,988 impressions and 31,485,663 candidates.
The original MPNet control differs from the later improved-MPNet baseline.
All six headline ranking metrics regressed, including both recall metrics.
The latest chronological quarter lost 0.010875 AUC. Sampled-pair selection AUC
rose slightly (+0.000231), illustrating its mismatch with impression ranking.
Decision: retain the original scorer. The saved evaluation remains at
`runs/mind_large_temporal_mpnet_no_dense_output_v1/eval/ranker_eval_val.json`.

## Current ranker loss and checkpoint-selection experiment

The standalone full-impression checkpoint-selector run reached AUC 0.689509,
below the promoted single model's 0.691440. Its training trajectory differed
because validation changed random-state consumption. That run was not promoted;
its setup config has been superseded, and its artifacts remain in
`runs/mind_large_temporal_mpnet_p1_impression_auc_v1/`.

The subsequent matched loss comparison completed eight epochs per arm. Both
arms use the same grouped sampled-impression batches and preserve training RNG
state across validation. Architecture, teacher, sampling and distillation are
fixed. Each epoch is evaluated by both selectors on the same model weights.

| Supervised objective | Best impression-AUC epoch | Mean impression AUC | Sampled-AUC selector's epoch |
| --- | ---: | ---: | ---: |
| BCE control | 4 | 0.687608 | 4 |
| BCE + 0.25 * impression-normalized pairwise loss | 3 | 0.692355 | 2 |

The mixture gains +0.004747 over its matched BCE control and +0.000915 over
the historical promoted single model. In the mixture run, impression-AUC
selection adds +0.001478 relative to the sampled selector's chosen epoch.
Both evaluations cover 807,988 impressions and 31,485,663 candidates.
These matched-loss results are temporal validation evidence. The subsequent
two-model maximum-data submission achieved user-reported Large Test AUC 0.6990,
then replacing 0.6972 as the best reported submission; the later coefficient-0.5
candidate reached 0.7011. Pure supervised
pairwise loss, higher mixture weights and sampled listwise loss remain untested.

Active configs: `configs/mind_large_temporal_mpnet_p1_dual_bce.yaml` and
`configs/mind_large_temporal_mpnet_p1_dual_pairwise.yaml`.
See [the comparison workflow](ranker_loss_comparison.md). The completed report
is `runs/mind_large_temporal_mpnet_p1_loss_comparison_v1/comparison.json`.

The subsequent frozen ensemble experiment completed. A 50% pairwise-model
share, keeping the previous ensemble's internal 75/25 ratio, selected effective
weights 37.5/12.5/50. November 14 AUC improved from 0.698596 to 0.701471;
November 15 AUC improved from 0.686049 to 0.687562. These reused validation
diagnostics support a maximum-data candidate, not a new Large Test claim.
The saved report is
`runs/mind_large_ensemble_mpnet_p1_minilm_pairwise_share_v1/ensemble/search.json`.
The unused three-model final-fit workflow was removed in favor of the
frozen two-model candidate. Its completed temporal diagnostic is retained.

The completed follow-up uses exactly two mixed-loss students. It adds a
matched MiniLM BCE versus BCE + 0.25 pairwise comparison using the retained
MiniLM teacher and the same eight-epoch dual-selector protocol. The runner
accepts `--backbone minilm`; existing MPNet defaults remain unchanged. The
standalone `configs/mind_large_ensemble_two_mixed_loss.yaml` then searches
weights for the retained mixed-loss MPNet and new mixed-loss MiniLM on Nov 14,
freezes selection, and reports Nov 15 against the saved champion temporal
metrics. Temporal training, ensemble search and maximum-data execution completed;
the user reported Large Test AUC `0.6990` on 2026-10-29. See [the current workflow](two_model_mixed_loss_ensemble.md).

## Latest Completed Large Temporal Baseline

| Validation impressions | Teacher best epoch | Ranker best epoch | AUC | MRR | nDCG@5 | nDCG@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 807,988 | 4 | 1 | 0.646597 | 0.304210 | 0.330920 | 0.393785 |

This run trains on `MINDlarge_train` impressions before Nov 14 and evaluates the full validation window formed from the Nov 14 tail of `MINDlarge_train` plus all of `MINDlarge_dev` on Nov 15. The teacher stopped at epoch 6 and selected epoch 4; the ranker stopped at epoch 3 and selected epoch 1. These are full impression-ranking metrics from `ranker_eval_val.json`; the ranker's sampled-pair early-stopping AUC is a different quantity.

## Latest Completed Large Temporal Hard-Negative V4

This result uses the same 807,988 validation impressions as the baseline.

| AUC | MRR | nDCG@5 | nDCG@10 |
| ---: | ---: | ---: | ---: |
| 0.654199 | 0.304433 | 0.331107 | 0.394609 |

V4 uses `configs/mind_large_temporal_tune.yaml`. Its zero-history groups use four random negatives; cold users with usable history use one teacher-hard plus three random negatives.

The previously recorded `0.645785` AUC belongs to the separate `mind_large_temporal_hard_neg_v4_ranker_lr_3em04` run, not the canonical `mind_large_temporal_hard_neg_v4` artifact.

## Completed MiniLM Text Adaptation V1

Phase 1 adapted `sentence-transformers/all-MiniLM-L6-v2` on Large Temporal Train and evaluated the raw history-mean/candidate-cosine objective every 1,000 optimizer updates on 785,325 evaluable Large Temporal Val impressions. The base encoder scored `0.623156` AUC. Update 6,000 was selected at `0.670009` AUC; update 7,000 reached `0.670094`, but its `+0.000084` change did not meet the configured `1e-4` minimum improvement. Early stopping fired at update 9,000.

Phase 2 froze the selected update-6,000 encoder and trained the temporal teacher/ranker. These metrics use the full 807,988-impression Large Temporal Val evaluation and are directly comparable with the canonical frozen-MiniLM hard-negative v4 run:

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Frozen MiniLM hard-negative v4 | 0.654199 | 0.304433 | 0.331107 | 0.394609 |
| Adapted MiniLM v1 | 0.664328 | 0.311520 | 0.341145 | 0.403632 |
| Absolute gain | +0.010130 | +0.007087 | +0.010038 | +0.009023 |

Phase 3 loaded the selected update-6,000 encoder and continued adaptation for exactly 2,000 updates on Large Temporal Val with a fresh optimizer. The final encoder therefore records 8,000 cumulative staged updates. The maximum-data teacher and ranker were then fitted on `MINDlarge_train + MINDlarge_dev` with four teacher epochs and one ranker epoch. No labeled local validation split was retained for this final fit.

## Controlled MPNet Backbone Experiment

`configs/mind_large_temporal_mpnet.yaml` changes the selected temporal pipeline's text backbone to `sentence-transformers/all-mpnet-base-v2`. It otherwise retains the natural Large Temporal Train/Val distribution, text-adaptation loss and early stopping, cold-user 1-hard/3-random policy, teacher settings, and the candidate-aware ranker. The 768-dimensional encoder is inferred at runtime; the teacher still projects to the fixed 384-dimensional hidden space.

The tuned MiniLM configured training schedule is retained: physical batch size 16, four gradient-accumulation steps, up to 10,000 optimizer updates, and validation every 1,000 updates. On the 6 GB target GPU, MPNet instead saves memory through FP16 autocasting, whole-encoder activation checkpointing, and chunked article encoding. Article embeddings are reassembled before the loss, so batch membership and the 16-example in-batch-negative set remain unchanged. Offline text-encoding batches are reduced to 64 independently of training.

Candidate-attention scoring caches item and history states, then runs only the small candidate-conditioned attention operation per candidate. These implementation details support the [three-phase workflow](../README.md#3-end-to-end-large-mpnet-workflow) within the target GPU's memory budget.

Phase 1 selected update 9,000. The full Phase 2 evaluation produced AUC `0.688880`, MRR `0.336397`, nDCG@5 `0.371215`, and nDCG@10 `0.431291` across 807,988 impressions, exceeding the matched MiniLM candidate-attention AUC of `0.671593`. This passed the promotion gate.

The original `lr=2e-5` Phase 3 workflow on this branch is implemented by `scripts/run_mpnet_backbone.ps1 -Phase phase3`. It is locked to the selected update-9,000 encoder and exactly 2,000 successful continuation optimizer updates on Large Temporal Val (11,000 cumulative), followed by four fixed teacher epochs and two fixed maximum-data candidate-attention ranker epochs. It retains `lr=1e-4`, `weight_decay=3e-5`, four attention heads, zero attention dropout, full representation distillation, and submission recency `alpha=0.02`. The workflow verifies temporal results and stage metadata before reusing artifacts; all MPNet artifacts are isolated from the MiniLM submission runs.

The completed continuation used the locked batch size 16 and four accumulation steps. Four FP16-overflow attempts were skipped and replaced, so the artifact records exactly 2,000 successful optimizer updates (11,000 cumulative), 8,016 microbatches, and 128,256 examples versus nominal counts of 8,000 and 128,000. The configured update count and logical batch construction therefore remained unchanged.

The maximum-data teacher completed four fixed epochs with 768-dimensional text features and a 384-dimensional teacher space. The ranker completed two fixed epochs, loaded `mind_large_submission_mpnet_v1`, used candidate attention and the full representation-distillation target, and processed 19,062,280 selected pairs. The submission used recency `alpha=0.02` and exact-rank guarding for 79,898 impressions.

The competition platform reported Large Test AUC **`0.6948`** on 2026-08-30. This is `+0.0079` over the selected MiniLM candidate-attention champion (`0.6869`), `+0.0100` over text-adapt v1 (`0.6848`), and `+0.0224` over frozen MiniLM (`0.6724`). The matched temporal MPNet gain was `+0.017287`; approximately 46% of that AUC advantage transferred to the hidden test set. Hidden-test MRR and nDCG values were not reported, so no claims are made for those metrics.

Local structural validation confirmed one `prediction.txt` entry in the ZIP, 2,370,727 sequential impression IDs from 1 through 2,370,727, valid sampled rank permutations, a clean ZIP CRC, and an exact hash match between zipped and external prediction text. The prediction SHA-256 is `1705ef49e0ecec2492cd5d25890bc566b1ce9f0c98d07630c533f966013cc68c`.

## MPNet Reranking

The reranker selects the highest full-tuning nDCG@10 subject to aggregate requirements: at most 2% relative nDCG loss, non-decreasing new-item exposure, at least +0.25 mean category coverage, and at least 0.03 pool-KL reduction. The score combines relevance, category/entity coverage, and pool KL. There is no L1 or new-item shortfall penalty and no ILD guardrail.

The final 15-setting search evaluated all combinations on 431,517 tuning impressions; 11 qualified. The frozen winner uses coverage 0.035 and KL penalty 0 (derived relevance 0.965), with nDCG@10 0.423648 and 0.122% relative loss. Category coverage gains 0.299184, new-item exposure gains 0.217670 percentage points, and pool KL decreases by 0.030268. All four tuning guardrails pass; the KL margin is narrow. Teacher-cosine ILD remains diagnostic only.

Reporting is complete on 376,471 November 15 impressions. nDCG@10 decreased from 0.439457 to 0.438302 (0.262693% relative loss); category coverage gained 0.280850 and new-item exposure gained 0.071620 percentage points. These three guardrails pass. Pool KL decreased from 0.442187 to 0.413450, an improvement of 0.028736 that **fails the required 0.03** by 0.001264. Teacher-cosine ILD increased by 0.001823 as a diagnostic result.

The frozen config and measured artifacts remain unchanged for reproduction; the result is not approved under all reporting requirements. November 15 was reused while refining requirements and remains follow-up reporting. See [reranking](reranking.md) for the formula, grid process, and commands.

## MPNet Ranker Experiments

### Rejected controlled student-width 56 -> 96 experiment

This ranker-only experiment reused the selected temporal MPNet encoder and 384-dimensional teacher and changed the learned student width from 56 to 96. The corresponding full student representation grew from 168 to 288 dimensions, while the full teacher target remained 768-dimensional. Data, hard/random negative selection, batch size, learning rate, weight decay, ID width, bottom/top MLPs, semantic feed-forward settings, candidate attention, full distillation weights, epoch cap, and early-stopping rule were unchanged.

The resolved training data was identical to width 56: 12,990,042 selected rows per epoch, including 10,277,430 negatives (219,513 hard and 10,057,917 random), at batch size 1,024. This retained 12,686 batches per epoch and the configured eight-epoch maximum of 101,488 batches. The unchanged early-stopping rule chose epoch 5 and stopped after epoch 7 for width 96 (88,802 completed batches), versus epoch 4 and epoch 6 for width 56 (76,116 completed batches). Thus the configured schedule was unchanged, while the metric-driven completed count differed by one epoch.

Full 807,988-impression Large Temporal Val results were:

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Selected MPNet student width 56 | **0.688880** | **0.336397** | **0.371215** | **0.431291** |
| MPNet student width 96 | 0.688316 | 0.329523 | 0.362792 | 0.424056 |
| Width-96 delta | -0.000564 | -0.006873 | -0.008423 | -0.007235 |

Width 96 marginally improved the sampled-pair early-stopping AUC (`0.687490` versus `0.686866`) but regressed every full-impression headline metric. This indicates that the extra capacity fitted the sampled pairwise objective better without improving listwise ranking.

The time-period AUC deltas were `+0.007366`, `-0.001730`, `+0.001355`, and `-0.009245` from the earliest to latest quarter. The severe latest-period regression was especially unfavorable for hidden-test transfer. Width 96 also improved the 22,663 zero-history impressions, but MRR and nDCG regressed for the much larger history-bearing population.

The promotion rule required at least `+0.001` over width 56, or AUC `0.689879567`. Width 96 instead trailed width 56 and missed that threshold by `0.001563523`. Phase 3 was therefore not run, the width-56 MPNet submission remained the `0.6948` champion, and the experimental config/script/code were removed. The local rejected-run metrics remain under `runs/mind_large_temporal_mpnet_candidate_attention_student_width_96_v1`.

### Rejected Phase 3 replay experiment

A follow-up Phase 3 experiment tested whether replaying older temporal-training examples would reduce forgetting during the 2,000-update continuation. It used the same Phase 1 update-6,000 checkpoint and an exact 20-sample mixture cycle:

| Continuation source | Fraction | Samples |
| --- | ---: | ---: |
| Large Temporal Val (Nov 14-15) | 85% | 108,800 |
| Large Temporal Train, Nov 13 | 10% | 12,800 |
| Large Temporal Train, Nov 12 | 5% | 6,400 |
| **Total** | **100%** | **128,000** |

The maximum-data teacher/ranker and post-hoc recency setting remained fixed at four teacher epochs, one ranker epoch, and `alpha=0.02`. Large Test AUC fell to `0.6800`, which is `-0.0048` versus the pure-Temporal-Val Phase 3 result of `0.6848` (but still `+0.0076` versus the frozen-MiniLM baseline).

The replay data had already been seen during Phase 1, while it displaced 15% of the newer Nov 14-15 continuation samples. The result therefore provides no evidence that this run needed protection from catastrophic forgetting and is consistent with older replay weakening adaptation to the later target period. The replay implementation and configuration were discarded; it is a rejected result, not a supported current pipeline. Historical local artifacts, if kept, are under `runs/mind_large_submission_text_adapt_replay_85_10_5_v1` and the submitted ZIP is under `runs/mind_large_submission_text_adapt_replay_85_10_5_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 Nov 15 overweight experiment

A second follow-up kept the Phase 1 update-6,000 checkpoint, 2,000 continuation updates, learning rate, hard-negative policy, and downstream pipeline fixed. It used only Large Temporal Val but changed its clicked-positive sample mix with an exact 20-sample cycle:

| Continuation date | Fraction | Samples |
| --- | ---: | ---: |
| Nov 14 | 45% | 57,600 |
| Nov 15 | 55% | 70,400 |
| **Total** | **100%** | **128,000** |

The maximum-data teacher/ranker again used four teacher epochs and one ranker epoch, followed by post-hoc recency `alpha=0.02`. Large Test AUC was `0.6796`, which is `-0.0052` versus the natural Large Temporal Val mixture at `0.6848` (but still `+0.0072` versus the frozen-MiniLM baseline).

The earlier observation that adaptation improved Nov 15 evaluation more than Nov 14 did not imply that Nov 15 examples deserved greater training weight. Overweighting Nov 15 displaced Nov 14 topical/source diversity and did not generalize to the later hidden test period. Together with the older-data replay result, this provides evidence against further manual date-mixture sweeps; keep the natural shuffled Large Temporal Val distribution. The implementation and configuration were discarded. Historical local artifacts, if kept, are under `runs/mind_large_submission_text_adapt_nov15_weighted_v1`, and the submitted ZIP is under `runs/mind_large_submission_text_adapt_nov15_weighted_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 2,500-update experiment

A third follow-up changed only the natural Large Temporal Val continuation length. It started again from the Phase 1 update-6,000 checkpoint, retained the globally shuffled unweighted validation distribution and `lr=2e-5`, and trained for 2,500 updates (160,000 samples) instead of 2,000. The resulting encoder recorded 8,500 cumulative staged updates. The maximum-data teacher/ranker and post-hoc recency settings remained fixed at four teacher epochs, one ranker epoch, and `alpha=0.02`.

Large Test AUC was `0.6842`, which is `-0.0006` versus the then-current 2,000-update text-adapt v1 result at `0.6848` (and `+0.0118` versus the frozen-MiniLM baseline). Downstream training losses improved slightly despite the hidden-test regression: teacher loss moved from `6.139574` to `6.138398`, and ranker loss from `0.834028` to `0.833814`. This is consistent with mild overfitting or overshooting during the additional 500 constant-learning-rate updates. Keep 2,000 Phase 3 updates; do not extend the continuation to 2,500 at `lr=2e-5`.

The implementation and configuration were discarded. Historical local artifacts, if kept, are under `runs/mind_large_submission_text_adapt_updates_2500_v1`, and the submitted ZIP is under `runs/mind_large_submission_text_adapt_updates_2500_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 teacher-guided hard-negative experiment

A fourth follow-up changed only Phase 3 adaptation hard-negative scoring. It loaded the selected Phase 2 teacher at epoch 4 and used its frozen projected item vectors plus transformer/attention history representation to score each sampled negative pool. The Phase 1 update-6,000 source, natural shuffled Large Temporal Val, 2,000 updates, `lr=2e-5`, cold-user-only policy, hard fraction `0.25`, pool size `20`, consistency guard, and downstream pipeline remained fixed. MiniLM still optimized the original history-mean/clicked-article contrastive objective.

Large Test AUC was `0.6835`, which is `-0.0013` versus the then-current snapshot-mined text-adapt v1 result at `0.6848` (and `+0.0111` versus the frozen-MiniLM baseline). The teacher changed hard-negative identity rather than quantity: both runs mined 11,809 groups, while the teacher-guided run selected 10,632 hard negatives versus 10,635 for the reference run. Its consistency guard rejected 63,876 of 177,471 scored pool negatives because the teacher placed them above the clicked positive.

The likely problem was objective/representation mismatch. The teacher judged difficulty with learned attention and projected item vectors, but the MiniLM adaptation loss represented history with a mean of raw MiniLM vectors. A teacher-hard negative was therefore not necessarily a useful contrastive target for the representation MiniLM was actually trained to produce. Downstream losses were essentially unchanged (teacher `6.139544` versus `6.139574`; ranker `0.834134` versus `0.834028`), providing no evidence of an operational failure. Keep the original update-6,000 MiniLM snapshot scorer.

The implementation and configuration were discarded. Historical local artifacts, if kept, are under `runs/mind_large_submission_text_adapt_teacher_guided_v1`, and the submitted ZIP is under `runs/mind_large_submission_text_adapt_teacher_guided_recency_alpha_002_v1/submission/prediction.zip`.

### Rejected Phase 3 learning-rate 1.5e-5 experiment

A fifth follow-up changed only the Phase 3 MiniLM learning rate from `2e-5` to `1.5e-5`. It retained the Phase 1 update-6,000 checkpoint, natural shuffled Large Temporal Val data, 2,000 continuation updates (128,000 samples), original snapshot-mined hard negatives, maximum-data teacher/ranker fit, and post-hoc recency `alpha=0.02`.

Large Test AUC was `0.6797`, which is `-0.0051` versus the then-current `lr=2e-5` text-adapt v1 result at `0.6848` (but still `+0.0073` versus the frozen-MiniLM baseline). The run used the intended data and artifact routing, and its sample and negative-policy counts matched the reference run. Its teacher loss was marginally lower (`6.139317` versus `6.139574`), while ranker loss was slightly worse (`0.834304` versus `0.834028`); neither suggests an operational failure. Compared with the reference submission, only 283,858 of 2,370,727 impressions (11.97%) retained an identical full ranking, and 1,646,229 (69.44%) retained the same top-ranked candidate, confirming that the learning-rate change materially propagated through the downstream pipeline.

With a fixed 2,000-update continuation, `1.5e-5` reduces the nominal Phase 3 update scale by 25%. The hidden-test regression is therefore more consistent with insufficient adaptation to the recent Nov 14-15 distribution than with successful overfitting control. Together with the rejected 2,500-update run, the evidence supports keeping `lr=2e-5` and 2,000 Phase 3 updates rather than continuing a constant-learning-rate or update-count sweep.

The implementation and configuration were discarded. Historical local artifacts, if kept, are under `runs/mind_large_submission_text_adapt_lr_1p5em05_v1`, and the submitted ZIP is under `runs/mind_large_submission_text_adapt_lr_1p5em05_recency_alpha_002_v1/submission/prediction.zip`.

## Rejected Candidate-Attention Follow-ups

All temporal results below use the same 807,988-impression validation protocol and retain the 4-head, zero-dropout direct candidate-attention run at `0.671593` AUC as the reference. Most earlier experimental implementations and configs were discarded; the item-only distillation config is retained so its negative result remains reproducible.

This statement describes upstream architecture experiments. Reranking uses Nov 14 (`431,517` impressions) for tuning and Nov 15 (`376,471` impressions) for reporting. Both days were used in upstream model development, and Nov 15 was reused while refining reranking requirements; it is not an independent test.

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

The item-only representation-distillation run completed on 2026-08-27. It kept candidate-aware scoring, teacher-logit distillation, the teacher and data artifacts, optimizer schedule, and all configured loss weights unchanged. It replaced the joint projection from `[user_sem, item_sem, sem_fused]` to `[teacher_user, teacher_item]` with `item_sem -> teacher_item`; item representation loss also remained active for zero-history rows. The selected sampled-pair checkpoint moved from epoch 3 and `0.674491` AUC to epoch 2 and `0.671052` AUC, and the full-impression result regressed across every primary ranking metric:

| Metric | Direct candidate attention | Item-only distillation | Delta |
| --- | ---: | ---: | ---: |
| AUC | 0.671593 | 0.666388 | -0.005205 |
| MRR | 0.321535 | 0.316708 | -0.004828 |
| nDCG@5 | 0.352780 | 0.346539 | -0.006241 |
| nDCG@10 | 0.414064 | 0.408609 | -0.005455 |
| Recall@5 | 0.489773 | 0.482717 | -0.007056 |
| Recall@10 | 0.664185 | 0.659521 | -0.004665 |

The intended cold slices did not recover: AUC changed by `-0.003285` for empty history, `-0.003851` for cold users, and `-0.002040` for impressions whose clicked item was new. Larger losses on clicked warm items (`-0.016775` AUC), popularity 20+ (`-0.016911`), and history length 21+ (`-0.006257`) indicate that removing user/fused representation supervision discarded useful signal. Item-only distillation still exceeded the older mean-pooling text-adapt model by `+0.002060` AUC, so candidate-aware pooling remains beneficial; the failed component is the item-only objective. This variant is not promoted to maximum-data training or submission, and the verified full-distillation candidate-attention workflow remained the selected architecture at that stage.

The reverse-position run selected epoch 3 and learned a substantial newest-click signal, but its gains were not stable across history lengths. Relative to direct attention, AUC changed by `-0.004674` for empty history, `-0.003251` for history length 1--4, `+0.002937` for length 5--20, and only `+0.000384` for length 21+. Its sampled-pair validation AUC also fell from `0.674491` to `0.672318`. Because the small overall impression-AUC gain came with broad top-ranking and short/cold-history regressions, neither the model nor a further recency-logit bias was promoted. The implementation and config were discarded; historical artifacts, if kept, are under `runs/mind_large_temporal_text_adapt_candidate_attention_reverse_position_v1`.

The head/dropout comparison is preserved in `runs/mind_large_temporal_text_adapt_candidate_attention_impression_auc_v1/tuning/candidate_attention_sweep/sweep.json`. The gated and four-interest evaluations remain under `runs/mind_large_temporal_text_adapt_gated_candidate_attention_v1` and `runs/mind_large_temporal_text_adapt_multi_interest_candidate_attention_v1`. No variant was promoted to maximum-data training. Separately, the original one-epoch candidate-attention maximum-data schedule tied text-adapt v1 at `0.6848` Large Test AUC; the selected two-epoch lower-learning-rate schedule superseded it at `0.6869`.

## Rejected Historical Retrieval Experiments

These experiments predate the Large temporal architecture-search protocol and do not have directly comparable Large temporal metrics. They are retained as historical evidence for the current text-only retrieval design.

| Experiment | Change | Outcome and decision |
| --- | --- | --- |
| Full-strength KG in retrieval/item base | Concatenated a strongly weighted KG vector with the text vector used by the teacher and base retrieval index, allowing entity embeddings, one-hop neighbors, and relation messages to affect nearest-neighbor retrieval directly. | Did not generalize reliably; rejected. |
| Weakened KG in retrieval | Retained text-plus-KG retrieval but reduced the KG, neighbor, and relation weights so text remained dominant. | Did not generalize reliably; rejected. |
| Reserved KG candidate slots | Generated candidates from entities and their one-hop neighbors, then reserved a fixed quota for them in the final top-`K` set. | Did not generalize reliably; rejected. |
| KG-aware retrieval score bonus | Kept candidate generation unchanged and added a small bonus when candidate entities matched or neighbored entities in the clicked history. | Did not generalize reliably; rejected. |
| Category/subcategory prefixes in teacher text | Added category and subcategory prefixes to the teacher's article text input. | Improved teacher retrieval validation `Recall@200` in the historical Small demo but hurt downstream student-ranker quality; rejected. Keep `teacher.text.include_category_prefix = false`. |

The selected architecture keeps retrieval text-only and uses KG features only in the downstream ranker.

## Additional Current Metrics

Values below were read from the listed local evaluation artifacts on 2026-07-03. They describe completed runs, not untrained working-tree configuration changes.

| Label | Eval split | Impressions | Teacher best epoch | Ranker best epoch | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small dev split | test | 14,631 | 2 | 5 | 0.6619 | 0.3589 | 0.3463 | 0.4050 |
| Large dev only | val | 376,471 | 2 | 1 | 0.6686 | 0.3313 | 0.3659 | 0.4252 |
| Large temporal hard-negative v3 | val | 807,988 | 4 (reused) | 1 | 0.6498 | 0.3082 | 0.3344 | 0.3969 |
| Small temporal | val | 103,422 | 1 | 3 | 0.6297 | 0.2995 | 0.3263 | 0.3867 |

The hard-negative v3 run applied 1 teacher-hard plus 3 random negatives to all cold users, including zero-history groups.

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
| Original MPNet candidate attention, selected schedule | Phase 1 update 9,000 + 2,000 Phase 3 updates; adapted `all-mpnet-base-v2` | `MINDlarge_train + MINDlarge_dev` | 4 | 2 | 0.02 | 0.6948 | 2,370,727 | `runs/mind_large_submission_mpnet_candidate_attention_low_lr_2ep_recency_alpha_002_v1/submission/prediction.zip` |
| MPNet + selected candidate-attention MiniLM weighted-Borda ensemble | 75% MPNet ranks + 25% MiniLM ranks; MPNet breaks exact fusion ties | Same two maximum-data members above | 4 | 2 | Included in both members | 0.6960 | 2,370,727 | `runs/mind_large_ensemble_mpnet_minilm_rank_v1/submission/prediction.zip` |
| Improved MPNet single model (`31ab682`) | MPNet text-encoder lr=1e-5, selected update 9,000, then 2,000 continuation updates | `MINDlarge_train + MINDlarge_dev` | 4 | 2 | 0.02 | 0.6960 | 2,370,727 | `runs/mind_large_submission_mpnet_p1_output/submission/prediction.zip` |
| Improved MPNet + selected candidate-attention MiniLM weighted-Borda ensemble (previous champion) | 75% improved MPNet (`31ab682`) ranks + 25% MiniLM ranks; MPNet breaks exact fusion ties | Same improved MPNet and selected MiniLM maximum-data members | 4 | 2 | Included in both members (`0.02`) | **0.6972** | 2,370,727 | `runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/submission/prediction.zip` |
| **Mixed-loss MPNet + MiniLM, coefficient 0.75 (best reported)** | 75% MPNet / 25% MiniLM; BCE + 0.75 * pairwise + distillation per student | `MINDlarge_train + MINDlarge_dev` | 4 (reused per member) | 2 per member | 0.02 per member | **0.7025** | 2,370,727 | `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1/submission/prediction.zip` |
| Asymmetric MPNet 0.75 / MiniLM 0.5 (not promoted; cleaned up) | Reuse final MPNet-0.75 and MiniLM-0.5 ranks at 75/25 | Existing `MINDlarge_train + MINDlarge_dev` fits | 4 (reused per member) | 2 per member (reused) | Already included; not reapplied | 0.7021 | 2,370,727 | Combined ZIP removed at the user's request; verified SHA256 recorded above |
| Mixed-loss MPNet + MiniLM, coefficient 0.5 (previous champion) | 75% MPNet / 25% MiniLM; BCE + 0.5 * pairwise + distillation per student | `MINDlarge_train + MINDlarge_dev` | 4 (reused per member) | 2 per member | 0.02 per member | **0.7011** | 2,370,727 | `runs/mind_large_ensemble_two_mixed_loss_w050_final_75_25_v1/submission/prediction.zip` |
| Mixed-loss MPNet + MiniLM, coefficient 0.25 (previous champion) | 75% MPNet / 25% MiniLM; both students use BCE + 0.25 * pairwise + distillation | `MINDlarge_train + MINDlarge_dev` | 4 (reused per member) | 2 per member | 0.02 per member | **0.6990** | 2,370,727 | `runs/mind_large_ensemble_two_mixed_loss_final_75_25_v1/submission/prediction.zip` |
| Rejected attentive multi-view v1 | Separate title/abstract encoding with a jointly trained attentive gate | `MINDlarge_train + MINDlarge_dev` | 1 | 3 | 0.02 | 0.6746 | 2,370,727 | `runs/mind_large_submission_text_adapt_multiview_low_lr_3ep_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected replay 85/10/5 | Phase 1 update 6,000 + 2,000 replay-mixture updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6800 | 2,370,727 | `runs/mind_large_submission_text_adapt_replay_85_10_5_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected Nov 15 overweight 45/55 | Phase 1 update 6,000 + 2,000 date-weighted updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6796 | 2,370,727 | `runs/mind_large_submission_text_adapt_nov15_weighted_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected 2,500 updates | Phase 1 update 6,000 + 2,500 Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6842 | 2,370,727 | `runs/mind_large_submission_text_adapt_updates_2500_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected teacher-guided negatives | Phase 1 update 6,000 + 2,000 teacher-mined Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6835 | 2,370,727 | `runs/mind_large_submission_text_adapt_teacher_guided_recency_alpha_002_v1/submission/prediction.zip` |
| Rejected Phase 3 `lr=1.5e-5` | Phase 1 update 6,000 + 2,000 lower-learning-rate Phase 3 updates | `MINDlarge_train + MINDlarge_dev` | 4 | 1 | 0.02 | 0.6797 | 2,370,727 | `runs/mind_large_submission_text_adapt_lr_1p5em05_recency_alpha_002_v1/submission/prediction.zip` |

The adapted-text submission first improved Large Test AUC from `0.6724` to `0.6848` (`+0.0124`). Candidate-aware pooling then improved matched temporal validation AUC from `0.664328` to `0.671593`. Its original one-epoch maximum-data schedule remained at `0.6848`, while transferring the selected `lr=1e-4`, `weight_decay=3e-5` schedule for two maximum-data epochs reached **`0.6869`**. Original MPNet then raised Large Test AUC to **`0.6948`**. That historical model gained `+0.0079` over selected candidate-attention MiniLM, `+0.0100` over text-adapt v1, and `+0.0224` over the frozen-MiniLM baseline.

The current best single model is the `lr=1e-5` MPNet from `31ab682`, at `0.6960`; the original MPNet+MiniLM ensemble independently achieved `0.6960`. The previous improved-MPNet+MiniLM ensemble reached **`0.6972`**, gaining `+0.0012` over both of those results. The coefficient-0.25 mixed-loss ensemble reached **`0.6990`** (+0.0018). Increasing both coefficients to 0.5 reached **`0.7011`** (+0.0021); coefficient 0.75 then reached the current best **`0.7025`**, a further **`+0.0014`** gain.

The original rank-only ensemble selected 75% MPNet and 25% MiniLM on the Nov 14 `rerank_tune` view. Mean impression AUC increased from MPNet's `0.693301` to `0.694412` (`+0.001111`). With that weight frozen, Nov 15 AUC increased from `0.683811` to `0.684632` (`+0.000820`). Nov 15 MRR and nDCG were essentially flat but slightly lower, so this is specifically an AUC-targeted leaderboard candidate. The two hidden rank archives agree on the top candidate for only `49.30%` of impressions, providing useful model diversity. The output ZIP and hashes are recorded under `runs/mind_large_ensemble_mpnet_minilm_rank_v1`; its user-reported hidden Large Test AUC is `0.6960`.

The follow-up config `configs/mind_large_ensemble_mpnet_p1_minilm.yaml` uses the improved MPNet from commit `31ab682` (`feature/lr_sweep_2`) and the same MiniLM member. The temporal checkpoint is `mind_large_temporal_mpnet_p1_promoted`; the hidden-test member is `mind_large_submission_mpnet_p1_output`. Search and submission generation are complete. The independent Nov 14 search selected 75% improved MPNet and 25% MiniLM from the 0.00--1.00 grid (step 0.05), then froze that choice before the Nov 15 report:

| Split | Impressions | MiniLM AUC | Improved MPNet AUC | 75/25 ensemble AUC | Ensemble delta vs MPNet |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nov 14 weight selection | 431,517 | 0.675268 | 0.697527 | **0.698596** | +0.001069 |
| Nov 15 frozen-weight report | 376,471 | 0.667200 | 0.684624 | **0.686049** | +0.001425 |

On Nov 15, ensemble MRR was `0.339251` (`-0.000665` vs improved MPNet), nDCG@5 was `0.377157` (`-0.000137`), and nDCG@10 was `0.438932` (`+0.000131`). Existing November splits remain reused diagnostics, not an independent holdout. These are local temporal metrics, not the hidden Large Test score.

The user reported **Large Test AUC `0.6972`** for the completed submission. The generated metadata records 2,370,727 impressions and 75/25 weights; the actual ZIP checksum matches that metadata. The score and artifact identity are preserved in [leaderboard_result.json](../runs/mind_large_ensemble_mpnet_p1_minilm_rank_v1/ensemble/leaderboard_result.json). The submission command reads the frozen search result automatically and verifies its source fingerprint; the YAML's null weight and `frozen: false` are intentional in `search_artifact` mode. No search rerun or MiniLM ZIP regeneration is needed for unchanged inputs. See the [ensembling guide](ensembling.md) for the method, artifacts, and CLI commands.

The new search applies recency `alpha=0.02` before ranking each temporal member and shares exact integer Borda sorting with submission. Historical ensemble validation numbers above did not include recency and should not be compared as matched-protocol results. The current source fingerprints cover actual checkpoints, embeddings, validation data, age artifacts and prediction contents; old search artifacts must be regenerated. The original-model phase runner is retained for historical reproduction, not training the new champion.

The attentive multi-view promotion is rejected. Its encouraging temporal AUC of `0.675783` did not transfer: Large Test AUC was `0.6746`, which is `-0.0123` versus the candidate-attention champion and only `+0.0022` over frozen MiniLM. The submission was structurally valid. Post-result analysis found that its one-epoch maximum-data teacher remained less mature than the champion's four-epoch teacher, while the hidden ranking over-promoted sports and weather, two categories that had already regressed in temporal slices. The retained artifacts are historical negative evidence and must not be treated as the active submission workflow.

The follow-up temporal-only diagnostic used a now-removed experimental config. It trained the title/abstract gate during epoch 1, froze it, then continued the teacher through fixed epoch 4 before validation-selected ranker training.

That redesigned temporal schedule is also rejected. The implementation behaved as configured: its saved gate and attention arrays are tensor-identical to the original epoch-1 gate, the gate was frozen for epochs 2--4, and the fixed teacher checkpoint records epoch 4. Against the matched candidate-attention reference, results were:

| Metric | Candidate-attention reference | Gate-1/teacher-4 v2 | Delta |
| --- | ---: | ---: | ---: |
| Teacher Recall@200 | 0.041199 | 0.034746 | -0.006453 |
| Full-impression AUC | 0.671593 | 0.672267 | +0.000675 |
| MRR | 0.321535 | 0.320110 | -0.001426 |
| nDCG@5 | 0.352780 | 0.351188 | -0.001592 |
| nDCG@10 | 0.414064 | 0.412651 | -0.001413 |

The AUC gain remained below the required `+0.001`, teacher Recall@200 fell by `15.7%`, and all three top-ranking metrics regressed. Although zero-popularity, period-4, and sports slices recovered, the errors shifted to new items (`-0.000868` AUC), popularity 1--4 (`-0.009515`), period 1 (`-0.008519`), and weather (`-0.047077`). This is distribution instability rather than a robust improvement. The pure attentive multi-view branch is closed: do not promote this schedule to maximum-data training and do not proceed to cross-field contrastive adaptation.

Large Test scores were returned by the competition platform; hidden test labels remain unavailable locally. The MiniLM candidate-attention result was reported on 2026-08-16 and the MPNet result on 2026-08-30. Local MPNet artifact validation confirmed 2,370,727 impressions and a valid submission ZIP. The replay, Nov 15 overweight, 2,500-update, teacher-guided, and learning-rate `1.5e-5` rows are retained only as negative experimental evidence.

## Sanity Rules

- For temporal configs, `preprocess_meta.json:n_validation_eval_impressions` must equal `ranker_eval_val.json:n_impressions`.
- The older Small dev split is a historical demo result, not the current architecture-search baseline.
- Use Small temporal for fast architecture experiments.
- Use `configs/mind_large_temporal_baseline.yaml` to reproduce the random-negative baseline and `configs/mind_large_temporal_tune.yaml` to reproduce hard-negative v4.
- Use `configs/mind_large_submission.yaml` only after choosing fixed settings; it trains on `MINDlarge_train + MINDlarge_dev` and writes hidden-test submission ranks.
- Use `configs/mind_large_submission_mpnet_candidate_attention.yaml` and its recency companion to reproduce the original `0.6948` MPNet. The MiniLM candidate-attention configs reproduce the previous `0.6869` fallback.
- Use `scripts/run_pairwise_weight_075_final_fit.py` with `configs/mind_large_two_model_final_fit_w075.yaml` for the best reported `0.7025` two-student ensemble. Keep `configs/mind_large_two_model_final_fit_w050.yaml` and its `0.7011` submission as the previous champion, plus `configs/mind_large_two_model_final_fit.yaml` and its `0.6990` submission as an older champion. Preserve the frozen 75/25 selection and submission identities, and retain `configs/mind_large_ensemble_mpnet_p1_minilm.yaml` and its `0.6972` submission as an older reference.

## Known Metadata Note

`data/processed/MINDlarge_temporal_tune/preprocess_meta.json` may show the older internal mode name `leaderboard_temporal_tune` if it was generated before the config was renamed to `temporal_tune`. The directory name, split counts, and eval file identify the intended Large temporal protocol. Rerunning preprocess with the current config refreshes the mode string.

The two-model search completed at 75/25, gaining +0.000788 AUC on Nov 14
and +0.001076 on Nov 15. The implemented `two_model_final_fit` command refits
both students for two fixed epochs, reuses final teachers, generates hidden-test
predictions and stages a verified 75/25 ZIP for manual upload. Maximum-data
execution completed; the user reported platform Large Test AUC **0.6990** on
2026-10-29, an absolute gain of **+0.0018** over the previous 0.6972 champion.
