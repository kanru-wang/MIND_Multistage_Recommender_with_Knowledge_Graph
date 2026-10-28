# Matched loss and checkpoint-selector comparison

The current follow-up applies this protocol to MiniLM with `--backbone minilm`,
then tests exactly two mixed-loss students in an ensemble. See
[the two-model workflow](two_model_mixed_loss_ensemble.md). The MPNet results
and original commands below remain unchanged.

Completed on 2026-10-21 (Australia/Sydney). Both arms finished all eight epochs
and full-candidate evaluation on 807,988 impressions / 31,485,663 candidates.

| Objective | Selected epoch | Mean impression AUC |
| --- | ---: | ---: |
| Matched grouped-batch BCE control | 4 | 0.687608 |
| BCE + 0.25 * pairwise loss | 3 | 0.692355 |
| Historical promoted MPNet (different training trajectory) | 4 | 0.691440 |

The mixture improves AUC by 0.004747 over the matched control and 0.000915
over the historical single model. Both selectors choose epoch 4 in the control;
in the mixture, mean impression AUC chooses epoch 3 instead of the sampled
selector's epoch 2, gaining 0.001478 AUC. Large Test performance remains
unmeasured for the mixture; the 0.6972 ensemble is still the confirmed champion.
Higher pairwise weights and replacement of supervised BCE remain future tests.

The previous full-impression selector run reached AUC 0.689509 versus the
promoted baseline's 0.691440. Its training trajectory differed, so that result
does not isolate the checkpoint rule. Keep the existing submission champion.

This experiment uses one fixed eight-epoch trajectory per loss arm. Both arms
reuse the promoted teacher, processed temporal data, hard-negative policy,
student architecture, learning rate, distillation weights and seed. They differ
only in an auxiliary loss weight: control 0, treatment 0.25. Calibration is
disabled for both because this experiment targets ranking.

Both arms use the same shuffled **complete sampled impressions** in each
batch, packed up to 1,024 candidate rows. This grouped sampler is necessary to
form within-impression comparisons. It changes the historical row-shuffling
procedure, so the newly trained BCE control is the matched reference for the
pairwise arm. Impression IDs are retained through negative mining, including
pending hard-negative pools and multiple clicks from one impression. Old
preprocessed pair tables without these IDs require regeneration if used with
grouped batches; these configs regenerate pairs through hard-negative mining.

The treatment objective is:

```text
BCE + 0.25 * mean_over_impressions(
    mean_over_sampled_positive_negative_pairs(softplus(negative_logit - positive_logit))
) + unchanged_teacher_distillation
```

No pairs cross impression boundaries. The training candidate rows are the
existing sampled rows, not full logged candidate sets. In multi-click
impressions, a negative sampled for more than one positive retains that
multiplicity, as in the BCE training data. The pairwise loss is normalized
within each sampled impression before averaging across impressions. The 0.25
weight is a prespecified first experiment, not an established optimum.

At the end of every epoch, the **same model weights** are evaluated using:

- Full-candidate mean impression AUC (primary selector).
- Pooled sampled-validation-pair AUC (historical selector).

Python, NumPy, CPU PyTorch and initialized CUDA random states are restored after
validation, including on failure. Iterating validation loaders therefore cannot
change subsequent training shuffles or dropout draws. This preservation now
defaults to true; explicitly set `checkpoint_selection.preserve_training_rng:
false` only when reproducing the old validation RNG behavior.

Early stopping is disabled so both selectors and arms see the same eight-epoch
budget. Each arm saves:

- `ranker/best_mean_impression_auc.pt`
- `ranker/best_sampled_pair_auc.pt`
- `ranker/best.pt` (copy of the primary mean-impression checkpoint)
- `ranker/epochs.json` and `ranker/train_summary.json`, including both metrics
  at each epoch and cross-metrics at each selector's chosen epoch.

Commands below describe the retained workflow. The completed outputs already
exist; training again requires fresh output names. Run from the repository root:

```powershell
# Read-only preflight.
.\.venv\Scripts\python.exe -u scripts/run_ranker_loss_comparison.py --check

# Original training command (refuses to overwrite the completed outputs).
.\.venv\Scripts\python.exe -u scripts/run_ranker_loss_comparison.py
```

Training, negative mining, item encoding and both validation passes have
progress bars. This is two complete eight-epoch ranker fits with two validation
passes per epoch; it is more expensive than the earlier single-arm experiment.
No encoder/teacher training or hidden-test submission is performed.

Outputs are isolated under:

```text
runs/mind_large_temporal_mpnet_p1_dual_bce_v1/
runs/mind_large_temporal_mpnet_p1_dual_bce_pairwise_w025_v1/
runs/mind_large_temporal_mpnet_p1_loss_comparison_v1/comparison.json
```

The runner refuses to overwrite existing training directories. Run one arm
with `--arm control` or `--arm pairwise`, or evaluate completed arms with
`--evaluate-only`. It does not resume partially completed training. If a fit
is interrupted, preserve its directory and choose a new run_name before
restarting that arm. `--check` never launches a fit.
It checks that arm configs differ only in output name and pairwise weight.
Training saves `ranker/training_config.json`; evaluation-only mode and result
comparison reject configs edited since training, incomplete epoch budgets,
missing selector checkpoints, and inconsistent primary evaluation metrics.
The comparison also reports `selector_mean_auc_gain` within each arm, measuring
mean impression AUC at its own best epoch minus that at the sampled selector's
best epoch. These checks compare saved settings and counts, not hashes of the
input dataset contents.

To inspect the sampled selector's full evaluation separately:

```powershell
.\.venv\Scripts\python.exe -u -m mindrec.cli evaluate --config configs/mind_large_temporal_mpnet_p1_dual_bce.yaml --checkpoint best_sampled_pair_auc.pt
.\.venv\Scripts\python.exe -u -m mindrec.cli evaluate --config configs/mind_large_temporal_mpnet_p1_dual_pairwise.yaml --checkpoint best_sampled_pair_auc.pt
```

Alternate-checkpoint reports go to `eval/best_sampled_pair_auc/`, preserving
the primary report. They do not reuse the primary checkpoint's temperature.
For the selection comparison, `best_by_metric` already provides mean impression
AUC at both selected epochs without requiring another inference pass.

Compare treatment against the new matched control first, including new-item,
zero-prior-click, cold-user and latest-period slices. A temporal improvement
would justify further testing; it does not establish Large Test improvement.
The combined November 14-15 validation set has been reused for model selection
and is not an independent test set. Repeated-seed uncertainty is not measured
by this single-seed comparison.
