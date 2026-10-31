# Pairwise loss weight 0.5 on both backbones

This experiment compares **BCE + 0.5 * pairwise + teacher distillation** with
the completed 0.25 mixture, for both MPNet and MiniLM. Exactly two students
form the candidate ensemble. The current best user-reported Large Test AUC
remains **0.6990**, from the 0.25/0.25 students at 75/25 ensemble weights.

Only the loss coefficient and output run names change. Each new temporal fit
keeps the retained teacher and embeddings, architecture, negative sampling,
grouped sampled-impression batches, optimizer, seed, eight-epoch budget,
distillation, disabled calibration, and both checkpoint selectors. Training
RNG is preserved across validation. `best.pt` is selected by full-candidate
mean impression AUC, not pooled sampled-pair AUC. The existing 0.25 runs are
the comparators and are not retrained. This remains a single-seed experiment.

Run from the repository root:

```powershell
# Read-only check of both configs, retained inputs and completed comparators.
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_comparison.py --check

# Train/evaluate both students, then compare their fixed 75/25 ensemble.
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_comparison.py
```

Progress bars cover input fingerprinting, negative mining, training, both
validation selectors, full evaluation, item encoding and ensemble inference.
Training and full evaluation use separate subprocesses to release GPU memory.
No encoder/teacher is trained, and no maximum-data fit or hidden-test
prediction is launched. No competition upload occurs.

For separate stages:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_comparison.py --stage train --backbone mpnet
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_comparison.py --stage train --backbone minilm
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_comparison.py --stage ensemble
```

The ensemble stage requires both completed candidate fits and evaluations.
Appending `--check` to `--stage ensemble` checks those prerequisites without
inference. The default command performs all stages for both backbones.

Outputs:

```text
runs/mind_large_temporal_mpnet_p1_pairwise_w050_v1/
runs/mind_large_temporal_minilm_pairwise_w050_v1/
runs/mind_large_pairwise_w050_comparison_v1/
  context.json
  mpnet_comparison.json
  minilm_comparison.json
  mpnet_evaluation.json
  minilm_evaluation.json
  fixed_weights.json
  tune_metrics.json
  report_metrics.json
  ensemble_comparison.json
```

Student comparisons report selected epochs, ranking deltas and slice AUC
deltas against the respective 0.25 runs. Review new-item, low-popularity,
cold-user and latest-period slices alongside overall AUC. The two-model
comparison evaluates **75% new MPNet / 25% new MiniLM** on both November 14
and November 15 against the completed 0.25/0.25 ensemble. `fixed_weights.json`
records the prespecified weights before inference. Neither day chooses new
ensemble weights. Both days have influenced checkpoint selection and previous
development, so these are reused validation diagnostics, not independent
holdouts or new Large Test scores. A local gain is not a guaranteed gain over
0.6990. A coefficient of 0.5 is not a 50% share of the total training gradient.

Repeat the same command after an interruption. Completed, matching fits are
reused, completed evaluations are checksummed, and completed split inference
is reused only when context and result hashes match. A partial fit restarts
at epoch 1; a partial evaluation or split restarts that stage. There is no
optimizer-state resume. Source configs, data, teachers and code must remain
unchanged while it runs. Run one instance at a time. Changed inputs or
unowned output directories fail verification: preserve them and choose new
candidate run names and a new script `ROOT` for a changed experiment.

The 0.6990 champion, its two maximum-data students and ZIP, the 0.25 temporal
results, and slide files remain untouched.

## Completed temporal result and 0.5 final-fit candidate

Both eight-epoch fits completed. MPNet selected epoch 3 and reached AUC
0.695935 (+0.003579 versus 0.25); MiniLM selected epoch 4 and reached 0.679036
(+0.002792). At fixed 75/25, ensemble AUC rose from 0.699384 to 0.702605 on
November 14 and from 0.687125 to 0.690140 on November 15. Top-ranking metrics
and several cold/new-item slices regressed. These are AUC-targeted reused
validation results, not a new Large Test score.

Run the maximum-data candidate using the shared final-fit command with a new
config. Optional read-only preflight:

```powershell
.\.venv\Scripts\python.exe -u -m mindrec.cli two_model_final_fit --config configs/mind_large_two_model_final_fit_w050.yaml --check
```

Then fit both students for two fixed epochs, score hidden test and fuse 75/25:

```powershell
.\.venv\Scripts\python.exe -u -m mindrec.cli two_model_final_fit --config configs/mind_large_two_model_final_fit_w050.yaml
```

This reuses the final MPNet and MiniLM teachers and embeddings, uses coefficient
0.5 for both students, retains BCE and distillation, and applies recency
alpha=0.02 once per member. It creates separate output directories; the two
0.25 champion students and their submission are never overwritten. The new
student configs inherit the corresponding 0.25 final configs and change only
run name and pairwise coefficient. Epochs 3 and 4 from temporal selection do
not change the two-epoch Train+Dev schedule.

The new final-fit verifier checks the recorded temporal comparison, its
checksummed inference stages, actual checkpoint bytes, and recorded inputs.
It permits the final-fit module extension made after temporal evaluation;
that module was not used for temporal training or scoring. Other recorded
input and code hashes remain mandatory. The temporal comparison artifacts
are preserved without rewriting their context to fit the new code.

The ZIP to upload manually will be:

```text
runs/mind_large_ensemble_two_mixed_loss_w050_final_75_25_v1/submission/prediction.zip
```

Progress bars cover final input verification, mining, training, encoding,
scoring and fusion. Separate subprocesses release GPU memory between stages.
Rerun the same command after an interruption: checksummed completed stages
are reused, partial fits restart from epoch 1, and partial scoring/fusion
restarts that stage. Run one instance and keep sources unchanged. Altered
inputs or damaged outputs fail verification. No upload integration is present;
compare the competition's returned AUC with the preserved 0.6990 champion.
No large run is launched by --check, and slides remain untouched.
