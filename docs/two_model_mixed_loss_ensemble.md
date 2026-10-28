# Two-student mixed-loss ensemble

This is the current requested experiment: exactly two students, MPNet and
MiniLM, each trained with `BCE + 0.25 * pairwise + teacher distillation`.
The unused three-model final-fit implementation has been removed.

The mixed-loss MPNet student already exists. Only MiniLM rankers need training.
Both MiniLM arms reuse `mind_large_temporal_text_adapt_v1` teacher artifacts.
They have identical architecture, distillation, negative mining, grouped
sampled-impression batches, seed and eight-epoch budget. The control uses BCE;
treatment adds impression-normalized pairwise weight 0.25. Both selectors
evaluate each epoch's weights, preserving training RNG across validation.
`best.pt` aliases the full-candidate mean impression AUC checkpoint. Early
stopping and calibration are disabled. No encoder or teacher is trained.

## Commands

Run from the repository root. Optional read-only input check:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_ranker_loss_comparison.py --backbone minilm --check
```

First train and evaluate both MiniLM arms (two eight-epoch student fits):

```powershell
.\.venv\Scripts\python.exe -u scripts/run_ranker_loss_comparison.py --backbone minilm
```

Then search weights for the two mixed-loss students:

```powershell
.\.venv\Scripts\python.exe -u -m mindrec.cli two_model_ensemble_search --config configs/mind_large_ensemble_two_mixed_loss.yaml
```

Progress bars cover negative mining, training, encoding, validation and
ensemble inference. Appending `--check` to the ensemble command verifies
completed checkpoints without inference; before MiniLM training completes it
reports the missing prerequisite. No hidden-test prediction ZIP is required.

The training runner preserves existing directories. Use `--arm control` or
`--arm pairwise` for just the missing arm; `--evaluate-only` evaluates completed
arms without retraining. An interrupted fit requires a fresh `run_name` in its
config to restart; optimizer state is not resumed. Keep both arm configs
identical except run name and pairwise weight. The ensemble references the
treatment config and resolves its current run name. Without `--backbone
minilm`, the runner retains its original MPNet defaults and report paths.

## Selection and outputs

The ensemble contains **only the mixed-loss MPNet and mixed-loss MiniLM**.
MPNet weights are searched from 0% to 100% in steps of 5%, including 75/25
without assuming it remains optimal. Exact integer weighted Borda fusion
prefers MPNet rank on exact ties. Single-model endpoints are safeguards: if
one wins, this experiment has not established a benefit for a two-model blend.

All candidates use the champion's near-tie reference guard and `alpha=0.02`
recency adjustment, applied once per member before ranking. Weight selection
maximizes mean impression AUC on November 14 (431,517 impressions). Exact AUC
ties use nDCG@10, then the larger MPNet weight, as in the previous two-model
protocol. Selection is saved before November 15 scoring (376,471 impressions).
Reporting evaluates only the frozen weight and both single-model endpoints.

The report compares against the **champion's saved two-model temporal
metrics**, with matching data, recency and scoring fingerprints. The champion
is a reference, not another inference member. Inspect MiniLM treatment versus
matched control separately to determine whether the mixed loss helps MiniLM.
Both dates influenced checkpoint selection and earlier development; this is
reused validation, not an independent holdout or a Large Test score.

```text
runs/mind_large_temporal_minilm_dual_bce_v1/
runs/mind_large_temporal_minilm_dual_bce_pairwise_w025_v1/
runs/mind_large_temporal_minilm_loss_comparison_v1/comparison.json
runs/mind_large_ensemble_two_mixed_loss_v1/ensemble/search.json
```

The ensemble directory also saves `context.json`, `tune_metrics.json`,
`selection.json` and `report_metrics.json`. All tuning weights share model
predictions. Completed split metrics can be reused after an interruption;
unfinished splits restart scoring. Changed inputs/code require a new ensemble
run name. Context and result hashes, checkpoint aliases, completed budgets,
split dates and impression counts are checked. Run one instance at a time and
keep inputs unchanged while it runs. Use the dedicated
`two_model_ensemble_search` command with this config.

The temporal commands above do not perform maximum-data fits. The completed
search selected 75/25: AUC gains over the champion were +0.000788 on Nov 14
and +0.001076 on Nov 15. Freeze those weights for the final candidate.

## Maximum-data refit and submission

Optional read-only verification:

```powershell
.\.venv\Scripts\python.exe -u -m mindrec.cli two_model_final_fit --config configs/mind_large_two_model_final_fit.yaml --check
```

Train both new students, score both on hidden test, and fuse at frozen 75/25:

```powershell
.\.venv\Scripts\python.exe -u -m mindrec.cli two_model_final_fit --config configs/mind_large_two_model_final_fit.yaml
```

Both students fit Train+Dev for two fixed epochs using their retained final
teachers and embeddings. No teacher or encoder is trained. Temporal selected
epochs 3 and 4 are not transferred to maximum-data fitting. Both use BCE plus
0.25 times pairwise loss and existing distillation. Recency alpha=0.02 is
applied once during each student's scoring; fusion combines ranks without
applying recency again. Exact fusion ties prefer MPNet.

The output to upload manually to the MIND Large competition is:

```text
runs/mind_large_ensemble_two_mixed_loss_final_75_25_v1/submission/prediction.zip
```

Progress bars cover fingerprinting, mining, training, encoding, scoring and
fusion. The runner uses separate subprocesses to release GPU memory between
stages. Repeat the command after an interruption: verified completed stages
are reused; an incomplete fit restarts from epoch 1, and incomplete scoring
or fusion restarts that stage. Keep inputs and code unchanged, and run only
one instance. Changed inputs or damaged completed outputs fail verification.
Use new output names for a changed experiment. Source models, the 0.6972
champion, and slides are preserved. Predictions are staged and read back for
integrity checks before the final ZIP is published. No large run is launched
by --check.

There is no competition upload integration in this repository. After upload,
record the returned Large Test AUC and the ZIP SHA256 from submission_meta.json.
Only that platform result can establish improvement over the 0.6972 champion.
