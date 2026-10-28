# Frozen pairwise-model ensemble comparison

The [two-model mixed-loss workflow](two_model_mixed_loss_ensemble.md) is now
the requested next experiment. The three-model diagnostic below is completed,
and its unused final-fit path has been removed.

This is an inference-only experiment. It uses the existing temporal MPNet,
MiniLM, and BCE + 0.25 * pairwise student checkpoints. It trains no model and
does not create or change any Large Test submission.

The baseline is the completed improved-MPNet/MiniLM ensemble associated with
the reported 0.6972 Large Test AUC. Its internal weight is read from its saved,
frozen selection artifact, currently 75% MPNet and 25% MiniLM; it is not retuned.
The extra member is the pairwise student's mean-impression-AUC checkpoint
selected at epoch 3. Its maximum-data counterpart does not yet exist.

For each candidate, all three frozen models use the same recency adjustment
(`alpha=0.02`), candidate order, and near-tie reference-scoring guard as the
existing ensemble. Recency is applied once, before conversion to rank.

The experiment tests these **flat three-member Borda weights**:

| Pairwise share | Existing MPNet | Existing MiniLM | Pairwise student |
| --- | ---: | ---: | ---: |
| 0% | 75% | 25% | 0% |
| 10% | 67.5% | 22.5% | 10% |
| 25% | 56.25% | 18.75% | 25% |
| 50% | 37.5% | 12.5% | 50% |

Costs are weighted sums of the three member ranks, computed with exact integer
arithmetic. Lower cost ranks first. Exact cost ties follow the original
MPNet/MiniLM fused ranking, which itself prefers MPNet on ties. The original
blend is not converted into an additional rank before mixing; its two internal
coefficients retain their fixed ratio. The 0% row exactly reproduces the old
two-member fusion.

The share is selected by mean impression AUC on November 14, 2019
(`rerank_tune`); an exact AUC tie chooses the smaller pairwise share. Selection
is saved to `selection.json` before the November 15 (`rerank_test`) evaluation.
Reporting evaluates only the frozen share and 0% baseline. Both dates have
already influenced development and remain reused diagnostics, not independent
holdouts. MRR and nDCG@5/@10 are reported, but do not select the share.

Run from the repository root:

```powershell
# Read-only verification of inputs and the frozen baseline; no model inference.
.\.venv\Scripts\python.exe -u -m mindrec.cli pairwise_ensemble_search --config configs/mind_large_ensemble_mpnet_p1_minilm_pairwise.yaml --check

# Score the frozen models, select on Nov 14, then report Nov 15.
.\.venv\Scripts\python.exe -u -m mindrec.cli pairwise_ensemble_search --config configs/mind_large_ensemble_mpnet_p1_minilm_pairwise.yaml
```

Progress bars cover item encoding, buffered model inference, and evaluation of
the share grid. Scoring covers the full 431,517 tuning and 376,471 reporting
impressions. All four shares reuse the same model predictions; no model is
rescored separately for each share. The existing hidden-test prediction ZIPs
are not required for this temporal inference experiment.

Outputs are isolated under:

```text
runs/mind_large_ensemble_mpnet_p1_minilm_pairwise_share_v1/ensemble/
  context.json
  rerank_tune_ranks.npz
  rerank_tune_cache.json
  rerank_test_ranks.npz
  rerank_test_cache.json
  selection.json
  search.json
```

`search.json` records tuning results, selected share, effective weights,
baseline and selected reporting metrics, reporting deltas and whether AUC
improved on both days. These local metrics are not Large Test AUC.

The two completed split caches together use approximately 420 MB, storing
three int32 rank arrays, click labels and impression offsets. Repeating the
same command reuses verified completed caches and requires no model inference
when both are present. An interrupted split restarts its scoring; a completed
split is reused. Reruns still recompute the inexpensive share metrics.

Cache verification hashes checkpoints, embeddings, validation data, age
artifacts, the original frozen selection, and inference implementations.
Changed sources require a new output run_name. Do not edit model/data files or
run the experiment concurrently while it is scoring. Partial cache writes do
not count as completed caches. The source ensemble's selection and submissions
are never overwritten. This config does not inherit writable submission or
selection paths; the old two-member ensemble commands reject it. Each day's
0% metrics must reproduce the saved baseline
within 1e-6 before the comparison is accepted.

If tuning selects 0%, the experiment found no added value on the tuning day.
If a nonzero share improves both days, the next candidate step is a separate
maximum-data pairwise-student refit using the retained final teacher/embeddings,
initially keeping the champion's two-epoch final-fit schedule. A subsequent
submission is needed to establish any improvement over Large Test AUC 0.6972.

## Completed result and maximum-data candidate

The completed search selected 50% pairwise share on November 14. The existing
ensemble's AUC increased from 0.698595877 to 0.701470709 (+0.002874832). The
frozen blend then improved November 15 AUC from 0.686048946 to 0.687562172
(+0.001513226); MRR and both nDCG metrics improved on that reporting day too.
The effective weights are **37.5% MPNet / 12.5% MiniLM / 50% pairwise student**.
The 0% results exactly reproduce the previous ensemble. These are reused
temporal validation results, not a new Large Test score.

The unused three-model final-fit code and command were removed. Use the
[two-model final-fit workflow](two_model_mixed_loss_ensemble.md) instead.
