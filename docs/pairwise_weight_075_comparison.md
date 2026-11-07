# Pairwise coefficient 0.75 versus 0.5

This bounded comparison trains both MPNet and MiniLM students with **BCE +
0.75 * impression-normalized pairwise loss + teacher distillation**. It compares
them with the completed coefficient-0.5 temporal students and evaluates exactly
two students at fixed **75% MPNet / 25% MiniLM** ensemble weights. Coefficient
0.75 is a loss multiplier, not the fraction of supervision or gradients.

Only the coefficient and output run names change. Each student retains the
same architecture, teachers, embeddings, mining, grouped sampled-impression
batches, optimizer, seed and eight-epoch training budget. Early stopping and
calibration remain disabled. The full-candidate mean impression AUC selector
determines `best.pt`; the sampled-pair selector remains a diagnostic. Recency
alpha 0.02 is applied once per member before the existing weighted-Borda fusion,
with MPNet rank breaking exact ties. No ensemble weight search is performed.

Run from the repository root:

```powershell
# Optional read-only preflight: no training, inference or output writes.
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_comparison.py --check

# Two new temporal student fits, full evaluations and fixed-weight ensemble.
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_comparison.py
```

The default command reuses the verified 0.5 comparators without retraining
them. Progress bars cover evidence verification, input fingerprints, mining,
training, full-candidate validation, evaluation, encoding and ensemble inference.
Training and evaluation run in separate subprocesses to release GPU memory
between stages. It does not train teachers, refit on maximum labeled data,
generate a submission, or change the **0.7011** Large Test champion.

Optional separate stages:

```powershell
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_comparison.py --stage train --backbone mpnet
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_comparison.py --stage train --backbone minilm
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_comparison.py --stage ensemble
```

The ensemble stage requires both completed students and their evaluations.
Re-running the default command verifies and reuses completed training,
evaluations and ensemble stages. Interrupted training restarts that student's
fit from epoch 1; it is not optimizer-state resumption. Changed inputs,
configurations, checkpoints or damaged completion records are rejected.
Existing unowned output directories are preserved.
Preflight also checks both saved candidate runs and any sealed evaluations
before either student starts training, so a conflicting MiniLM run is detected
before spending time on MPNet (and conversely).

Outputs:

- Students: `runs/mind_large_temporal_mpnet_p1_pairwise_w075_v1/` and
  `runs/mind_large_temporal_minilm_pairwise_w075_v1/`.
- Comparison: `runs/mind_large_pairwise_w075_comparison_v1/`.
- `mpnet_comparison.json` / `minilm_comparison.json`: full validation metrics,
  selected epochs and available slice AUC deltas against coefficient 0.5.
- `ensemble_comparison.json`: November 14 and November 15 metrics and deltas
  against the coefficient-0.5 ensemble at the same 75/25 weights.
- `fixed_weights.json`: weights recorded before inference, with checkpoint-bound
  context; split metric stages have context and result hashes.

Consider a maximum-data final fit only if ensemble AUC improves on **both**
dates. Report MRR/nDCG and slices even when AUC improves. The two dates have
already influenced development, so they remain reused validation diagnostics,
not independent estimates of hidden-test improvement. This is the last planned
coefficient-only comparison for now; no further sweep or automatic promotion is
part of this command.

The earlier 0.5 runner is retained byte-for-byte because its source hash is
recorded in completed experiments. This isolated runner reuses its
coefficient-independent checks and the same scoring implementation. The rejected
1.0 runner was removed during cleanup; its results are consolidated in the
[experiment registry](experiment_registry.md#rejected-pairwise-weight-variants).

The comparison completed: ensemble AUC gained **+0.001133** on November 14
and **+0.000819** on November 15. MPNet gained +0.001002 individual AUC;
MiniLM lost 0.001513. Both ran eight epochs, selecting epochs 3 and 4.
Ensemble MRR/nDCG declined on November 14 and improved on November 15.
The agreed [isolated coefficient-0.75 final fit](pairwise_weight_075_final_fit.md)
is implemented, with two fixed epochs on Train+Dev and a separate 75/25 ZIP.
Its Large Test result is pending; the champion remains **0.7011**.
