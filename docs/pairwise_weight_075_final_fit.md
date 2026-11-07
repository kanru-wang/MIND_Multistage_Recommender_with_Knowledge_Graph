# Coefficient-0.75 final fit and Large Test submission

The completed temporal comparison improved fixed-75/25 ensemble AUC on both
reused development dates:

| Temporal AUC | Coefficient 0.5 | Coefficient 0.75 | Delta |
| --- | ---: | ---: | ---: |
| November 14 | 0.702605 | 0.703738 | +0.001133 |
| November 15 | 0.690140 | 0.690959 | +0.000819 |

Both students completed eight epochs. MPNet selected epoch 3 and gained
0.001002 individual AUC; MiniLM selected epoch 4 and lost 0.001513. Ensemble
MRR/nDCG declined on November 14 and improved on November 15. These are reused
validation diagnostics, not a new Large Test result. The current best reported
Large Test AUC remains **0.7011**.

This workflow refits both students on existing **Train+Dev** data for **two
fixed epochs**, with BCE + 0.75 * impression-normalized pairwise loss + teacher
distillation. It reuses the completed four-epoch final teachers, embeddings,
architecture, optimizer and mining settings from the coefficient-0.5 final
students. Calibration and early stopping remain disabled. The temporal best
epochs are not copied to the larger final-fit dataset.

Run from the repository root:

```powershell
# Optional read-only verification of evidence, final teachers and data.
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_final_fit.py --check

# Train and score both students, then write the separate 75/25 ensemble ZIP.
.\.venv\Scripts\python.exe -u scripts/run_pairwise_weight_075_final_fit.py
```

The default configuration is `configs/mind_large_two_model_final_fit_w075.yaml`;
an explicit `--config` is also supported. The separate script keeps completed
experiments' recorded source hashes intact. The earlier CLI final-fit commands
continue to support their existing 0.25 and 0.5 configurations. This runner
reuses their training checks, stage checksums, ZIP verification and staged
fusion writer without changing those source files.

The ensemble remains exactly **75% MPNet / 25% MiniLM**, using the existing
integer weighted-Borda procedure and MPNet rank to break exact ties. Recency
alpha 0.02 is applied once per student before rank fusion. No new weight search,
teacher training, temporal inference or automatic platform upload occurs.
Progress bars cover evidence checks, fingerprints, negative mining, training,
item encoding, scoring and fusion.

New outputs:

- MPNet: `runs/mind_large_submission_mpnet_p1_pairwise_w075_2ep_v1/`.
- MiniLM: `runs/mind_large_submission_minilm_pairwise_w075_2ep_v1/`.
- Workflow: `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1/workflow/`.
- Upload ZIP: `runs/mind_large_ensemble_two_mixed_loss_w075_final_75_25_v1/submission/prediction.zip`.
- Submission metadata: the same submission directory's `submission_meta.json`,
  recording coefficient 0.75, effective weights, source selection hash, prediction
  content hash and the preserved 0.7011 champion.

Repeat the command after an interruption. Completed training/scoring/fusion
stages are reused only after context and output checksum checks. Incomplete
training restarts that student's two-epoch fit from epoch 1; incomplete scoring
restarts scoring. Each subprocess releases its GPU memory when it exits.
Preflight checks both students' existing configurations and completed stages
before launching work, and rejects changed inputs/code, conflicting output
names and damaged completion records. Run one instance at a time and keep the
inputs and source files unchanged during execution.

After completion, manually upload the new ZIP and report Large Test AUC.
Preserve the 0.7011 submission regardless of this candidate's score. All earlier
models, results and slide-related files remain in place.
