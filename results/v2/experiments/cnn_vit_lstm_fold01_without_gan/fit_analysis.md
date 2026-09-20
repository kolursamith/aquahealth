# Fit analysis — cnn_vit_lstm · fold 01 · WITHOUT-GAN (dataset configuration v3)

Source: `history.csv` (30 epochs), `metrics.json` (best epoch 29), `run_summary.json`.
This is an observation record. **No hyper-parameter was changed**; any correction goes
through the Phase 13 procedure with a separate `_corrected` experiment id.

## Curves

| quantity | value |
|---|---|
| train loss / Macro-F1 at best epoch (29) | 0.027 / 0.992 |
| validation loss / Macro-F1 at best epoch | 0.447 / 0.896 |
| train − validation Macro-F1 gap at best epoch | **0.095** |
| validation loss | 2.04 (ep 1) → 0.62 (ep 11) → **0.45 (ep 29, minimum)** → 0.48 (ep 30); never trends upward |
| validation Macro-F1 | 0.181 (ep 1) → 0.829 (ep 11) → 0.876 (ep 16) → **0.896 (ep 29)**; last-10-epoch mean 0.883 ± 0.011 (min 0.866, max 0.896) |
| early stopping | not triggered (patience 8 on Macro-F1); best epoch 29 of 30 |
| stages | head-only epochs 1–3 (Macro-F1 ≤ 0.26), full fine-tuning from epoch 4 (cosine schedule) |
| runtime | 1,820 s (30.3 min) on Tesla T4, 60.7 s/epoch |

## Assessment

- **Underfitting: no.** Training Macro-F1 reaches 0.99; validation Macro-F1 0.896 with every class ≥ 0.85 F1.
- **Unstable training: no.** Validation Macro-F1 varies by ±0.011 over the last ten epochs (315 validation images → one image ≈ 0.3 accuracy points); no divergence, no loss spikes.
- **Overfitting: mild generalisation gap, not harmful overfitting.** The training loss is near zero from epoch ~13 while the validation loss stays flat in 0.45–0.65 and keeps *falling* to its minimum at epoch 29 (validation Macro-F1 still rising 0.829 → 0.896 from epoch 11 to 29). The gap (0.095 Macro-F1) therefore reflects capacity relative to 2,729 training images, not a validation deterioration that early stopping would have caught.
- **Convergence: reasonable but possibly incomplete.** The best epoch is the second-to-last and the cosine schedule ended at epoch 30; whether extra epochs or a different schedule would add anything is a question for Phase 13, to be decided across folds/models, not from this single run.

## Per-class (best epoch, validation fold 1, 315 images)

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.967 | 0.763 | 0.853 | 38 |
| Aeromoniasis | 0.900 | 0.900 | 0.900 | 40 |
| Bacterial Gill Disease | 0.949 | 0.902 | 0.925 | 41 |
| EUS Disease | 0.889 | 0.865 | 0.877 | 37 |
| Saprolegniasis | 0.837 | 1.000 | 0.911 | 36 |
| Parasitic Disease | 0.897 | 0.897 | 0.897 | 39 |
| White Tail Disease | 0.889 | 0.865 | 0.877 | 37 |
| Healthy Fish | 0.885 | 0.979 | 0.929 | 47 |

Weakest recall: Bacterial Red Disease (29/38; 4 → EUS Disease, 2 → Saprolegniasis). Confusion matrix: `confusion_matrix.csv`.

**Decision: no correction is applied from this run. Record kept; the remaining matrix runs with the same configuration so the comparison is controlled.**
