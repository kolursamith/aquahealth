# Experiments (all artifact-verified; validation split only)

All six runs: same frozen split, 8 classes, EfficientNet-B0, batch 64, seed 42, AMP, 30 epochs, selection by validation
Macro-F1, Colab Tesla T4. Records: `results/experiments/<ID>/{config.json, metrics.csv, inference_benchmark.json,
experiment_report.md, training_curves.png, confusion_matrix*.png, eval_val/metrics.json}`; comparison:
`results/verified_experiment_comparison.csv`, `results/FINAL_MODEL_COMPARISON.csv`.

| ID | Config | Single factor vs EXP-001 | Paper | Best stage/epoch | Val acc | Macro P / R | **Val Macro-F1** | Weakest class (F1) | ECE | ms/img (T4) | Train time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | `exp001_efficientnet_b0_baseline` | — (CLAHE on, AdamW, 3 stages) | repository defaults | partial / 9 | 0.9004 | 0.9018 / 0.9010 | 0.9004 | Bacterial Red (0.829) | 0.0555 | 9.004 | 25.8 min |
| EXP-002 | `exp002_clahe_off` | CLAHE off | P1 / P4 preprocessing | partial / 10 | 0.9180 | 0.9207 / 0.9185 | 0.9182 | EUS (0.859) | 0.0316 | 8.719 | 14.4 min |
| EXP-003 | `exp003_p1_two_phase_adam` | Adam, head 1e-3 → last 3 blocks 25 ep @ 1e-5, no full stage | P1 | finetune / 25 | 0.8789 | 0.8798 / 0.8798 | 0.8783 | Bacterial Red (0.787) | 0.0862 | 12.099 | 25.1 min |
| **EXP-004** | `exp004_p4_sgd_steplr` | SGD 0.9, lr 0.01, StepLR ×0.1 / 5 ep | P4 optimiser | full / 3 | 0.9336 | 0.9345 / 0.9347 | **0.9330** | EUS (0.850) | 0.0215 | 10.483 | 24.7 min |
| EXP-005 | `exp005_p4_augmentation` | flip 0.5, crop 0.8–1.0, jitter ±0.2 b/c/saturation, no rotation | P4 augmentation | full / 15 | 0.9336 | 0.9342 / 0.9343 | 0.9335 | EUS (0.845) | 0.0218 | 10.941 | 23.6 min |
| EXP-006 | `exp006_p3_adam_3e5` | Adam 3e-5 everywhere | P3 | full / 14 | 0.9023 | 0.9047 / 0.9033 | 0.9014 | Bacterial Red (0.810) | 0.0330 | 9.183 | 25.7 min |

Findings: the P4 optimiser (EXP-004) and P4 augmentation (EXP-005) each add ≈ +0.033 Macro-F1 over the baseline and make
the full-unfreeze stage useful (it hurt EXP-001); CLAHE off (EXP-002) beats CLAHE on with AdamW (+0.018) and trains
faster; P1's lr 1e-5 recipe (EXP-003) is too slow for 30 epochs; P3's Adam 3e-5 (EXP-006) matches the baseline.
Combinations (e.g. P4 optimiser + P4 augmentation ± CLAHE off) were **not run** and remain future work.

Paper techniques: 18 identified in `results/final_paper_technique_matrix.csv`; 5 executed (the rows above); SLCAM
attention, alternative heads/backbones, fusion+SVM, DINOv2, LSTM and YOLO were not executed (no support, no annotations,
or new architecture required) — see `results/paper_reproducibility.md`. **0.904** seen in a Colab log is EXP-004's
full-stage epoch-2 value (acc 0.9043 / F1 0.9035), not a result of EXP-001 (0.9004).

## Execution history
EXP-001 17:07 IST 2026-09-12 (25.8 min). EXP-003's first run was interrupted from the frontend by a Colab VM recycle;
after the environment was rebuilt, EXP-004, EXP-005, EXP-006, EXP-002 and a clean EXP-003 completed (21:13–23:08 IST).
Artifacts were downloaded from Google Drive, every checkpoint was loaded and re-evaluated on the validation split
locally (`src.evaluate --split val`), and the numbers agree with the training logs and the checkpoints' stored
`best_metric` (≤ 1e-3). Full timeline: `results/AquaHealthAI_EXPERIMENT_TIMELINE.md`.
