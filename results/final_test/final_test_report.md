# Official frozen-test report — AquaHealth AI final model (Gate 8)

**OFFICIAL TEST RESULT.** Executed exactly once on 2026-09-13 15:19 IST with
`python -m src.evaluate --checkpoint models/final_model.pth --split test --out-dir results/final_test --batch-size 64 --workers 4`
(the repository's existing evaluator, Apple-silicon MPS, float32). **TEST RUN COUNT = 1.** No model, preprocessing,
threshold or class-mapping change was made after this run.

## Model under test
- Experiment **EXP-004** (P4-informed SGD 0.9 / lr 0.01 / StepLR; CLAHE on), stage `full`, epoch 3
- `models/final_model.pth`, SHA-256 `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae` (verified before and after the run)
- torchvision EfficientNet-B0 (IMAGENET1K_V1) + Dropout(0.2) → Linear(1280, 8); 4,017,796 parameters
- Preprocessing (from checkpoint): CLAHE(2.0, 8) on LAB-L → Resize 256 → CenterCrop 224 → ImageNet normalisation; no random transform
- Frozen split manifest `data/split_manifest.csv` (sha256 b7d1fccb…), **TEST SAMPLE COUNT = 509**

## Validation results (model selection — NOT test)
Macro-F1 0.9330 · accuracy 0.9336 · macro P/R 0.9345 / 0.9347 · ECE 0.0215 (512 validation images; `results/experiments/EXP-004/eval_val/metrics.json`).

## OFFICIAL TEST RESULTS (509 images, never used before this run)

| Metric | Value |
|---|---|
| Accuracy | **0.9450** (481 correct / 28 incorrect / 509 total) |
| Macro precision | 0.9470 |
| Macro recall | 0.9458 |
| **Macro-F1** | **0.9453** |
| Weighted F1 | 0.9450 |
| Cross-entropy loss | 0.1916 |
| ECE (10 bins) | 0.0123 |
| Mean confidence | 0.9495 (correct 0.9678 · incorrect 0.6362) |
| Inference latency | 13.52 ms/image model-only, batch 1, this Mac (MPS); 10.483 ms/image on the Tesla T4 training benchmark |

### Per-class (test)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.9831 | 0.9355 | 0.9587 | 62 |
| Aeromoniasis | 0.9403 | 0.9545 | 0.9474 | 66 |
| Bacterial Gill Disease | 0.9688 | 0.9394 | 0.9538 | 66 |
| EUS Disease | 0.9516 | 0.8939 | 0.9219 | 66 |
| Saprolegniasis | 0.9672 | 1.0000 | 0.9833 | 59 |
| Parasitic Disease | 0.9655 | 0.8889 | 0.9256 | 63 |
| White Tail Disease | 0.8592 | 1.0000 | 0.9242 | 61 |
| Healthy Fish | 0.9403 | 0.9545 | 0.9474 | 66 |

Strongest class: Saprolegniasis (F1 0.9833); weakest: EUS Disease (F1 0.9219).

### Confusion matrix (rows = true, columns = predicted; BRD Bacterial Red, Aero Aeromoniasis, BGD Bacterial Gill, EUS, Sapro Saprolegniasis, Para Parasitic, WTD White Tail)
| true \ pred | BRD | Aero | BGD | EUS | Sapro | Para | WTD | Healthy |
|---|---|---|---|---|---|---|---|---|
| BRD | 58 | 1 | 0 | 1 | 0 | 0 | 2 | 0 |
| Aero | 0 | 63 | 1 | 0 | 0 | 0 | 2 | 0 |
| BGD | 0 | 0 | 62 | 1 | 0 | 1 | 1 | 1 |
| EUS | 1 | 3 | 0 | 59 | 0 | 0 | 1 | 2 |
| Sapro | 0 | 0 | 0 | 0 | 59 | 0 | 0 | 0 |
| Para | 0 | 0 | 1 | 1 | 2 | 56 | 2 | 1 |
| WTD | 0 | 0 | 0 | 0 | 0 | 0 | 61 | 0 |
| Healthy | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 63 |

Diseased images predicted Healthy Fish: **4** of 443; Healthy Fish predicted as a disease: **3** of 66.
Largest confusions (true → predicted, count): EUS Disease → Aeromoniasis (3); Parasitic Disease → White Tail Disease (2); Parasitic Disease → Saprolegniasis (2); Healthy Fish → White Tail Disease (2); EUS Disease → Healthy Fish (2).
Plots: `confusion_matrix.png`, `confusion_matrix_normalized.png`.

## Artifacts (`results/final_test/`)
`metrics.json` (evaluator output: summary, per-class, confusion raw+normalised, confidence bins, checkpoint provenance), `metrics.csv`,
`classification_report.csv`, `classification_report.txt`, `confusion_matrix.csv`, `confusion_matrix_normalized.csv`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `predictions.csv` (one row per test image), `misclassified.csv`, `inference_benchmark.json`, this report.

## Comparison with validation (for context only — no tuning followed)
Test Macro-F1 0.9453 vs validation 0.9330; test accuracy 0.9450 vs validation 0.9336; test ECE 0.0123 vs validation 0.0215.
The test split is the held-out 15 % of the same de-duplicated, group-aware split; the difference is within the sampling
variation of ~510-image splits (one image ≈ 0.2 %). The validation number remains the selection evidence; this test number
is the reported generalisation estimate.

**Post-test tuning: NONE. Model frozen. TEST evaluation: COMPLETE.**
