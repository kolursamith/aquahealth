# AquaHealth AI — Final project report

Date: 2026-09-13 · Repository `kolursamith/aquahealth` (branch `feature/real-data-training`) · Team: Students 1–4

## 1. Problem and objective
Aquaculture disease is recognised late; a photograph is available early. AquaHealth AI is a software-only screening
aid: one fish image → 8-class classification (7 diseases + Healthy Fish) → confidence → application-defined risk →
farmer-facing recommendation → Grad-CAM visual explanation, served by a Streamlit app. It is not a veterinary diagnosis.

## 2. Data
Public Kaggle fish-disease images (3,503 files). Audit: 0 corrupt, 68 exact-duplicate groups (74 copies dropped),
334 near-duplicate groups (59 leaking across the vendor's train/test folders), 8 label-conflict groups (24 excluded).
Frozen group-aware 70/15/15 split of **3,405 images — 2,384 / 512 / 509** (manifest SHA-256 `b7d1fccb…`). Test split
never used for training, tuning or selection. (`results/data_audit_report.md`, `docs/DATASET.md`)

## 3. Model and training
torchvision EfficientNet-B0 (ImageNet-1K weights) + Dropout(0.2) → Linear(1280, 8); 4,017,796 parameters. Canonical
preprocessing CLAHE → Resize 256 → CenterCrop 224 → ImageNet normalisation (train-only augmentation on top). Staged
fine-tuning head → last 3 blocks → all, AMP, batch 64, seed 42, selection by validation Macro-F1; Colab Tesla T4,
~25 min per 30-epoch run. (`docs/MODEL_AND_TRAINING.md`)

## 4. Experiments (validation results, artifact-verified)
| ID | Technique (paper) | Val Macro-F1 | Val acc | ECE | ms/img T4 |
|---|---|---|---|---|---|
| EXP-001 | baseline, CLAHE on, AdamW | 0.9004 | 0.9004 | 0.0555 | 9.0 |
| EXP-002 | CLAHE off (P1/P4) | 0.9182 | 0.9180 | 0.0316 | 8.7 |
| EXP-003 | two-phase Adam 1e-3 → 1e-5 (P1) | 0.8783 | 0.8789 | 0.0862 | 12.1 |
| **EXP-004** | SGD 0.9 / 0.01 / StepLR (P4) | **0.9330** | 0.9336 | 0.0215 | 10.5 |
| EXP-005 | P4 augmentation | 0.9335 | 0.9336 | 0.0218 | 10.9 |
| EXP-006 | Adam 3e-5 (P3) | 0.9014 | 0.9023 | 0.0330 | 9.2 |

Five research papers were audited (18 techniques; 5 executed above). SLCAM attention, other backbones/heads,
fusion+SVM, DINOv2, LSTM and YOLO were not executed — no data/paper support or new architecture required; YOLO in
particular is infeasible because the dataset has zero bounding boxes. (`docs/EXPERIMENTS.md`,
`results/final_paper_technique_matrix.md`, `results/YOLO12_AUDIT.md`)

## 5. Final model selection
EXP-004 and EXP-005 tie on the primary criterion (same 34 misclassified validation images; Δ F1 0.0005); all
tie-breaks (weakest-class F1, ECE, latency, fewer diseased→Healthy errors) favour **EXP-004**. Frozen checkpoint
`models/final_model.pth`, SHA-256 `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae`.
(`results/final_model_selection.md`)

## 6. Official test result (509 images, evaluated once, 2026-09-13 15:19 IST)
**Accuracy 0.9450 · macro precision 0.9470 · macro recall 0.9458 · Macro-F1 0.9453 · weighted F1 0.9450 · ECE 0.0123**
(481 correct / 28 incorrect). Per-class F1 from 0.922 (EUS) to 0.983 (Saprolegniasis); Healthy Fish 0.947; 4 of 443
diseased images predicted Healthy. No tuning after the test. (`results/final_test/final_test_report.md`)

## 7. Backend, explainability, frontend
`src/predict.py` (cached predictor, canonical transform from the checkpoint, error contract, quality flags, risk and
recommendation), `src/explainability.py` (Grad-CAM on `features[-1]` of the frozen model, overlays in input and original
coordinates), Streamlit UI with five product pages, cinematic aquatic design, real-model integration and honest
wording throughout. (`docs/FRONTEND.md`, `docs/EXPLAINABILITY.md`, `results/final_ui_ux_integration.md`)

## 8. Quality
469 automated tests pass; ruff / black / mypy clean; end-to-end QA with the real model across healthy, disease,
corrupt, dark, blurry, low-confidence, large, grayscale, reset, multi-analysis and refresh cases
(`results/final_end_to_end_qa.md`); release audit `results/final_release_audit.md`.

## 9. Limitations
Small single-source dataset with pre-augmented low-resolution close-ups and class-correlated resolution; ~510-image
validation/test splits (one image ≈ 0.2 %); single seed; strong image-quality dependence; AI screening only, not a
diagnosis; Grad-CAM shows activation, not lesions; local single-user deployment with an out-of-git checkpoint.

## 10. Future work
Combination experiments (P4 optimiser + P4 augmentation ± CLAHE off), multi-seed variance, larger and more diverse data,
SLCAM-style attention as a controlled experiment, on-device/mobile packaging, and a field study with veterinary
ground truth.
