# AquaHealth AI — Project overview

## What it is
A software-only, single-fish image classification system for aquaculture disease screening: one photograph in,
an 8-class prediction (seven diseases + Healthy Fish), the model's confidence, an application-defined risk level,
a farmer-facing recommendation and a Grad-CAM visual explanation out — presented in a Streamlit application.

## Why
Disease is recognised late on fish farms; a photo is available long before a specialist. An honest AI indication
lets a farmer isolate fish, check water and call a professional earlier. The system is a screening aid, not a
diagnosis.

## Scope (delivered)
- Data audit and frozen, leakage-controlled 70/15/15 split of the delivered dataset (3,405 images).
- EfficientNet-B0 transfer learning with a staged schedule; six validation-only experiments (baseline + five
  paper-informed single-factor variants); evidence-based selection of EXP-004.
- One official evaluation of the frozen test split: accuracy 0.9450, Macro-F1 0.9453 (509 images).
- Inference backend (`src/predict.py`) with risk engine, recommendations, image-quality screening and Grad-CAM.
- Streamlit product UI (Home, Analyze, Results, Dashboard, How it works).
- 469 automated tests, lint/format/type checks, full documentation and audit trail in `results/`.

Out of scope (by the PRD and by evidence): fish detection on pond images (no bounding boxes exist), water-quality
prediction, LSTM/Transformer hybrids (no sequence data; no paper support), production ensembles, cGAN augmentation.

## Key numbers (all traceable)
| Item | Value | Source |
|---|---|---|
| Source files | 3,503 | `results/data_audit_report.md` |
| Frozen split | 2,384 / 512 / 509 (train / val / test) | `data/split_manifest.csv` (sha256 b7d1fccb…) |
| Baseline EXP-001 val Macro-F1 | 0.9004 | `results/experiments/EXP-001/` |
| Final EXP-004 val Macro-F1 | 0.9330 | `results/experiments/EXP-004/eval_val/metrics.json` |
| **Official test Macro-F1 / accuracy** | **0.9453 / 0.9450** | `results/final_test/metrics.json` |
| Parameters / latency | 4,017,796 / 10.5 ms per image (T4) | `results/experiments/EXP-004/inference_benchmark.json` |

## Team
Student 1 (core AI), Student 2 (data & evaluation backend), Student 3 (frontend), Student 4 (UX/dashboard);
`src/predict.py` jointly owned by Students 1 and 2.

## Limitations
- **Dataset size and origin**: ~3.4 k images from one public Kaggle set, with many pre-augmented, low-resolution
  (128 px) lesion close-ups; resolution correlates with class (a shortcut risk that was audited but not removed).
- **Domain**: the classes and appearance are those of the source dataset; other species, other diseases, other
  cameras and lighting are outside the model's experience.
- **Split size**: 512 validation / 509 test images — one image is ≈ 0.2 %; differences below ~0.01 Macro-F1 are noise.
- **Single seed** (42) for every run; no repeated-seed variance estimate.
- **Image quality dependence**: dark or blurry inputs drop to LOW confidence; the app warns but still predicts.
- **AI screening only**: no treatment advice, no certification, no substitute for a veterinarian.
- **Grad-CAM**: shows where the last convolutional block activated for the predicted class; it is not a lesion
  detector and can highlight background context.
- **Deployment**: local single-user Streamlit; the checkpoint is distributed outside git; no authentication,
  no persistence beyond the browser session.
