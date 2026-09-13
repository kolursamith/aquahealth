# AquaHealth AI

**AI screening support for aquaculture: single-fish image → 8-class disease classification → confidence → risk level → farmer-facing recommendation, with a Grad-CAM visual explanation — served through a Streamlit application.**

> AI screening support only — not a veterinary diagnosis. AquaHealth AI never prescribes treatment and does not replace a fish-health professional.

## Problem statement
Fish disease outbreaks cause large economic losses in aquaculture and are often recognised late. Farmers can photograph a fish long before a specialist can visit. AquaHealth AI turns one photograph into an AI-assisted screening indication so that isolation, water checks and a professional consultation can start earlier.

## Objectives
1. Classify a fish photo into seven disease conditions or *Healthy Fish* with a model small enough to run locally.
2. Report an honest confidence, an application-defined risk level and a practical next step.
3. Show *why* — a Grad-CAM activation map from the same model — without over-claiming.
4. Be scientifically defensible: de-duplicated, leakage-controlled data split; validation-only model selection; one official test run; every number traceable to an artifact.

## Features
- 8-class EfficientNet-B0 classifier (ImageNet-pretrained, fine-tuned) — `src/model.py`, `src/train.py`, `src/finetune.py`
- Canonical preprocessing (CLAHE → Resize 256 → CenterCrop 224 → ImageNet normalisation) rebuilt from the checkpoint at inference — `src/preprocessing.py`, `src/predict.py`
- Risk engine (<50 % LOW / UNCERTAIN · 50–80 % MODERATE · >80 % HIGH; Healthy Fish shown as HEALTHY) and per-class husbandry recommendations — `src/risk_engine.py`
- Image-quality screening (dark / blurry, advisory) — `src/image_quality.py`
- Grad-CAM visual explanation on the frozen model — `src/explainability.py`
- Premium dark aquatic Streamlit UI: Home · Analyze · Results · Dashboard · How it works — `app/`
- 469 automated tests (`tests/`), config-driven experiment runner (`scripts/run_experiment.py`)

## The eight classes (fixed mapping)
| 0 | Bacterial Red Disease | 4 | Saprolegniasis |
|---|---|---|---|
| 1 | Aeromoniasis | 5 | Parasitic Disease |
| 2 | Bacterial Gill Disease | 6 | White Tail Disease |
| 3 | EUS Disease | 7 | Healthy Fish |

## Architecture
```
Fish image ─► validate/decode ─► quality check ─► CLAHE → Resize 256 → CenterCrop 224 → ImageNet norm
          ─► EfficientNet-B0 (frozen final model) ─► 8 softmax probabilities ─► predicted class + confidence
          ─► risk engine (LOW / MODERATE / HIGH / HEALTHY) ─► recommendation ─► Grad-CAM overlay ─► Streamlit UI
```
Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Dataset
Public Kaggle fish-disease image set (delivered as `DATASET.zip`, 3,503 files). Audit found 0 corrupt files, 68 exact-duplicate groups, 334 near-duplicate groups (59 straddling the vendor's own train/test folders) and 8 label-conflict groups. After de-duplication and exclusion the frozen split has **3,405 images: 2,384 train / 512 validation / 509 test**, near-duplicate groups kept inside one split, manifest SHA-256 `b7d1fccb…d708021d43`. See [docs/DATASET.md](docs/DATASET.md). The images are **not** in this repository.

## Training approach
Staged transfer learning (head → last 3 blocks → all blocks) on a Colab Tesla T4 with AMP, selection by validation Macro-F1; five paper-informed single-factor experiments against the baseline (CLAHE off, P1 two-phase Adam, P4 SGD + StepLR, P4 augmentation, P3 Adam 3e-5). See [docs/MODEL_AND_TRAINING.md](docs/MODEL_AND_TRAINING.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Final model
**EXP-004** — EfficientNet-B0, SGD momentum 0.9 / lr 0.01 / StepLR ×0.1 every 5 epochs, CLAHE on, best at stage `full` epoch 3. Selected on validation only (Macro-F1 0.9330, accuracy 0.9336, ECE 0.0215; tie-break over EXP-005 on weakest-class F1, ECE, latency and fewer diseased→Healthy errors). Checkpoint `models/final_model.pth`, SHA-256 `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae`, 4,017,796 parameters, 10.5 ms/image on a T4.

## Official TEST result (509 held-out images, evaluated once)
| Accuracy | Macro precision | Macro recall | **Macro-F1** | Weighted F1 | ECE |
|---|---|---|---|---|---|
| **0.9450** | 0.9470 | 0.9458 | **0.9453** | 0.9450 | 0.0123 |

Per-class F1: Bacterial Red 0.959 · Aeromoniasis 0.947 · Bacterial Gill 0.954 · EUS 0.922 · Saprolegniasis 0.983 · Parasitic 0.926 · White Tail 0.924 · Healthy Fish 0.947. Artifacts: [results/final_test/](results/final_test/). Validation numbers (model selection) and test numbers (generalisation estimate) are kept separate throughout — see [docs/EVALUATION.md](docs/EVALUATION.md).

## Frontend
Streamlit product UI with a cinematic aquatic design system, custom navigation, an analysis console (drop zone → preview → scanning state), a results page (large prediction, confidence, risk, recommendation, Original / AI-explanation toggle, probability distribution, "why did the model predict this?"), a session dashboard and a How-it-works timeline. [docs/FRONTEND.md](docs/FRONTEND.md).

## Grad-CAM
Post-hoc Grad-CAM on `model.features[-1]` of the frozen EfficientNet-B0: it shows **model activation patterns** for the predicted class — not guaranteed lesion localisation and not a biological finding. [docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md).

## Installation
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/app.txt        # runtime + Streamlit
pip install -r requirements/dev.txt        # + experiments, pytest, ruff, mypy, black (optional)
python scripts/verify_environment.py
```
Place the final checkpoint at `models/final_model.pth` (git-ignored; copy the EXP-004 `best_model.pth`) or set `AQUAHEALTH_CHECKPOINT=/path/to/best_model.pth`.

## Run
```bash
python -m streamlit run app/main.py        # http://localhost:8501
pytest -q                                  # 469 tests (no test-split evaluation)
python -m src.evaluate --checkpoint models/final_model.pth --split val --out-dir results/eval_val   # validation re-evaluation
```
Prediction API:
```python
from src.predict import Predictor
result = Predictor("models/final_model.pth").predict("fish.jpg")
result.predicted_class, result.confidence, result.risk, result.message, result.recommendation
```
Deployment notes: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Repository structure
```
app/            Streamlit UI (main.py, theme.py, state.py, components/)
src/            data · preprocessing · model · training · evaluation · predict · risk_engine · image_quality · explainability
scripts/        build_split_manifest.py · smoke_train.py · run_experiment.py · compare_experiments.py · verify_environment.py
configs/        exp001…exp006 experiment definitions
data/           split_manifest.csv (+ .sha256); images stay outside the repo
models/         final_model.pth (git-ignored) + README
results/        data audit · experiments/<ID>/ records · verified comparison · final_model_selection · final_test/ · audits
docs/           PROJECT_OVERVIEW · ARCHITECTURE · DATASET · MODEL_AND_TRAINING · EXPERIMENTS · EVALUATION · FRONTEND · EXPLAINABILITY · DEPLOYMENT · layers/
tests/          469 tests
notebooks/      Colab training notebook
```

## Limitations
~3.4 k images from one public dataset (one species/region mix, many pre-augmented close-ups); 512/509-image validation/test splits (one image ≈ 0.2 %); single seed; performance depends on image quality (dark/blurry photos fall to LOW confidence); Grad-CAM shows activation, not lesions; local single-user deployment only. Full list: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md#limitations).

## Disclaimer
AquaHealth AI is an AI screening aid. Its output is an indication from a single photograph, not a certified veterinary diagnosis, and it must not be used to decide on treatment. Consult a fish-health professional.

## Team
Student 1 — core AI (`src/model.py`, `src/train.py`) · Student 2 — data/evaluation backend (`src/dataset.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/evaluate.py`, `src/risk_engine.py`) · Student 3 — frontend (`app/components/`) · Student 4 — UX/dashboard. Students 1 + 2 jointly own `src/predict.py`. Branches: `main` (protected), `develop`, `student-1..4`, `feature/real-data-training`.
