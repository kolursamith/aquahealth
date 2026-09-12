# AquaHealth AI

AI-Based Smart Aquaculture Disease Detection

## Project

Image-based fish disease classification system for aquaculture support.

## Current MVP

```
Fish Image
    ↓
CLAHE
    ↓
EfficientNet-B0
    ↓
8-Class Disease Classification
    ↓
Confidence
    ↓
Risk Assessment
    ↓
Streamlit UI
```

## Team

| Role | Owns |
|---|---|
| Student 1 — Core AI | `src/model.py`, `src/train.py` |
| Student 2 — Data/Evaluation Backend | `src/dataset.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/evaluate.py`, `src/risk_engine.py` |
| Student 3 — Frontend | `app/components/` |
| Student 4 — UX/Dashboard | `app/components/dashboard.py`, `app/assets/` |

Student 1 and Student 2 jointly own the `src/predict.py` integration interface between the AI backend and the frontend.

See [docs/team/](docs/team/) for individual responsibilities.

## Branches

```
main
develop
student-1
student-2
student-3
student-4
```

`main` and `develop` are protected. Student branches merge into `develop` via pull request; only `develop` → `main` is merged directly by the project maintainer.

## Build Status

The project is built one validated layer at a time. Each layer is only started
once the previous one passes its acceptance gate; see [docs/layers/](docs/layers/).

| Layer | Scope | Status |
|---|---|---|
| 0 | Python environment | PASS |
| 1 | PyTorch | PASS |
| 2 | TorchVision | PASS |
| 3 | Pretrained EfficientNet-B0 | PASS |
| 4 | Disease classifier head | PASS |
| 5 | Dataset / DataLoader | PASS |
| 6 | Preprocessing + CLAHE | PASS |
| 7 | Training loop | PASS |
| 8 | Validation | PASS |
| 9 | Fine-tuning | PASS |
| 10 | Evaluation | PASS |
| 11 | Prediction API | PASS |
| 12 | YOLO integration | DEFERRED — decision needs the real dataset ([record](docs/layers/layer-12-yolo-decision.md)) |

All layers 0–11 are built and gated; Layer 12 is an evidence-gated decision.

## Local Setup

Requires Python 3.11 (pinned in `.python-version`).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements/dev.txt
python scripts/verify_environment.py
```

`verify_environment.py` checks the interpreter against `.python-version` and
every distribution declared in `requirements/`, and exits non-zero if the
environment does not match. It is the same check CI runs.

## Testing

```bash
pytest -rs
```

Tests belonging to layers that are not built yet skip with an explicit reason
rather than failing.

## Running the App

```bash
pip install -r requirements/app.txt        # adds Streamlit on top of base.txt
streamlit run app/main.py
```

With no checkpoint at `models/final_model.pth` (or `AQUAHEALTH_CHECKPOINT`)
the page runs on the mock predictor and says so; with one, it serves the
real model through `src/predict.py`.

## Prediction API

```python
from src.predict import Predictor
result = Predictor("models/final_model.pth").predict("fish.jpg")
result.predicted_class, result.confidence, result.risk, result.ranked_predictions
result.to_dict()   # JSON-serialisable; status "ok" | "error", model/preprocessing versions, warnings
```

## Project Layout

See [docs/architecture/system_architecture.md](docs/architecture/system_architecture.md) for a full breakdown of the pipeline and module ownership.
