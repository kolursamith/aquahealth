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

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

streamlit run app/app.py
```

## Testing

```bash
pytest
```

## Project Layout

See [docs/architecture/system_architecture.md](docs/architecture/system_architecture.md) for a full breakdown of the pipeline and module ownership.
