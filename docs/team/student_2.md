# Student 2 — Data / Evaluation Backend

## Owns

- `src/dataset.py` — Dataset/DataLoader
- `src/preprocessing.py` — CLAHE, resize, normalize
- `src/augmentation.py` — training-time augmentation
- `src/evaluate.py` — metrics
- `src/risk_engine.py` — confidence → risk mapping
- `src/predict.py` — jointly with Student 1

## Branch

`student-2`, merged into `develop` via pull request.

## Getting started

```bash
git checkout develop
git checkout student-2
pip install -r requirements-dev.txt
python scripts/audit_dataset.py
```
