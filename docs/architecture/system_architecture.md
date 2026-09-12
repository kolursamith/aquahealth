# System Architecture

## Overview

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

## Module Ownership

| Module | Responsibility | Owner |
|---|---|---|
| `src/config.py` | Single source of truth: classes, image size, seed, paths, model/training params | Shared |
| `src/dataset.py` | Images → labels → Dataset → DataLoader | Student 2 |
| `src/preprocessing.py` | Image → CLAHE → resize → normalize → tensor | Student 2 |
| `src/augmentation.py` | Training-only augmentation | Student 2 |
| `src/model.py` | EfficientNet-B0 → 8-class classifier | Student 1 |
| `src/train.py` | Training loop → best checkpoint | Student 1 |
| `src/evaluate.py` | Accuracy, precision, recall, macro-F1, confusion matrix, inference time | Student 2 |
| `src/predict.py` | `predict(image)` — the AI ↔ frontend integration interface | Student 1 + Student 2 |
| `src/risk_engine.py` | Confidence → risk level | Student 2 |
| `app/main.py` + `app/components/` | Streamlit UI | Student 3 (components) + Student 4 (dashboard/UX) |

## Integration Contract

`predict(image)` returns a dict whose original four keys are unchanged and
which now also carries provenance and status (see `src/predict.py`):

```json
{
  "status": "ok",
  "predicted_class": "Aeromoniasis",
  "confidence": 0.87,
  "risk": "HIGH",
  "message": "High-confidence disease indication.",
  "ranked_predictions": [{"rank": 1, "class_name": "Aeromoniasis", "probability": 0.87}, "..."],
  "model_version": "best.pt@6b1cc32580ad epoch 12/full",
  "preprocessing_version": "preprocess-30991495576a",
  "api_version": "1.0",
  "device": "mps",
  "warnings": [],
  "error": null
}
```

Input problems come back as `status: "error"` with `error` set, never as an
exception; a missing or incompatible checkpoint raises `ModelLoadError` when
the predictor is created.

Any change to this schema must be coordinated across `src/predict.py`,
`app/mock_prediction.py`, and every `app/components/*.py` file that consumes
it, since the frontend depends on it. See
[training_pipeline.md](training_pipeline.md) and
[inference_pipeline.md](inference_pipeline.md) for the two flows that
produce and consume this interface.

While no checkpoint exists, `app/main.py` uses `app/mock_prediction.py`,
which builds the same `PredictionResult` type with a fixed response, so
frontend work is never blocked on training and the schema cannot drift.

## Modularity

Code should stay modular by responsibility, not by student. Ownership above
is a starting-point guide for the hackathon, not a hard boundary — any
student can contribute to any module through a pull request.
