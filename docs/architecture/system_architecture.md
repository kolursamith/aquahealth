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
| `app/app.py` + `app/components/` | Streamlit UI | Student 3 (components) + Student 4 (dashboard/UX) |

## Integration Contract

`predict(image)` returns:

```json
{
  "predicted_class": "Aeromoniasis",
  "confidence": 0.87,
  "risk": "HIGH",
  "message": "High-confidence disease indication."
}
```

Any change to this schema must be coordinated across `src/predict.py`,
`app/mock_prediction.py`, and every `app/components/*.py` file that consumes
it, since the frontend depends on it. See
[training_pipeline.md](training_pipeline.md) and
[inference_pipeline.md](inference_pipeline.md) for the two flows that
produce and consume this interface.

While the model is untrained, `app/app.py` uses
`app/mock_prediction.py`, which returns a fixed response matching this
schema, so frontend work is never blocked on training.

## Modularity

Code should stay modular by responsibility, not by student. Ownership above
is a starting-point guide for the hackathon, not a hard boundary — any
student can contribute to any module through a pull request.
