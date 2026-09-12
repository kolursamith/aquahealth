# Inference Pipeline

```
Uploaded image (app/components/upload.py)
    ↓
src/predict.py: predict(image)
    ↓
src/preprocessing.py  (CLAHE, resize, normalize)
    ↓
models/final_model.pth (loaded EfficientNet-B0)
    ↓
softmax → predicted_class, confidence
    ↓
src/risk_engine.py: get_risk_level(confidence)
    ↓
{predicted_class, confidence, risk, message}
    ↓
app/components/result_card.py, app/components/risk_card.py
```

## Development mode

`app/main.py` selects the predictor at start-up: the real one when a
checkpoint exists at `AQUAHEALTH_CHECKPOINT` or `models/final_model.pth`,
otherwise `app/mock_prediction.py`, which returns a fixed response of the
identical schema. The page shows which one is active.
