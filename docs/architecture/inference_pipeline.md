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

Until `models/final_model.pth` exists, `app/app.py` calls
`app/mock_prediction.py` instead of `src/predict.py`. Both expose the same
function signature and return schema, so swapping one for the other is a
single import change in `app/app.py`.
