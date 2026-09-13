# Architecture

```
                 ┌────────────────────────── training (Colab T4, scripts/run_experiment.py) ──────────────────────────┐
DATASET.zip ─► scripts/build_split_manifest.py ─► data/split_manifest.csv (frozen, sha256) ─► src.manifest.build_split_dataset
                                                                                              │ train (augmented) / val (eval transform)
                                                                                              ▼
                                                    src.finetune.run_schedule (head → partial → full) + src.train.fit
                                                                                              ▼
                                       results/experiments/<ID>/{config.json, metrics.csv, best_model.pth, …}
                 └────────────────────────────────────────────────────────────────────────────────────────────────────┘

                 ┌────────────────────────── serving (local, app/main.py) ──────────────────────────┐
upload bytes ─► src.predict.load_input ─► src.image_quality.assess_quality ─► build_eval_transform(checkpoint.preprocess)
             ─► EfficientNet-B0 (models/final_model.pth, eval + inference_mode) ─► softmax ─► ranked classes
             ─► src.risk_engine.get_risk_level / get_risk_message / get_recommendation ─► PredictionResult
             ─► src.explainability.generate_gradcam (same frozen model) ─► overlay images
             ─► Streamlit components render the dict (no model logic in the UI)
                 └────────────────────────────────────────────────────────────────────────────────┘
```

## Modules
| Module | Responsibility |
|---|---|
| `src/manifest.py` | canonical 8-class mapping, manifest read/write/digest, group-aware stratified split, `ManifestDataset` |
| `src/dataset.py` | Pillow decoding, `Sample`, seeded `build_dataloader` |
| `src/preprocessing.py` | `CLAHE` (OpenCV, LAB-L), `PreprocessConfig`, `build_eval_transform` (single source of truth for inference input) |
| `src/augmentation.py` | train-only `AugmentConfig` (crop, flip, rotation, colour jitter incl. optional saturation) |
| `src/validation.py` | refuses random transforms on eval/test pipelines |
| `src/model.py` | torchvision EfficientNet-B0 + `Dropout(0.2) → Linear(1280, K)` head, block freezing, summaries |
| `src/train.py` | `TrainConfig`, optimizers (AdamW/Adam/SGD), StepLR, AMP, `fit`, checkpoint format v2 (model, optimizer, scheduler, scaler, RNG, class names, preprocessing) |
| `src/finetune.py` | staged unfreezing schedule; each stage starts from the previous best |
| `src/metrics.py`, `src/evaluate.py` | accuracy / macro P-R-F1 / confusion / ECE; CLI `--split val|test`, `test` is held-out |
| `src/predict.py` | `Predictor` (cached checkpoint, canonical transform, error contract), `PredictionResult` |
| `src/risk_engine.py` | thresholds, Healthy wording, per-class recommendations |
| `src/image_quality.py` | brightness / Laplacian-variance flags |
| `src/explainability.py` | Grad-CAM on `features[-1]`, overlays in model-input and original coordinates |
| `scripts/` | data audit + split, smoke test, experiment runner, experiment comparison, environment check |
| `app/` | `main.py` (nav + pages + state), `theme.py` (design system, 3D-style scene), `state.py` (session history), `components/` |

## Contracts
- **Prediction** (`PredictionResult.to_dict()`): `predicted_class, confidence, risk, message` + `healthy, recommendation,
  quality, ranked_predictions, model_version, preprocessing_version, api_version, status, warnings, error, device`.
- **Checkpoint**: carries `class_names` and `preprocess`; the predictor rebuilds the exact validation transform from it.
- **Risk**: `< 0.50 → LOW`, `0.50–0.80 → MODERATE`, `> 0.80 → HIGH`; `healthy = predicted_class == "Healthy Fish"`.

## Stack
Python 3.11 (Colab ran 3.13), PyTorch 2.14.0, torchvision 0.29.0, OpenCV-headless 5.0 (CLAHE), Pillow 12.3, NumPy 2.4,
Matplotlib (plots only), Streamlit 1.63, pytest / ruff / black / mypy. No timm, no Albumentations, no scikit-learn.
