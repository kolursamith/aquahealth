# Final backend integration (Gate 6) — 2026-09-13 12:35 IST

## Final checkpoint
- `models/final_model.pth` — EXP-004, byte-identical copy of `drive_export/experiments/EXP-004/best_model.pth`
  (SHA-256 `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae`, 32,524,991 bytes), stage `full`,
  epoch 3, stored best_metric 0.9329857 (validation Macro-F1). Weights untouched. Git-ignored (`models/**/*.pth`).
- Loaded with `src.train.load_checkpoint` → `Checkpoint.build_model()`: `torchvision.models.EfficientNet`
  (EfficientNet-B0, ImageNet-pretrained architecture, `IMAGENET1K_V1` weights at training time), classifier
  `Dropout(0.2) → Linear(1280, 8)`, 4,017,796 parameters, logits shape (1, 8). Verified by
  `tests/test_final_model.py::test_final_checkpoint_metadata_matches_the_project_contract`.

## Class mapping
Checkpoint `class_names` == `src.manifest.CANONICAL_CLASSES` == `src.config.CLASS_NAMES`
(0 Bacterial Red Disease … 7 Healthy Fish). The predictor reads names from the checkpoint; `app/main.py` reads
`predictor.class_names` (and `src.config.CLASS_NAMES` only as labels when no model is loaded). No mapping is
duplicated in the UI.

## Preprocessing (from checkpoint metadata, rebuilt by `build_eval_transform`)
Pillow decode → RGB → `CLAHE(clip 2.0, tile 8)` on the LAB L-channel (OpenCV) → `Resize(256, bicubic)` →
`CenterCrop(224)` → `PILToTensor` → `ToDtype(float32, scale)` → `Normalize(mean (0.485, 0.456, 0.406),
std (0.229, 0.224, 0.225))`. Identical to the validation pipeline used for every reported metric; no random
transform present (`tests/test_final_model.py::test_predictor_uses_the_checkpoint_preprocessing_and_is_in_eval_mode`).

## Production inference path (`src/predict.py`)
`Predictor.predict(image)`: `load_input` (PIL / numpy / bytes / path → validate → RGB; too-small / undecodable →
`status="error"`) → `assess_quality` (dark/blur flags → warnings + `quality` dict) → `self.transform` →
`model.eval()` + `torch.inference_mode()` → logits → softmax → ranked classes → `confidence = top probability`
→ `get_risk_level(confidence)` → `get_risk_message(risk, class)` / `get_recommendation(class, risk)` →
`PredictionResult`. No training code is reachable from this path.

## Prediction contract
Minimum keys `predicted_class, confidence, risk, message` plus `healthy, recommendation, quality,
ranked_predictions (8 entries, sorted, sum 1), model_version, preprocessing_version, api_version, status,
warnings, error, device`. Enforced by `tests/test_prediction.py::test_prediction_result_contract`.

## Risk engine (`src/risk_engine.py`, unchanged)
`< 0.50 → LOW`, `0.50–0.80 → MODERATE`, `> 0.80 → HIGH` (application-defined screening thresholds, not
medically validated). Healthy Fish gets separate wording ("no disease detected") and `healthy=True`; the UI shows
it as HEALTHY. Boundary tests: 0.499/0.50/0.80/0.801 (`tests/test_risk_engine.py`).

## Image quality (`src/image_quality.py`, unchanged)
Brightness (< 0.15 → dark) and Laplacian-variance sharpness (< 25 → blurry) computed on the real input; flags and
human-readable warnings are carried in the result. No scores are invented.

## Placeholder audit (`grep -rni "mock|placeholder|dummy|fake|UNTRAINED|scratchpad|sample prediction" app src`)
Only two docstring sentences in `app/main.py` stating that there is no mock predictor and that the app never fakes
results, and a diagram line in `src/evaluate.py`. `app/mock_prediction.py` is deleted;
`tests/test_prediction.py::test_no_mock_predictor_remains` guards against its return. The earlier UI-smoke
checkpoint lived only in the session scratchpad and is not referenced anywhere in the repository.

## Tests (no TEST-split access)
`pytest tests/test_final_model.py tests/test_prediction.py tests/test_risk_engine.py tests/test_image_quality.py`
→ **60 passed**. Covers: checkpoint load + metadata, eval mode + preprocessing stages, real train-split images of all
8 classes through the full contract, deterministic repeated inference, invalid/corrupt inputs (empty bytes, garbage
bytes, 2×2 array, missing file), large (1800×2400) and grayscale inputs, risk boundaries, healthy handling, output
schema. Low-confidence UI state is covered by the labelled fixture in
`tests/test_prediction.py::test_healthy_fish_prediction_is_flagged_and_worded_distinctly` (monkeypatched
probabilities), not by fabricated production output.

## Real inference verification (frozen-manifest **train** images; the frozen test split was not touched)
Note: the frozen split re-partitioned the vendor folders, so some train-split images sit in the vendor's
`test_split/` directory; membership is defined by `data/split_manifest.csv` (`split == train`).

| Image (manifest split = train) | True label | Predicted | Confidence | Risk | Healthy | Quality |
|---|---|---|---|---|---|---|
| `…/test_split/Healthy Fish_Healthy Fish_183.jpg` | Healthy Fish | Healthy Fish | 0.9944 | HIGH (shown as HEALTHY) | True | ok |
| `…/test_split/Bacterial Red disease_…_1.jpeg` | Bacterial Red Disease | Bacterial Red Disease | 0.9997 | HIGH | False | ok |
| `…/test_split/Bacterial diseases - Aeromoniasis_…_103.jpg` | Aeromoniasis | Aeromoniasis | 0.9977 | HIGH | False | ok |
| `…/test_split/EUS_EUS_133.jpg` | EUS Disease | EUS Disease | 0.9875 | HIGH | False | ok |

model_version `final_model.pth@3045a949ff54 epoch 3/full`, preprocessing `preprocess-f0f42c7fca17`, device `mps`,
status `ok` for all four; messages/recommendations came from `src/risk_engine.py`. (These are training images, so
high confidence is expected; they verify the pipeline, not generalisation.)

## Streamlit connection
`app/main.py::get_predictor()` → `load_predictor(path)` decorated with `@st.cache_resource` → `Predictor(path)`;
path = `AQUAHEALTH_CHECKPOINT` env or `src.config.CHECKPOINT_PATH` (= `models/final_model.pth`). The Analyze button
calls `predictor.predict(bytes).to_dict()` exactly once per click; the UI renders the returned dict. Without a
checkpoint the app shows "Model not available" and never predicts. No training import exists in `app/`.
Model loads once per process (cache), not per interaction.

## Remaining blocker
None for the backend. Not yet done (later gates): official one-shot TEST evaluation, premium UI pass, end-to-end
QA with the running app, final documentation.
