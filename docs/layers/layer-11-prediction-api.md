# Layer 11 — Prediction API

## 1. Objective

A clean, version-aware inference interface — `Predictor(checkpoint).predict(image)`
— that validates input, reuses the checkpoint's own preprocessing and model,
returns a structured result (never raises on bad images), and sits behind
the Streamlit page with a schema-identical mock for checkpoint-less
development.

## 2. Architecture

```
src/predict.py
  load_input(image) → (RGB PIL, warnings)      PIL | numpy uint8 (H,W[,1|3|4]) | bytes | BytesIO | path
       rejects: None, empty, undecodable, non-uint8, bad shape, < 8 px side, unknown type, missing file
       warns:   mode/alpha/grayscale conversion
  Predictor(checkpoint_path, device=None)
       load_checkpoint → build_model (weights + freeze state) → device → eval
       transform = build_eval_transform(checkpoint.preprocess)     ← the checkpoint's, not config's
       model_version   = "<file>@<sha256[:12]> epoch N[/stage]"
       preprocessing_version = "preprocess-<sha256(PreprocessConfig)[:12]>"
       .predict(image, top_k=None) → PredictionResult
            ok:    predicted_class, confidence, risk (src/risk_engine), message, ranked_predictions, warnings
            error: status="error", error="…", versions still populated
  get_predictor(path) (lru_cache) · predict(image, checkpoint_path) → dict   frontend entry point

app/main.py (renamed from app/app.py)        selects real predictor (AQUAHEALTH_CHECKPOINT or config) or mock
app/mock_prediction.py                       builds the same PredictionResult type with a fixed response
app/components/*                             consume the dict; result card shows ranking + warnings
requirements/app.txt                         streamlit==1.63.0 (UI only); dev.txt includes it
```

## 3. Files

**Created**: `requirements/app.txt`, this document.
**Modified**: `src/predict.py` (scaffold replaced), `app/main.py` (renamed + rewritten), `app/mock_prediction.py`, `app/components/{dashboard,result_card,risk_card}.py`, `tests/test_prediction.py` (rewritten, 34 tests), `requirements/dev.txt`, `README.md`, `docs/architecture/*`, `docs/team/student_{3,4}.md`.

## 4. Dependencies

| Package | Version | Reason | Used by |
|---|---|---|---|
| streamlit | 1.63.0 | demo UI | `app/` only |

Declared in `requirements/app.txt`, not `base.txt`: `src/` never imports it.
Pulls 27 transitive packages (pandas, pyarrow, protobuf, …) — the reason it
is kept out of the runtime set.

## 5. Implementation Decisions

**D1 — The checkpoint defines serving.** Model, freeze state, class list and
`PreprocessConfig` all come from the file; `test_prediction_is_deterministic_and_matches_the_eval_transform_path`
and `test_api_matches_the_raw_model_path_on_every_fixture_file` prove the API
output equals the hand-built transform→model→softmax path to 1e-6.

**D2 — Input errors are results, load errors are exceptions.** A caller
feeding a bad image gets `status="error"` with a message and the same schema
(a UI can render it); a caller with no usable model cannot proceed at all,
so `ModelLoadError` is raised at construction (missing file, wrong format
version, undecodable file, unavailable device).

**D3 — Versioning is content-addressed.** `model_version` embeds a SHA-256
prefix of the checkpoint bytes; `preprocessing_version` hashes the
`PreprocessConfig`. Two checkpoints with the same epoch but different
weights, or the same weights with different preprocessing, are
distinguishable from the response alone.

**D4 — Mock shares the type, not just the keys.** `app/mock_prediction.py`
constructs `PredictionResult`, so the contract cannot drift; a parity test
asserts key sets are identical and the four legacy keys survive.

**D5 — `app/app.py` → `app/main.py` (bug B1).** See §9.

**D6 — Fixture claims are plumbing claims.** See §9 B2 for why the
"predict the class of a flat colour patch" test was wrong and what replaced it.

## 6. Tests

`tests/test_prediction.py` — 34 (1 parametrized × 2 devices):

Input — all 7 accepted forms decode to the same RGB image with no warnings; grayscale array / RGBA / single-channel / 4-channel convert with the documented warning; 11 invalid inputs rejected with specific messages

Construction — missing checkpoint; wrong format version; undecodable file; versions/classes/device exposed; preprocessing version changes with config; unavailable device rejected

Results — full contract key set, JSON-serialisable, risk consistent with confidence; ranking complete/sorted/normalised and consistent with top-1; `top_k`; **API ≡ raw model path on every fixture file** (+ sanity floor); deterministic; invalid input → error result with versions intact; small/grayscale input predicts with warnings; model untouched; every device agrees with CPU (1e-4)

Module API — cached `get_predictor`; missing default checkpoint raises

Mock — key parity with the real contract; error contract mirrored

Streamlit — `AppTest` runs `app/main.py` on the mock path (no checkpoint) and on the real path (checkpoint via env), no exceptions, correct captions/expander

## 7. Commands Executed

```bash
.venv/bin/python -m pip install streamlit==1.63.0
.venv/bin/python -m pytest -q tests/test_prediction.py
.venv/bin/python -m pytest -q -rs -p no:cacheprovider
.venv/bin/ruff check . ; .venv/bin/black --check . ; .venv/bin/mypy src app scripts
.venv/bin/python scripts/verify_environment.py
AQUAHEALTH_CHECKPOINT=<stage full/best.pt> .venv/bin/python -m streamlit run app/main.py --server.headless true --server.port 8765   # smoke
curl http://localhost:8765/                                                                                                   # → 200
.venv/bin/python -c "from src.predict import Predictor; ..."                                                                  # smoke on mps
```

## 8. Results

| Check | Result |
|---|---|
| Tests | **401 passed, 0 skipped, 0 warnings** (was 368) |
| Lint / format / types | clean (27 source files; `app/` type-checked) |
| Verifier | streamlit `[OK]`, exit 0 |
| Smoke (server) | headless Streamlit serves HTTP 200 within 1 s, no errors logged |
| Smoke (API, mps) | `status ok`, versions populated, `b""` → `error: empty input` |
| Regression (Layers 0–10) | pass unchanged |

## 9. Bugs Found and Fixed

**B1 — `app/app.py` could not run under Streamlit.** `AppTest`/`streamlit run`
failed with `No module named 'app.components'; 'app' is not a package`:
Streamlit puts the script's directory on `sys.path`, so `import app`
resolved to `app/app.py` itself, shadowing the package. Root cause: a module
named after its own package. Fix: renamed to `app/main.py` and the entry
point inserts the project root into `sys.path` (as `scripts/` already do).
References updated across docs. Regression: both `AppTest` tests.

**B2 — Test design: flat colour patches are out-of-distribution.** A first
test asserted the fitted model labels a flat 120×90 patch of each class
colour. It failed; diagnosis (`diag11.py`, `diag11b.py`) showed (a) the
fixture's class signal is colour *plus* ±5 noise texture at 48×40, and
pretrained features are texture-sensitive, so flat patches of another size
are OOD for a head fitted on 32 images; (b) with CLAHE in the pipeline the
fixture never fully fits within 30 epochs (CLAHE equalises away lightness),
while without CLAHE it fits at epoch 5. Neither is a code defect. The test
was replaced by the correct plumbing claim: API output ≡ raw model path on
every fixture file (1e-6), with an 0.8 accuracy floor against a garbage
checkpoint. Recorded here so nobody re-adds the wrong test.

**Test-side:** an assertion expected no warnings on 48×40 fixture files;
the API correctly warns that they are upscaled to 64. Assertion fixed.

## 10. Known Limitations

1. **Single-image API.** No batch endpoint; the UI needs one image at a time.
   Batch work goes through `src/evaluate.py`.
2. **No HTTP service.** The "API" is a Python interface; wrapping it in
   FastAPI/Flask is a thin layer if ever needed.
3. **File-upload path not exercised by `AppTest`** (Streamlit's harness does
   not simulate `file_uploader`); the upload component's decode logic is the
   same `PIL.Image.open(...).convert("RGB")` the API validates.
4. **Risk thresholds are the brief's** (0.5 / 0.8) via `src/config.py`;
   their meaning depends on calibration of the eventual real model (Layer 10
   reports ECE for exactly this reason).
5. **`predict()` convenience caches up to 4 predictors by path**; a rewritten
   checkpoint at the same path is not reloaded within a process.

## 11. Acceptance Gate

**PASS** — implementation, imports, 34 new tests, integration (checkpoint →
API → UI, AppTest on both paths), smoke (server + mps API), static checks,
error handling for every listed failure mode, no duplication of model or
preprocessing logic, B1 fixed at root, no regression, documentation updated.

## 12. Next Layer

Layer 12 (YOLO) is an evidence-gated decision, not a build task: inspect
whether the real images require localisation before adding anything.
