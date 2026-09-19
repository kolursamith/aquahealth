# Final end-to-end product QA (Gate 9) — 2026-09-13 15:30 IST

Application: `python -m streamlit run app/main.py` (fresh process, `models/final_model.pth`, SHA-256
`3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae`, predictor `final_model.pth@3045a949ff54
epoch 3/full`, cached once, device mps). All images below are frozen-manifest **train** images or fixtures derived
from them (never the test split). Uploads were injected into the real Streamlit uploader; every value quoted was read
from the rendered page. Statuses: PASS / FAIL / WARNING / NOT_APPLICABLE.

## User journey
| Step | Status | Evidence |
|---|---|---|
| HOME (hero, 3D centerpiece, feature strip, classes, footer) | PASS | rendered at 1400/820/400 px |
| ANALYZE (console, drop zone) | PASS | "02 ANALYZE" nav → drop zone + tips |
| UPLOAD → PREVIEW (image + IMAGE INFORMATION) | PASS | e.g. `FIXTURE_large_saprolegniasis.jpg · 2000 × 2000 px · JPEG · 256 KB` |
| ANALYZE (scanning state, one backend call) | PASS | "AI VISUAL ANALYSIS / Scanning image…" then Results page |
| REAL INFERENCE → RESULT | PASS | values from `Predictor.predict` (see cases) |
| RISK | PASS | badge = backend risk; Healthy Fish → HEALTHY |
| RECOMMENDATION | PASS | backend text per class / LOW guidance |
| PROBABILITIES | PASS | 8 backend probabilities, sum 100 % |
| GRAD-CAM (toggle + WHY section) | PASS | overlay from `src/explainability.py`; disabled toggle when unavailable |
| WHY THIS RESULT | PASS | "HOW THIS RESULT WAS PRODUCED" frame |
| DASHBOARD | PASS | real session records only |
| RESET / NEW ANALYSIS | PASS | image/result/explanation cleared, history kept |

## Test cases
| # | Case | Input (train-derived) | Observed | Status |
|---|---|---|---|---|
| 1 | Healthy image | `healthy_train_183_320px.jpg` | Healthy Fish 99.5 % → **HEALTHY** (green), healthy message + monitoring recommendation, Grad-CAM shown | PASS |
| 2 | Disease image | `bacterial_red_train_1.jpeg` | Bacterial Red Disease >99.9 % → HIGH RISK, disease recommendation, Grad-CAM shown | PASS |
| 3 | Invalid image | `FIXTURE_corrupt.jpg` (text bytes) | Results error card: "could not decode image: the file is not a readable image (unsupported or corrupt)", CHOOSE ANOTHER IMAGE; no stack trace | PASS |
| 4 | Corrupt image | same fixture (undecodable) | as above | PASS |
| 5 | Dark image | `FIXTURE_dark_aeromoniasis.jpg` (brightness ×0.08) | "IMAGE MAY BE TOO DARK AND BLURRY" warning; prediction still computed (Saprolegniasis 42.9 %, LOW) | PASS |
| 6 | Blurry image | `FIXTURE_blurry_aeromoniasis.jpg` (Gaussian r=10) | "IMAGE MAY BE BLURRY"; Aeromoniasis 38.8 %, LOW | PASS |
| 7 | Low-confidence image | dark/blurry/noise fixtures | "ANALYSIS UNCERTAIN" state, LOW badge, re-shoot guidance (42.9 %, 38.8 %, 46.3 %) | PASS |
| 8 | Large image | `FIXTURE_large_saprolegniasis.jpg` 2000 × 2000 | Saprolegniasis 99.9 % HIGH; preview and analysis fine | PASS |
| 9 | Grayscale image | `FIXTURE_grayscale_whitetail.png` (mode L) | White Tail Disease 96.6 % HIGH; note "image mode L converted to RGB" | PASS |
| 10 | Reset | ANALYZE ANOTHER FISH / CHOOSE ANOTHER | returns to drop zone; session count preserved | PASS |
| 11 | Multiple analyses | 4 consecutive analyses | dashboard: total 4, disease 4, healthy 0, latest 38.8 %, recent cards, class frequency | PASS |
| 12 | Browser refresh | reload http://localhost:8501 | new Streamlit session: home page, "ANALYSES THIS SESSION: 0", model still cached/loaded | PASS (expected reset of session state) |

## Verification
Prediction / risk / recommendation / probabilities come from `src.predict.Predictor` (single call per Analyze) — PASS.
Grad-CAM from real model activations (`src/explainability.py`, `features[-1]`) — PASS. Dashboard from
`app/state.SessionHistory` — PASS. No fake data, no mock (`app/mock_prediction.py` absent, guarded by test) — PASS.

## UI (desktop 1400 px, tablet 820 px, mobile 400 px)
No horizontal overflow, no clipped text, buttons readable and reachable, cards/frames intact, images scale — PASS.
Mobile: nav buttons stack compactly; outer scan ring and floating nodes hidden by design.

## Accessibility
Risk conveyed by text + colour; SVG `role="img"`/`aria-label`; scanning `role="status" aria-live="polite"`;
focus outlines; reduced-motion support — PASS (automated screen-reader testing NOT_APPLICABLE in this environment).

## Code checks
`pytest -q` 469 passed / 0 failed (Gate 7C run; unchanged code since) · `ruff check src app scripts` clean ·
`black --check src app scripts tests` clean · `mypy src app scripts` clean.

TEST rerun: NO. Model modified: NO (SHA unchanged).
