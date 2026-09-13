# Frontend (Streamlit)

Run: `python -m streamlit run app/main.py` → http://localhost:8501. Requires `models/final_model.pth`
(or `AQUAHEALTH_CHECKPOINT`); without it the UI shows "Model not available" and never predicts (no mock).

## Design
Premium dark aquatic identity (`app/theme.py`): near-black navy background with layered radial glows and a particle
field; technical glass frames with corner brackets and state-coloured glows; uppercase technical labels and monospace
section numbers; large gradient headline; luminous gradient primary buttons, glass secondary buttons; a procedural
3D-style centerpiece (perspective grid plane, rotating scan rings, layered SVG fish, bubbles, floating nodes) built from
CSS/SVG only — no CDN, no external assets; `prefers-reduced-motion` respected.

## Navigation and pages (`app/main.py`, Streamlit session state, no routing framework)
| Page | Content |
|---|---|
| 01 Home | hero (headline, subheading, ANALYZE FISH / HOW IT WORKS), large aquatic scene, feature strip, supported classes, footer |
| 02 Analyze | AI inspection console: framed drop zone (JPG/JPEG/PNG/WEBP/BMP) → large image + IMAGE INFORMATION (real filename, resolution, format, size) → ANALYZE IMAGE / CHOOSE ANOTHER; scanning state (rings, grid, scan line, pipeline stages — no fake percentages) |
| 03 Results | ORIGINAL / AI EXPLANATION image toggle; AI ANALYSIS panel (large class name, huge confidence, risk badge, backend message); RECOMMENDATION; SCREENING RISK frame with gauge; WHY DID THE MODEL PREDICT THIS? (model input vs Grad-CAM overlay); MODEL CONFIDENCE DISTRIBUTION (8 bars); HOW THIS RESULT WAS PRODUCED; ANALYZE ANOTHER FISH |
| 04 Dashboard | SESSION INTELLIGENCE from `app/state.SessionHistory` only: total, disease, healthy, latest confidence, recent analyses, class frequency, confidence history; empty state with START FIRST ANALYSIS |
| 05 How it works | 9-step glowing timeline (upload → quality → preprocessing → EfficientNet-B0 → classification → confidence → risk → Grad-CAM → recommendation), model card from artifacts (validation + official test), limits, classes |

## Analysis workflow
Upload bytes → preview metadata (`image_facts`) → ANALYZE → `run_analysis`: exactly one `Predictor.predict` call and one
`generate_gradcam` call → result and explanation stored in session state → Results page. Undecodable files go straight
to the backend error contract and render as a clean error card (no stack traces).

## Result presentation rules
- Prediction, confidence, risk, message, recommendation, probabilities and quality flags are rendered verbatim from
  the backend dict; the UI computes nothing (confidence ≥ 99.95 % is shown as ">99.9%" to avoid a rounded 100 %).
- Healthy Fish is shown as **HEALTHY** (green) with the backend's "no disease detected" wording — never HIGH RISK.
- LOW risk renders the **ANALYSIS UNCERTAIN** state with re-shoot guidance.
- Dark/blurry flags render an IMAGE QUALITY warning; the prediction is still shown.
- Grad-CAM failure renders "AI visual explanation unavailable." and disables the toggle; the prediction survives.

## State and reset
`history`, `result`, `explanation`, `image_bytes`, `image_name`, `upload_key`, `page`, `view`. "Analyze another fish" /
"Choose another" clear image, result and explanation and re-key the uploader; the session history is preserved.
A full browser reload starts a new Streamlit session (history empty; the model stays cached in the process).

## Responsiveness and accessibility
Grids collapse 4 → 2 → 1 columns (1000 px / 680 px breakpoints), nav compacts on mobile, no horizontal overflow at
1400 / 820 / 400 px. Risk is conveyed by text and colour; SVGs carry `role="img"`/`aria-label`; the scanning state is an
`aria-live` status; buttons have visible focus.

## Tests
`tests/test_app_components.py` (badge, quality summary, gauge formatting, HTML escaping / no remote assets, image
metadata), `tests/test_app_state.py`, plus the backend tests the UI depends on. Manual QA record:
`results/final_end_to_end_qa.md`.
