# Final UI/UX integration (Gate 7C redesign) — 2026-09-13 15:05 IST

## 1. Page architecture
Streamlit only. `app/main.py` renders a product navbar and five pages driven by `st.session_state["page"]`:
**01 Home** (cinematic hero + large 3D-style aquatic centerpiece + feature strip + supported classes + footer),
**02 Analyze** (AI inspection console: framed drop zone → large image + IMAGE INFORMATION frame → ANALYZE IMAGE /
CHOOSE ANOTHER; scanning state), **03 Results** (2-column: framed image panel with ORIGINAL / AI EXPLANATION
toggle, AI ANALYSIS panel with very large class name + confidence + risk badge + backend message, RECOMMENDATION,
SCREENING RISK frame with gauge, WHY DID THE MODEL PREDICT THIS? (original vs Grad-CAM overlay), MODEL CONFIDENCE
DISTRIBUTION, HOW THIS RESULT WAS PRODUCED), **04 Dashboard** (SESSION INTELLIGENCE: metric frames, recent-analysis
cards, class frequency, confidence history; empty state with START FIRST ANALYSIS), **05 How it works** (9-step
glowing timeline, model card from artifacts, limits, supported classes). The Streamlit sidebar is hidden.

## 2. Navigation
Custom navbar: brand block + five numbered `st.button`s inside `st.container(key="aq-nav")`, styled as product tabs
(active = primary with cyan underline). Analyze/Results/Dashboard reachable from CTAs as well; the Results page
redirects to Analyze when no result exists. No radio widgets remain in the primary navigation.

## 3. Design system (`app/theme.py`)
Near-black navy background with three layers — background (water gradient + fixed particle field), midground
(perspective grid plane, orb glow, rotating scan rings), foreground (fish, glass frames, controls). Technical
frames (`.aq-frame`) with corner brackets and luminous edges; state glows (healthy green / high coral / moderate
amber / low blue); small uppercase technical labels and monospace section numbers; large gradient headline
(`clamp(2.4rem, 5.2vw, 4.6rem)`); buttons: primary luminous gradient, secondary dark glass with border, tertiary
text-only; hover elevation/glow and focus outlines. `prefers-reduced-motion` disables all animation.

## 4. 3D visual implementation
Procedural, repository-local, no CDN: `scene_html()` = water layer + tilted technical grid + orb + three rotating
rings (`rotateX(66deg)` under `perspective:1100px`) + a 560-px-wide layered SVG fish (gradient body, scale
pattern, belly highlight, translucent fins, eye, gill line, floor shadow) + rising bubbles + floating numbered
nodes + corner brackets; height 520 px on desktop, 420 px tablet, 300 px mobile (nodes and outer ring hidden).
A `small` variant decorates the Analyze console; the scanning state reuses rings, grid and a sweeping scan line.

## 5. Analysis workspace
Drop zone (Streamlit uploader styled as a technical frame, "Upload" control) → after upload: large image (left),
IMAGE INFORMATION (filename, resolution, format, file size — real values), ANALYZE IMAGE (primary), CHOOSE ANOTHER.
Undecodable files are routed to the backend error result (never to `st.image`).

## 6. Result page
All values verbatim from `Predictor.predict(...).to_dict()`: class, confidence (`>99.9%` cap avoids a rounded
100 %), risk badge (HEALTHY for a confident Healthy Fish — never HIGH RISK), message, recommendation, quality flags,
ranked probabilities (bars + two-decimal percentages), model/preprocessing version. Uncertain state
("ANALYSIS UNCERTAIN") appears whenever the backend risk is LOW.

## 7. Grad-CAM
`src/explainability.generate_gradcam` on `model.features[-1]` of the frozen EfficientNet-B0 (unchanged since
Gate 7B). The Results page shows the original-image overlay in the image panel (toggle) and, in the WHY section,
the 224×224 model input next to the overlay with the caption "AI ACTIVATION VISUALIZATION" and the statement:
"The highlighted regions represent areas with stronger model activation for the predicted class … a post-hoc model
explanation and not guaranteed lesion localization." If Grad-CAM fails the page shows "AI visual explanation
unavailable." and the prediction still renders (the toggle button is disabled).

## 8. Dashboard
`app/state.SessionHistory` only: total analyses, disease detected, healthy, latest confidence, recent analyses
(class, file, time, badge, confidence), class frequency for the eight classes, confidence history line (≥ 2
records). No farms, ponds, water quality, IoT or geographic data.

## 9. Responsive design
Checked at 1400 × 900 (desktop), 820 × 1000 (tablet) and 400 × 900 (mobile): grids collapse 4 → 2 → 1, hero and
scene scale, nav buttons compact and stack, no horizontal scrolling.

## 10. Accessibility
Risk conveyed by text badge + colour; SVGs carry `role="img"`/`aria-label`; scanning state uses `role="status"
aria-live="polite"`; buttons have explicit labels and visible focus outlines; light text on dark surfaces, dark text
on saturated badges; reduced-motion respected.

## 11. Backend integration
Unchanged contract: `Predictor` cached with `@st.cache_resource`; one `predict` + one `generate_gradcam` per Analyze
click; class names from `predictor.class_names`; no thresholds, recommendations or preprocessing in the UI; no mock,
no placeholder model; without a checkpoint the UI shows "Model not available".

## 12. Testing
`ruff check .`, `black --check`, `mypy src app scripts`: clean. Full suite (see Gate 7C report) incl.
`tests/test_app_components.py` (badge, quality summary, gauge formatting, HTML escaping / no remote assets, image
metadata), `tests/test_app_state.py`, `tests/test_explainability.py`, `tests/test_final_model.py`,
`tests/test_prediction.py`, `tests/test_risk_engine.py`, `tests/test_image_quality.py`.
Manual QA on the running app with the real checkpoint (train-split images and derived fixtures only): Home,
Analyze (drop zone, preview, information), scanning state, Results (Bacterial Red Disease >99.9 % HIGH; Healthy Fish
99.5 % HEALTHY), ORIGINAL/AI EXPLANATION toggle, Grad-CAM section, probability distribution, risk frame, Dashboard
(real records), How it works, reset (history preserved), tablet and mobile widths.

## 13. Model integrity
`models/final_model.pth` SHA-256 `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae` (unchanged);
architecture, class mapping, preprocessing and risk thresholds untouched. The checkpoint is git-ignored; supply it
by copying the selected EXP-004 `best_model.pth` to `models/final_model.pth` (or set `AQUAHEALTH_CHECKPOINT`).
TEST split: not accessed.

Run: `python -m streamlit run app/main.py`
