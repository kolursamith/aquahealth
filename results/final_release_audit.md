# Final release audit (Gate 10) — 2026-09-13 15:40 IST

Repository `kolursamith/aquahealth`, branch `feature/real-data-training`, HEAD `598273c` + uncommitted release work
(49 changed paths). No model change, no TEST rerun in this gate.

## Architecture (as shipped)
`src/` — data (`manifest.py`, `dataset.py`), preprocessing (`preprocessing.py`, `augmentation.py`, `validation.py`),
model (`model.py`), training (`train.py`, `finetune.py`), evaluation (`evaluate.py`, `metrics.py`), serving
(`predict.py`, `risk_engine.py`, `image_quality.py`, `explainability.py`). `scripts/` — audit/split, smoke test,
experiment runner, comparison. `app/` — Streamlit presentation layer only (`main.py`, `theme.py`, `state.py`,
`components/`). `configs/` — six experiment definitions. `results/` — evidence. `models/` — final checkpoint
(git-ignored). `tests/` — 469 tests.

## Final model
EXP-004 · EfficientNet-B0 (torchvision IMAGENET1K_V1) + Dropout(0.2) → Linear(1280, 8) · `models/final_model.pth` ·
SHA-256 `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae` · selected on validation
(`results/final_model_selection.{md,json}`) · **official test** (`results/final_test/`, one run): accuracy 0.9450,
macro-F1 0.9453, macro P/R 0.9470/0.9458, weighted F1 0.9450, ECE 0.0123 on 509 images.

## Audit checklist
| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | final checkpoint exists | PASS | `models/final_model.pth`, 32,524,991 bytes |
| 2 | SHA matches | PASS | `shasum -a 256` = 3045a949…c0dae (verified after test and after UI work) |
| 3 | EXP-004 is the selected experiment | PASS | `results/final_model_selection.json` `selected_experiment` |
| 4 | 8 classes | PASS | checkpoint `class_names` == `CANONICAL_CLASSES` (test `test_final_model.py`) |
| 5 | Predictor is the production path | PASS | `app/main.py::load_predictor` → `src.predict.Predictor` |
| 6 | `mock_prediction.py` absent | PASS | deleted; `test_no_mock_predictor_remains` |
| 7 | no placeholder model | PASS | grep UNTRAINED/placeholder/dummy over app/src: only "never shows placeholder results" copy |
| 8 | no hard-coded predictions | PASS | UI renders `PredictionResult` dict only |
| 9–12 | no fake dashboard / IoT / water-quality / analytics | PASS | dashboard reads `SessionHistory`; UI states "not connected to ponds, sensors…" |
| 13 | no TEST images committed | PASS | `git ls-files` contains no image files except 3 EXP-001 plot PNGs; `results/final_test/predictions.csv` lists paths only |
| 14 | dataset images not committed | PASS | `data/original/aquahealth` is a symlink; only the manifest is tracked |
| 15 | model weights not committed | PASS | `models/**/*.pth`, `results/**/*.pth`, `drive_export/` ignored |
| 16 | secrets/API keys absent | PASS | grep token/secret/credential/key over tracked files: none |
| 17 | `.env` ignored | PASS | `.gitignore` lines `.env`, `.env.*` |
| 18 | temporary files removed | PASS | scratchpad outside repo; `.claude/launch.json` holds only the production launcher |
| 19 | debugging code removed | PASS | no `print`/`breakpoint`/`pdb` in `app/`; scripts print reports by design |
| 20 | Streamlit starts | PASS | health endpoint `ok`; manual QA (`results/final_end_to_end_qa.md`) |
| 21 | requirements documented | PASS | `requirements/{base,app,experiments,dev}.txt` (pinned); stale `app/app.py` comment fixed |
| 22 | README accurate | WARNING → fixed in Gate 11 | README rewritten (mock reference removed, final results added) |
| 23 | run instructions accurate | PASS | `python -m streamlit run app/main.py`; checkpoint placement documented |
| 24 | results artifacts organised | PASS | `results/experiments/<ID>/`, `results/final_test/`, audit/selection/QA reports at `results/` root |

## Code quality
`pytest -q`: 469 passed · `ruff check src app scripts`: pass · `black --check src app scripts tests`: pass ·
`mypy src app scripts`: pass.

## Git
`git ls-files`: 123 tracked files; nothing tracked that should not be. Untracked-but-intended: app/src/tests sources,
`.streamlit/config.toml`, `.claude/launch.json`, `results/**` evidence (≈3 MB incl. 17 small PNG plots and the
official test artifacts), `docs/*.md`. Excluded by `.gitignore`: `models/final_model.pth`, `drive_export/`, per-epoch
checkpoints, dataset. Nothing is committed automatically — `git add -A && git commit` is the operator's call.

## Frontend / backend / explainability / testing
See `results/final_ui_ux_integration.md`, `results/final_backend_integration.md`, `src/explainability.py`
(Grad-CAM on `features[-1]`), `results/final_end_to_end_qa.md`.

## Known limitations
Single public dataset (~3.4 k images, one region/species mix); 512/509-image validation/test splits (one image ≈ 0.2 %);
single seed; image-quality dependence (dark/blurry inputs drop to LOW confidence); AI screening aid, not a veterinary
diagnosis; Grad-CAM shows activation, not lesion localisation; local Streamlit deployment only (no auth, no
multi-user persistence); checkpoint distributed outside git.

## Deployment
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/app.txt          # torch, torchvision, opencv-headless, pillow, numpy, streamlit
cp <EXP-004 best_model.pth> models/final_model.pth   # or: export AQUAHEALTH_CHECKPOINT=/path/to/best_model.pth
python -m streamlit run app/main.py          # http://localhost:8501
```

## GitHub readiness
Ready to commit and push (`feature/real-data-training` → PR to `main`): sources, tests, configs, docs and evidence
are in the tree; weights and data stay out; CI-relevant checks are green.

## RELEASE READY = YES
