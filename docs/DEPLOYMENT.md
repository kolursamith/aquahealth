# Deployment

## Requirements
Python 3.11 (Colab used 3.13 for training; inference works on both), pinned packages in `requirements/base.txt`
(torch 2.14.0, torchvision 0.29.0, numpy 2.4.6, pillow 12.3.0, opencv-python-headless 5.0.0.93) and
`requirements/app.txt` (+ streamlit 1.63.0). CPU, Apple-silicon MPS and CUDA are auto-detected (`src/device.py`).

## Install and run (local)
```bash
git clone https://github.com/kolursamith/aquahealth.git && cd aquahealth
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/app.txt
# Supply the frozen final checkpoint (not in git):
cp /path/to/EXP-004/best_model.pth models/final_model.pth      # SHA-256 3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae
#   or: export AQUAHEALTH_CHECKPOINT=/path/to/best_model.pth
python -m streamlit run app/main.py                              # http://localhost:8501
```
Verify the checkpoint: `shasum -a 256 models/final_model.pth`. Streamlit settings live in `.streamlit/config.toml`
(dark theme, 20 MB upload limit, no usage statistics).

## Checks before a demo
```bash
python scripts/verify_environment.py
pytest -q                                   # 469 tests; skips final-model tests if the checkpoint is absent
python -m ruff check src app scripts && python -m black --check src app scripts tests && python -m mypy src app scripts
```

## Operational notes
- The model loads once per Streamlit process (`st.cache_resource`); the first Grad-CAM call warms the device (~8 s on
  MPS), later calls take tens of milliseconds.
- Session history lives in the browser session only; a reload starts fresh. No database, no authentication, no
  multi-user isolation beyond Streamlit's per-session state — suitable for a local demo or a single-tenant kiosk.
- To serve on a LAN: `python -m streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501`.
- Training/experiments are separate (`scripts/run_experiment.py`, Colab notebook) and are never triggered by the app.

## Reproducing the model
1. Obtain the dataset zip and place/extract it so that `data/original/aquahealth` points at the `New Dataset` folder.
2. The frozen manifest `data/split_manifest.csv` is committed; `scripts/build_split_manifest.py` refuses to overwrite it.
3. `python scripts/run_experiment.py --id EXP-004 --config configs/exp004_p4_sgd_steplr.json --out-root <dir> --batch-size 64 --workers 2 --amp`
   reproduces the selected run (seed 42; GPU recommended; exact numbers depend on hardware/AMP).
4. `python -m src.evaluate --checkpoint <dir>/EXP-004/best_model.pth --split val --out-dir <dir>/eval_val`.
   Do not evaluate `--split test` again for tuning; the official test result is recorded in `results/final_test/`.
