# Layer 10 — Evaluation

## 1. Objective

Evaluate a *saved checkpoint* on an image folder, independently of training,
with deterministic inference, and export everything a reviewer needs:
metrics, per-sample predictions, misclassifications, confusion matrices
(raw and normalised), a classification report, and a confidence/calibration
analysis. Built entirely on Layer 8's validation pass and metrics — no
second implementation of either.

## 2. Architecture

```
src/evaluate.py
  evaluate_checkpoint(ckpt, data_root, device, batch_size, workers, bins)
      load_checkpoint → build_model (weights + freeze state) → eval
      ImageFolderDataset(data_root, eval transform FROM THE CHECKPOINT, class list FROM THE CHECKPOINT)
      unshuffled loader → evaluate_model
  evaluate_model(model, dataset, loader, device, checkpoint)
      guards: SequentialSampler; dataset classes == checkpoint classes; sample count
      run_validation (Layer 8) → EvaluationReport
  confidence_analysis(probabilities, targets, bins)
      mean top-1 confidence (all / correct / incorrect), equal-width reliability bins, ECE
  export_report(report, out_dir) → metrics.json, predictions.csv, misclassified.csv,
      confusion_matrix.csv, confusion_matrix_normalized.csv, classification_report.txt
  python -m src.evaluate --checkpoint … --data-root … --out-dir …
```

## 3. Files

**Created**: `tests/test_evaluate.py` (19 tests), this document.
**Modified**: `src/evaluate.py` (scaffold replaced), `results/README.md` (actual outputs), `README.md`.

## 4. Dependencies

None new. Plots are deliberately not produced (no matplotlib dependency);
the CSV/JSON exports are plot-ready.

## 5. Implementation Decisions

**D1 — The checkpoint is the source of truth.** Preprocessing config and
class list come from the checkpoint, never from `src/config.py` or the
folder being evaluated; a folder whose classes differ is rejected. This is
what makes an evaluation reproducible from the file alone.

**D2 — Sample alignment is guarded.** Per-sample rows are built by zipping
the dataset's sorted `samples` with the validation outputs, which is only
valid for an unshuffled loader; `evaluate_model` refuses any other sampler
and checks the counts agree.

**D3 — Confidence analysis is standard and hand-verified.** Top-1
probability as confidence; equal-width bins `(lower, upper]` with 0.0
folded into the first bin; `ECE = Σ n_b/N · |acc_b − conf_b|`. A 4-sample
case is checked to the arithmetic.

**D4 — Test split is one-shot.** Evaluating a directory named `test` logs a
warning stating the policy; nothing prevents it (it must be possible once)
but nothing about the pipeline uses it for selection — model selection is
`fit`'s `best.pt` on the validation loader.

**D5 — Evaluation metrics equal validation metrics** for the same model and
data (`test_metrics_agree_with_the_validation_pass_on_the_same_model`), so
there is one number, not two.

## 6. Tests

`tests/test_evaluate.py` — 19 (1 parametrized × 2 devices):

Confidence — hand-computed means/bins/ECE; all-correct / all-wrong edge cases (`None` means, ECE 0.775); bins cover 0 and 1; bad bin counts and shape mismatches rejected

Evaluation — model and preprocessing (incl. CLAHE config) rebuilt from the file; stage/epoch/classes recorded; per-sample path/target/predicted/confidence/correct/probabilities all consistent with the files on disk; misclassified ≡ incorrect ≡ N − trace(confusion); deterministic across runs and batch sizes; agrees with `run_validation`; model untouched (state_dict byte-equal, no grads); runs on every device

Guards — shuffled loader rejected; class mismatch rejected; `test` directory logs the one-shot warning, `val` does not

Export — all six artefacts written and mutually consistent (JSON ↔ report, CSV rows ↔ samples, confusion CSV ↔ matrix, report text); **CLI** end-to-end

## 7. Commands Executed

```bash
.venv/bin/python -m pytest -q tests/test_evaluate.py
.venv/bin/python -m pytest -q -rs -p no:cacheprovider
.venv/bin/ruff check . ; .venv/bin/black --check . ; .venv/bin/mypy src app scripts
.venv/bin/python -m src.evaluate --checkpoint <stage full/best.pt> --data-root <fixture> --out-dir … --workers 2   # smoke, mps
```

## 8. Results

| Check | Result |
|---|---|
| Tests | **368 passed, 0 skipped, 0 warnings** (was 349) |
| Lint / format / types | clean (27 source files) |
| Regression (Layers 0–9) | pass unchanged |
| Smoke | Layer 9 `full/best.pt` evaluated on `mps` with 2 workers; six artefacts written |

Fixture metric values are not reported: they describe synthetic colour
patches and a few epochs of plumbing, not a disease model.

## 9. Bugs Found and Fixed

None in project code; one lint fix (lambda → def).

## 10. Known Limitations

1. **No figures.** Reliability diagram / confusion heat-map are left to a
   notebook or a later plotting dependency; all inputs are exported.
2. **Confidence = top-1 softmax probability.** No temperature scaling or
   other calibration is applied; ECE reports how miscalibrated the raw
   output is.
3. **`test` warning is name-based** and advisory.
4. **Whole dataset's probabilities are held in memory** — fine at the
   project's scale (thousands of images × K classes).

## 11. Acceptance Gate

**PASS** — implementation, imports, 19 new tests, integration (checkpoint →
evaluation → export → CLI), smoke, static checks, determinism and
model-immutability proofs, no regression, documentation updated.

## 12. Next Layer Prerequisites

Layer 11 (Prediction API) must:

- provide `predict(image) → structured result` built on `Checkpoint.build_model` + `build_eval_transform(checkpoint.preprocess)`; no duplicated preprocessing or model code
- accept PIL images, numpy arrays, file paths and raw bytes; validate: empty, corrupt, unsupported mode/dimensions
- return `predicted_class`, `confidence`, `ranked_predictions`, `model_version` (checkpoint identity), `preprocessing_version`, `status`, `warnings`, plus the `risk` mapping from `src/risk_engine.py` for the frontend contract
- handle model-loading failures and incompatible checkpoints explicitly; run on CPU and on the auto-selected device
- replace `app/mock_prediction.py`'s role with the real predictor behind the same schema
- re-run Layers 0–10 as regression
