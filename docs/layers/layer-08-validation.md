# Layer 8 — Validation

## 1. Objective

A validation subsystem that is separate from training, provably inert
(changes nothing in the model), computes the full metric set from one
confusion matrix, and plugs into `fit` to drive best-checkpoint selection.
Fixture numbers are machinery checks, not disease results.

## 2. Architecture

```
src/metrics.py                       pure metric arithmetic (no model, no loader)
  confusion_matrix(t, p, K)          K×K int64, rows = true, cols = predicted
  normalize_confusion_matrix(C)      row-normalised, empty rows → 0
  metrics_from_confusion(C, names)   per-class P/R/F1/support; accuracy; macro P/R/F1; weighted F1
  compute_metrics(t, p, names)       the two above
  classification_report(m)           text table (per class, accuracy, macro avg, weighted avg)

src/validation.py                    running the model over a loader, eval-mode, inference_mode
  contains_random_transform / assert_deterministic_loader     refuses training augmentation
  run_validation(model, loader, device, class_names) → ValidationResult
      .summary   ValidationSummary (scalars: loss, acc, macro P/R/F1, weighted F1, samples, seconds)
      .metrics   ClassificationMetrics incl. per_class
      .confusion, .targets, .predictions, .probabilities   (all on CPU, loader order preserved)

src/train.py (extended)
  TrainConfig.selection_metric ∈ {"f1_macro", "accuracy", "loss"}
  EpochStats.validation: ValidationSummary | None
  fit(..., val_loader=)   per-epoch validation → best tracking → best.pt (+ last.pt)
  checkpoint format v2    stores history with validation, best_epoch, best_metric
  CLI: --val-root, --selection-metric
```

## 3. Files

**Created**
- `src/metrics.py`, `src/validation.py`
- `tests/test_metrics.py` — 15 tests; `tests/test_validation.py` — 18 tests
- `docs/layers/layer-08-validation.md`

**Modified**
- `src/train.py` — validation hook, best tracking, checkpoint v2, CLI flags
- `README.md` — build-status table

`src/evaluate.py` (scaffold, `NotImplementedError`) is untouched; Layer 10 replaces it on top of these modules.

## 4. Dependencies

None new. Metrics are implemented from the confusion matrix in torch rather
than adding scikit-learn: the arithmetic is small, is verified against
hand-computed values, and averaging conventions are stated explicitly
(macro over all K classes, zero-division → 0 — equivalent to
`sklearn.metrics.*(labels=range(K), zero_division=0)`).

## 5. Implementation Decisions

**D1 — One confusion matrix, fixed K.** Every metric derives from
`C[true, pred]` over the known class list, so results do not depend on which
classes happen to appear in a batch or split, and absent classes are
handled deterministically (zero precision/recall, still counted in macro
averages, zero weight in weighted averages — each behaviour is tested).

**D2 — Validation cannot run on augmented data.** `run_validation` inspects
the loader's transform and raises if any stage name starts with `Random`,
`ColorJitter`, `Gaussian`, `Elastic` or `AutoAugment`. The train pipeline
from Layer 6 is rejected; the eval pipeline passes.

**D3 — Inert by construction and by proof.** `model.eval()` +
`torch.inference_mode()`; no optimiser is ever referenced.
`test_validation_changes_no_parameter_or_buffer` asserts byte-equality of the
entire `state_dict` (parameters *and* BatchNorm buffers) before/after, and
that no `.grad` is created; `test_validation_uses_eval_mode_batchnorm_running_stats`
does the same for a model handed over in train mode.

**D4 — Best selection is strict improvement on a named metric.** `loss` is
minimised, the others maximised; ties do not replace the incumbent, so the
earliest best epoch wins. `best.pt` is written only on improvement; `last.pt`
every epoch. Both carry `best_epoch`/`best_metric` so a resumed run continues
tracking correctly (tested: resumed run equals the uninterrupted run in
validation losses and best epoch/metric).

**D5 — Outputs land on the CPU.** Probabilities/predictions are moved off the
compute device inside the loop so callers (Layer 10 exports, Layer 11)
never hold device tensors.

**D6 — Checkpoint format bumped to 2.** History entries now nest a
`validation` dict; `load_checkpoint` rejects other versions. No v1
checkpoint exists outside test temp dirs, so no migration path is needed.

## 6. Tests

`tests/test_metrics.py` — 15: confusion orientation, completeness, absent
classes; 4 invalid-input cases; row normalisation incl. empty rows;
per-class and aggregate metrics vs. hand computation on a 10-sample case;
perfect predictions; never-predicted class; absent class macro vs weighted;
shape mismatch; report layout and numbers.

`tests/test_validation.py` — 18 (1 parametrized × 2 devices): random-transform
detection; training augmentation refused; class-count mismatch refused;
result shapes/aggregation order/probability rows; summary ≡ metrics and loss
≡ full-batch cross-entropy; deterministic and batch-size independent;
**no parameter/buffer/grad change**; model left in eval; BatchNorm stats
untouched from train mode; every device with CPU-side outputs;
`is_improvement` directions; unknown selection metric rejected; `fit` with
validation records summaries, picks the best epoch, writes `best.pt`/`last.pt`
with matching metadata, and the reloaded best model reproduces the metric;
worse epochs do not overwrite `best.pt` (loss metric); resume keeps best
tracking intact; without `val_loader` nothing is tracked; **CLI** `--val-root`
logs validation and writes `best.pt`.

## 7. Commands Executed

```bash
.venv/bin/python -m pytest -q tests/test_metrics.py tests/test_validation.py
.venv/bin/python -m pytest -q -rs -p no:cacheprovider
.venv/bin/ruff check . ; .venv/bin/black --check . ; .venv/bin/mypy src app scripts
```

## 8. Results

| Check | Result |
|---|---|
| Tests | **319 passed, 0 skipped, 0 warnings** (was 286) |
| Lint / format / types | clean (26 source files) |
| Regression (Layers 0–7) | 286 earlier tests pass; Layer 7 `fit` extended without changing its resume/determinism guarantees |
| Smoke | CLI train + `--val-root` on fixture: validation logged per epoch, `best.pt` written (covered by subprocess test) |

## 9. Bugs Found and Fixed

None in project code. Two test-authoring errors (a wrong expected macro-F1
in the report test; a garbled fixture line) were corrected before the tests
first passed; one `TypeError` from building the `EpochStats` replacement by
hand was replaced with `dataclasses.replace`.

## 10. Known Limitations

1. **No sklearn cross-check.** Metric correctness rests on hand-computed
   cases; the conventions are documented so a cross-check can be added
   later without changing behaviour.
2. **Random-transform guard is name-based.** A custom random transform not
   matching the markers would slip through; `build_train_transform` output is
   caught, which is the realistic mistake.
3. **Validation loss is mean cross-entropy over samples** (sum-reduced per
   batch, divided by total), so it is batch-size independent; it is not
   label-smoothed or weighted.
4. **Fixture is tiny;** best-epoch selection on 12 validation images is
   noisy by nature. This only exercises the mechanism.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `src/metrics.py`, `src/validation.py`, `fit` hook |
| Unit tests pass | 33 new |
| Integration tests pass | `fit` + validation + best/last checkpoints; resume; CLI |
| Validation is inert | byte-equal state_dict, no grads, BN buffers untouched |
| No augmentation / no leak | training transform refused; val split has its own images |
| Lint / format / types | clean |
| No regression | 286 earlier tests pass |
| Documentation | this file; README |

## 12. Next Layer Prerequisites

Layer 9 (Fine-tuning) must:

- add an explicit staged freeze/unfreeze API on `src/model.py` (whole backbone; last N `features` blocks) and per-group learning rates in the optimiser
- run stages through `fit` without duplicating the loop; keep checkpoint/resume compatible (stage recorded in the checkpoint)
- prove: frozen params receive no gradient and do not change; unfrozen ones do; param groups carry the intended LRs; a stage transition changes `requires_grad` exactly as specified
- re-run Layers 0–8 as regression
