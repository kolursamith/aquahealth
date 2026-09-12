# Layer 7 — Training Loop

## 1. Objective

A reusable training engine: epoch loop, loss, backward, optimiser step,
mode handling, checkpoint save/load, exact resume, seed reproducibility,
device handling and a CLI entry point — proven on the synthetic fixture.
Fixture results are a software sanity check, not a disease result.

## 2. Architecture

```
src/train.py
  TrainConfig(epochs, learning_rate, weight_decay, seed)     defaults from src/config.py
  build_optimizer(model, cfg)      AdamW over requires_grad params only; error if none
  set_train_mode(model)            model.train(); frozen backbone → features.eval()
  train_one_epoch(model, loader, optimizer, device, epoch) → EpochStats
       per batch: zero_grad(set_to_none) → forward → cross_entropy → backward → step
  save_checkpoint(path, ...)       atomic write (tmp → rename) of:
       model_state, optimizer_state, model_spec, class_names, preprocess,
       train_config, history, RNG states (python/numpy/torch cpu/cuda/mps),
       loader shuffle-generator state
  load_checkpoint(path) → Checkpoint   version-checked; .build_model(), .restore_optimizer(), .restore_rng(loader)
  fit(model, loader, *, config, class_names, preprocess, device, checkpoint_dir, resume_from)
       → TrainingResult(history, last_checkpoint); leaves model in eval()
  main()                           python -m src.train --train-root … --checkpoint-dir … [--resume]

src/dataset.py  build_dataloader(..., persistent_workers=)    (Layer 5 extension)
```

## 3. Files

**Created**
- `tests/test_train.py` — 26 tests
- `docs/layers/layer-07-training.md` — this document

**Modified**
- `src/train.py` — rewritten from the `NotImplementedError` scaffold
- `src/model.py` — head init changed to fan-in scaling (bug B1, a Layer 4 defect)
- `src/dataset.py` — `persistent_workers` option on `build_dataloader`
- `tests/test_classifier.py` — init test corrected; new initial-loss regression test
- `docs/layers/layer-04-classifier-head.md` — amendment recording the superseded decision
- `README.md` — build-status table

## 4. Dependencies

None new.

## 5. Implementation Decisions

**D1 — Frozen backbone stays in `eval()` during training.** `model.train()`
alone would let the backbone's BatchNorm running statistics drift even
though its parameters are frozen. `set_train_mode` puts the trainable part in
train mode and a frozen `features` in eval. Verified by
`test_frozen_backbone_batchnorm_statistics_do_not_drift`; the hazard was
re-demonstrated when a test called bare `model.train()` and eval predictions
changed afterwards.

**D2 — Exact resume, not approximate.** The checkpoint carries the RNG
states of Python, numpy, torch CPU, CUDA (if present) and MPS (if present),
plus the DataLoader's shuffle generator state. Resuming after epoch 1 and
training to epoch 3 reproduces the uninterrupted 3-epoch loss curve
**bit-for-bit** on CPU (`test_resume_reproduces_the_uninterrupted_trajectory_exactly`).

**D3 — Checkpoint is self-describing.** It stores `model_spec`
(num_classes, dropout, freeze flag), `class_names`, and the full
`PreprocessConfig`, so `Checkpoint.build_model()` needs nothing from
`src/config.py` and Layer 11 can serve it with the identical preprocessing.
Writes are atomic (temp file + rename) so a crash never leaves a truncated
`last.pt`. A `format_version` is checked on load.

**D4 — `EpochStats.train_accuracy` is named for what it is.** It is
accumulated batch-by-batch with dropout active while weights change. It is
a progress signal. "Is the model fitted?" is answered by an eval-mode pass —
the fixture test does exactly that and
`test_epoch_train_accuracy_is_a_train_mode_estimate_not_an_evaluation`
guards the distinction.

**D5 — `fit` returns the model in `eval()`.** Leaving a trained model in
train mode is a classic footgun for whoever calls it next (Layer 8/11).

**D6 — Best-model tracking deferred to Layer 8, deliberately.** "Best" needs
a validation metric; selecting on training loss would pick the most
over-fitted epoch. Layer 7 writes `last.pt` every epoch; Layer 8 adds a
validation pass and `best.pt`.

**D7 — No LR scheduler yet.** Fine-tuning schedules are Layer 9's decision;
adding one here without a validation signal would be untestable.

**D8 — CLI is a real module with a `__main__` guard** (Layer 5's macOS
`spawn` constraint) and uses `persistent_workers=True`, so worker start-up
is paid once per run: measured 1.8 s for epoch 1, 0.1 s for epochs 2–3.

## 6. Tests

`tests/test_train.py` — 26 tests (1 parametrized × 2 devices):

Config — invalid `epochs`/`lr`/`weight_decay` rejected; defaults come from `src/config.py`

Modes / optimiser — `set_train_mode` keeps a frozen backbone in eval, trains everything when unfrozen; optimiser holds exactly the trainable params with the configured lr/wd; fully frozen model rejected

One epoch — returns `EpochStats`, updates the head, **leaves every frozen backbone tensor byte-identical**; BatchNorm running stats do not drift; unfrozen backbone does update; **`zero_grad` and `step` each called exactly once per batch** (4 = `len(loader)`), gradients present after the epoch

Fixture training — initial loss ≈ ln 4 (±0.3), final loss < 75 % of initial, train-accuracy ≥ 0.9, and **eval-mode predictions equal the fixture labels exactly**; `fit` leaves the model in eval; train-mode vs eval-mode measurement distinction; runs on every available device; seed reproduces the loss curve exactly, different seed differs

Checkpoints — round trip preserves weights, optimiser state, epoch, history, class names, `PreprocessConfig` (incl. nested `CLAHEConfig`), `TrainConfig`, `model_spec`, frozen flag; no `.tmp` left behind; wrong format version rejected; `model_spec` reflects the model; `fit` writes `last.pt` each epoch; **resume reproduces the uninterrupted trajectory exactly**; resuming at the target epoch trains nothing more

CLI — `python -m src.train` trains 1 epoch with 2 worker processes, then resumes to epoch 2 (log lines, checkpoint epoch/history/preprocess/class names verified); `--device cuda` on a CUDA-less machine exits non-zero with "not available on this machine"

`tests/test_classifier.py` (Layer 4, revalidated) — 39 tests including the new `test_initial_loss_is_close_to_uniform_chance_for_any_class_count` for K ∈ {2, 4, 8, 13}.

## 7. Commands Executed

```bash
.venv/bin/python <scratchpad>/probe7.py      # convergence / determinism / resume probe
.venv/bin/python <scratchpad>/diag7.py       # root-cause isolation (aug vs lr vs dropout vs init)
.venv/bin/python -m pytest -q tests/test_classifier.py tests/test_model.py   # Layer 3–4 regression
.venv/bin/python -m pytest -q tests/test_train.py
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check . ; .venv/bin/black --check . ; .venv/bin/mypy src app scripts
.venv/bin/python -m src.train --train-root … --checkpoint-dir … --epochs 3 --image-size 64 \
    --batch-size 8 --freeze-backbone --workers 2      # smoke, auto device → mps
```

## 8. Results

Fixture (4 colour classes × 8 images, 64 px, frozen backbone, AdamW 1e-3, eval transform, CPU):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| loss | 1.361 | 1.252 | 1.157 | 1.074 | 1.039 | 0.977 | 0.906 | 0.865 |
| train_acc | 0.38 | 0.50 | 0.63 | 0.75 | 0.72 | 0.81 | 0.88 | 0.97 |

Eval-mode predictions after 10 epochs: 32/32 correct. **This is a plumbing
check on synthetic colour patches, not a disease result.**

Determinism / resume (CPU): identical loss curves for identical seeds; resumed
run equals uninterrupted run to the last bit. MPS: seed-reproducible across
two runs.

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **286 passed, 0 skipped, 0 warnings** (was 256) |
| Layer 4 regression | `pytest tests/test_classifier.py tests/test_model.py` | 60 passed |
| Lint / format / types | ruff, black, mypy | clean (24 source files) |
| Smoke | CLI, 3 epochs, mps, 2 persistent workers | checkpoint written; epochs 2–3 at 0.1 s |
| Regression (Layers 0–6) | 256 earlier tests | pass; one Layer 4 test corrected (see B1) |

## 9. Bugs Found and Fixed

**B1 — Classifier-head initialisation was scaled by output count (Layer 4).**
The first fixture run did not converge: loss started at ≈ 4.3 (ln 4 = 1.39)
and bounced. `diag7.py` isolated it: pooled features have std ≈ 0.35 over
1280 dims, and the head init `uniform(±1/√num_classes)` — torchvision's
scheme, sized for a 1000-way head — gave bound 0.5 for 4 classes and initial
logit std 4.2. Fix in `src/model.py`: `uniform(±1/√in_features)` (fan-in),
which gives logit std ≈ 0.2 for any class count. After the fix the same run
converged (loss 1.38 → 0.87, 31 % → 100 %). The Layer 4 test that encoded
the old scheme was corrected, and a regression test now asserts initial loss
≈ ln K (±0.3) and logit std < 0.5 for K ∈ {2, 4, 8, 13}. Layer 4's doc
carries an amendment. Layers 3–4 revalidated: 60 passed.

**B2 — Fixture test judged "fitted" by train-mode accuracy.** With dropout
active the per-epoch accuracy plateaued at 97 %. The criterion was changed
to an eval-mode pass over the fixture (32/32), the field was renamed
`train_accuracy`, and a regression test guards the distinction (D4).

**B3 — Test used bare `model.train()` on a frozen backbone.** BatchNorm
running stats drifted and eval predictions changed. Corrected to
`set_train_mode`; this is the exact hazard D1 addresses.

**Test-side only:** `class_names` expectation had to be `sorted(CLASSES)`
because `discover_classes` sorts (Layer 5 D1). Not a code defect.

**Static:** two mypy `union-attr` errors from typing `model: nn.Module`
where EfficientNet structure is assumed; annotated as `EfficientNet`.

## 10. Known Limitations

1. **Augmentation slows the tiny fixture.** With default augmentation on 32
   images, 3 epochs (12 steps) barely move; the convergence proof uses the
   eval transform. That is a property of 32 samples under heavy augmentation,
   not of the loop — the real dataset is two orders of magnitude larger.
2. **Exact resume proven on CPU.** MPS is seed-reproducible run-to-run, but
   bit-exact resume across MPS RNG restore was not separately asserted.
3. **No validation, no best-model selection, no scheduler** — by design
   (D6, D7); Layers 8 and 9.
4. **No mixed precision, gradient clipping or accumulation.** Add when the
   real data shows a need.
5. **Checkpoint stores optimiser state unconditionally** (~16 MB for a
   frozen-backbone model). Fine for this project's scale.
6. **Logger name shows `__main__` when run as `python -m src.train`.**
   Cosmetic.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `src/train.py` engine + CLI |
| Imports work | full suite |
| Unit tests pass | 26 new tests |
| Integration tests pass | fixture training end-to-end; CLI train + resume with workers |
| Smoke tests pass | CLI on mps with persistent workers |
| Lint / format / types | clean |
| Runtime behaviour verified | learning, mode handling, zero_grad/step accounting, checkpoint, exact resume, determinism, devices |
| No blocking bug | B1–B3 fixed at root, regression-tested |
| No unexplained error | 0 warnings |
| No regression | Layer 4 revalidated after its fix; Layers 0–6 pass |
| Documentation updated | this file; Layer 4 amendment; README |

## 12. Next Layer Prerequisites

Layer 8 (Validation) must:

- add a separate `src/evaluate.py`-based validation subsystem: eval-mode, `no_grad`, eval transform only, no optimiser
- compute loss, accuracy, precision/recall/F1 (macro and weighted), per-class metrics, confusion matrix, classification report, and aggregate predictions
- prove validation changes **no** parameter or buffer, performs no optimiser step, and never sees the train transform
- integrate into `fit` as an optional per-epoch hook that drives `best.pt` selection on a validation metric
- re-run Layers 0–7 as regression
