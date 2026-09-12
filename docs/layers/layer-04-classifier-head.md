# Layer 4 — Disease Classifier Head

## 1. Objective

Put a task-specific classification head on the Layer 3 backbone without
committing to a class count. `num_classes` is a required argument, the
backbone is untouched by the swap, and the backbone can be frozen so only the
head trains. All validation uses random inputs; no disease data, no training
run.

## 2. Architecture

```
src/model.py
  build_classifier(num_classes, *, dropout=0.2, freeze_backbone=False, pretrained=True)
      │
      ├── build_efficientnet_b0(pretrained)          Layer 3 backbone, weights untouched
      ├── classifier = Dropout(dropout) → Linear(1280, num_classes)
      │       init: uniform(±1/√num_classes), bias 0   — torchvision's own Linear scheme
      └── freeze_backbone → set_backbone_trainable(model, False)

  set_backbone_trainable(model, bool)   toggles requires_grad on model.features
  summarize(model)                      Layer 3; trainable count now meaningful

Input contract unchanged: (N, 3, 224, 224) ImageNet-normalized → logits (N, num_classes)
```

Parameter count for `num_classes = k`: 5,288,548 − (1280·1000 + 1000) + (1280·k + k).
For k = 8: 4,017,796 total, 10,248 trainable when frozen.

## 3. Files

**Created**
- `tests/test_classifier.py` — 35 tests
- `docs/layers/layer-04-classifier-head.md` — this document

**Modified**
- `src/model.py` — `build_classifier`, `set_backbone_trainable`, constants `STOCK_DROPOUT`, `MIN_NUM_CLASSES`
- `src/config.py` — comment only: `CLASS_NAMES` is now explicitly marked as the brief's *expected* list, pending verification by `scripts/audit_dataset.py`; the model does not read it
- `README.md` — build-status table

## 4. Dependencies

None new.

## 5. Implementation Decisions

**D1 — `num_classes` is required and positional; everything else keyword-only.**
There is no default because there is no verified value to default to (model
rule: do not hardcode the disease class count before the dataset is
inspected). `test_num_classes_is_a_required_positional_argument` inspects the
signature so a future "convenience default" fails the suite.

**D2 — Head init replicates torchvision's scheme.** torchvision initialises
its `Linear` layers as `uniform(±1/√out_features)` with zero bias
(`efficientnet.py:328-331`). Using the same scheme means the new head starts
exactly the way the reference recipe's head did, rather than PyTorch's
generic `kaiming_uniform` default. Verified by bound, std (±5 %), and zero
bias.

**D3 — Only the head is replaced.** `model.features` is byte-identical to a
freshly built pretrained backbone after `build_classifier`
(`test_pretrained_backbone_is_unchanged_by_head_replacement`), so the swap
cannot silently perturb pretrained weights.

**D4 — Freezing is a flag plus a reversible function.** `freeze_backbone=True`
is the common fine-tuning start; `set_backbone_trainable(model, True)` lets
Layer 9 unfreeze for a second phase without rebuilding. Both directions are
tested via `summarize().trainable_parameters` and via actual gradient
presence/absence after `backward()`.

**D5 — Learnability is demonstrated, not inferred.** One SGD step on a frozen-
backbone model reduces cross-entropy on a fixed random batch (3.35 → 2.21 in
the probe). This proves the head is wired into the optimiser and the gradient
direction is correct; it says nothing about disease accuracy.

**D6 — `src/config.py` left functional, annotated.** Removing `CLASS_NAMES`
would break scaffold modules (`dataset.py`, `audit_dataset.py`,
`create_split.py`, the Streamlit dashboard) that Layers 5–11 will rewrite
anyway. The comment now states its provenance and that the model ignores it.
`tests/test_dataset.py::test_class_count` (a Layer 0 placeholder asserting 8)
is untouched and will be replaced by Layer 5.

## 6. Tests

`tests/test_classifier.py` — 35 tests (1 parametrized × 2 devices here):

Interface — required positional `num_classes`, keyword-only options; `num_classes` ∈ {0, 1, −3} rejected; dropout ∈ {−0.1, 1.0, 1.5} rejected

Head structure — `Dropout(0.2) → Linear(1280, k)` for k ∈ {2, 5, 8, 13}; dropout configurable; parameter count formula for k ∈ {2, 8}; init bound/std/zero-bias; deterministic under seed and different across seeds

Backbone preservation — `features` identical to stock pretrained after swap; `pretrained=False` gives a different backbone

Freezing — default fully trainable; `freeze_backbone` → trainable == 1280·8+8; toggle both ways; frozen backbone gets **no** gradients while head gets finite ones; unfrozen gets all

Forward pass — `(4, 8)` finite logits on every device; output width tracks k; `eval()` deterministic; softmax rows sum to 1

Learnability — one SGD step on the head reduces loss on a fixed batch

Checkpoint contract — `state_dict` round trip between same-k classifiers is exact; loading a k=8 checkpoint into k=5 raises `size mismatch`

## 7. Commands Executed

```bash
sed -n '328,332p' .venv/lib/python3.11/site-packages/torchvision/models/efficientnet.py   # init scheme
.venv/bin/python -m pytest -q -rs tests/test_classifier.py
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
```

## 8. Results

| Property | Value |
|---|---|
| k = 8 total / trainable (frozen) | 4,017,796 / 10,248 |
| Head init std (k = 8) | 0.2038 measured vs 0.2041 expected |
| One head step, frozen backbone, CE loss | 3.351 → 2.206 |
| Smoke on `mps` | `(3, 8)` probabilities, row sums 1.0 |

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **155 passed, 1 skipped, 0 warnings** (was 120 + 1) |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 35 files unchanged |
| Types | `mypy src app scripts` | no issues in 24 source files |
| Smoke | `build_classifier(8, freeze_backbone=True)` on resolved device | summary correct, distribution valid |
| Regression (Layers 0–3) | 120 earlier tests | pass unchanged |

## 9. Bugs Found and Fixed

None. Three `E501` failures on over-long section-divider comments were
shortened; no behaviour change.

## 10. Known Limitations

1. **Head-only architecture.** A single `Linear` head is the reference recipe.
   No hidden layer / extra normalisation is offered; if fine-tuning (Layer 9)
   shows the need, it is a contained change in `build_classifier`.
2. **Freezing granularity is all-or-nothing on `features`.** Partial unfreezing
   (e.g. last N blocks) is not implemented; Layer 9 decides whether it is needed.
3. **Learnability test is one step on one random batch.** It proves plumbing,
   not convergence. Tiny-fixture training belongs to Layer 7.
4. **`src/config.py` still carries the brief's 8-class list** as an annotated
   placeholder, consumed only by scaffold code.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `build_classifier`, `set_backbone_trainable` |
| Imports work | full suite imports |
| Unit tests pass | 35 new tests |
| Integration tests pass | pretrained backbone → new head → forward → softmax; checkpoint round trip |
| Smoke tests pass | resolved-device build + forward + distribution check |
| Lint / format / types | ruff, black, mypy clean |
| Runtime behaviour verified | shapes, init, freezing, gradient routing, learnability, determinism |
| No blocking bug | none |
| No unexplained error | 0 warnings |
| No regression | Layers 0–3 pass unchanged |
| Documentation updated | this file; README; config annotation |

## 12. Next Layer Prerequisites

Layer 5 (Dataset / DataLoader) must:

- replace the `src/dataset.py` scaffold: directory-per-class layout, Pillow decoding (Layer 2 decision), labels derived from the directory listing — not from `config.CLASS_NAMES`
- expose the discovered class list so `num_classes` for `build_classifier` comes from the data
- build a controlled synthetic fixture (generated images, known labels) under `tmp_path` for tests; no real dataset
- verify: length, item shape/dtype, label range, class ↔ index mapping stability (sorted, deterministic), DataLoader batching, `shuffle` reproducibility under seed, worker behaviour, rejection of corrupt/unreadable files
- re-run Layers 0–4 as regression

---

## Amendment (Layer 7, 2026-09-12) — D2 superseded

D2 above chose torchvision's `uniform(±1/√out_features)` head initialisation.
Layer 7's fixture training exposed that this scheme is only well-conditioned
for torchvision's 1000-way head: for K = 4 it yields initial logits with
std ≈ 4.2 and an initial loss ≈ 4.3 (ln 4 = 1.39), and training did not
converge. The head is now initialised `uniform(±1/√in_features)` (fan-in),
which gives initial logits near zero for any K. `tests/test_classifier.py`
was corrected accordingly and gained
`test_initial_loss_is_close_to_uniform_chance_for_any_class_count`.
All 39 Layer 4 tests pass after the change. See
[layer-07-training.md](layer-07-training.md) §9 B1.
