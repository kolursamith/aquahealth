# Layer 6 — Preprocessing + CLAHE

## 1. Objective

Define the one canonical way an image becomes model input, make CLAHE an
explicit and configurable option (off by default), and give training its
own augmentation that lands in exactly the same output space. Validated on
synthetic images; no claim is made about whether CLAHE helps on fish data.

## 2. Architecture

```
src/preprocessing.py
  PreprocessConfig(image_size=224, resize_size=256, resize_mode="crop"|"squash",
                   clahe=None|CLAHEConfig, mean, std)
  CLAHEConfig(clip_limit=2.0, tile_grid_size=8)
  CLAHE                       PIL→PIL on LAB lightness; picklable; lazy cv2 handle
  build_eval_transform(cfg)   [CLAHE?] → resize stage → PILToTensor → float → Normalize
  denormalize(t, cfg)         inverse of Normalize, for inspection
  resize_transforms / tensor_transforms   shared stage builders

src/augmentation.py
  AugmentConfig(crop_scale=(0.8,1), crop_ratio=(0.9,1.1), horizontal_flip=0.5,
                rotation_degrees=15, brightness=0.2, contrast=0.2)
  build_train_transform(cfg, aug)
      [CLAHE?] → RandomResizedCrop → HFlip → Rotation → ColorJitter → tensor_transforms(cfg)

Output of both: float32 (3, image_size, image_size), ImageNet-normalised.
```

Train and eval share `PreprocessConfig` and the identical final tensor
stage, so a model trained with the train pipeline sees the same input space
at inference. Only the geometric/photometric middle differs.

## 3. Files

**Created**
- `tests/test_augmentation.py` — 17 tests
- `docs/layers/layer-06-preprocessing.md` — this document

**Modified**
- `src/preprocessing.py` — rewritten from the scaffold (which used OpenCV resize/normalise directly and had no config)
- `src/augmentation.py` — rewritten from the scaffold (albumentations, never declared) onto `transforms.v2`
- `tests/test_preprocessing.py` — rewritten; the `cv2` skip guard is gone (37 tests)
- `requirements/base.txt` — declares `opencv-python-headless==5.0.0.93`
- `README.md` — build-status table

## 4. Dependencies

| Package | Version | Reason | Used by |
|---|---|---|---|
| opencv-python-headless | 5.0.0.93 | reference CLAHE implementation + RGB↔LAB | `src/preprocessing.py` |

No new transitive packages (numpy already present). Verified to import
cleanly under `-W error` alongside numpy 2.4.6. `albumentations` and `timm`
from the initial scaffold are now referenced nowhere.

## 5. Implementation Decisions

**D1 — CLAHE via OpenCV, not a re-implementation.** CLAHE is tiled
histogram equalisation with a clip limit *and* bilinear interpolation across
tile borders. Re-implementing that in numpy risks a subtly different
operator that is still called "CLAHE". OpenCV's is the reference used in the
literature, so results are comparable. Headless build: no GUI toolkit.

**D2 — CLAHE on LAB lightness only.** Equalising RGB channels independently
shifts hue. Applying it to L and leaving a/b untouched preserves chroma;
`test_clahe_changes_lightness_but_not_chroma_much` measures this (mean |ΔL|
> 5, mean |Δab| < |ΔL|/4).

**D3 — CLAHE is off by default and configurable.** `PreprocessConfig().clahe
is None`; enabling it is an explicit `CLAHEConfig(...)`. Whether it improves
disease classification is an experiment on the real data (preprocessing
rule), not a baked-in assumption. It sits at the same position (first) in
both train and eval pipelines, so a model trained with CLAHE is served with
CLAHE.

**D4 — Default resize is the torchvision preset; "squash" is offered.**
`resize_mode="crop"` reproduces `EfficientNet_B0_Weights.transforms()` —
bicubic shorter-side 256, centre-crop 224 — to within 7 × 10⁻⁷ (verified
against the preset itself). It is the preprocessing the pretrained weights
were validated with. Centre-cropping does discard borders; for images where
lesions may sit at the edge, `resize_mode="squash"` keeps the whole frame at
the cost of aspect distortion. Which is right depends on the real images and
is a one-field switch; `test_crop_mode_crops_and_squash_mode_keeps_everything`
proves the two really differ.

**D5 — Augmentation on `transforms.v2`, not albumentations.** Everything the
brief asks for (flip, rotation, brightness/contrast, controlled crop/zoom)
exists in v2, keeps the pipeline in one library, and is seedable through
`torch.manual_seed` — verified reproducible. Vertical flip is deliberately
absent (fish orientation is meaningful); it is one line to add if the data
argues otherwise.

**D6 — `CLAHE` is picklable (bug B1).** See §9.

## 6. Tests

`tests/test_preprocessing.py` — 37 tests (1 parametrized × 2 devices):

- OpenCV declared at the installed version; constants match `config.IMAGE_SIZE` and the pretrained preset
- `CLAHEConfig` / `PreprocessConfig` reject invalid values (5 cases); squash mode relaxes the size constraint; CLAHE off by default
- eval output contract `(3, 224, 224)` float32 finite for 5 input sizes × 2 modes
- eval crop mode equals the torchvision preset (atol 1e-5)
- eval bit-deterministic, with and without CLAHE
- normalisation exact on a flat grey image; `denormalize` inverts to the un-normalised tensor
- crop mode discards a border stripe, squash keeps it; custom image size honoured
- CLAHE: >2× contrast on a flat image; size/mode preserved and deterministic; non-RGB rejected; clip limit and tile size change output; lightness-not-chroma
- **CLAHE pickles and still works; eval + train pipelines with CLAHE run in 2 worker processes** (regression for B1)
- with vs without CLAHE differ
- integration: dataset + eval transform → loader → `build_classifier` → `(8, 4)` logits; tensor moves to every device

`tests/test_augmentation.py` — 17 tests:

- `AugmentConfig` rejects 6 invalid forms
- train output contract for 3 input sizes
- train and eval share the identical tensor stage (type and repr)
- train output statistics lie near eval's
- seed-reproducible and otherwise varying; all-randomness-off is deterministic; flip p=1 mirrors exactly
- stage order fixed; CLAHE placed first when enabled; config values reach the transforms; custom size

## 7. Commands Executed

```bash
.venv/bin/python -m pip install opencv-python-headless==5.0.0.93
.venv/bin/python -W error -c "import cv2, numpy, torch; ..."            # compat probe
.venv/bin/python -m pytest -q -rs tests/test_preprocessing.py tests/test_augmentation.py
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
.venv/bin/python scripts/verify_environment.py
.venv/bin/python <scratchpad>/smoke6.py     # train+eval pipelines, CLAHE on, 2 workers, mps
```

## 8. Results

| Property | Value |
|---|---|
| eval("crop") vs torchvision preset | max abs diff 7.2 × 10⁻⁷ |
| CLAHE on flat image (std 100±20) | pixel std 5.8 → 19.6 |
| Train pipeline | seed-reproducible; varies without seed |
| Smoke (3 classes, CLAHE on, 2 workers, mps) | train batch `(8,3,224,224)` loss 2.92, finite head grads; val logits `(8,3)` |

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **256 passed, 0 skipped, 0 warnings** (was 202 + 1 skip) |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 38 files unchanged |
| Types | `mypy src app scripts` | no issues in 24 source files |
| Smoke | file-based, both pipelines, worker processes | pass |
| Verifier | `verify_environment.py` | opencv `[OK]`, exit 0 |
| Regression (Layers 0–5) | 202 earlier tests | pass; the last remaining skip is gone |

## 9. Bugs Found and Fixed

**B1 — `CLAHE` could not be sent to DataLoader workers.** The file-based
smoke with `num_workers=2` failed: `TypeError: cannot pickle 'cv2.CLAHE'
object`. Root cause: the transform stored the OpenCV handle as instance
state, and macOS `spawn` workers receive the dataset (transform included) by
pickling. The Layer 5 worker test passed only because it used a pure-v2
transform. Fix: `CLAHE` keeps only its `CLAHEConfig` as state
(`__getstate__`/`__setstate__`) and builds the cv2 handle lazily per
process. Regression tests: pickle round-trip of `CLAHE` and of a full eval
pipeline, plus eval and train pipelines with CLAHE iterated through 2
workers. This is exactly the class of bug the smoke step exists to catch.

## 10. Known Limitations

1. **No evidence CLAHE helps.** Verified only that it does what CLAHE does.
   The real dataset decides whether it is enabled.
2. **Crop vs squash is a judgement call deferred to the data.** Default is
   the pretrained preset; the audit's size/aspect statistics should inform
   the choice before training.
3. **CLAHE runs at native resolution** before resize, so tile size in pixels
   varies with input resolution. Consistent between train and eval, but if
   the real images vary wildly in size, applying it after resize would be
   the alternative — one-line change, needs data to decide.
4. **Augmentation defaults are conventional, not tuned.** Ranges (±15°, 0.8–1.0
   crop scale, ±20 % brightness/contrast) are reasonable starting points for
   fine-tuning; Layer 9 may revisit with validation evidence.
5. **`ColorJitter` hue/saturation are off.** Colour is likely diagnostic for
   disease; changing it was judged riskier than not. Revisit with data.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `src/preprocessing.py`, `src/augmentation.py` rewritten |
| Imports work | full suite; `cv2` declared and importable |
| Unit tests pass | 54 new tests |
| Integration tests pass | dataset → transform → loader → classifier; worker processes with CLAHE |
| Smoke tests pass | file-based train + eval with CLAHE on `mps` |
| Lint / format / types | ruff, black, mypy clean |
| Runtime behaviour verified | preset equivalence, determinism, seeding, CLAHE effect, pickling |
| No blocking bug | B1 found by smoke, fixed at root, regression-tested |
| No unexplained error | 0 warnings |
| No regression | Layers 0–5 pass; former skip now a real passing test |
| Documentation updated | this file; README |

## 12. Next Layer Prerequisites

Layer 7 (Training Loop) must:

- replace the `src/train.py` scaffold with a training loop that takes a `build_classifier` model, train/val loaders, an optimiser, and a device from `resolve_device`
- be runnable as `python -m src.train` with an `if __name__ == "__main__":` guard (macOS `spawn` constraint from Layer 5), and support `persistent_workers=True` in `build_dataloader`
- one epoch = forward → loss → backward → step over the train loader, then a no-grad validation pass with the eval transform
- checkpoint: save `{model state_dict, optimizer state_dict, epoch, class_names, PreprocessConfig, metrics}` and prove resume reproduces the same next-step loss
- tiny-fixture training: on the colour-per-class synthetic data (which is linearly separable), loss must fall and train accuracy must reach ~100 % within a few epochs — a plumbing proof, explicitly not a disease result
- verify: no parameter updates during validation; frozen backbone stays frozen; seed → identical loss curve; MPS and CPU both run
- re-run Layers 0–6 as regression
