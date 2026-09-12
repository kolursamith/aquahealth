# Layer 2 — TorchVision

## 1. Objective

Install TorchVision paired with the Layer 1 torch, prove it is
binary-compatible with that torch, and verify the three capabilities later
layers take from it: image → tensor conversion (`transforms.v2`), the
pretrained-model registry (source of EfficientNet-B0 in Layer 3), and image
I/O. No weights download, no model, no dataset.

## 2. Architecture

```
requirements/base.txt
  torch==2.14.0            (Layer 1)
  torchvision==0.29.0      (Layer 2)  — compiled ops built against exactly torch 2.14.0
  numpy==2.4.6             (Layer 2)  — hard dep of torchvision; imported by src/utils
  pillow==12.3.0           (Layer 2)  — hard dep of torchvision; the image entry format

Image I/O path decided at this layer:

  bytes / file ──► Pillow (Image.open → .convert("RGB")) ──► transforms.v2 ──► tensor
                   NOT torchvision.io (deprecated in 0.29, see D2)
```

Layer 2 adds **no new project module by design**. TorchVision's value is
consumed by Layer 3 (model registry) and Layer 6 (transforms); this layer
proves the dependency is sound and fixes the I/O decision so those layers do
not build on a deprecated API.

## 3. Files

**Created**
- `tests/test_torchvision.py` — 21 tests
- `docs/layers/layer-02-torchvision.md` — this document

**Modified**
- `requirements/base.txt` — declares `torchvision`, `numpy`, `pillow`
- `src/utils.py` — numpy guard removed from `set_seed` (numpy is now declared)
- `tests/test_prediction.py` — numpy `importorskip` guard removed (reason was no longer true)
- `tests/test_preprocessing.py` — numpy guard removed; the `cv2` guard remains (Layer 6)
- `README.md` — build-status table

## 4. Dependencies

| Package | Version | Reason | Used by |
|---|---|---|---|
| torchvision | 0.29.0 | transforms.v2, model registry, pretrained weights | Layers 3, 6 |
| numpy | 2.4.6 | torch/torchvision hard dependency; array bridge; seeding | `src/utils.py` |
| pillow | 12.3.0 | torchvision hard dependency; image decoding entry point | I/O path |

numpy and pillow were installed by torchvision's own metadata, not chosen
independently; they are pinned so `verify_environment.py` checks them and so
the versions are recorded. No other new transitive packages.

## 5. Implementation Decisions

**D1 — torchvision 0.29.0.** The pairing with torch 2.14.0 was established by
dry-run in Layer 1 and is now proven at runtime: `torchvision.ops.nms` (a C++
op compiled against one exact torch ABI) executes and returns the correct
result. A mismatched pair fails at this call, not silently.

**D2 — Image I/O goes through Pillow, not `torchvision.io`.** The first run of
the Layer 2 tests surfaced a `DeprecationWarning` from
`torchvision/io/image.py`: *"The image decoding and encoding capabilities of
TorchVision are deprecated since torchvision 0.29 and will be removed in a
future release. They are superseded by … TorchCodec."* Building the
dataset/inference pipeline on `torchvision.io.decode_image` would therefore
lock the project onto an API scheduled for removal. Decision: images are
decoded by Pillow (`Image.open(...).convert("RGB")`) and handed to
`transforms.v2` (`pil_to_tensor` / `ToImage`). The two `torchvision.io` tests
were replaced with Pillow-path equivalents, and a guard test fails if any file
under `src/` ever imports `torchvision.io`. The warning was not filtered; the
dependency on it was removed.

**D3 — numpy and pillow declared explicitly.** They arrive as hard
dependencies of torchvision regardless, but project code imports them directly
(`src/utils.py` seeds numpy; images enter as PIL objects). Declaring them makes
the environment verifier check them and removes the conditional-import
workaround from Layer 1.

**D4 — Registry facts checked without downloading.** `EfficientNet_B0_Weights`
metadata (5,288,548 params, 1000 ImageNet categories, 77.692 % acc@1, and the
preset transform: resize 256 → crop 224, ImageNet mean/std, bicubic) is
verified from the enum alone. The architecture is instantiated with
`weights=None` and its parameter count matches the metadata. The weight file
itself is downloaded and validated in Layer 3, which owns that concern.

**D5 — Two flavours of determinism tested.** A deterministic pipeline
(`ToImage → ToDtype → Resize`) must be bit-identical across calls; a random
transform (`RandomHorizontalFlip`) must be reproducible under `set_seed` *and*
must actually vary within the seeded sequence (guards against a test that
passes because nothing random ever happens).

## 6. Tests

`tests/test_torchvision.py` — 21 tests (1 parametrized × 2 devices here):

- declared pins for `torchvision`, `numpy`, `pillow` equal the installed versions
- `import torch, torchvision` under `-W error` succeeds — the Layer 1 warning is gone
- compiled op `nms` runs and is correct (binary compatibility with torch)
- `torch.from_numpy` / `Tensor.numpy` round trip
- `pil_to_tensor`: CHW, uint8, pixel values preserved
- `ToImage + ToDtype(scale=True)`: `tv_tensors.Image`, float32, exact 0 / 128⁄255 / 1
- `Resize` yields requested spatial size
- `Normalize` applies mean/std exactly
- deterministic pipeline is bit-identical across calls
- random transform reproducible under `set_seed`, and genuinely random
- converted image moves to every available device and keeps its values
- `efficientnet_b0` in `models.list_models()`
- `EfficientNet_B0_Weights.DEFAULT is IMAGENET1K_V1`; params / categories / acc@1 match
- weight preset transforms declare crop 224 / resize 256 / ImageNet mean & std
- `efficientnet_b0(weights=None)` builds with 5,288,548 params and a 1000-way head
- PNG bytes → Pillow → tensor is lossless; JPEG bytes → Pillow → tensor has expected shape/dtype
- **guard:** no file under `src/` imports `torchvision.io`

## 7. Commands Executed

```bash
.venv/bin/python -m pip install torchvision==0.29.0      # → + numpy 2.4.6, pillow 12.3.0
.venv/bin/python -W error::UserWarning -c "import torch, torchvision, ..."   # probe
.venv/bin/python -m pytest -q -rs tests/test_torchvision.py
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
.venv/bin/python scripts/verify_environment.py
```

## 8. Results

Measured on this machine (Apple M1 Pro, macOS 14.8.1, Python 3.11.15):

| Property | Value |
|---|---|
| torchvision | 0.29.0 |
| numpy / pillow | 2.4.6 / 12.3.0 |
| `nms` on torch 2.14.0 | runs, correct |
| `import torch` warning from Layer 1 | **gone** (verified under `-W error`) |
| `EfficientNet_B0_Weights.DEFAULT` | IMAGENET1K_V1 — 5,288,548 params, 1000 classes, 77.692 % acc@1 |

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **99 passed, 1 skipped, 0 warnings** (was 77 + 2 + 1 warning) |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 33 files unchanged |
| Types | `mypy src app scripts` | no issues in 24 source files |
| Smoke | `verify_environment.py` | exit 0; torchvision, numpy, pillow all `[OK]` |
| Regression (Layers 0–1) | 77 earlier tests | pass; `test_prediction.py` now runs instead of skipping |

The remaining skip is `tests/test_preprocessing.py` (OpenCV, Layer 6).

## 9. Bugs Found and Fixed

None in project code. One upstream fact changed the design (D2).

## 10. Known Limitations

1. **Pretrained weights not yet downloaded or validated.** Only the registry
   metadata and the bare architecture were checked. Layer 3 downloads
   `IMAGENET1K_V1`, verifies its checksum/loadability, and runs a forward pass.
2. **CI still unverified on GitHub.** All gate commands ran locally; the
   Linux CPU-index install path was validated by dry-run in Layer 1 and now
   additionally resolves torchvision (torchvision `+cpu` wheels exist on the
   same index), but no Actions run has executed.
3. **`torchvision.io` guard is text-based.** It greps `src/` for the string
   `torchvision.io`; an aliased import would evade it. Acceptable for a
   guard whose purpose is to catch the obvious mistake.
4. **Scaffold modules still reference undeclared packages.** `src/model.py`
   (`timm`), `src/preprocessing.py` (`cv2`), `src/augmentation.py`
   (`albumentations`) are untested skeleton from the initial commit and will be
   replaced by Layers 3 and 6. mypy passes because `ignore_missing_imports`
   is set; nothing imports them at runtime.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | dependency declared; I/O path decided and guarded |
| Imports work | `torchvision`, `transforms.v2`, `models`, `numpy`, `PIL` |
| Unit tests pass | 21 new tests |
| Integration tests pass | `-W error` subprocess import; verifier smoke |
| Smoke tests pass | verifier exit 0 with all four new distributions OK |
| Lint / format / types | ruff, black, mypy clean |
| Runtime behaviour verified | compiled op, conversions, normalize, determinism, device moves |
| No blocking bug | none |
| No unexplained error | **0 warnings** in the full suite |
| No regression | Layers 0–1 tests pass; one former skip now runs and passes |
| Documentation updated | this file; README status table |

## 12. Next Layer Prerequisites

Layer 3 (Pretrained EfficientNet-B0) must:

- download `EfficientNet_B0_Weights.IMAGENET1K_V1` via torchvision (cached under `~/.cache/torch/hub`), and verify it loads
- replace `src/model.py` (timm scaffold) with a torchvision-based builder; keep the head *configurable* — do not hardcode the disease class count
- forward-pass a batch on the resolved device and on CPU; assert logits shape `(N, 1000)` for the stock head
- verify the model is deterministic in `eval()` mode and that `train()`/`eval()` differ where dropout/BN make them differ
- record parameter count, and the input contract (`(N, 3, 224, 224)`, ImageNet-normalized) as the model's canonical input
- re-run Layers 0–2 as regression
