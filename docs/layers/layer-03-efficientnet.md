# Layer 3 — Pretrained EfficientNet-B0

## 1. Objective

Obtain the official TorchVision EfficientNet-B0 ImageNet-1K weights, prove
their provenance, and verify that the stock model (1000-way head) forward-
passes correctly on every backend available here. The disease classifier
head is **not** built at this layer — Layer 4 owns it and keeps it
configurable.

The weights are general-purpose ImageNet classification weights. They are not
fish-disease-trained and nothing in this layer measures disease performance.

## 2. Architecture

```
src/model.py
  PRETRAINED_WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1   (single source of truth)
  build_efficientnet_b0(pretrained=True) -> torchvision EfficientNet, on CPU
  summarize(model)                      -> ModelSummary(arch, total, trainable, out_features)
  logits_to_probabilities(logits)       -> softmax over the last dim (no_grad)

Input contract (stock model):
  (N, 3, H, W) float32, ImageNet-normalized  → logits (N, 1000)
  canonical H = W = 224 (the weight preset resizes 256 → centre-crops 224)
  other H, W accepted (adaptive pooling); 1-channel and unbatched inputs raise
```

The builder returns the model on the CPU; callers move it with
`.to(resolve_device())`. This keeps device policy in `src/device.py` (Layer 1)
and out of model construction.

## 3. Files

**Created**
- `tests/test_model.py` — 21 tests
- `docs/layers/layer-03-efficientnet.md` — this document

**Modified**
- `src/model.py` — rewritten: `timm` scaffold replaced by a torchvision builder
- `README.md` — build-status table

**Not modified, deliberately**
- `src/config.py` — still carries the 8-class scaffold (`CLASS_NAMES`,
  `NUM_CLASSES`, `MODEL_NAME`). `src/model.py` no longer imports any of it.
  Layer 4 decides how the head is parameterised; the real class list is fixed
  only after the dataset is inspected (real-dataset rule).

## 4. Dependencies

No new package. Runtime artefact:

| Artefact | Source | Size | Location |
|---|---|---|---|
| `efficientnet_b0_rwightman-7f5810bc.pth` | `download.pytorch.org/models/` | 21,444,401 bytes | `~/.cache/torch/hub/checkpoints/` |

Downloaded once on first `build_efficientnet_b0(pretrained=True)`; not
committed (outside the repo, and `models/**/*.pth` is git-ignored anyway).

## 5. Implementation Decisions

**D1 — torchvision weights, not timm.** Layer 0 already dropped `timm`. The
torchvision enum gives a hash-verified download, documented metadata
(`num_params`, `categories`, `acc@1`, preset transforms) and a registry the
tests can assert against without the network.

**D2 — Provenance is verified, not assumed.** torchvision calls
`get_state_dict(check_hash=True)` (confirmed in the installed source,
`efficientnet.py:361`), which checks the SHA-256 prefix embedded in the
filename. `test_pretrained_weight_file_is_cached_and_hash_verified` recomputes
the SHA-256 of the cached file and asserts the `7f5810bc` prefix
independently, so a corrupted or substituted cache is caught by the suite, not
only by torchvision's download path.

**D3 — Head left at 1000 classes.** Replacing it here would pre-empt Layer 4
and tempt a hardcoded class count before the dataset exists. The layer instead
pins down the *facts* Layer 4 needs: the head is `Dropout(0.2) → Linear(1280, 1000)`,
and `summarize()` already distinguishes total from trainable parameters so a
frozen-backbone configuration can be verified later.

**D4 — Device agreement tolerance 1e-4.** Measured max |CPU − MPS| on random
input was 1.8 × 10⁻⁶ across 2000 logits; 1e-4 leaves headroom for other
backends without hiding a real numerical fault.

**D5 — Train-mode stochasticity asserted, not assumed.** EfficientNet-B0 has
dropout (p = 0.2) in the head and stochastic depth in the blocks. The suite
asserts that two `train()` passes on the same input differ, that they differ
from `eval()`, and that they are reproducible under `set_seed` — the three
properties Layer 7 will rely on.

## 6. Tests

`tests/test_model.py` — 21 tests (1 parametrized × 2 devices here):

Provenance
- weights enum is `IMAGENET1K_V1` from `download.pytorch.org`; 1000 categories
- weight file exists in the hub cache and its SHA-256 starts with the declared `7f5810bc`
- pretrained weights differ from random init; two pretrained loads are identical

Architecture
- returns a torchvision `EfficientNet`
- `summarize` == (EfficientNet, 5,288,548 total, 5,288,548 trainable, 1000 out) and equals the enum's `num_params`
- with `features` frozen, trainable count equals the classifier's parameter count
- stock head is `Dropout(0.2)` → `Linear(1280, 1000)`

Forward pass
- `(2, 3, 224, 224)` → `(2, 1000)`, all finite — every device
- non-CPU backends match CPU logits within 1e-4
- `eval()` is bit-deterministic
- `train()` differs between passes and from `eval()`; reproducible under `set_seed`
- a single image's logits equal its logits inside a batch
- softmax rows sum to 1, all ≥ 0

Input contract
- weight preset transform turns a 640×480 PIL image into `(3, 224, 224)` float32
- 256×320 input accepted; 1-channel input raises; unbatched input raises

Gradients
- after `backward()`, every parameter has a finite gradient

## 7. Commands Executed

```bash
.venv/bin/python -c "from src.model import build_efficientnet_b0; build_efficientnet_b0()"   # download
shasum -a 256 ~/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth
grep -n check_hash .venv/lib/python3.11/site-packages/torchvision/models/efficientnet.py
.venv/bin/python -m pytest -q -rs tests/test_model.py
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
```

## 8. Results

| Property | Value |
|---|---|
| Weight file SHA-256 prefix | `7f5810bc96def8f7…` — matches filename hash |
| Parameters (total / trainable) | 5,288,548 / 5,288,548 |
| Head | `Dropout(p=0.2)` → `Linear(1280 → 1000)` |
| Pretrained build time (cached) | ≈ 0.5 s; first download ≈ 2.7 s |
| max \|CPU − MPS\| logits | 1.8 × 10⁻⁶ |
| Softmax row sums | 1.0 |
| Auto-selected device (smoke) | mps |

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **120 passed, 1 skipped, 0 warnings** (was 99 + 1) |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 34 files unchanged |
| Types | `mypy src app scripts` | no issues in 24 source files |
| Smoke | build → `.to(mps)` → forward → softmax | `(1, 1000)`, sum 1.0 |
| Regression (Layers 0–2) | 99 earlier tests | pass unchanged |

## 9. Bugs Found and Fixed

None. Two `E501` line-length lint failures on the first gate run were
reformatted; no behaviour change.

## 10. Known Limitations

1. **No semantic check of the pretrained weights.** Provenance is verified by
   hash, and pretrained ≠ random-init is verified, but no natural image was
   classified to confirm ImageNet accuracy — the foundation-only rule forbids
   real data here and a synthetic image has no meaningful ImageNet label. The
   hash check is the evidence that these are the official weights.
2. **First build needs network access** to fetch 20 MB from
   `download.pytorch.org`; subsequent builds use the cache. CI runners will
   download on every run until a cache step is added (deferred until CI is
   first exercised).
3. **`src/config.py` still declares an 8-class scaffold** that nothing in
   Layers 0–3 uses. Left for Layer 4 / the dataset inspection to resolve.
4. **Only CPU and MPS exercised.** CUDA agreement is covered by the same test
   but has not been run on CUDA hardware.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `src/model.py` torchvision builder + summary + softmax |
| Imports work | `from src.model import …`; `timm` reference gone |
| Unit tests pass | 21 new tests |
| Integration tests pass | pretrained download → cache → hash → load → forward |
| Smoke tests pass | forward on resolved device, softmax sums to 1 |
| Lint / format / types | ruff, black, mypy clean |
| Runtime behaviour verified | shape, finiteness, determinism, train/eval, device agreement, gradients |
| No blocking bug | none |
| No unexplained error | 0 warnings |
| No regression | Layers 0–2 pass unchanged |
| Documentation updated | this file; README status table |

## 12. Next Layer Prerequisites

Layer 4 (Disease Classifier Head) must:

- add a builder that replaces `classifier[1]` with `Linear(1280, num_classes)` where `num_classes` is a **required parameter**, not a constant
- keep dropout configurable (default the stock 0.2) and initialise the new head deterministically under `set_seed`
- support freezing the backbone (`features`) and verify via `summarize()` that trainable == head parameters
- verify: output `(N, num_classes)` for several `num_classes` values; backbone weights unchanged after head replacement; gradients reach the head (and the backbone only when unfrozen)
- resolve what `src/config.py` should hold about classes: at most a *placeholder* that is explicitly marked as pending dataset inspection
- re-run Layers 0–3 as regression
