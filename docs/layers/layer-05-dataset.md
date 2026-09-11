# Layer 5 — Dataset / DataLoader

## 1. Objective

Turn a directory of class-labelled images into a `Dataset` and `DataLoader`
whose labels are derived from the data, with tooling to audit and split that
data. Validated entirely on generated images whose pixel content encodes
their class, so label correctness is checked against content — not just
counts.

## 2. Architecture

```
<root>/<class_name>/<image>.{jpg,jpeg,png,bmp,webp}

src/dataset.py
  discover_classes(root)            sorted dir names → the class list (data-derived)
  index_samples(root, class_names)  sorted image files → [Sample(path, label)]
                                    unknown dir under root → ValueError
  load_image(path)                  Pillow → RGB; failure → ImageDecodeError naming the file
  ImageFolderDataset(root, *, transform=None, class_names=None)
      .class_names / .class_to_idx  discovered, or supplied to share a mapping across splits
      [i] → (transform(PIL) | pil_to_tensor(PIL), label)
      .targets, .class_counts()
  build_dataloader(dataset, *, batch_size, shuffle, num_workers, seed, drop_last)
      seeded torch.Generator → reproducible shuffle order

scripts/audit_dataset.py   counts / extensions / modes / sizes / corrupt / duplicates; exit 1 on problems
scripts/create_split.py    stratified, seeded copy into train/ val/ test/; refuses to overwrite
```

Both scripts sit on `src.dataset`; there is one indexing implementation.

## 3. Files

**Created**
- `tests/conftest.py` — `write_image_folder` / `make_image_folder` synthetic fixture (colour-per-class)
- `tests/test_dataset_scripts.py` — 12 tests
- `docs/layers/layer-05-dataset.md` — this document

**Modified**
- `src/dataset.py` — rewritten from the scaffold (which hard-wired `config.CLASS_NAMES` and had an unimplemented `__getitem__`)
- `tests/test_dataset.py` — rewritten; the Layer 0 placeholder `test_class_count` (asserting the brief's 8) is gone
- `scripts/audit_dataset.py`, `scripts/create_split.py` — rewritten on `src.dataset`
- `README.md` — build-status table

## 4. Dependencies

None new. `config.BATCH_SIZE` (32) and `config.NUM_WORKERS` (4) are the loader defaults.

## 5. Implementation Decisions

**D1 — Classes come from directory names, sorted.** Nothing in this layer
reads `config.CLASS_NAMES`. `discover_classes` is the only source of the
class list, and `len(dataset.class_names)` is what feeds
`build_classifier(num_classes=…)`. Sorting makes the mapping deterministic
across machines and filesystems.

**D2 — Splits share a mapping explicitly.** `ImageFolderDataset(val_root,
class_names=train.class_names)` reuses the train mapping. A directory under a
split that is *not* in the mapping raises (a silent drop would hide a label
mismatch); a mapped class with no directory is allowed (count 0), since a
small val split may legitimately lack a rare class.

**D3 — No preprocessing decisions here.** Without a `transform`, an item is
the raw uint8 CHW tensor of whatever size the file has. Layer 6 owns the
canonical transform; the test `test_mixed_sizes_without_resize_cannot_be_batched`
documents that the loader alone does not make images batchable.

**D4 — Corrupt files fail loudly at access, naming the file.** Indexing is
by extension only (cheap, no decode); decoding happens in `__getitem__`. A
bad file raises `ImageDecodeError("could not decode image <path>: …")` rather
than being skipped. `audit_dataset.py` exists to find these *before*
training, and exits 1 when it does.

**D5 — Duplicates are an audit failure.** The audit hashes every file
(SHA-256) and reports byte-identical groups, including across classes.
Duplicates straddling a split boundary are leakage; the audit blocks
splitting until they are resolved.

**D6 — `create_split` copies, is seeded, stratifies per class with
`round()`, and refuses to overwrite.** A second run into a populated
destination raises `FileExistsError` so two splits can never be mixed.

**D7 — macOS worker processes need a real `__main__` file.** Discovered while
smoke-testing: a script fed via stdin crashed every worker
(`FileNotFoundError: …/<stdin>`) because the default `spawn` start method
re-imports `__main__` from its path. The pytest worker test passes since
pytest's main is a file. Not a project bug; recorded as a hard constraint
for Layer 7's training entry point (must be a module/file with an
`if __name__ == "__main__":` guard).

## 6. Tests

`tests/test_dataset.py` — 36 tests:

Discovery — sorted; hidden dirs and loose files ignored; missing root → `FileNotFoundError`; no class dirs → `ValueError`

Indexing — `is_image_file` over 10 names (case-insensitive extensions, `.gitkeep`, `.DS_Store`, `.tif` rejected, hidden rejected); non-image files skipped; order deterministic and sorted; unknown class dir rejected; declared class with no dir tolerated

Decoding — L and RGBA → RGB; corrupt file → `ImageDecodeError` naming it

Dataset — length / names / mapping / counts; empty root rejected; item is uint8 CHW + `int`; **every item's mean colour matches its label's colour (±3)**; `targets` matches iteration; transform applied; split mapping shared via `class_names`; unknown split class rejected; corrupt file indexed but fails at access

DataLoader — batch shapes/dtypes incl. partial last batch; `drop_last`; unshuffled preserves order; seeded shuffle reproducible and differs across seeds and is a permutation; **2 worker processes yield identical batches to the main process**; mixed sizes without resize raise at collate; defaults come from config

`tests/test_dataset_scripts.py` — 12 tests:

Audit — counts/extensions/modes/sizes from data, `RESULT: OK`; corrupt file flagged, `RESULT: FAIL`; cross-class byte duplicate flagged; **CLI**: exit 0 + valid `--json`, exit 1 with "Corrupt files: 1"

Split — ratio validation (sum ≠ 1, negative); partition exhaustive/disjoint with expected sizes; stratified counts exactly 7/2/1 per class, source untouched, splits disjoint and exhaustive; reproducible under seed; refuses to overwrite; outputs load with a shared mapping

## 7. Commands Executed

```bash
.venv/bin/python -m pytest -q -rs tests/test_dataset.py tests/test_dataset_scripts.py
.venv/bin/python -m pytest -q -rs
.venv/bin/ruff check .
.venv/bin/black --check .
.venv/bin/mypy src app scripts
.venv/bin/python <scratchpad>/smoke5.py     # file-based smoke, see D7
```

## 8. Results

Smoke (5 synthetic classes × 10 images, 240×180, config defaults):

| Step | Result |
|---|---|
| `discover_classes` | `['Columnaris', 'Dropsy', 'Fin_Rot', 'Healthy', 'White_Spot']` (sorted) |
| loader, batch 32, 4 workers, seeded shuffle | 2 batches: `(32, 3, 224, 224)`, `(18, 3, 224, 224)` in 11.5 s |
| `build_classifier(len(class_names))` on `mps` | logits `(32, 5)` |

The 11.5 s is dominated by `spawn` start-up of 4 workers that each import
torch; see limitation 1.

Gate results:

| Check | Command | Result |
|---|---|---|
| Tests | `pytest -q -rs` | **202 passed, 1 skipped, 0 warnings** (was 155 + 1) |
| Lint | `ruff check .` | All checks passed |
| Format | `black --check .` | 37 files unchanged |
| Types | `mypy src app scripts` | no issues in 24 source files |
| Smoke | file-based end-to-end (above) | data-derived classes → loader → classifier |
| Regression (Layers 0–4) | 154 earlier tests | pass unchanged (the Layer 0 `test_class_count` placeholder was replaced by design) |

## 9. Bugs Found and Fixed

**Scaffold defects replaced (not patched):** the original `src/dataset.py`
derived labels from `config.CLASS_NAMES`, silently skipped unknown
directories, globbed `*.*` (would index `.DS_Store`, `README.md`), and had
`__getitem__` raising `NotImplementedError`. The audit and split scripts
duplicated that indexing logic. All are now one implementation.

One harness issue (D7) was diagnosed to root cause and is not a code defect.

## 10. Known Limitations

1. **Worker start-up cost.** `spawn` boots each worker with a full torch
   import (~10 s for 4 workers here). Per-epoch iteration in Layer 7 should
   use `persistent_workers=True` so this is paid once; `build_dataloader`
   does not expose it yet because nothing iterates more than once.
2. **Extension allow-list is fixed** (`jpg jpeg png bmp webp`). TIFF/GIF are
   excluded on purpose; the real dataset's audit will show whether the list
   must change.
3. **Duplicate detection is byte-exact.** Near-duplicates (re-encoded,
   resized) are not detected; that needs perceptual hashing and a decision
   informed by the real data.
4. **No sampler / class-weighting support.** `targets` is exposed so a
   `WeightedRandomSampler` can be added if the real class distribution is
   imbalanced; not built until that is known.
5. **Audit reads every file twice** (hash + decode). Fine for thousands of
   images; would need a single pass for millions.

## 11. Acceptance Gate

**PASS**

| Criterion | Evidence |
|---|---|
| Implementation exists | `src/dataset.py`; both scripts rewritten on it |
| Imports work | full suite; scripts importable and runnable |
| Unit tests pass | 48 new tests |
| Integration tests pass | split → two datasets with shared mapping; audit CLI subprocess; worker processes |
| Smoke tests pass | file-based end-to-end into the Layer 4 classifier |
| Lint / format / types | ruff, black, mypy clean |
| Runtime behaviour verified | content ↔ label, determinism, shuffle seeding, workers, error paths |
| No blocking bug | none |
| No unexplained error | 0 warnings; the stdin/spawn crash is explained (D7) |
| No regression | Layers 0–4 pass |
| Documentation updated | this file; README |

## 12. Next Layer Prerequisites

Layer 6 (Preprocessing + CLAHE) must:

- replace the `src/preprocessing.py` / `src/augmentation.py` scaffolds (OpenCV + albumentations, both undeclared)
- define **one** canonical eval transform (resize → centre-crop 224 → float → ImageNet normalise) used by training-validation and inference alike, built on `transforms.v2`
- make CLAHE a configurable, off-by-default stage; implement it against numpy/Pillow *or* declare `opencv-python-headless` explicitly with a reason — decide from what the CLAHE implementation actually needs
- keep train-time augmentation separate from, and composable with, the canonical eval transform
- verify: output `(3, 224, 224)` float32 for arbitrary input sizes and modes; eval transform bit-deterministic; train transform seed-reproducible; CLAHE changes the image and is idempotent-configurable; numerics match `EfficientNet_B0_Weights.transforms()` for the non-CLAHE path
- remove the `cv2` skip guard in `tests/test_preprocessing.py` once the module is rewritten
- re-run Layers 0–5 as regression
