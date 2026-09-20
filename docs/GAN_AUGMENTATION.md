# Training-fold-only GAN augmentation — Layer 3 record (Phase 10)

Implementation: `src/gan_augmentation.py`; drivers `scripts/run_gan_fold.py` (one
fold) and `scripts/run_gan_all_folds.py` (every fold + `GAN_MANIFEST.csv` +
verification); verifier `scripts/verify_gan_outputs.py`; configuration
`configs/gan_v2/{default,smoke}.json`; tracked records under `results/v2/gan/`
(`registry.csv`, `GAN_MANIFEST.csv`, `verification.json`, `generation_summary.json`);
proofs `tests/test_gan_isolation.py`, `tests/test_gan_manifest.py`
(+ `tests/test_gan_augmentation.py`); Colab notebook
`notebooks/AquaHealthAI_Layer3_GAN_Colab.ipynb`. Consumer: `src/cv_runner.py`
(`--data-arm with_gan` reads `data/gan/fold_XX/fold_XX_train_gan.csv`, with its
own leakage checks). Images and checkpoints live under `data/gan/` and are
git-ignored.

**Execution rule (Layer 3).** Real GAN training runs on Google Colab CUDA only;
`run_gan_fold.py` / `run_gan_all_folds.py --require-cuda` exit 2 on CPU/MPS. The
local machine runs the unit tests (16 × 16, CPU fixture) and the smoke config.
The GAN is an augmentation *arm* (WITH-GAN vs WITHOUT-GAN in the experiment
matrix), not a replacement for the real data: every WITH-GAN manifest is the
fold's real training rows plus that fold's synthetic rows.

## 1. Is a specific GAN architecture required? — No (verified)

Every supplied document was read for GAN/cGAN/DCGAN/StyleGAN/CycleGAN/WGAN/
"generative adversarial"/"synthetic" on 2026-09-19:

| source | what it says about GAN |
|---|---|
| `ROADMAO/NEW_ROADMPA_ENTIRE HACKATHON.docx` (Phases 12–14) | GAN is a training-data augmentation mechanism, not one of the five models; per fold: training fold → GAN → synthetic training images → hybrid model; validation fold and final test get NO GAN; "GAN should not automatically generate the same number of images for every class … based on your cleaned class distribution". **No architecture named.** |
| `ROADMAO/AquaHealthAI_Phase_Roadmap_and_Code_Locations_2.pdf` (Phase 10) | "GAN Augmentation … For every fold: current training fold → GAN training → synthetic training images … Validation fold receives no GAN-generated images. The frozen final test set … is never used to train the GAN." Planned module `src/gan_augmentation.py`; "**Do not invent the architecture if the source does not specify it.**" **No architecture named.** |
| `Review_dcouments/ai_hackathon_method.docx` (§9, §16, Phase 10) | "GAN augmentation on TRAINING data only"; per-fold generation inside the training portion; test ❌ GAN, validation ❌ GAN-generated samples. **No architecture named.** |
| `AquaHealthAI_Phase1_5_Summary.pdf` | GAN augmentation "strictly within each training fold" is the not-yet-implemented next step. **No architecture named.** |
| Build guides / team workflow / master experiment docs / Student guides | "cGAN" appears only as the item **cut** from the old baseline scope ("cGAN work from the original plan is cut"; "Do not train a GAN/cGAN during the critical first hours"). **No architecture specified.** |
| Research papers (MobileNetV3, EfficientNet-B1, empirical evaluation, SLCAM-AquaNet, DINOv2) | none uses a GAN; "GAN" occurs only inside reference lists (CBIR-GAN, steganography GAN). |

Conclusion: the project **requires** GAN augmentation (Phase 10, training fold
only) and **does not specify** the architecture. The architecture below is the
project's own documented choice; it is attributed to no paper.

## 2. Architecture chosen and why — cDCGAN

Class-conditional DCGAN: DCGAN generator/discriminator blocks (Radford et al.
2015) with the label injected as in Mirza & Osindero 2014 (label embedding
concatenated to the latent in G; one-hot label maps concatenated to the image
in D). Non-saturating BCE-with-logits loss, Adam for both networks.

| item | value (from `configs/gan_v2/default.json`, recorded in every run) |
|---|---|
| architecture | `cDCGAN (class-conditional DCGAN)` |
| latent dimension | 100 |
| resolution | 64 × 64 RGB (tanh output → PNG; the classifier later resizes to 224 through the SAME preprocessing as real images, CLAHE included) |
| generator | z(100) ++ embed(label,100) → ConvT 4×4 (1024) → 8 → 16 → 32 → 64, BatchNorm + ReLU, tanh |
| discriminator | image ++ 8 one-hot label maps → Conv 4/2/1 ×4 (64→128→256→512), BatchNorm + LeakyReLU(0.2) → 1 logit |
| epochs | 30 |
| batch size | 64 |
| optimizer | Adam(lr = 2e-4, betas = (0.5, 0.999)) for generator and discriminator |
| learning rate | 2e-4 |
| seed | 42 (`src/config.py::SEED`) — `set_seed`, seeded DataLoader shuffle, seeded latent generator for generation |
| class conditioning | yes — label embedding (100-d) concatenated to z in G; 8 one-hot label maps concatenated to the image in D; generation asks for a class explicitly |
| output format | PNG, RGB, 64 × 64, tanh output mapped to [0, 255]; `data/gan/fold_XX/train/<class>/synthetic_NNNNN.png` |
| generation count | class-aware per fold (§4), computed from the fold's real class counts: fold 1 = 1,971, fold 2 = 1,806, folds 3–10 = 2,474–2,487 each; **23,616 planned over 10 folds** (largest class of each fold gets 0). Actual counts: `results/v2/gan/registry.csv` |
| one model per | CV fold (10 models), trained on that fold's training rows only |
| dependencies | torch + torchvision only (none added) |

Why this and not more: a training fold holds 3.8–4.3 k images over 8 classes
(≈ 340–1,000 per class) — too few for one model per class, so one conditional
model per fold learns all classes from the fold's pooled data; it trains in
minutes on Apple MPS / a Colab T4; and it is honest about what it is: at this
data size a DCGAN produces low-detail 64 × 64 images, so WITH-GAN is an
**ablation arm** against WITHOUT-GAN in the experiment matrix, not a claim of
photo-realism. StyleGAN2-ADA would suit few-shot data better but needs an
external dependency and far more compute, and no document asks for it.

## 3. Isolation — how "training data only" is enforced (not just intended)

```
data/audit/cv_v2/fold_XX_train.csv  ──►  FoldTrainingImages  ──►  cDCGAN (fold XX)  ──►  data/gan/fold_XX/train/<class>/synthetic_NNNNN.png
                                          │  aborts if any offered id ∈ forbidden
forbidden ids = fold_XX_validation ∪ final_test (built from the digest-verified manifests; the files are read for ids only, never as images)
```

1. `read_folds` / `read_split` verify `folds.sha256` and `split_manifest.sha256` before anything is read.
2. `forbidden_ids_for_fold` = this fold's validation ids ∪ all final-test ids; `FoldTrainingImages` raises before decoding a single image if any training id is in that set (`forbidden_ids_checked` is stored in the run record).
3. Synthetic files may only be written under a path containing `train` and `fold_XX` and none of `validation | val | test | final_test` (`assert_train_only_path`, `validate_synthetic_rows`); a checkpoint trained on fold *k* refuses to generate for fold *j*.
4. Synthetic ids are `gan-fXX-<sha256(path)[:12]>` — disjoint from every real id; in the WITH-GAN manifest they carry `fold = -1` (never a validation sample), `group_id = image_id`, `source_dataset = gan`.
5. The consumer (`src/cv_runner.py`) re-checks all of this when it loads a WITH-GAN manifest (`LeakageError` on any synthetic row outside `data/gan/fold_XX/train/`, in validation or in test).
6. Smoke runs (`--smoke`) write under `data/gan/smoke/` and `results/v2/smoke/` so the real fold directories and the real registry cannot be touched by an infrastructure check.

## 4. Class-aware amount policy (`plan_synthetic_counts`)

From the **actual** class distribution of the fold's training rows: every class
is topped up to the fold's largest class, capped at 1.0 × its real count; a
class already at the target gets 0. This follows the roadmap (Phase 14): raise
under-represented classes, never the same number for every class. Explicit
per-class counts (`synthetic_per_class`) or a fixed target
(`target_per_class`) can override it in the config; both are recorded.

## 5. Provenance written for every run

| file | content |
|---|---|
| `data/gan/fold_XX/gan_run.json` | architecture, config, seed, device, optimizer, cache flag, real images seen + per class, forbidden ids checked, epochs/batches run, final losses, seconds, checkpoint + SHA-256, source training manifest + SHA-256, torch version, start time |
| `data/gan/fold_XX/synthetic_manifest.csv` | one row per synthetic image: image_id, fold, class, label, path, index, seed, generator checkpoint + SHA-256, architecture |
| `data/gan/fold_XX/fold_XX_train_gan.csv` | WITH-GAN training list (real rows `synthetic=False` + synthetic rows `synthetic=True`); the WITHOUT-GAN arm is `fold_XX_train.csv` itself |
| `data/gan/fold_XX/summary.json` | counts, paths, config file, smoke flag |
| `results/v2/gan/registry.csv` (tracked) | one line per fold: architecture, latent_dim, resolution, epochs, batch_size, optimizer, learning_rate, betas, seed, device, **GPU name, CUDA version**, real images + per class, generated count + per class, source training fold + SHA-256, forbidden ids checked, generator SHA-256, synthetic/with-GAN manifest SHA-256s, torch version, start, seconds, smoke flag |
| `results/v2/gan/GAN_MANIFEST.csv` (tracked) | **one line per synthetic image across every fold**: `synthetic_id`, `fold`, `class`, `label`, `generation_seed`, `generation_config` (JSON of the GANConfig), `config_file`, `training_source` (`data/audit/cv_v2/fold_XX_train.csv`), `training_source_sha256`, `path`, `sha256` (of the PNG), `width`, `height`, `format`, `source_dataset` (= `gan`), `generator_checkpoint`, `generator_sha256`, `architecture`, `device`, `gpu_name`, `generated_at`, `index` — built by `build_gan_manifest` from the per-fold manifests and run records; a row whose generator digest differs from its fold's run record is refused |
| `results/v2/gan/verification.json` (tracked) | the `verify_gan_outputs` report (§6b) over every completed fold |
| `results/v2/gan/generation_summary.json` (tracked) | device, GPU, CUDA, torch version, config file, folds requested/completed, total images, per-fold status / training seconds / wall seconds |

## 6. Tests (`tests/test_gan_isolation.py`)

| # | proof | test |
|---|---|---|
| 1 | test IDs never enter GAN | `test_1_test_ids_never_enter_gan` — every final-test id is forbidden for every fold; none is in the GAN's rows; a smuggled test id aborts before decoding |
| 2 | validation IDs never enter GAN | `test_2_validation_ids_never_enter_gan` — same for each fold's validation ids; the GAN input is exactly the other folds' rows |
| 3 | generated data belongs only to the training fold | `test_3_generated_data_belongs_only_to_training_fold` — paths under `fold_XX/train`, ids disjoint from all real ids, WITH-GAN manifest = real train + synthetic, `fold=-1`, validation/test manifests digest-intact |
| 4 | fold isolation holds | `test_4_fold_isolation_holds_across_folds` — three folds generated: no shared paths/ids, no cross-fold rows in any WITH-GAN manifest, checkpoint of fold *k* refuses fold *j*, every PNG under exactly one fold |
| 5 | generated images have valid dimensions | `test_5_generated_images_have_valid_dimensions` — every PNG opens, RGB, `image_size × image_size`, values in [0, 255], matches the run record |
| 6 | generated files are traceable | `test_6_generated_files_are_traceable` — manifest → checkpoint SHA-256, run record → source manifest SHA-256, registry fields and digests, replace-not-duplicate, committed configs load into the same dataclass |
| 7 | raw data remains unchanged | `test_7_raw_data_remains_unchanged` — SHA-256 + size of every raw file and every audit manifest identical after a run; nothing new outside `data/gan` |
| — | class-aware policy | `test_policy_never_generates_the_same_count_for_every_class` |

`tests/test_gan_manifest.py` (Layer 3):

| requirement | test |
|---|---|
| class mapping | `test_class_mapping_label_index_equals_class` — label index ↔ `CANONICAL_CLASSES` ↔ class directory for every synthetic row and in `GAN_MANIFEST.csv`; a mismatched label is rejected |
| manifest consistency | `test_gan_manifest_lists_every_synthetic_image_with_provenance` — every image of every fold with seed, config, training source + digest, path, PNG digest, generator digest, dimensions, timestamp; unique ids/paths; foreign generator refused |
| isolation proven from files | `test_verifier_passes_on_clean_outputs_and_catches_tampering` — clean outputs VERIFIED; a validation id in a WITH-GAN manifest, a replaced PNG, a PNG under `validation/`, an unlisted PNG and a synthetic id shared by two folds are each caught |
| raw-data / source protection | `test_verifier_detects_changed_training_source` — run record digest ≠ fold manifest, and an edited fold manifest failing the digest-guarded readers, are caught |
| fold isolation + resume | `test_all_folds_driver_generates_skips_complete_and_verifies` — the driver generates the missing folds, keeps a complete fold byte-identical, writes registry / `GAN_MANIFEST.csv` / `verification.json` / `generation_summary.json`, is idempotent; the standalone verifier agrees and fails when a registry line is missing |
| CUDA-only rule | `test_require_cuda_refuses_cpu_and_mps` — both drivers exit 2 with `--require-cuda` off CUDA and write nothing |

### 6b. `scripts/verify_gan_outputs.py` — what "verified" means for generated data

Per fold: run record is for the fold; generator SHA-256 == run record; training source is
`fold_XX_train.csv` and its digest is unchanged; forbidden ids checked == |validation| + |final test|;
real images seen == training rows; synthetic rows valid (fold, class, label, train-only path, file
present); ids unique and disjoint from every real id; class directory == class; no PNG outside
`fold_XX/train`; PNGs on disk == manifest rows; PNGs open as RGB at the recorded size; WITH-GAN
manifest == real training ids + synthetic ids, with no validation and no final-test id; registry
digests / counts / device agree. Across folds: no shared id or path; label index == class.
`GAN_MANIFEST.csv`: rows == union of the per-fold manifests; classes, labels, sources, dimensions
consistent; image digests match the files. Also: split and fold manifests still verify against
their digests. Exit 0 = VERIFIED, 1 = NOT VERIFIED; report in `results/v2/<gan|smoke>/verification.json`.
Raw-image integrity after generation is re-proven by `scripts/colab_dataset.py verify` (Layer 2).

## 7. Procedure on Colab (`notebooks/AquaHealthAI_Layer3_GAN_Colab.ipynb`)

1. Clone `develop`, install the pinned requirements; `gpu_smoke.py --require-cuda`,
   `colab_preflight.py --env-only --require-cuda` (CUDA, GPU, VRAM, torch version).
2. `AQUAHEALTH_COLAB_DATASET_ROOT` → extracted bundle; `colab_dataset.py attach`;
   `colab_dataset.py verify --hash all` (5,942 images, manifest / split / fold digests).
3. **Smoke**: `run_gan_fold.py --fold 1 --config configs/gan_v2/smoke.json --smoke --require-cuda`
   (1 epoch, 256 images, 16 images) → `verify_gan_outputs.py --smoke --images all` → grid inspection.
   Never counted as a result.
4. **Full**: `run_gan_all_folds.py --config configs/gan_v2/default.json --require-cuda --images all`
   (10 folds, resumable) → `verify_gan_outputs.py --images all`, `colab_dataset.py verify --hash sample`.
5. Export `results/v2/gan/*` + per-fold records + PNGs (+ generators) as a tar to Drive; extract at the
   repository root locally; `verify_gan_outputs.py --images all` locally; commit `results/v2/gan/`.

### 7a. Local run before the Layer 3 rule (NOT a Layer 3 result)

Before this layer's CUDA-only rule was enforced, fold 1 was trained locally on Apple MPS with
`default.json` (30 epochs, 3,811 real images, 1,971 synthetic, 531 s, generator SHA-256
`ccaa67d3…`, started 2026-09-19T23:44). It proved the real-data path end to end but violates the
"all real GAN training on Colab" rule, so its registry line was removed from `results/v2/gan/registry.csv`
(kept only as an untracked backup) and its outputs are not used; the Colab run of fold 1 replaces it.

## 8. Runs on the real data (Colab)

NOT VERIFIED until `results/v2/gan/registry.csv`, `GAN_MANIFEST.csv`, `verification.json` and
`generation_summary.json` hold the Colab run; this section is filled from them afterwards.
