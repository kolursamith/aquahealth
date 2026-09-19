# FINAL DATASET PROTOCOL — Layer 1 gate (dataset / split / 10-fold CV)

Every number below was **measured on 2026-09-19** from the filesystem and the committed
artefacts on `develop` (`ebad68f`), not copied from earlier reports. Nothing was trained; no raw
file was modified. Measurement script output: reproduced by the commands in §9.

Raw dataset root (outside git): `/Users/sogo/genai/TEMP/Projects/AI_TECHTAHON_DOCUMENTS/Dataset`
(re-packed delivery: parts under `Dataset/Dataset/…`), reached through the symlinks
`data/raw/<key>` created by `scripts/link_raw_datasets.py`.

---

## 1. Raw images — five sources, counted on disk

| key | delivered folder | images on disk | structure | extensions |
|---|---|---|---|---|
| `current_freshwater` | `Fresh_water_disease/` | **3,503** | `train_split/` 3,200 (8 class folders × 400) + `test_split/` 303 flat files labelled by `test.csv` (303 rows) | .jpg 3,453 · .jpeg 30 · .png 20 |
| `kaptai` | `Fresh Water Fish Dataset/` | **133** | 7 class folders (EUS 26, Redspot 31, Argulus 23, THE BACTERIAL GILL ROT 6, Broken antennae and rostrum 7, Tail And Fin Rot 9, Healthy Fish 31) | .jpg 131 · .png 2 |
| `mendeley` | `MatsyaDx-BD/` | **2,137** | 4 class folders (EUS 332, Bacterial Gill 275, Bacterial Red 653, Healthy 877) + `metadata.csv` | .jpg 2,137 |
| `paper_dataset` | `SalmonScan/` | **1,208** | `InfectedFish/` 752, `FreshFish/` 456 | .png 1,208 |
| `roboflow` | `Fish Disease.v1i.folder/` | **454** | vendor `train/` 365, `valid/` 44, `test/` 45 | .jpg 454 |
| **total** | | **7,435** | | |

`master_dataset.csv` has 7,435 rows = the on-disk count; **0 corrupt** (every file decoded).

**Why `current_freshwater` is 3,503 (measured, not assumed):** 3,200 images in `train_split`
(exactly 400 per class in 8 folders) + 303 images in the flat `test_split` folder, whose labels
come from `test.csv` (303 rows, columns `filename,label`). 1,936 of the 3,200 training files carry
`aug` in the file name — the vendor shipped pre-augmented copies; they are counted as raw images
and flagged (`pre_augmented_filename`, §5), never treated as independent specimens.

## 2. Label harmonisation (`label_mapping.csv`, 27 rows)

| source | EXACT_MATCH | SUPPORTED_MAPPING | UNRESOLVED | EXCLUDED |
|---|---|---|---|---|
| current_freshwater | 3,503 | | | |
| kaptai | 57 | | 69 | 7 |
| mendeley | 877 | 1,260 | | |
| paper_dataset | | | 1,208 | |
| roboflow | 454 | | | |

Canonical classes (`src/manifest.py::CANONICAL_CLASSES`, index = model label):
`0 Bacterial Red Disease · 1 Aeromoniasis · 2 Bacterial Gill Disease · 3 EUS Disease ·
4 Saprolegniasis · 5 Parasitic Disease · 6 White Tail Disease · 7 Healthy Fish`.
UNRESOLVED classes (SalmonScan *InfectedFish*/*FreshFish*, kaptai *Redspot*/*Argulus*/*Tail And Fin
Rot*/*Broken antennae*) are **never accepted**; `review_status` marks them "ambiguous (requires
review)". `paper_dataset` therefore contributes 0 clean images.

## 3. Duplicates (SHA-256 exact, dHash perceptual)

| quantity | measured |
|---|---|
| distinct SHA-256 among 7,435 files | 7,257 |
| exact-duplicate groups (SHA shared by >1 file) | 171 (178 surplus copies) |
| exact groups spanning two datasets | 88 |
| perceptual (dHash) groups with different bytes | 693 (1,627 images) |
| perceptual groups spanning two datasets | 211 |
| `duplicate_report.csv` pairs | 1,405 = 185 sha256 + 1,220 dhash; cross-dataset 475; label relation SAME 1,247 / UNRESOLVED 104 / CONFLICT 54 |
| label-conflicting perceptual groups | 12 groups, 48 images |

Policy applied (`src/dataset_cleaning.py`): exact copies collapse to the first eligible path
(162 rows `EXACT_DUPLICATE`, each with `representative_image_id`; the other 16 surplus copies
are excluded under a label reason that takes precedence); perceptual duplicates are **kept** but
bound into one `group_id`; label-conflicting groups leave the corpus entirely (47 rows
`LABEL_CONFLICT_GROUP`; the 48th image is UNRESOLVED and excluded under that reason). No label is
ever "chosen" for a conflicting pair.

## 4. Clean manifest — `data/audit/clean_manifest.csv` (SHA-256 `84f90bb2…725b`)

| check | result |
|---|---|
| rows | 7,435 (one per raw image) |
| included | **5,942** |
| excluded | 1,493 = LABEL_UNRESOLVED 1,277 · EXACT_DUPLICATE 162 · LABEL_CONFLICT_GROUP 47 · LABEL_EXCLUDED 7 |
| excluded by source | current_freshwater 102 (74 exact, 28 conflict) · kaptai 80 · paper_dataset 1,208 · roboflow 103 (88 exact, 15 conflict) · mendeley 0 |
| all 7,435 `filepath`s resolve on disk | yes (0 missing) |
| included labels ∈ canonical classes | yes |
| `group_id` present on every row | yes; 3,301 included groups, 564 multi-image, largest 396 |
| included SHA-256 unique | yes |
| included groups with mixed labels | 0 |
| raw files unchanged | seeded sample (seed 42) of 300 included files re-hashed: **0 mismatches**; tests re-hash further samples |
| manifest change since the split was cut | only two columns added (`leakage_flags`, `preprocessing_status`); every `image_id`, `group_id`, label, path and `included` flag identical — verified field-by-field against `split_v2` and `cv_v2` (0 mismatches) |

**Final class distribution (5,942 included)**

| class | images | sources |
|---|---|---|
| Healthy Fish | 1,440 | current_freshwater 440 · mendeley 877 · roboflow 92 · kaptai 31 |
| Bacterial Red Disease | 1,106 | mendeley 653 · current_freshwater 414 · roboflow 39 |
| EUS Disease | 791 | current_freshwater 437 · mendeley 332 · kaptai 22 |
| Bacterial Gill Disease | 768 | current_freshwater 439 · mendeley 275 · roboflow 54 |
| Aeromoniasis | 485 | current_freshwater 441 · roboflow 44 |
| Parasitic Disease | 461 | current_freshwater 424 · roboflow 37 |
| White Tail Disease | 449 | current_freshwater 409 · roboflow 40 |
| Saprolegniasis | 442 | current_freshwater 397 · roboflow 45 |

**Source distribution:** current_freshwater 3,401 · mendeley 2,137 · roboflow 351 · kaptai 53 ·
paper_dataset 0. Four classes come from `current_freshwater` + `roboflow` only.

## 5. Leakage audit (`leakage_report.*`, re-measured)

| finding | measured | status |
|---|---|---|
| target leakage — label in path/filename/metadata | folder = class in all sources; filename reveals label for 3,793 included images (100 % current_freshwater & roboflow, 47 % kaptai, 0 % mendeley); `mendeley/metadata.csv` carries the label | **not a leak in the pipeline**: `src/dataset.py` decodes pixels only; tests guard it. Would become a leak if any loader read names. |
| duplicate leakage | 160 included images in exact-dup groups (kept copy), 1,429 in perceptual groups, 527 in cross-dataset groups | **resolved by group_id** and enforced (0 group crossing in split and folds, §6–7) |
| same-specimen leakage (MatsyaDx-BD) | 2,137 mendeley images carry `specimen_id`: 86 specimen ids → 79 groups (dHash chaining merges some specimens; largest groups 396/259/222 are Healthy Fish) | **resolved where ids exist**: 0 specimens straddle dev/test or any fold. For the other four sources no specimen id exists — only exact/perceptual screening is possible (documented limitation). |
| resolution ↔ class shortcut | resolution alone predicts the class with an in-sample upper bound **0.3514 vs 0.2423 majority** (38 distinct resolutions; e.g. Aeromoniasis mostly 128×128, MatsyaDx classes only 4000×3000/3000×4000) | **NOT FIXED.** All images are resized to 224×224 but sharpness/compression traces survive. Mitigation in place is methodological only: (class, source)-stratified partitions and per-source evaluation. A resolution-only control classifier is the proposed quantification. Do not report this as removed. |
| EXIF orientation | 658 mendeley images stored rotated (orientation tag ≠ 1); loader does not `exif_transpose` | **open decision**, unchanged (documented; not silently altered) |
| vendor splits | 200 groups straddle the vendors' own train/valid/test folders | vendor folders ignored entirely |

## 6. Development / final-test split — `data/audit/split_v2/`

**Ratio — project decision.** No supplied document fixes a ratio for the new experiment
(`data/audit/split_policy_proposal.md` §1: the roadmap says "the exact percentage can be decided";
the team method note gives **80/20 development/test as an example**). The experiment used
`test_ratio = 0.20`, seed 42. **This document records 80/20 as the project decision**; it is not
changed here. Because whole groups are indivisible (largest 396), the realised share is 21.6 %.

| check | measured |
|---|---|
| unit / stratum | `group_id` / (unified class, majority source of the group) — `src/split_v2.py::assign_groups` |
| seed | 42 (`src/config.py::SEED`) |
| development | **4,658** images, 2,647 groups (78.4 %) |
| final_test | **1,284** images, 654 groups (21.6 %) |
| dev ∪ test == included rows | yes; excluded rows never enter |
| groups straddling dev/test | **0** |
| specimens straddling dev/test | 0 |
| every class in both partitions | yes — dev/test: Aeromoniasis 388/97 · Gill 612/156 · Red 879/227 · EUS 630/161 · Healthy 1,068/372 · Parasitic 369/92 · Saprolegniasis 353/89 · White Tail 359/90 |
| sources dev/test | current_freshwater 2,719/682 · mendeley 1,614/523 · roboflow 281/70 · kaptai 44/9 |
| `label` index == canonical index | yes, all 5,942 rows |
| digests `split_manifest.sha256` | development.csv `a2e9c500…`, final_test.csv `1a2f6aea…` — **both verify** |
| reproducibility | `build_dev_test_split(read_clean_manifest(...), test_ratio=0.2, seed=42)` on the **current** manifest reproduces the committed assignment **exactly** |

`final_test.csv` is frozen: `read_development()` verifies the digest and never opens it; the test
ids are read only to assert they are absent from every training/validation manifest.

## 7. 10-fold cross-validation — `data/audit/cv_v2/` (seed 42, on development only)

| check | measured (all 10 folds) |
|---|---|
| every development image validates exactly once | yes (folds.csv 4,658 rows == development ids) |
| groups in >1 fold | **0** |
| specimens in >1 fold | **0** |
| final-test ids inside any fold file | **0** |
| per fold: train/val id overlap · group overlap · specimen overlap | 0 · 0 · 0 for folds 01–10 |
| per fold: validation == fold rows, train ∪ val == development | true for folds 01–10 |
| digests `folds.sha256` (21 files) | all verify |
| reproducibility | `assign_folds(development, n_folds=10, seed=42)` on the current manifest reproduces `folds.csv` **exactly** |

| fold | train | validation | validation classes (Aero/Gill/Red/EUS/Healthy/Para/Sapro/WhiteTail) | validation sources (cf/kaptai/mendeley/roboflow) |
|---|---|---|---|---|
| 01 | 3,811 | 847 | 40/64/92/96/443/39/36/37 | 278/5/532/32 |
| 02 | 4,028 | 630 | 40/63/81/67/269/37/36/37 | 277/5/319/29 |
| 03 | 4,245 | 413 | 39/62/90/67/45/37/36/37 | 272/6/105/30 |
| 04 | 4,247 | 411 | 39/62/89/66/45/37/36/37 | 275/3/103/30 |
| 05 | 4,255 | 403 | 39/60/88/63/45/37/35/36 | 268/8/99/28 |
| 06 | 4,259 | 399 | 39/60/88/59/45/37/35/36 | 272/5/95/27 |
| 07 | 4,258 | 400 | 39/63/88/58/44/37/35/36 | 275/2/97/26 |
| 08 | 4,269 | 389 | 38/59/86/56/44/36/35/35 | 268/5/90/26 |
| 09 | 4,274 | 384 | 38/60/89/48/44/36/35/34 | 266/3/87/28 |
| 10 | 4,276 | 382 | 37/59/88/50/44/36/34/34 | 268/2/87/25 |

Fold sizes are uneven (847 … 382) because the three MatsyaDx Healthy-Fish super-groups
(396, 259, 222 images) are indivisible and land in folds 1–2. This is the price of zero specimen
leakage and is accepted; Macro-F1 (not accuracy) is the selection metric for that reason.

## 8. CLAHE — verified from code (`src/preprocessing.py`), not rewritten

| property | value in code |
|---|---|
| colour space | LAB (`cv2.COLOR_RGB2LAB`), CLAHE applied to channel 0 = **L** only; A/B untouched; back via `COLOR_LAB2RGB` |
| clip limit | `CLAHEConfig.clip_limit = 2.0` |
| tile grid | `tile_grid_size = 8` → (8, 8) |
| implementation | OpenCV `cv2.createCLAHE` (opencv-python-headless 5.0.0.93) |
| placement | **first stage** of `build_eval_transform`, on the full-resolution decoded RGB image, before Resize(256, bicubic) → CenterCrop(224) → PILToTensor → float32 [0,1] → ImageNet Normalize |
| default | off (`PreprocessConfig.clahe=None`); enabled by `configs/preprocess_v2_clahe.json`, referenced by `configs/cv_v2/default.json` |
| raw files | CLAHE is applied in memory on the fly (`preprocessing_status` column); `clahe_validation.csv` 24 samples, all `ok`; test `test_clahe_does_not_modify_raw_files` |

## 9. Tests and how to re-measure

Dataset-layer + Phase-12 structural tests on this tree: **145 passed, 1 skipped, 0 failed**
(`tests/test_dataset_layer_verification.py test_dataset_cleaning.py test_multi_dataset.py
test_split_v2.py test_leakage_and_clahe.py test_frozen_manifest.py test_manifest.py
test_dataset_scripts.py test_cv_runner.py test_hybrid_models.py test_gan_augmentation.py`);
ruff, black and `mypy src scripts` clean. Re-measure with `scripts/audit_leakage.py`,
`scripts/build_dev_test_split.py --check` semantics as encoded in `src/split_v2.py::validate_split`
/ `validate_folds`, and the verification tests above.

## 10. Gate result

No critical isolation failure: 0 group / specimen / test-id crossings anywhere; digests verify;
split and folds reproduce from the current manifest. Open, documented, **not** fixed:
resolution↔class shortcut, EXIF orientation decision, 1,277 UNRESOLVED labels.

**LAYER 1 = PASS**
