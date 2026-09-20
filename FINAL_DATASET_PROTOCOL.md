# FINAL DATASET PROTOCOL — dataset configuration v3 (four sources; MatsyaDx-BD/Mendeley excluded)

**Dataset configuration v3 excludes MatsyaDx-BD/Mendeley** (project decision, 2026-09-20).
Measured on 2026-09-20 by re-running the Phase 1–9 scripts on `develop` against the
local delivery drop with the `mendeley` source removed from
`src/multi_dataset.py::DATASET_SOURCES` (kept in `EXCLUDED_SOURCES`). Methodology,
taxonomy, label mapping, duplicate definitions, split rule and seed are unchanged;
no count below was edited by hand. Full record: `data/audit/DATASET_CONFIG_V3.md`;
the v2 artefacts are archived unchanged under `data/audit/archive/v2_mendeley/`.

## v3.1 Raw — four sources, 5,298 images (`dataset_inventory.csv`)

| key | delivery folder | images | corrupt |
|---|---|---|---|
| current_freshwater | `Fresh_water_disease/` (3,200 train_split + 303 test_split) | **3,503** | 0 |
| kaptai | `Fresh Water Fish Dataset/` | **133** | 0 |
| roboflow | `Fish Disease.v1i.folder/` | **454** | 0 |
| paper_dataset | `SalmonScan …/SalmonScan/` | **1,208** | 0 |
| **total** | | **5,298** | 0 |

## v3.2 Clean corpus (`clean_manifest.csv`, `dedup_report.json`)

**Included: 3,805** (current_freshwater 3,401 · kaptai 53 · roboflow 351 · paper_dataset 0).
Excluded 1,493: unresolved label 1,277 (paper_dataset 1,208 + kaptai 69) · exact duplicate 162 ·
label-conflict group 47 · excluded class 7. Exact duplicate groups 171 (88 cross-dataset);
near-duplicate groups 540 (211 cross-dataset); label-conflict groups 12 (48 images); corrupt 0.
Per class: Aeromoniasis 485 · Bacterial Gill 493 · Bacterial Red 453 · EUS 459 · Healthy 563 ·
Parasitic 461 · Saprolegniasis 442 · White Tail 449.

## v3.3 Leakage (`leakage_report.json`)

3,222 leakage groups (485 multi-image, 1,068 images); 180 groups span datasets; 200 span the
vendors' own splits; largest group 6; 0 included groups with mixed labels; specimen ids: none
in the active sources. Resolution-shortcut upper bound 0.271 vs 0.148 baseline (documented,
open). EXIF orientation tags: 27 kaptai images (decision open, unchanged from v2).

## v3.4 Frozen split (`split_v3/`, seed 42, test ratio 0.20, group-aware, class|source strata)

development **3,044** (2,585 groups) · **final_test 761** (637 groups); per-class test share
0.198–0.201; groups straddling partitions 0. Digests (`split_v3/split_manifest.sha256`):
`68b5f8f7…6599  development.csv`, **`baa33034…6599  final_test.csv`** (frozen).

## v3.5 10-fold CV (`cv_v3/`, seed 42)

Validation 315 / 311 / 308 / 308 / 304 / 304 / 303 / 299 / 297 / 295 (train 2,729–2,749);
every class and source in every fold; groups shared with train 0/10 folds. Digests in
`cv_v3/folds.sha256`.

## v3.6 CLAHE (unchanged)

LAB, L channel, clipLimit 2.0, tileGridSize (8, 8), before resize/crop — `configs/preprocess_v2_clahe.json`,
applied inside the transform, never to files on disk.

## v3.7 Known unresolved (carried over, not silently fixed)

resolution/class shortcut; EXIF orientation decision (27 kaptai files); 1,277 unresolved labels
(all of SalmonScan + 69 kaptai).

---

# HISTORICAL — dataset configuration v2 (five sources incl. MatsyaDx-BD), superseded 2026-09-20

The section below is the v2 gate record as measured on 2026-09-19. Its artefacts live under
`data/audit/archive/v2_mendeley/`. Nothing in it applies to v3.

## (v2) FINAL DATASET PROTOCOL — Layer 1 gate (dataset / split / 10-fold CV)

Measured on 2026-09-19 against the filesystem and the committed Phase 1–9
artefacts on `develop` (`ebad68f`). Nothing was trained; no GAN exists; no
raw file, manifest, split or fold assignment was modified by this gate.
Every number below was measured by re-reading the artefacts and the raw
files (script: full re-hash of all 7,435 files, all manifest paths opened,
split and folds re-validated and re-generated from source with the recorded
seed). Where a documented value could not be confirmed it is said so.

Raw dataset root (outside git, symlinked from `data/raw/<key>` by
`scripts/link_raw_datasets.py`):
`/Users/sogo/genai/TEMP/Projects/AI_TECHTAHON_DOCUMENTS/Dataset/Dataset/`
(re-packed delivery: the current dataset's parts sit under `Fresh_water_disease/`).

---

## 1. Raw dataset — five sources, 7,435 images (measured)

| key | delivery folder | images | non-image files | structure | classes (original) |
|---|---|---|---|---|---|
| current_freshwater | `Fresh_water_disease/` | **3,503** | 2 (`test.csv`, `.DS_Store`) | `train_split/<8 class folders>` + flat `test_split/` labelled by `test.csv` | 8 |
| kaptai | `Fresh Water Fish Dataset/` | **133** | 1 | class folders | 7 |
| mendeley | `MatsyaDx-BD …/MatsyaDx-BD/` | **2,137** | 11 (`metadata.csv` …) | class folders + `metadata.csv` (specimen_id, health_condition) | 4 |
| paper_dataset | `SalmonScan …/SalmonScan/` | **1,208** | 1 | class folders | 2 |
| roboflow | `Fish Disease.v1i.folder/` | **454** | 6 (Roboflow READMEs) | `train/valid/test/<class>` | 7 |
| **total** | | **7,435** | | | |

Bytes: 5.72 GB (sum of `file_size` in `master_dataset.csv`; the "12 GB" figure
used in planning is not what is on disk). Corrupt/undecodable: **0**.

**Why current_freshwater is 3,503 (measured, not assumed):**
`train_split/` holds exactly 8 class folders × 400 files = **3,200**
(3,164 `.jpg` + 18 `.jpeg` + 18 `.png`; one `.DS_Store` is not an image), and
`test_split/` holds **303** flat files (289 `.jpg` + 12 `.jpeg` + 2 `.png`)
whose labels come from the 303 rows of `test.csv`. 3,200 + 303 = 3,503.
The delivery has no `validation` folder. 1,853 of the 3,200 train files carry
`aug`/`flip`/`rot` in their names, i.e. the vendor's own augmentation is
already baked into the raw set (see `pre_augmented_filename` flag, §3).

### Duplicates (measured from SHA-256 / dHash over all 7,435 decodable rows)

| quantity | value |
|---|---|
| exact-duplicate SHA-256 clusters (size ≥ 2) | **171** — 349 images, 178 redundant copies |
| exact clusters spanning two datasets | 88 |
| perceptual (dHash) clusters with ≥ 2 distinct byte streams | **693** |
| perceptual clusters spanning two datasets | 211 |
| duplicate pairs listed in `duplicate_report.csv` | 1,405 (185 sha256 pairs, 1,220 dhash pairs; 475 cross-dataset) |
| dHash clusters whose *mapped* members disagree on the label | **12** clusters, 48 images |
| label relation of listed pairs | SAME_LABEL 1,247 · UNRESOLVED_LABEL 104 · LABEL_CONFLICT 54 |

Accounting of the 349 images in exact clusters: 160 kept as the
representative (first eligible path, sorted), **162 excluded as
`EXACT_DUPLICATE`**, 27 excluded earlier for label reasons (11 clusters have no
eligible member at all). 162 + (27 − 11) = 178 redundant copies. ✔

### Exclusions → clean corpus

| exclusion reason | images | by source |
|---|---|---|
| LABEL_UNRESOLVED | **1,277** | paper_dataset 1,208 (FreshFish, InfectedFish — binary labels, no supported mapping), kaptai 69 (Argulus, Redspot, Bacterial gill rot, Tail & fin rot) |
| EXACT_DUPLICATE | **162** | roboflow 88, current_freshwater 74 |
| LABEL_CONFLICT_GROUP | **47** | current_freshwater 28, roboflow 15, kaptai 4 (whole dHash group leaves; no label is picked) |
| LABEL_EXCLUDED | **7** | kaptai 7 (Broken antennae and rostrum — not a disease class) |
| **excluded total** | **1,493** | |
| **clean (included)** | **5,942** | |

Label mapping (`label_mapping.csv`, 28 rows): 18 EXACT_MATCH, 3
SUPPORTED_MAPPING, 6 UNRESOLVED, 1 EXCLUDED; every row carries a
`review_status`; UNRESOLVED is never accepted (test-guarded).
**paper_dataset contributes 0 clean images** until its binary labels are
resolved — a documented open decision, not a silent drop.

### Final class × source distribution of the clean corpus and the frozen split

| unified class | current_freshwater | kaptai | mendeley | roboflow | clean total | development | final_test | test share |
|---|---|---|---|---|---|---|---|---|
| Aeromoniasis | 441 | 0 | 0 | 44 | 485 | 388 | 97 | 0.200 |
| Bacterial Gill Disease | 439 | 0 | 275 | 54 | 768 | 612 | 156 | 0.203 |
| Bacterial Red Disease | 414 | 0 | 653 | 39 | 1106 | 879 | 227 | 0.205 |
| EUS Disease | 437 | 22 | 332 | 0 | 791 | 630 | 161 | 0.204 |
| Healthy Fish | 440 | 31 | 877 | 92 | 1440 | 1068 | 372 | 0.258 |
| Parasitic Disease | 424 | 0 | 0 | 37 | 461 | 369 | 92 | 0.200 |
| Saprolegniasis | 397 | 0 | 0 | 45 | 442 | 353 | 89 | 0.201 |
| White Tail Disease | 409 | 0 | 0 | 40 | 449 | 359 | 90 | 0.200 |
| **total** | 3401 | 53 | 2137 | 351 | **5942** | **4658** | **1284** | **0.2161** |

Canonical label order (`src/manifest.py::CANONICAL_CLASSES`, unchanged from
the baseline): Bacterial Red Disease, Aeromoniasis, Bacterial Gill Disease,
EUS Disease, Saprolegniasis, Parasitic Disease, White Tail Disease, Healthy Fish.

---

## 2. Clean manifest — `data/audit/clean_manifest.csv`

SHA-256 `84f90bb225bb51f7a26e235f96ba0dfd854d2374e5afe397fa5083f29c96725b`,
7,435 rows (one per raw image), 25 columns.

| check | result |
|---|---|
| every `filepath` resolves to a regular file | **7,435 / 7,435**, 0 missing |
| every file re-hashes to its recorded SHA-256 (full re-hash, not a sample) | **7,435 / 7,435**, 0 mismatches → **no raw file was modified** |
| every included row has a unified class in `CANONICAL_CLASSES` | yes (5,942 / 5,942) |
| every row has a non-empty `group_id`; `group_size` consistent | yes |
| excluded exact duplicates name their kept `representative_image_id` | yes (162 / 162) |
| included rows with two labels in one leakage group | **0** |
| two included rows sharing a SHA-256 | **0** |
| leakage groups among included images | 3,301 (564 with > 1 image; largest 396, 259, 222, 59, 41) |
| specimen-bearing groups (mendeley only, 86 specimen ids → 79 groups after dHash chaining) | 79 |
| groups spanning two datasets | 180 |
| `leakage_flags` populated | exact_duplicate_group 349 · near_duplicate_group 1,627 · cross_dataset_group 691 · specimen_group 2,137 · filename_reveals_label 5,240 · pre_augmented_filename 1,937 |

Provenance note: the split (§4) was cut from an earlier revision of this file
(SHA `544ebc79…`, recorded in `split_v2/split_report.json`). The revision now
on `develop` (`84f90bb2…`, commit `ebad68f`) changed only the resolved raw
paths (re-packed delivery) and added the `leakage_flags` /
`preprocessing_status` columns. This gate re-validated the split against the
**current** manifest: identical image-id set, identical `group_id`, identical
`unified_class`, identical `filepath` for all 5,942 rows, and the split
regenerates identically from the current manifest (§4). The frozen files and
their digests were not touched.

---

## 3. Leakage findings (what is and is not fixed)

| # | check | measured | status |
|---|---|---|---|
| 1 | duplicate leakage within a dataset | 162 exact copies excluded; 564 multi-image groups kept whole under one `group_id` | **resolved** — 0 SHA-256 and 0 dHash values shared across the dev/test boundary or any fold boundary (§4, §5) |
| 2 | cross-dataset duplicate leakage | 180 included groups span ≥ 2 sources (roboflow re-exports of current_freshwater, kaptai copies, SalmonScan photos inside current_freshwater); 47 images in label-conflicting groups excluded | **resolved** by group assignment; the label disagreements are evidence of label noise in the sources (human review recommended, unchanged) |
| 3 | same-specimen (same fish) leakage | specimen ids exist only in mendeley (86 ids). Union-find joins `specimen:<id>` with dHash clusters → 79 specimen groups; three chained super-groups of 396 / 259 / 222 *Healthy Fish* images | **resolved where ids exist** — 0 specimens shared across dev/test or across any fold. For current_freshwater, kaptai, roboflow, paper_dataset no specimen id exists; only exact/perceptual screening is possible (**documented limitation, not fixable from metadata**) |
| 4 | target leakage (label outside the pixels) | label is in the folder path for all sources; in file names for current_freshwater 100 %, roboflow 100 %, paper_dataset 62.3 %, kaptai 47.4 %, mendeley 0 %; in `mendeley/metadata.csv` and `current_freshwater/test.csv` | **not a leak in the pipeline** — loaders decode pixels only (`src/dataset.py::load_image`); guarded by tests; would become a leak if any loader read names |
| 5 | EXIF / metadata shortcuts | mendeley: 2,137 / 2,137 EXIF (camera OPPO A52, 658 with orientation ≠ 1); kaptai 27 / 53; others 0 | no metadata reaches the model; orientation handling remains an **open preprocessing decision** (not silently changed) |
| 6 | class ↔ resolution shortcut | 38 distinct resolutions; resolution-only in-sample upper bound **0.3514** vs majority baseline **0.2423** (source-only 0.2425). Aeromoniasis is mostly 128×128; mendeley classes are 4000×3000 / 3000×4000 only | **NOT fixed.** Resizing to 224 does not remove sharpness/compression traces. Mitigation is methodological only: (class, source)-stratified partitions and per-source metrics. A resolution-only control classifier is the proposed quantification and has not been run. |
| 7 | class ↔ source confound | Aeromoniasis, Parasitic, Saprolegniasis, White Tail come from current_freshwater + roboflow only; mendeley covers 4 classes; 59 % of Bacterial Red is mendeley | **documented, not removable** without dropping sources; stratification by (class, majority source) is applied |
| 8 | vendor train/valid/test folders | 200 included groups straddle the vendors' own splits | vendor splits are **ignored entirely** (they leak) |

---

## 4. Development / final-test split — `data/audit/split_v2/`

**Ratio — project decision record.** No supplied document fixes the ratio for
this experiment: the roadmap (Phase 8) says "the exact percentage can be
decided based on your final dataset size/class distribution"; the team method
note gives "for example: 80 % development, 20 % final test"; the 70/15/15 in
the PRD applies to the old 2,400-image baseline only
(`data/audit/split_policy_proposal.md`). The experiment was therefore cut with
`test_ratio = 0.20` (`src/split_v2.py::DEFAULT_TEST_RATIO`), and this document
**records 80/20 development/final-test as the project's experimental
decision**. It is not re-cut here; the achieved share is 0.2161 because whole
groups are indivisible.

| property | value |
|---|---|
| unit of assignment | `group_id` (exact ∪ near-duplicate ∪ same-specimen) — never single images |
| stratum | `<unified class>\|<majority source of the group>` |
| rule | per stratum: shuffle groups with the seed, largest first, each to the part with the largest relative deficit (largest-remainder quotas) — `src/split_v2.py::assign_groups`, the baseline rule of `src/manifest.py` generalised |
| seed | **42** (`src/config.py::SEED`) |
| development | **4,658** images, 2,647 groups (0.7839) |
| final_test | **1,284** images, 654 groups (**0.2161**) |
| groups straddling partitions | **0** |
| specimen ids straddling | **0** |
| SHA-256 values shared | **0** · dHash values shared **0** |
| every class in both partitions | yes (per-class test share 0.200–0.258; Healthy Fish 0.258 because its 396-image group landed in dev and the 259-image group in test) |
| every source in both partitions | yes (current_freshwater 2719/682, kaptai 44/9, mendeley 1614/523, roboflow 281/70) |
| digests (`split_manifest.sha256`) | development.csv `a2e9c500…`, final_test.csv `1a2f6aea…` — **verified intact** |
| reproducibility | `build_dev_test_split(current clean manifest, test_ratio=0.2, seed=42)` reproduces the committed assignment **exactly** (5,942 / 5,942 ids) |

`final_test.csv` is frozen: `read_development` verifies both digests and never
opens it; the builder refuses to overwrite it.

---

## 5. 10-fold cross-validation — `data/audit/cv_v2/`

Source: `development.csv` only (`final_test.csv` never read; **0** test ids in
`folds.csv`). Same rule as §4 with 10 equal quotas, seed **42**; every
development image is the validation sample **exactly once** (4,658 = Σ
validation sizes). `folds.sha256` lists all 21 files — **verified intact**;
`fold_XX_train.csv` / `fold_XX_validation.csv` agree with `folds.csv` for all
10 folds; `assign_folds(development, n_folds=10, seed=42)` reproduces the
committed folds **exactly**.

| fold | train | validation | val share | Aerom. | B.Gill | B.Red | EUS | Healthy | Parasitic | Sapro. | White Tail | current_fw | kaptai | mendeley | roboflow | val groups | specimen groups | largest group | groups ∩ train | specimens ∩ train | SHA ∩ | dHash ∩ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3811 | 847 | 0.182 | 40 | 64 | 92 | 96 | 443 | 39 | 36 | 37 | 278 | 5 | 532 | 32 | 259 | 6 | 396 | 0 | 0 | 0 | 0 |
| 2 | 4028 | 630 | 0.135 | 40 | 63 | 81 | 67 | 269 | 37 | 36 | 37 | 277 | 5 | 319 | 29 | 263 | 6 | 222 | 0 | 0 | 0 | 0 |
| 3 | 4245 | 413 | 0.089 | 39 | 62 | 90 | 67 | 45 | 37 | 36 | 37 | 272 | 6 | 105 | 30 | 262 | 6 | 30 | 0 | 0 | 0 | 0 |
| 4 | 4247 | 411 | 0.088 | 39 | 62 | 89 | 66 | 45 | 37 | 36 | 37 | 275 | 3 | 103 | 30 | 267 | 6 | 29 | 0 | 0 | 0 | 0 |
| 5 | 4255 | 403 | 0.086 | 39 | 60 | 88 | 63 | 45 | 37 | 35 | 36 | 268 | 8 | 99 | 28 | 266 | 6 | 26 | 0 | 0 | 0 | 0 |
| 6 | 4259 | 399 | 0.086 | 39 | 60 | 88 | 59 | 45 | 37 | 35 | 36 | 272 | 5 | 95 | 27 | 267 | 6 | 26 | 0 | 0 | 0 | 0 |
| 7 | 4258 | 400 | 0.086 | 39 | 63 | 88 | 58 | 44 | 37 | 35 | 36 | 275 | 2 | 97 | 26 | 269 | 7 | 24 | 0 | 0 | 0 | 0 |
| 8 | 4269 | 389 | 0.084 | 38 | 59 | 86 | 56 | 44 | 36 | 35 | 35 | 268 | 5 | 90 | 26 | 264 | 6 | 23 | 0 | 0 | 0 | 0 |
| 9 | 4274 | 384 | 0.082 | 38 | 60 | 89 | 48 | 44 | 36 | 35 | 34 | 266 | 3 | 87 | 28 | 264 | 6 | 21 | 0 | 0 | 0 | 0 |
| 10 | 4276 | 382 | 0.082 | 37 | 59 | 88 | 50 | 44 | 36 | 34 | 34 | 268 | 2 | 87 | 25 | 266 | 7 | 21 | 0 | 0 | 0 | 0 |

All 8 classes and all 4 sources are present in every fold's training *and*
validation portion. Fold sizes are deliberately unequal (382–847): the
396-image and 222-image Healthy-Fish super-groups are indivisible and land
whole in folds 1 and 2. Per-fold metrics must therefore be reported with fold
sizes, and fold 1 / fold 2 validation accuracy on *Healthy Fish* is dominated
by one specimen chain each.

---

## 6. CLAHE — verified from code, not rewritten

Implementation: `src/preprocessing.py::CLAHE` (OpenCV `cv2.createCLAHE`).
Configuration actually used by the CV runner:
`configs/preprocess_v2_clahe.json` → `configs/cv_v2/default.json` /
`smoke.json` (`preprocess_config`), loaded by `src/cv_runner.py` into the
existing `CLAHEConfig` / `PreprocessConfig` dataclasses.

| parameter | value (from code / config) |
|---|---|
| colour space | **LAB** (`cv2.COLOR_RGB2LAB`) |
| channel | **L (lightness) only**; A and B untouched |
| clip limit | **2.0** |
| tile grid | **8 × 8** |
| position | **first stage**, on the full-resolution decoded RGB image, **before** resize/crop, in both the eval pipeline (CLAHE → Resize 256 bicubic → CenterCrop 224 → PILToTensor → float32 → ImageNet normalise) and the train pipeline (CLAHE → RandomResizedCrop 224 → flip → rotation → ColorJitter → tensor → normalise) |
| default in `PreprocessConfig` | `clahe=None` (off); the experiment enables it via the config above |
| effect check | `data/audit/clahe_validation.csv`: 12 samples (one per class + extremes), status ok, L-channel std ratio after/before 0.917–1.189, output 3×224×224 float32 |
| raw files | never written (CLAHE runs on the fly; `test_clahe_does_not_modify_raw_files`) |

Note kept from Phase 7: the team method note sketches Resize → CLAHE; the
project applies CLAHE → Resize. Kept as implemented; changing the order is a
decision for approval, not something this gate silently alters.

---

## 7. Reproducibility summary

| artefact | seed | regenerates identically from source |
|---|---|---|
| SHA-256 / dHash / duplicate groups | deterministic | yes (`test_hashes_and_duplicate_groups_are_deterministic`) |
| master → clean manifest | deterministic | yes (`test_master_manifest_regenerates_identically`) |
| dev / final-test split | 42 | yes (this gate, from the current manifest) |
| 10 folds | 42 | yes (this gate, from `development.csv`) |

---

## 8. Gate verdict

Critical isolation checks (all measured): 0 group / specimen / SHA-256 / dHash
crossings on the dev–test boundary and on all 10 fold boundaries; 0 test ids
in the folds; all raw files unchanged. The remaining findings (resolution
shortcut, class ↔ source confound, missing specimen ids outside mendeley,
1,277 unresolved labels, EXIF orientation) are **documented limitations and
open decisions**, not isolation failures.

Tests run for this gate (2026-09-19, `develop`): `tests/test_dataset_layer_verification.py`, `test_dataset_cleaning.py`, `test_multi_dataset.py`, `test_split_v2.py`, `test_leakage_and_clahe.py`, `test_preprocessing.py`, `test_frozen_manifest.py`, `test_manifest.py`, `test_dataset_scripts.py`, `test_image_quality.py` → **135 passed, 1 skipped, 0 failed**.

**LAYER 1 = PASS**
