# Phase 10 status report — verification before hybrid-model training

Dataset configuration **v3** (MatsyaDx-BD/Mendeley excluded). Generated 2026-09-20 from the
committed artefacts (`data/audit/`, `results/v2/gan/`); no number here was typed by hand.
Gate: everything below must hold before Phase 12 (five hybrid architectures × 10 folds × 2 arms).

## DATASETS

| item | value |
|---|---|
| active source datasets | current_freshwater (`Fresh_water_disease/`), kaptai (`Fresh Water Fish Dataset/`), roboflow (`Fish Disease.v1i.folder/`), paper_dataset (`SalmonScan …/SalmonScan/`); **MatsyaDx-BD excluded** (`EXCLUDED_SOURCES`) |
| total source images | **5,298** (3,503 / 133 / 454 / 1,208), 0 corrupt |
| usable images | **3,805** |
| excluded images | 1,493 — unresolved label 1,277 (SalmonScan 1,208 + kaptai 69) · exact duplicate 162 · label-conflict group 47 · excluded class 7 |
| images per class (usable) | Aeromoniasis 485 · Bacterial Gill Disease 493 · Bacterial Red Disease 453 · EUS Disease 459 · Healthy Fish 563 · Parasitic Disease 461 · Saprolegniasis 442 · White Tail Disease 449 |
| images per source (usable) | current_freshwater 3,401 · kaptai 53 · roboflow 351 · paper_dataset 0 |
| Drive delivery (Colab, `results/v2/drive_delivery.json`) | mapped with the committed `DATASET_SOURCES`: 3,503 / 133 / 454 found (all rows resolve; included 3,401 / 53 / 351); SalmonScan 1,223 found vs 1,208 audited (+15 surplus, 0 included); every one of the 3,805 usable images SHA-256-verified on Colab (`verify --hash all`, 18/18 checks) |

## LEAKAGE (`data/audit/leakage_report.json`)

| item | value |
|---|---|
| exact duplicate groups | 171 (88 cross-dataset) — 162 copies excluded, representatives kept |
| near-duplicate groups (distinct bytes, dHash) | 540 (211 cross-dataset) — kept, bound by `group_id` |
| cross-source duplicate groups (included) | 180 |
| label conflicts | 12 groups / 48 images → 47 excluded (`LABEL_CONFLICT_GROUP`) |
| leakage groups (included corpus) | 3,222 (485 multi-image holding 1,068 images; largest 6; 0 mixed-label) |
| specimen ids | none in the active sources |
| open, documented (unchanged methodology) | resolution→class shortcut upper bound 0.271 vs 0.148 baseline; 27 kaptai EXIF orientation tags; 1,277 unresolved labels |

## SPLITS (`data/audit/split_v3/`, `data/audit/cv_v3/`)

| item | value |
|---|---|
| development | **3,044** images (2,585 groups) |
| frozen final test | **761** images (637 groups) — `final_test.csv` SHA-256 `baa330347a89305212b08ac4c79a1bb6a98cdd7447172f4aa3aea113c16cb7a2`; opened once, at the end; present in no fold manifest (checked on Colab) |
| fold counts (validation) | 315 / 311 / 308 / 308 / 304 / 304 / 303 / 299 / 297 / 295 (train 2,729–2,749); every class and source in every fold; groups shared with train 0/10 |
| seed | 42 |
| split policy | group-aware (exact/near duplicates ∪ same specimen), stratified by unified class \| majority source, largest-remainder quotas, 80/20 development/test, 10-fold on development only; 0 groups straddle any boundary |

## GAN (`results/v2/gan/`, `docs/GAN_AUGMENTATION.md` §8)

| item | value |
|---|---|
| smoke test | PASS on Colab CUDA (fold 1, 1 epoch, 256 images, 16 PNGs; 31/31 checks VERIFIED) |
| folds ready | **10/10** — cDCGAN per fold, 64×64, z=100, Adam 2e-4 (0.5, 0.999), batch 64, 30 epochs, seed 42, `--require-cuda`, Tesla T4 / CUDA 13.0 / torch 2.14.0+cu130 |
| synthetic images planned / generated / failed | 5,004 / **5,004** / 0 (495, 491, 504, 504, 500, 500, 507, 503, 501, 499) |
| isolation | training source = that fold's `fold_XX_train.csv` only (registry digests == `cv_v3/folds.sha256`); forbidden ids (validation + final test) checked per fold, 0 offered; synthetic ids disjoint from every real id; no id/path shared between folds; WITH-GAN manifests = real train ids + synthetic, no validation / final-test id |
| verification | `verify_gan_outputs.py --images all`: 229/229 checks, RESULT: VERIFIED; raw images re-hashed after generation: VERIFIED |
| persistence | `MyDrive/AquaHealth/results_v2_datasetv3/`, `MyDrive/AquaHealth/aquahealth_gan_layer3_a9bc3c8.tar.gz` (689.6 MB: records + PNGs + generators); tracked records in `results/v2/gan/` |
| unofficial, excluded | old local MPS fold-1 run (v2 data) quarantined in `data/gan_archive_unofficial/` |

## REPRODUCIBILITY

| item | value |
|---|---|
| git commit (data + code) | `6fa8e8e` dataset configuration v3; `a9bc3c8` executed on Colab; records committed after |
| clean manifest SHA-256 | `e90b43a0c5a5cf9ea61ec3140c048d1e8ef854874492eaeafaeed04b56c86b68` |
| development / final_test SHA-256 | `68b5f8f7cd60f0fda6ecad2241f27a1fa2af4c0c35ceca9fcfc5ba4d940e6599` / `baa330347a89305212b08ac4c79a1bb6a98cdd7447172f4aa3aea113c16cb7a2` |
| folds.csv SHA-256 | `5cac023f2a1d12d0dd2edb28717c0271a7019d712f834f027d4eb9af83aa4a7f` (per-fold digests in `cv_v3/folds.sha256`) |
| preprocessing config (`configs/preprocess_v2_clahe.json`) SHA-256 | `4c35039a73c0d8740c736de00339c240d405849b885b0231d4688a3508499ab7` — CLAHE LAB-L clip 2.0 tile 8×8 → Resize 256 → CenterCrop 224 → ImageNet normalisation, applied in the transform only |
| device / CUDA | official runs: Colab `cuda`, Tesla T4, CUDA 13.0, torch 2.14.0+cu130; local machine: tests and manifests only |
| seed | 42 |

**Gate result: PASS — Phase 12 (hybrid-model CV training) may begin.**
