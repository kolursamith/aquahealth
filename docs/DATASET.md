# Dataset

## Source
Public Kaggle fish-disease image dataset (referenced by paper P1 as `irfanulhuda/fish-disease-detection-dataset`),
delivered as `AI_TECHTAHON_DOCUMENTS/DATASET.zip` (106 MB, 3,504 entries: 3,503 images + `test.csv`). Vendor layout:
`train_split/<8 class folders>/`, flat `test_split/`, `test.csv` (image-level labels only — **no bounding boxes**).
The repository references it read-only through the symlink `data/original/aquahealth`; no image is committed.

## Audit (`scripts/build_split_manifest.py` → `results/data_audit.csv`, `results/data_audit_report.md`)
| Quantity | Value |
|---|---|
| Files discovered | 3,503 (3,453 jpg · 30 jpeg · 20 png) |
| Corrupt / unreadable | 0 |
| Exact-duplicate groups (SHA-256) | 68 (142 images) → 74 duplicate copies dropped |
| Near-duplicate groups (dHash) | 334; 59 straddled the vendor train/test folders (leakage) |
| Label-conflict near-dup groups | 8 → 24 images excluded |
| Resolutions | 640×640: 2,099 · 128×128: 949 · 224×224: 427 · other: 4 (class-correlated: EUS and Healthy are all 640 px, Aeromoniasis 81 % 128 px) |

## Frozen split (`data/split_manifest.csv`, SHA-256 `b7d1fccbb21e73a847e078aed921c02339505bc735199d311103f4d708021d43`)
Seed 42, per-class 70/15/15 largest-remainder quotas, exact duplicates collapsed, near-duplicate groups kept inside a
single split. 3,405 rows; every run verifies the digest.

| Class | train | val | test |
|---|---|---|---|
| Bacterial Red Disease | 291 | 62 | 62 |
| Aeromoniasis | 310 | 66 | 66 |
| Bacterial Gill Disease | 307 | 66 | 66 |
| EUS Disease | 307 | 66 | 66 |
| Saprolegniasis | 278 | 60 | 59 |
| Parasitic Disease | 297 | 64 | 63 |
| White Tail Disease | 286 | 62 | 61 |
| Healthy Fish | 308 | 66 | 66 |
| **total** | **2,384** | **512** | **509** |

## Class mapping (canonical, never changed)
0 Bacterial Red Disease · 1 Aeromoniasis · 2 Bacterial Gill Disease · 3 EUS Disease · 4 Saprolegniasis ·
5 Parasitic Disease · 6 White Tail Disease · 7 Healthy Fish (`src/manifest.py::CANONICAL_CLASSES`).

## Test-split policy
`split == test` was never read by training, tuning or model selection (the runner builds train/val only). It was
evaluated exactly once (`results/final_test/`). Frontend QA used train-split images and derived fixtures only.
