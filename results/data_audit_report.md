# Dataset audit and frozen split

Source (read-only, symlinked): `/Users/sogo/genai/TEMP/Projects/AI_HACKATHON/data/original/aquahealth`

## Totals

- files discovered: 3503
- valid images: 3479
- corrupt/unreadable: 0
- unmapped class folders: 0
- exact-duplicate groups (SHA-256): 68 (142 images involved)
- near-duplicate groups (dHash, distinct bytes): 334
- near-duplicate groups spanning the delivered train_split/test_split: 59 (the delivered test split leaks; superseded by the group-aware split below)
- near-duplicate groups with conflicting labels: 8 (24 images excluded from the split as `label_conflict`)
- unexpected directories: none
- unexpected files: none
- YOLO / bounding-box annotations: none found (only `test.csv` with image labels)

## Class mapping (source folder → canonical class, index)

| Source folder | Canonical class | Label |
|---|---|---|
| Bacterial Red disease | Bacterial Red Disease | 0 |
| Bacterial diseases - Aeromoniasis | Aeromoniasis | 1 |
| Bacterial gill disease | Bacterial Gill Disease | 2 |
| EUS | EUS Disease | 3 |
| Fungal diseases Saprolegniasis | Saprolegniasis | 4 |
| Parasitic diseases | Parasitic Disease | 5 |
| Viral diseases White tail disease | White Tail Disease | 6 |
| Healthy Fish | Healthy Fish | 7 |

## Per-class counts (valid images, before de-duplication)

| Class | train_split | test_split | total |
|---|---|---|---|
| Bacterial Red Disease | 397 | 28 | 425 |
| Aeromoniasis | 397 | 49 | 446 |
| Bacterial Gill Disease | 397 | 44 | 441 |
| EUS Disease | 395 | 53 | 448 |
| Saprolegniasis | 397 | 19 | 416 |
| Parasitic Disease | 399 | 37 | 436 |
| White Tail Disease | 398 | 22 | 420 |
| Healthy Fish | 400 | 47 | 447 |

## Image dimensions (valid images)

- 640x640: 2099
- 128x128: 949
- 224x224: 427
- 224x109: 2
- 260x194: 1
- 116x212: 1

### Resolution by class (shortcut-learning risk)

| Class | 640x640 | 128x128 | 224x224 | other |
|---|---|---|---|---|
| Bacterial Red Disease | 263 | 88 | 74 | 0 |
| Aeromoniasis | 20 | 363 | 61 | 2 |
| Bacterial Gill Disease | 301 | 74 | 66 | 0 |
| EUS Disease | 447 | 0 | 0 | 1 |
| Saprolegniasis | 181 | 142 | 93 | 0 |
| Parasitic Disease | 235 | 145 | 55 | 1 |
| White Tail Disease | 205 | 137 | 78 | 0 |
| Healthy Fish | 447 | 0 | 0 | 0 |

## Extensions

- .jpg: 3453
- .jpeg: 30
- .png: 20

## Frozen split

- seed: 42
- ratios: 0.70 / 0.15 / 0.15 per class; exact duplicates collapsed to one image; near-duplicate groups kept within one split
- images in manifest: 3405
- manifest: `data/split_manifest.csv` (sha256 b7d1fccbb21e73a8…)
- **`split == test` is frozen**: evaluate once, at the end, never for model selection

| Class | train | val | test | total |
|---|---|---|---|---|
| Bacterial Red Disease | 291 | 62 | 62 | 415 |
| Aeromoniasis | 310 | 66 | 66 | 442 |
| Bacterial Gill Disease | 307 | 66 | 66 | 439 |
| EUS Disease | 307 | 66 | 66 | 439 |
| Saprolegniasis | 278 | 60 | 59 | 397 |
| Parasitic Disease | 297 | 64 | 63 | 424 |
| White Tail Disease | 286 | 62 | 61 | 409 |
| Healthy Fish | 308 | 66 | 66 | 440 |
| **total** | 2384 | 512 | 509 | 3405 |

### Resolution by split

| Split | 640x640 | 128x128 | 224x224 | other |
|---|---|---|---|---|
| train | 1437 | 639 | 306 | 2 |
| val | 303 | 151 | 56 | 2 |
| test | 312 | 141 | 56 | 0 |
