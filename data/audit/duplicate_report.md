# Duplicate analysis (Phase 5) — exact SHA-256 and perceptual dHash, across all datasets

Definitions are the repository's established ones (`scripts/build_split_manifest.py`, `results/data_audit.csv`): **exact** = identical SHA-256; **near** = identical 8x8 dHash with different bytes. No other similarity threshold is applied. Nothing was deleted; `dedup_action` in `master_dataset.csv` records what the established policy *would* do.

## Totals

- images inspected: 5298 (5298 decodable, 0 corrupt)
- exact-duplicate groups (SHA-256): 171 — 185 pairs, 89 of them cross-dataset
- near-duplicate groups (dHash, distinct bytes): 540 — 946 pairs, 386 of them cross-dataset
- near-duplicate groups whose mapped members carry conflicting unified labels: 12
- pairs involving an unresolved/excluded label (evidence for Phase 3, no action taken): 104

## Pairs per dataset pair

| dataset A | dataset B | exact pairs | near pairs |
|---|---|---|---|
| current_freshwater | current_freshwater | 80 | 457 |
| current_freshwater | kaptai | 0 | 68 |
| current_freshwater | roboflow | 89 | 288 |
| current_freshwater | paper_dataset | 0 | 3 |
| kaptai | kaptai | 16 | 1 |
| kaptai | roboflow | 0 | 26 |
| kaptai | paper_dataset | 0 | 0 |
| roboflow | roboflow | 0 | 65 |
| roboflow | paper_dataset | 0 | 1 |
| paper_dataset | paper_dataset | 0 | 37 |

## Recorded `dedup_action` per image (annotation only)

| action | images |
|---|---|
| KEEP | 3906 |
| KEEP_GROUPED | 1166 |
| DROP_EXACT_DUPLICATE | 178 |
| EXCLUDE_LABEL_CONFLICT | 47 |
| REVIEW_LABEL_CONFLICT | 1 |

## Near-duplicate evidence for unresolved classes

Unresolved/excluded original classes whose images are byte- or dHash-identical to images of a mapped class. This is evidence for the Phase-3 decision, not a mapping.

| unresolved (dataset / class) | matches mapped class | pairs |
|---|---|---|
| kaptai / Argulus | Bacterial Red Disease | 10 |
| kaptai / Argulus | Healthy Fish | 5 |
| kaptai / Broken antennae and rostrum | EUS Disease | 1 |
| kaptai / Redspot | Aeromoniasis | 1 |
| kaptai / Redspot | Bacterial Red Disease | 7 |
| kaptai / Redspot | EUS Disease | 5 |
| kaptai / THE BACTERIAL GILL ROT | Bacterial Gill Disease | 11 |
| kaptai / Tail And Fin Rot | White Tail Disease | 12 |
| paper_dataset / InfectedFish | Aeromoniasis | 3 |
| paper_dataset / InfectedFish | EUS Disease | 1 |

## Informational: pairs at small dHash Hamming distance (≤ 8 bits)

Distance 0 is the established near-duplicate definition above. Distances > 0 are listed only so a reviewer can see how many borderline pairs exist; **no threshold above 0 is established in this project and none is applied.**

| distance | all pairs | cross-dataset pairs |
|---|---|---|
| 0 | 1131 | 475 |
| 1 | 469 | 161 |
| 2 | 290 | 63 |
| 3 | 195 | 22 |
| 4 | 181 | 16 |
| 5 | 207 | 24 |
| 6 | 246 | 40 |
| 7 | 359 | 71 |
| 8 | 485 | 81 |

## Policy applied (as annotation)

- exact duplicates: keep the first by sorted `filepath` (so a `current_freshwater` copy wins over any other dataset's copy, then `kaptai`, `paper_dataset`, `roboflow`), drop the rest — `DROP_EXACT_DUPLICATE`
- near-duplicate groups whose mapped members disagree on the unified label: `EXCLUDE_LABEL_CONFLICT` (unmapped members of such a group: `REVIEW_LABEL_CONFLICT`)
- other near-duplicate groups: `KEEP_GROUPED` (must land in one split)
- everything else: `KEEP`; undecodable files: `CORRUPT`

Files themselves were not touched. Whether a cross-dataset copy should be kept from the *other* dataset instead (e.g. to favour the higher-resolution original) is a decision the existing policy does not make and is left open.
