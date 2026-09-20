# De-duplication report — clean manifest for the multi-dataset experiment

Source: `data/audit/master_dataset.csv` (hashes reused from `src/manifest.py`, groups from `scripts/build_split_manifest.py`). Result: `data/audit/clean_manifest.csv`. **No raw file was deleted or modified**; exclusion is a manifest flag.

## Policy applied

- exact_duplicates: collapse to the first eligible path (sorted filepath)
- near_duplicates: kept, joined under one group_id (established project policy)
- label_conflicts: whole group excluded; no label chosen
- same_specimen: joined under one group_id where the dataset states a specimen id

## Before → after

| dataset | raw images | excluded | clean (included) |
|---|---|---|---|
| current_freshwater | 3503 | 102 | 3401 |
| kaptai | 133 | 80 | 53 |
| paper_dataset | 1208 | 1208 | 0 |
| roboflow | 454 | 103 | 351 |
| **total** | 5298 | 1493 | **3805** |

## Duplicate groups (all active datasets together)

| quantity | value |
|---|---|
| exact groups | 171 |
| exact groups cross dataset | 88 |
| near groups distinct bytes | 540 |
| near groups cross dataset | 211 |
| conflicting label groups | 12 |
| conflicting label group images | 48 |
| corrupt / unreadable | 0 |

## Exclusions by reason

| reason | images |
|---|---|
| LABEL_CONFLICT_GROUP | 47 |
| EXACT_DUPLICATE | 162 |
| LABEL_UNRESOLVED | 1277 |
| LABEL_EXCLUDED | 7 |

| dataset | EXACT_DUPLICATE | LABEL_CONFLICT_GROUP | LABEL_EXCLUDED | LABEL_UNRESOLVED |
|---|---|---|---|---|
| current_freshwater | 74 | 28 | 0 | 0 |
| kaptai | 0 | 4 | 7 | 69 |
| paper_dataset | 0 | 0 | 0 | 1208 |
| roboflow | 88 | 15 | 0 | 0 |

## Final class distribution (clean corpus)

| unified class | mapped before dedup | clean | current_freshwater | kaptai | paper_dataset | roboflow |
|---|---|---|---|---|---|---|
| Aeromoniasis | 498 | 485 | 441 | 0 | 0 | 44 |
| Bacterial Gill Disease | 502 | 493 | 439 | 0 | 0 | 54 |
| Bacterial Red Disease | 474 | 453 | 414 | 0 | 0 | 39 |
| EUS Disease | 482 | 459 | 437 | 22 | 0 | 0 |
| Healthy Fish | 655 | 563 | 440 | 31 | 0 | 92 |
| Parasitic Disease | 475 | 461 | 424 | 0 | 0 | 37 |
| Saprolegniasis | 465 | 442 | 397 | 0 | 0 | 45 |
| White Tail Disease | 463 | 449 | 409 | 0 | 0 | 40 |

- leakage groups in the clean corpus: 3222 (485 with more than one image; largest 6) — a later split must assign whole groups

## Unresolved / ambiguous

- images awaiting a label decision (`LABEL_UNRESOLVED`): 1277
| dataset / original class | images |
|---|---|
| kaptai/Argulus | 23 |
| kaptai/Broken antennae and rostrum | 7 |
| kaptai/Redspot | 31 |
| kaptai/THE BACTERIAL GILL ROT | 6 |
| kaptai/Tail And Fin Rot | 9 |
| paper_dataset/FreshFish | 456 |
| paper_dataset/InfectedFish | 752 |

- label-conflict group members excluded (47):

| group | dataset | original path | unified class |
|---|---|---|---|
| g2078 | current_freshwater | `test_split/Bacterial gill disease_Bacterial gill disease_33.jpg` | Bacterial Gill Disease |
| g594 | current_freshwater | `test_split/EUS_EUS_156.jpg` | EUS Disease |
| g1688 | current_freshwater | `test_split/EUS_EUS_293.jpg` | EUS Disease |
| g2667 | current_freshwater | `test_split/EUS_EUS_30.jpg` | EUS Disease |
| g594 | current_freshwater | `train_split/Bacterial Red disease/Bacterial Red disease_21.jpg` | Bacterial Red Disease |
| g4281 | current_freshwater | `train_split/Bacterial Red disease/Bacterial Red disease_36.jpg` | Bacterial Red Disease |
| g3395 | current_freshwater | `train_split/Bacterial Red disease/Bacterial Red disease_5.jpg` | Bacterial Red Disease |
| g2667 | current_freshwater | `train_split/Bacterial Red disease/Bacterial Red disease_8.jpg` | Bacterial Red Disease |
| g601 | current_freshwater | `train_split/Bacterial diseases - Aeromoniasis/Bacterial diseases - Aeromoniasis_176.jpg` | Aeromoniasis |
| g3851 | current_freshwater | `train_split/Bacterial diseases - Aeromoniasis/Bacterial diseases - Aeromoniasis_37.jpg` | Aeromoniasis |
| g4281 | current_freshwater | `train_split/Bacterial diseases - Aeromoniasis/Bacterial diseases - Aeromoniasis_aug_132.jpg` | Aeromoniasis |
| g603 | current_freshwater | `train_split/Bacterial diseases - Aeromoniasis/Bacterial diseases - Aeromoniasis_aug_227.jpg` | Aeromoniasis |
| g1688 | current_freshwater | `train_split/Bacterial gill disease/Bacterial gill disease_41.jpg` | Bacterial Gill Disease |
| g2078 | current_freshwater | `train_split/Bacterial gill disease/Bacterial gill disease_aug_147.jpg` | Bacterial Gill Disease |
| g1688 | current_freshwater | `train_split/Bacterial gill disease/Bacterial gill disease_aug_254.jpg` | Bacterial Gill Disease |
| g3614 | current_freshwater | `train_split/EUS/EUS_106.jpg` | EUS Disease |
| g2869 | current_freshwater | `train_split/EUS/EUS_207.jpg` | EUS Disease |
| g3851 | current_freshwater | `train_split/EUS/EUS_214.jpg` | EUS Disease |
| g603 | current_freshwater | `train_split/EUS/EUS_25.jpg` | EUS Disease |
| g2667 | current_freshwater | `train_split/EUS/EUS_31.jpg` | EUS Disease |
| g1688 | current_freshwater | `train_split/EUS/EUS_aug_161.jpg` | EUS Disease |
| g2869 | current_freshwater | `train_split/EUS/EUS_aug_79.jpg` | EUS Disease |
| g2078 | current_freshwater | `train_split/Fungal diseases Saprolegniasis/Fungal diseases Saprolegniasis_82.jpeg` | Saprolegniasis |
| g2078 | current_freshwater | `train_split/Fungal diseases Saprolegniasis/Fungal diseases Saprolegniasis_aug_183.jpg` | Saprolegniasis |
| g2078 | current_freshwater | `train_split/Fungal diseases Saprolegniasis/Fungal diseases Saprolegniasis_aug_317.jpg` | Saprolegniasis |
| g1688 | current_freshwater | `train_split/Parasitic diseases/Parasitic diseases_190.jpg` | Parasitic Disease |
| g3614 | current_freshwater | `train_split/Viral diseases White tail disease/Viral diseases White tail disease_86.jpeg` | White Tail Disease |
| g3614 | current_freshwater | `train_split/Viral diseases White tail disease/Viral diseases White tail disease_aug_105.jpg` | White Tail Disease |
| g2191 | kaptai | `EUS/EUS  (12).jpg` | EUS Disease |
| g601 | kaptai | `EUS/EUS  (2).jpg` | EUS Disease |
| g3395 | kaptai | `EUS/EUS  (5).jpg` | EUS Disease |
| g2667 | kaptai | `EUS/EUS  (7).jpg` | EUS Disease |
| g594 | roboflow | `test/Bacterial Red disease/Bacterial-Red-disease-7-_jpeg.rf.1d666a612fa8b59f6b53652808f56861.jpg` | Bacterial Red Disease |
| g3395 | roboflow | `train/Bacterial Red disease/Bacterial-Red-disease-13-_jpg.rf.8d9d9458f572a4969a411aebf4ae815d.jpg` | Bacterial Red Disease |
| g2667 | roboflow | `train/Bacterial Red disease/Bacterial-Red-disease-15-_jpg.rf.d821c53393b7c3aad4fb74a4b244c5b6.jpg` | Bacterial Red Disease |
| g594 | roboflow | `train/Bacterial Red disease/Bacterial-Red-disease-3-_jpg.rf.23e52adf5a7ac46d710c6a4484f77f79.jpg` | Bacterial Red Disease |
| g4281 | roboflow | `train/Bacterial Red disease/Bacterial-Red-disease-40-_jpg.rf.9c7e9f198ad68e496ac8e2b66a991d2c.jpg` | Bacterial Red Disease |
| g2869 | roboflow | `train/Bacterial diseases - Aeromoniasis/Bacterial-diseases-Aeromoniasis-12-_jpg.rf.838be4dc8b38264f75c0b3d8e1409dad.jpg` | Aeromoniasis |
| g3851 | roboflow | `train/Bacterial diseases - Aeromoniasis/Bacterial-diseases-Aeromoniasis-15-_jpg.rf.edec24ece0e4fa14f7e0bb812329dc67.jpg` | Aeromoniasis |
| g4281 | roboflow | `train/Bacterial diseases - Aeromoniasis/Bacterial-diseases-Aeromoniasis-3-_png.rf.03d47b537d4062bfe389406a20cc6e30.jpg` | Aeromoniasis |
| g601 | roboflow | `train/Bacterial diseases - Aeromoniasis/Bacterial-diseases-Aeromoniasis-39-_jpg.rf.0a63210715bef67d9a0adff7bcd72bfe.jpg` | Aeromoniasis |
| g2078 | roboflow | `train/Bacterial gill disease/Bacterial-gill-disease-20-_jpg.rf.fe56f7a406a0702447e677cd2705af9a.jpg` | Bacterial Gill Disease |
| g1688 | roboflow | `train/Bacterial gill disease/Bacterial-gill-disease-26-_jpg.rf.efaaeb47f667534d5db5f272668fd9ee.jpg` | Bacterial Gill Disease |
| g2078 | roboflow | `train/Fungal diseases Saprolegniasis/Fungal-diseases-Saprolegniasis-8-_jpeg.rf.d5c879ae00c42b0e93ab80405a27b482.jpg` | Saprolegniasis |
| g1688 | roboflow | `train/Parasitic diseases/Parasitic-diseases-2-_jpg.rf.23f7ff863b0dc00e057f606753b05994.jpg` | Parasitic Disease |
| g3614 | roboflow | `train/Viral diseases White tail disease/Viral-diseases-White-tail-disease-10-_jpeg.rf.84c1cf72997815fca39e5b5180da2d10.jpg` | White Tail Disease |
| g2191 | roboflow | `valid/Bacterial Red disease/Bacterial-Red-disease-5-_jpeg.rf.edfe46da9dfb870a2cac9fff763c6636.jpg` | Bacterial Red Disease |
