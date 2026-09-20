# Leakage audit (Phase 6, pre-split) — clean manifest

Counting and EXIF-header reads only; no model was trained. Nothing was modified or deleted as a result of this audit. Machine-readable: `leakage_report.csv` (one row per check) and `leakage_report.json` (numbers).

## Findings (one row per check)

| check | finding | affected images | affected classes | action | status |
|---|---|---|---|---|---|
| 1 duplicate leakage (within a dataset) | 162 byte-identical copies excluded from the clean corpus; 305 near-duplicate/same-picture groups remain inside single datasets | 635 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | exact copies excluded (representative kept); near-duplicates kept but a split/fold must assign whole group_id values | resolved by manifest; enforced only if the split honours group_id |
| 2 cross-dataset duplicate leakage | 180 included groups contain images from more than one dataset (e.g. roboflow re-exports of current_freshwater; kaptai copies; SalmonScan photos inside current_freshwater) | 433 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | same as 1: whole group_id per partition; 47 images in label-conflicting groups already excluded (LABEL_CONFLICT_GROUP) | resolved by manifest; the cross-dataset label disagreements are evidence of label noise in the sources — human review recommended |
| 3 same-specimen / same-fish leakage | specimen identifiers exist only for no dataset; 0 specimen-bearing groups joined by group_id. For ['current_freshwater', 'kaptai', 'paper_dataset', 'roboflow']: Cannot be established from available metadata. | 0 |  | keep specimens inside one partition via group_id; for datasets without ids only exact/perceptual screening is possible (documented limitation) | partially testable: resolved where ids exist; Cannot be established from available metadata. elsewhere. dHash chaining joins near-duplicates into groups of up to 6 images (conservative but coarse) |
| 4 target leakage | the label is written in folder paths (all datasets), in file names current_freshwater: 100.0%, kaptai: 47.4%, paper_dataset: 62.3%, roboflow: 100.0%; no active dataset ships a label-bearing metadata table | 4772 | all | no change needed; keep filename/path/metadata out of every model input (guarded by tests on the dataset classes) | not a leak in the current pipeline; would become one if any loader used names |
| 5 suspicious metadata / features | no metadata table in the active datasets; EXIF present: current_freshwater: 0/3401 (orientation tag 0, of which stored rotated (tag != 1) 0 ; camera none), kaptai: 27/53 (orientation tag 27, of which stored rotated (tag != 1) 0 ; camera none), roboflow: 0/351 (orientation tag 0, of which stored rotated (tag != 1) 0 ; camera none) | 27 | all | EXIF is not read by the pipeline (src/dataset.py::load_image = Pillow decode + convert('RGB'); no exif_transpose), so images whose orientation tag != 1 enter the model as stored, i.e. rotated relative to their intended view. Decision needed: apply PIL.ImageOps.exif_transpose in load_image (changes the existing loader; the baseline dataset has no EXIF so its results are unaffected) or leave as is. Camera identity per source reaches the model only through pixel statistics. | no metadata reaches the model; the orientation issue is a preprocessing decision for human review (not silently changed); camera differences fold into finding 7 |
| 6 class <-> image-resolution shortcut | resolution alone predicts the class with an in-sample upper bound of 0.271 vs majority baseline 0.148; see resolution_by_class_top4 (classes differ in their dominant resolutions, e.g. Aeromoniasis is mostly 128x128) | 3805 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | cannot be removed by the pipeline (all images are resized to 224x224, but sharpness/compression traces survive); mitigation is methodological: per-source evaluation and (class, source)-stratified partitions; a resolution-only control classifier would quantify it and is proposed for a later phase | documented; requires human decision (accept, or equalise sources by resampling) |
| 7 other dataset / source shortcuts | source_dataset predicts the class at 0.1482 (baseline 0.148); classes present in a single source: none; file extension: 0.1516; image mode: .jpeg: 28, .jpg: 3756, .png: 21; pre-augmented files in current_freshwater: 1853; 200 groups straddle the vendors' own train/valid/test folders | 3805 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | source_dataset stays metadata; stratify partitions by (class, source); report per-source metrics; ignore the vendors' train/valid/test folders entirely (they leak) | documented; per-source evaluation to be built into the CV phase |

Severity: n/a (the project defines no severity convention)

## A. Target leakage — where is the label written outside the pixels?

- **folder_path**: label source by construction (folder = class); never read as a feature
- **filename**: see per_dataset; only the decoded image reaches the model (src/dataset.py, src/manifest.py load pixels only)
- **metadata_csv**: no active dataset ships a metadata table (MatsyaDx-BD excluded in v3)
- **test_csv**: current_freshwater/test.csv carries the label of the flat test_split; used only to assign original_class

| dataset | images (raw) | filename reveals label | % | label in folder path |
|---|---|---|---|---|
| current_freshwater | 3503 | 3503 | 100.0 | True |
| kaptai | 133 | 63 | 47.4 | True |
| paper_dataset | 1208 | 752 | 62.3 | True |
| roboflow | 454 | 454 | 100.0 | True |

## B. Duplicate / specimen leakage — what a split must keep together

| quantity | value |
|---|---|
| included images | 3805 |
| leakage groups | 3222 |
| multi image groups | 485 |
| images in multi image groups | 1068 |
| groups spanning datasets | 180 |
| groups spanning delivered splits | 200 |
| specimen groups | 0 |
| largest group | 6 |
| group size histogram | {'2': 405, '3': 65, '4': 13, '5': 1, '6': 1} |
| included group with mixed labels | 0 |
| excluded exact copies | 162 |

Verdict: a split that assigns whole group_id values to one partition cannot leak exact, near-duplicate or same-specimen images; a random per-image split would (see groups_spanning_delivered_splits: the vendors' own splits already do)

## C. Shortcut features — in-sample upper bounds (majority class per feature value)

| feature | distinct values | majority baseline | feature-only accuracy (upper bound) |
|---|---|---|---|
| source_dataset | 3 | 0.148 | 0.1482 |
| resolution | 36 | 0.148 | 0.271 |
| extension | 3 | 0.148 | 0.1516 |
| resolution+extension | 39 | 0.148 | 0.2739 |
| source+resolution | 37 | 0.148 | 0.271 |

| class | resolutions (top 4) | sources |
|---|---|---|
| Aeromoniasis | 128x128: 359, 640x640: 64, 224x224: 60, 224x109: 2 | current_freshwater: 441, roboflow: 44 |
| Bacterial Gill Disease | 640x640: 353, 128x128: 74, 224x224: 66 | current_freshwater: 439, roboflow: 54 |
| Bacterial Red Disease | 640x640: 296, 128x128: 86, 224x224: 71 | current_freshwater: 414, roboflow: 39 |
| EUS Disease | 640x640: 436, 360x480: 5, 150x150: 2, 200x200: 2 | current_freshwater: 437, kaptai: 22 |
| Healthy Fish | 640x640: 532, 480x360: 9, 360x480: 8, 216x68: 1 | current_freshwater: 440, kaptai: 31, roboflow: 92 |
| Parasitic Disease | 640x640: 263, 128x128: 142, 224x224: 55, 116x212: 1 | current_freshwater: 424, roboflow: 37 |
| Saprolegniasis | 640x640: 217, 128x128: 136, 224x224: 89 | current_freshwater: 397, roboflow: 45 |
| White Tail Disease | 640x640: 240, 128x128: 134, 224x224: 75 | current_freshwater: 409, roboflow: 40 |

## D. EXIF header scan (clean images)

| dataset | images | with EXIF | with orientation tag | camera make/model (top) |
|---|---|---|---|---|
| current_freshwater | 3401 | 0 | 0 | none |
| kaptai | 53 | 27 | 27 | none |
| roboflow | 351 | 0 | 0 | none |
