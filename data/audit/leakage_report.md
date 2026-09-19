# Leakage audit (Phase 6, pre-split) — clean manifest

Counting and EXIF-header reads only; no model was trained. Nothing was modified or deleted as a result of this audit. Machine-readable: `leakage_report.csv` (one row per check) and `leakage_report.json` (numbers).

## Findings (one row per check)

| check | finding | affected images | affected classes | action | status |
|---|---|---|---|---|---|
| 1 duplicate leakage (within a dataset) | 162 byte-identical copies excluded from the clean corpus; 384 near-duplicate/same-picture groups remain inside single datasets | 2772 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | exact copies excluded (representative kept); near-duplicates kept but a split/fold must assign whole group_id values | resolved by manifest; enforced only if the split honours group_id |
| 2 cross-dataset duplicate leakage | 180 included groups contain images from more than one dataset (e.g. roboflow re-exports of current_freshwater; kaptai copies; SalmonScan photos inside current_freshwater) | 433 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | same as 1: whole group_id per partition; 47 images in label-conflicting groups already excluded (LABEL_CONFLICT_GROUP) | resolved by manifest; the cross-dataset label disagreements are evidence of label noise in the sources — human review recommended |
| 3 same-specimen / same-fish leakage | specimen identifiers exist only for ['mendeley']; 79 specimen-bearing groups joined by group_id. For ['current_freshwater', 'kaptai', 'paper_dataset', 'roboflow']: Cannot be established from available metadata. | 2137 | Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish | keep specimens inside one partition via group_id; for datasets without ids only exact/perceptual screening is possible (documented limitation) | partially testable: resolved where ids exist; Cannot be established from available metadata. elsewhere. Open decision: dHash chaining merges different mendeley specimens into groups of up to 396 images (conservative but coarse) |
| 4 target leakage | the label is written in folder paths (all datasets), in file names current_freshwater: 100.0%, kaptai: 47.4%, mendeley: 0.0%, paper_dataset: 62.3%, roboflow: 100.0%, and in mendeley/metadata.csv (health_condition) | 4772 | all | no change needed; keep filename/path/metadata out of every model input (guarded by tests on the dataset classes) | not a leak in the current pipeline; would become one if any loader used names |
| 5 suspicious metadata / features | mendeley metadata columns location, capture_condition, image_format, preprocessing, notes are constant (single value each) — no discriminative metadata; EXIF present: current_freshwater: 0/3401 (orientation tag 0, of which stored rotated (tag != 1) 0 ; camera none), kaptai: 27/53 (orientation tag 27, of which stored rotated (tag != 1) 0 ; camera none), mendeley: 2137/2137 (orientation tag 2137, of which stored rotated (tag != 1) 658 {'Bacterial Gill Disease': 43, 'Bacterial Red Disease': 55, 'EUS Disease': 101, 'Healthy Fish': 459}; camera {'OPPO OPPO A52': 2137}), roboflow: 0/351 (orientation tag 0, of which stored rotated (tag != 1) 0 ; camera none) | 2164 | all | EXIF is not read by the pipeline (src/dataset.py::load_image = Pillow decode + convert('RGB'); no exif_transpose), so images whose orientation tag != 1 enter the model as stored, i.e. rotated relative to their intended view. Decision needed: apply PIL.ImageOps.exif_transpose in load_image (changes the existing loader; the baseline dataset has no EXIF so its results are unaffected) or leave as is. Camera identity per source reaches the model only through pixel statistics. | no metadata reaches the model; the orientation issue is a preprocessing decision for human review (not silently changed); camera differences fold into finding 7 |
| 6 class <-> image-resolution shortcut | resolution alone predicts the class with an in-sample upper bound of 0.3514 vs majority baseline 0.2423; e.g. Aeromoniasis is mostly 128x128, MatsyaDx classes are 4000x3000/3000x4000 only | 5942 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | cannot be removed by the pipeline (all images are resized to 224x224, but sharpness/compression traces survive); mitigation is methodological: per-source evaluation and (class, source)-stratified partitions; a resolution-only control classifier would quantify it and is proposed for a later phase | documented; requires human decision (accept, or equalise sources by resampling) |
| 7 other dataset / source shortcuts | source_dataset predicts the class at 0.2425 (baseline 0.2423); classes present in a single source: none; file extension: 0.2447; image mode: .jpeg: 28, .jpg: 5893, .png: 21; pre-augmented files in current_freshwater: 1853; 200 groups straddle the vendors' own train/valid/test folders | 5942 | Aeromoniasis; Bacterial Gill Disease; Bacterial Red Disease; EUS Disease; Healthy Fish; Parasitic Disease; Saprolegniasis; White Tail Disease | source_dataset stays metadata; stratify partitions by (class, source); report per-source metrics; ignore the vendors' train/valid/test folders entirely (they leak) | documented; per-source evaluation to be built into the CV phase |

Severity: n/a (the project defines no severity convention)

## A. Target leakage — where is the label written outside the pixels?

- **folder_path**: label source by construction (folder = class); never read as a feature
- **filename**: see per_dataset; only the decoded image reaches the model (src/dataset.py, src/manifest.py load pixels only)
- **metadata_csv**: mendeley/metadata.csv carries health_condition (= label), fish_category, specimen_id; used for provenance and grouping only
- **test_csv**: current_freshwater/test.csv carries the label of the flat test_split; used only to assign original_class

| dataset | images (raw) | filename reveals label | % | label in folder path |
|---|---|---|---|---|
| current_freshwater | 3503 | 3503 | 100.0 | True |
| kaptai | 133 | 63 | 47.4 | True |
| mendeley | 2137 | 0 | 0.0 | True |
| paper_dataset | 1208 | 752 | 62.3 | True |
| roboflow | 454 | 454 | 100.0 | True |

## B. Duplicate / specimen leakage — what a split must keep together

| quantity | value |
|---|---|
| included images | 5942 |
| leakage groups | 3301 |
| multi image groups | 564 |
| images in multi image groups | 3205 |
| groups spanning datasets | 180 |
| groups spanning delivered splits | 200 |
| specimen groups | 79 |
| largest group | 396 |
| group size histogram | {'2': 405, '3': 65, '4': 15, '5': 2, '6': 2, '7': 3, '8': 4, '9': 4, '10': 7, '11': 4, '12': 6, '13': 2, '14': 7, '15': 4, '16': 3, '17': 2, '18': 2, '19': 2, '20': 1, '21': 3, '22': 1, '23': 1, '24': 1, '26': 5, '28': 1, '29': 3, '30': 2, '31': 1, '38': 1, '41': 1, '59': 1, '222': 1, '259': 1, '396': 1} |
| included group with mixed labels | 0 |
| excluded exact copies | 162 |

Verdict: a split that assigns whole group_id values to one partition cannot leak exact, near-duplicate or same-specimen images; a random per-image split would (see groups_spanning_delivered_splits: the vendors' own splits already do)

## C. Shortcut features — in-sample upper bounds (majority class per feature value)

| feature | distinct values | majority baseline | feature-only accuracy (upper bound) |
|---|---|---|---|
| source_dataset | 4 | 0.2423 | 0.2425 |
| resolution | 38 | 0.2423 | 0.3514 |
| extension | 3 | 0.2423 | 0.2447 |
| resolution+extension | 41 | 0.2423 | 0.3532 |
| source+resolution | 39 | 0.2423 | 0.3514 |

| class | resolutions (top 4) | sources |
|---|---|---|
| Aeromoniasis | 128x128: 359, 640x640: 64, 224x224: 60, 224x109: 2 | current_freshwater: 441, roboflow: 44 |
| Bacterial Gill Disease | 640x640: 353, 4000x3000: 232, 128x128: 74, 224x224: 66 | current_freshwater: 439, mendeley: 275, roboflow: 54 |
| Bacterial Red Disease | 4000x3000: 598, 640x640: 296, 128x128: 86, 224x224: 71 | current_freshwater: 414, mendeley: 653, roboflow: 39 |
| EUS Disease | 640x640: 436, 4000x3000: 231, 3000x4000: 101, 360x480: 5 | current_freshwater: 437, kaptai: 22, mendeley: 332 |
| Healthy Fish | 640x640: 532, 3000x4000: 459, 4000x3000: 418, 480x360: 9 | current_freshwater: 440, kaptai: 31, mendeley: 877, roboflow: 92 |
| Parasitic Disease | 640x640: 263, 128x128: 142, 224x224: 55, 116x212: 1 | current_freshwater: 424, roboflow: 37 |
| Saprolegniasis | 640x640: 217, 128x128: 136, 224x224: 89 | current_freshwater: 397, roboflow: 45 |
| White Tail Disease | 640x640: 240, 128x128: 134, 224x224: 75 | current_freshwater: 409, roboflow: 40 |

## D. EXIF header scan (clean images)

| dataset | images | with EXIF | with orientation tag | camera make/model (top) |
|---|---|---|---|---|
| current_freshwater | 3401 | 0 | 0 | none |
| kaptai | 53 | 27 | 27 | none |
| mendeley | 2137 | 2137 | 2137 | {'OPPO OPPO A52': 2137} |
| roboflow | 351 | 0 | 0 | none |
