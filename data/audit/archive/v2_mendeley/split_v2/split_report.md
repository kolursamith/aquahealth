# Development / frozen final-test split (Phase 8)

- strategy: group-aware stratified assignment (src/split_v2.py::assign_groups, the baseline rule of src/manifest.py generalised): unit = clean-manifest group_id (exact/near duplicates ∪ same specimen), stratum = (unified class, majority source of the group), largest-remainder quotas, largest groups first to the most deficient part
- seed: 42
- test ratio: 0.2 — not fixed by any project document; the team method note gives 80/20 development/test as an example (see data/audit/split_policy_proposal.md)
- clean manifest SHA-256: `544ebc791db5ce30af77b059e5e639146841d375ee283d6780fbfcbe862c19c9`
- clean images: 7435 total, 5942 included; excluded (not eligible for any partition): {'LABEL_CONFLICT_GROUP': 47, 'EXACT_DUPLICATE': 162, 'LABEL_UNRESOLVED': 1277, 'LABEL_EXCLUDED': 7}

**`final_test.csv` is FROZEN**: not for training, cross-validation, GAN generation, hyper-parameter tuning or choosing between models. Later phases read the development partition through `src/split_v2.py::read_development`, which verifies the digests and never opens the test file.

## Counts

| partition | images | groups | share |
|---|---|---|---|
| development | 4658 | 2647 | 0.7839 |
| final_test | 1284 | 654 | 0.2161 |

## Per class

| unified class | development | final_test | test share |
|---|---|---|---|
| Aeromoniasis | 388 | 97 | 0.2 |
| Bacterial Gill Disease | 612 | 156 | 0.203 |
| Bacterial Red Disease | 879 | 227 | 0.205 |
| EUS Disease | 630 | 161 | 0.204 |
| Healthy Fish | 1068 | 372 | 0.258 |
| Parasitic Disease | 369 | 92 | 0.2 |
| Saprolegniasis | 353 | 89 | 0.201 |
| White Tail Disease | 359 | 90 | 0.2 |

## Per source

| source | development | final_test |
|---|---|---|
| current_freshwater | 2719 | 682 |
| kaptai | 44 | 9 |
| mendeley | 1614 | 523 |
| roboflow | 281 | 70 |

## Group constraints

- groups straddling partitions: 0
- largest group in test: 259
- specimen groups in test: 17

The old baseline split (`data/split_manifest.csv`) and its results are untouched.
