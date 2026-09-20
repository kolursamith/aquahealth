# Development / frozen final-test split (Phase 8)

- strategy: group-aware stratified assignment (src/split_v2.py::assign_groups, the baseline rule of src/manifest.py generalised): unit = clean-manifest group_id (exact/near duplicates ∪ same specimen), stratum = (unified class, majority source of the group), largest-remainder quotas, largest groups first to the most deficient part
- seed: 42
- test ratio: 0.2 — not fixed by any project document; the team method note gives 80/20 development/test as an example (see data/audit/split_policy_proposal.md)
- clean manifest SHA-256: `e90b43a0c5a5cf9ea61ec3140c048d1e8ef854874492eaeafaeed04b56c86b68`
- clean images: 5298 total, 3805 included; excluded (not eligible for any partition): {'LABEL_CONFLICT_GROUP': 47, 'EXACT_DUPLICATE': 162, 'LABEL_UNRESOLVED': 1277, 'LABEL_EXCLUDED': 7}

**`final_test.csv` is FROZEN**: not for training, cross-validation, GAN generation, hyper-parameter tuning or choosing between models. Later phases read the development partition through `src/split_v2.py::read_development`, which verifies the digests and never opens the test file.

## Counts

| partition | images | groups | share |
|---|---|---|---|
| development | 3044 | 2585 | 0.8 |
| final_test | 761 | 637 | 0.2 |

## Per class

| unified class | development | final_test | test share |
|---|---|---|---|
| Aeromoniasis | 388 | 97 | 0.2 |
| Bacterial Gill Disease | 395 | 98 | 0.199 |
| Bacterial Red Disease | 362 | 91 | 0.201 |
| EUS Disease | 368 | 91 | 0.198 |
| Healthy Fish | 450 | 113 | 0.201 |
| Parasitic Disease | 369 | 92 | 0.2 |
| Saprolegniasis | 353 | 89 | 0.201 |
| White Tail Disease | 359 | 90 | 0.2 |

## Per source

| source | development | final_test |
|---|---|---|
| current_freshwater | 2721 | 680 |
| kaptai | 45 | 8 |
| roboflow | 278 | 73 |

## Group constraints

- groups straddling partitions: 0
- largest group in test: 4
- specimen groups in test: 0

The old baseline split (`data/split_manifest.csv`) and its results are untouched.
