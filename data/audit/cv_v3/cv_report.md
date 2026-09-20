# 10-fold cross-validation manifests (Phase 9)

- source: `data/audit/split_v3/development.csv` (3044 images); `final_test.csv` was not read
- strategy: group-aware stratified K-fold — `src/split_v2.py::assign_folds` over `assign_groups` (unit = clean-manifest group_id, stratum = unified class | majority source), seed 42; every image is validation exactly once
- sklearn: not installed / not used (its StratifiedGroupKFold would ignore the source stratum); the project's own baseline rule is generalised instead
- files: `folds.csv`, `fold_XX_train.csv`, `fold_XX_validation.csv`, digests in `folds.sha256`

| fold | train | validation | validation classes | validation sources | groups |
|---|---|---|---|---|---|
| 1 | 2729 | 315 | Aeromoniasis=40; Bacterial Gill Disease=41; Bacterial Red Disease=38; EUS Disease=37; Healthy Fish=47; Parasitic Disease=39; Saprolegniasis=36; White Tail Disease=37 | current_freshwater=273; kaptai=8; roboflow=34 | groups=253; specimen_groups=0; largest_group=6; groups_shared_with_train=0 |
| 2 | 2733 | 311 | Aeromoniasis=40; Bacterial Gill Disease=40; Bacterial Red Disease=37; EUS Disease=37; Healthy Fish=47; Parasitic Disease=37; Saprolegniasis=36; White Tail Disease=37 | current_freshwater=276; kaptai=5; roboflow=30 | groups=257; specimen_groups=0; largest_group=4; groups_shared_with_train=0 |
| 3 | 2736 | 308 | Aeromoniasis=39; Bacterial Gill Disease=40; Bacterial Red Disease=37; EUS Disease=37; Healthy Fish=45; Parasitic Disease=37; Saprolegniasis=36; White Tail Disease=37 | current_freshwater=277; kaptai=3; roboflow=28 | groups=256; specimen_groups=0; largest_group=4; groups_shared_with_train=0 |
| 4 | 2736 | 308 | Aeromoniasis=39; Bacterial Gill Disease=40; Bacterial Red Disease=37; EUS Disease=37; Healthy Fish=45; Parasitic Disease=37; Saprolegniasis=36; White Tail Disease=37 | current_freshwater=273; kaptai=4; roboflow=31 | groups=261; specimen_groups=0; largest_group=4; groups_shared_with_train=0 |
| 5 | 2740 | 304 | Aeromoniasis=39; Bacterial Gill Disease=39; Bacterial Red Disease=36; EUS Disease=37; Healthy Fish=45; Parasitic Disease=37; Saprolegniasis=35; White Tail Disease=36 | current_freshwater=275; kaptai=2; roboflow=27 | groups=260; specimen_groups=0; largest_group=3; groups_shared_with_train=0 |
| 6 | 2740 | 304 | Aeromoniasis=39; Bacterial Gill Disease=39; Bacterial Red Disease=36; EUS Disease=37; Healthy Fish=45; Parasitic Disease=37; Saprolegniasis=35; White Tail Disease=36 | current_freshwater=274; kaptai=6; roboflow=24 | groups=261; specimen_groups=0; largest_group=3; groups_shared_with_train=0 |
| 7 | 2741 | 303 | Aeromoniasis=39; Bacterial Gill Disease=39; Bacterial Red Disease=36; EUS Disease=37; Healthy Fish=44; Parasitic Disease=37; Saprolegniasis=35; White Tail Disease=36 | current_freshwater=271; kaptai=4; roboflow=28 | groups=262; specimen_groups=0; largest_group=3; groups_shared_with_train=0 |
| 8 | 2745 | 299 | Aeromoniasis=38; Bacterial Gill Disease=39; Bacterial Red Disease=35; EUS Disease=37; Healthy Fish=44; Parasitic Disease=36; Saprolegniasis=35; White Tail Disease=35 | current_freshwater=268; kaptai=5; roboflow=26 | groups=258; specimen_groups=0; largest_group=3; groups_shared_with_train=0 |
| 9 | 2747 | 297 | Aeromoniasis=38; Bacterial Gill Disease=39; Bacterial Red Disease=35; EUS Disease=36; Healthy Fish=44; Parasitic Disease=36; Saprolegniasis=35; White Tail Disease=34 | current_freshwater=268; kaptai=4; roboflow=25 | groups=258; specimen_groups=0; largest_group=3; groups_shared_with_train=0 |
| 10 | 2749 | 295 | Aeromoniasis=37; Bacterial Gill Disease=39; Bacterial Red Disease=35; EUS Disease=36; Healthy Fish=44; Parasitic Disease=36; Saprolegniasis=34; White Tail Disease=34 | current_freshwater=266; kaptai=4; roboflow=25 | groups=259; specimen_groups=0; largest_group=3; groups_shared_with_train=0 |

Nothing was trained; no GAN images exist; the frozen test set is untouched.
