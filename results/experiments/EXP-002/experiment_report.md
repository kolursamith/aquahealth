# EXP-002 — EXP-002 [P1/P4 preprocessing]: identical to EXP-001 except CLAHE OFF. P1 and P4 use plain resize + ImageNet normalisation with no contrast enhancement [A]; CLAHE-on is the AquaHealth project default [B]. Single-factor ablation of the preprocessing step.

Split: frozen manifest `split_manifest.csv` (sha256 b7d1fccbb21e…),
train 2384 / val 512. The frozen test split was not read.

## Configuration
- preprocess: {"image_size": 224, "resize_size": 256, "resize_mode": "crop", "clahe": null, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
- augmentation: {"crop_scale": [0.8, 1.0], "crop_ratio": [0.9, 1.1], "horizontal_flip": 0.5, "rotation_degrees": 15.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.0}
- stages: [{"name": "head", "epochs": 5, "trainable_blocks": 0, "learning_rate": 0.001, "backbone_learning_rate": null}, {"name": "partial", "epochs": 10, "trainable_blocks": 3, "learning_rate": 0.0003, "backbone_learning_rate": 3e-05}, {"name": "full", "epochs": 15, "trainable_blocks": -1, "learning_rate": 0.0003, "backbone_learning_rate": 1e-05}]
- optimizer: {"optimizer": "adamw", "weight_decay": 0.0001, "momentum": 0.9, "lr_step_size": null, "lr_gamma": 0.1, "amp": true}
- batch size 64, seed 42, device cuda, selection metric val Macro-F1

## Selected checkpoint
Stage **partial**, epoch 10 (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | 0.9180 |
| macro precision | 0.9207 |
| macro recall | 0.9185 |
| **macro F1** | **0.9182** |
| weighted F1 | 0.9180 |
| val loss | 0.2662 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.851 | 0.919 | 0.884 | 62 |
| Aeromoniasis | 0.926 | 0.955 | 0.940 | 66 |
| Bacterial Gill Disease | 1.000 | 0.848 | 0.918 | 66 |
| EUS Disease | 0.887 | 0.833 | 0.859 | 66 |
| Saprolegniasis | 0.966 | 0.933 | 0.949 | 60 |
| Parasitic Disease | 0.859 | 0.953 | 0.904 | 64 |
| White Tail Disease | 0.922 | 0.952 | 0.937 | 62 |
| Healthy Fish | 0.955 | 0.955 | 0.955 | 66 |

```
                        precision  recall  f1-score  support
Bacterial Red Disease       0.851   0.919     0.884       62
Aeromoniasis                0.926   0.955     0.940       66
Bacterial Gill Disease      1.000   0.848     0.918       66
EUS Disease                 0.887   0.833     0.859       66
Saprolegniasis              0.966   0.933     0.949       60
Parasitic Disease           0.859   0.953     0.904       64
White Tail Disease          0.922   0.952     0.937       62
Healthy Fish                0.955   0.955     0.955       66

accuracy                                      0.918      512
macro avg                   0.921   0.919     0.918      512
weighted avg                0.921   0.918     0.918      512
```

## Efficiency
- parameters: 4,017,796 total
- inference: 8.719 ms/image (batch 1, cuda, preprocessing excluded)
- training wall time: 14.4 min for 30 epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
