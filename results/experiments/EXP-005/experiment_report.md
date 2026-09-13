# EXP-005 — EXP-005 [P4 augmentation]: P4's train-time augmentation [A] - horizontal flip 0.5, random crop scale 0.8-1.0, colour jitter +-20% brightness/contrast/saturation, no rotation - replacing EXP-001's augmentation (which has +-15 deg rotation and no saturation jitter). Saturation jitter is a new optional AugmentConfig field [D], default off. Everything else identical to EXP-001.

Split: frozen manifest `split_manifest.csv` (sha256 b7d1fccbb21e…),
train 2384 / val 512. The frozen test split was not read.

## Configuration
- preprocess: {"image_size": 224, "resize_size": 256, "resize_mode": "crop", "clahe": {"clip_limit": 2.0, "tile_grid_size": 8}, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
- augmentation: {"crop_scale": [0.8, 1.0], "crop_ratio": [0.9, 1.1], "horizontal_flip": 0.5, "rotation_degrees": 0.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.2}
- stages: [{"name": "head", "epochs": 5, "trainable_blocks": 0, "learning_rate": 0.001, "backbone_learning_rate": null}, {"name": "partial", "epochs": 10, "trainable_blocks": 3, "learning_rate": 0.0003, "backbone_learning_rate": 3e-05}, {"name": "full", "epochs": 15, "trainable_blocks": -1, "learning_rate": 0.0003, "backbone_learning_rate": 1e-05}]
- optimizer: {"optimizer": "adamw", "weight_decay": 0.0001, "momentum": 0.9, "lr_step_size": null, "lr_gamma": 0.1, "amp": true}
- batch size 64, seed 42, device cuda, selection metric val Macro-F1

## Selected checkpoint
Stage **full**, epoch 15 (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | 0.9336 |
| macro precision | 0.9342 |
| macro recall | 0.9343 |
| **macro F1** | **0.9335** |
| weighted F1 | 0.9329 |
| val loss | 0.2081 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.891 | 0.919 | 0.905 | 62 |
| Aeromoniasis | 0.912 | 0.939 | 0.925 | 66 |
| Bacterial Gill Disease | 0.955 | 0.970 | 0.962 | 66 |
| EUS Disease | 0.912 | 0.788 | 0.846 | 66 |
| Saprolegniasis | 0.967 | 0.983 | 0.975 | 60 |
| Parasitic Disease | 0.968 | 0.938 | 0.952 | 64 |
| White Tail Disease | 0.968 | 0.968 | 0.968 | 62 |
| Healthy Fish | 0.901 | 0.970 | 0.934 | 66 |

```
                        precision  recall  f1-score  support
Bacterial Red Disease       0.891   0.919     0.905       62
Aeromoniasis                0.912   0.939     0.925       66
Bacterial Gill Disease      0.955   0.970     0.962       66
EUS Disease                 0.912   0.788     0.846       66
Saprolegniasis              0.967   0.983     0.975       60
Parasitic Disease           0.968   0.938     0.952       64
White Tail Disease          0.968   0.968     0.968       62
Healthy Fish                0.901   0.970     0.934       66

accuracy                                      0.934      512
macro avg                   0.934   0.934     0.933      512
weighted avg                0.934   0.934     0.933      512
```

## Efficiency
- parameters: 4,017,796 total
- inference: 10.941 ms/image (batch 1, cuda, preprocessing excluded)
- training wall time: 23.6 min for 30 epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
