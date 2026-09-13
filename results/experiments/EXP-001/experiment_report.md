# EXP-001 — EXP-001 EfficientNet-B0 baseline: existing AquaHealth preprocessing (CLAHE on, resize 256 -> centre-crop 224, ImageNet normalisation), existing train-only augmentation, existing Layer 9 staged fine-tuning (head -> last 3 blocks -> all), AdamW, selection on validation Macro-F1. Values are the existing repository defaults [C], not paper-derived.

Split: frozen manifest `split_manifest.csv` (sha256 b7d1fccbb21e…),
train 2384 / val 512. The frozen test split was not read.

## Configuration
- preprocess: {"image_size": 224, "resize_size": 256, "resize_mode": "crop", "clahe": {"clip_limit": 2.0, "tile_grid_size": 8}, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
- augmentation: {"crop_scale": [0.8, 1.0], "crop_ratio": [0.9, 1.1], "horizontal_flip": 0.5, "rotation_degrees": 15.0, "brightness": 0.2, "contrast": 0.2}
- stages: [{"name": "head", "epochs": 5, "trainable_blocks": 0, "learning_rate": 0.001, "backbone_learning_rate": null}, {"name": "partial", "epochs": 10, "trainable_blocks": 3, "learning_rate": 0.0003, "backbone_learning_rate": 3e-05}, {"name": "full", "epochs": 15, "trainable_blocks": -1, "learning_rate": 0.0003, "backbone_learning_rate": 1e-05}]
- optimizer: {"optimizer": "adamw", "weight_decay": 0.0001, "momentum": 0.9, "lr_step_size": null, "lr_gamma": 0.1, "amp": true}
- batch size 64, seed 42, device cuda, selection metric val Macro-F1

## Selected checkpoint
Stage **partial**, epoch 9 (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | 0.9004 |
| macro precision | 0.9018 |
| macro recall | 0.9010 |
| **macro F1** | **0.9004** |
| weighted F1 | 0.9002 |
| val loss | 0.3381 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.836 | 0.823 | 0.829 | 62 |
| Aeromoniasis | 0.833 | 0.909 | 0.870 | 66 |
| Bacterial Gill Disease | 0.983 | 0.879 | 0.928 | 66 |
| EUS Disease | 0.885 | 0.818 | 0.850 | 66 |
| Saprolegniasis | 0.934 | 0.950 | 0.942 | 60 |
| Parasitic Disease | 0.877 | 0.891 | 0.884 | 64 |
| White Tail Disease | 0.910 | 0.984 | 0.946 | 62 |
| Healthy Fish | 0.955 | 0.955 | 0.955 | 66 |

```
                        precision  recall  f1-score  support
Bacterial Red Disease       0.836   0.823     0.829       62
Aeromoniasis                0.833   0.909     0.870       66
Bacterial Gill Disease      0.983   0.879     0.928       66
EUS Disease                 0.885   0.818     0.850       66
Saprolegniasis              0.934   0.950     0.942       60
Parasitic Disease           0.877   0.891     0.884       64
White Tail Disease          0.910   0.984     0.946       62
Healthy Fish                0.955   0.955     0.955       66

accuracy                                      0.900      512
macro avg                   0.902   0.901     0.900      512
weighted avg                0.902   0.900     0.900      512
```

## Efficiency
- parameters: 4,017,796 total
- inference: 9.004 ms/image (batch 1, cuda, preprocessing excluded)
- training wall time: 25.8 min for 30 epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
