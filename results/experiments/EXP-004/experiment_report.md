# EXP-004 — EXP-004 [P4 optimiser]: SGD momentum 0.9, lr 0.01, step decay x0.1 [A] applied to EXP-001's stage structure. P4 decays every 20 of 100 epochs; scaled to every 5 epochs within each stage here [D] because the budget is 30 epochs. lr 0.01 is used for head and backbone as in the paper. Preprocessing, augmentation, stages and block schedule identical to EXP-001.

Split: frozen manifest `split_manifest.csv` (sha256 b7d1fccbb21e…),
train 2384 / val 512. The frozen test split was not read.

## Configuration
- preprocess: {"image_size": 224, "resize_size": 256, "resize_mode": "crop", "clahe": {"clip_limit": 2.0, "tile_grid_size": 8}, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
- augmentation: {"crop_scale": [0.8, 1.0], "crop_ratio": [0.9, 1.1], "horizontal_flip": 0.5, "rotation_degrees": 15.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.0}
- stages: [{"name": "head", "epochs": 5, "trainable_blocks": 0, "learning_rate": 0.01, "backbone_learning_rate": null}, {"name": "partial", "epochs": 10, "trainable_blocks": 3, "learning_rate": 0.01, "backbone_learning_rate": 0.01}, {"name": "full", "epochs": 15, "trainable_blocks": -1, "learning_rate": 0.01, "backbone_learning_rate": 0.01}]
- optimizer: {"optimizer": "sgd", "weight_decay": 0.0001, "momentum": 0.9, "lr_step_size": 5, "lr_gamma": 0.1, "amp": true}
- batch size 64, seed 42, device cuda, selection metric val Macro-F1

## Selected checkpoint
Stage **full**, epoch 3 (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | 0.9336 |
| macro precision | 0.9345 |
| macro recall | 0.9347 |
| **macro F1** | **0.9330** |
| weighted F1 | 0.9326 |
| val loss | 0.2161 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.879 | 0.935 | 0.906 | 62 |
| Aeromoniasis | 0.913 | 0.955 | 0.933 | 66 |
| Bacterial Gill Disease | 0.984 | 0.924 | 0.953 | 66 |
| EUS Disease | 0.944 | 0.773 | 0.850 | 66 |
| Saprolegniasis | 0.952 | 0.983 | 0.967 | 60 |
| Parasitic Disease | 0.938 | 0.938 | 0.938 | 64 |
| White Tail Disease | 0.925 | 1.000 | 0.961 | 62 |
| Healthy Fish | 0.941 | 0.970 | 0.955 | 66 |

```
                        precision  recall  f1-score  support
Bacterial Red Disease       0.879   0.935     0.906       62
Aeromoniasis                0.913   0.955     0.933       66
Bacterial Gill Disease      0.984   0.924     0.953       66
EUS Disease                 0.944   0.773     0.850       66
Saprolegniasis              0.952   0.983     0.967       60
Parasitic Disease           0.938   0.938     0.938       64
White Tail Disease          0.925   1.000     0.961       62
Healthy Fish                0.941   0.970     0.955       66

accuracy                                      0.934      512
macro avg                   0.934   0.935     0.933      512
weighted avg                0.935   0.934     0.933      512
```

## Efficiency
- parameters: 4,017,796 total
- inference: 10.483 ms/image (batch 1, cuda, preprocessing excluded)
- training wall time: 24.7 min for 30 epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
