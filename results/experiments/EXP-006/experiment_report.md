# EXP-006 — EXP-006 [P3 training]: Adam with lr 3e-5 for every trainable parameter in every stage [A] (P3 fine-tunes EfficientNetB1 at 3e-5 with EarlyStopping patience 5). Early stopping is not in the engine; best-checkpoint selection on validation Macro-F1 plays the same role [D]. Stage structure, preprocessing and augmentation identical to EXP-001; weight_decay 0 because P3 reports none.

Split: frozen manifest `split_manifest.csv` (sha256 b7d1fccbb21e…),
train 2384 / val 512. The frozen test split was not read.

## Configuration
- preprocess: {"image_size": 224, "resize_size": 256, "resize_mode": "crop", "clahe": {"clip_limit": 2.0, "tile_grid_size": 8}, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
- augmentation: {"crop_scale": [0.8, 1.0], "crop_ratio": [0.9, 1.1], "horizontal_flip": 0.5, "rotation_degrees": 15.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.0}
- stages: [{"name": "head", "epochs": 5, "trainable_blocks": 0, "learning_rate": 3e-05, "backbone_learning_rate": null}, {"name": "partial", "epochs": 10, "trainable_blocks": 3, "learning_rate": 3e-05, "backbone_learning_rate": 3e-05}, {"name": "full", "epochs": 15, "trainable_blocks": -1, "learning_rate": 3e-05, "backbone_learning_rate": 3e-05}]
- optimizer: {"optimizer": "adam", "weight_decay": 0.0, "momentum": 0.9, "lr_step_size": null, "lr_gamma": 0.1, "amp": true}
- batch size 64, seed 42, device cuda, selection metric val Macro-F1

## Selected checkpoint
Stage **full**, epoch 14 (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | 0.9023 |
| macro precision | 0.9047 |
| macro recall | 0.9033 |
| **macro F1** | **0.9014** |
| weighted F1 | 0.9012 |
| val loss | 0.3128 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.797 | 0.823 | 0.810 | 62 |
| Aeromoniasis | 0.925 | 0.939 | 0.932 | 66 |
| Bacterial Gill Disease | 0.964 | 0.818 | 0.885 | 66 |
| EUS Disease | 0.944 | 0.773 | 0.850 | 66 |
| Saprolegniasis | 0.935 | 0.967 | 0.951 | 60 |
| Parasitic Disease | 0.910 | 0.953 | 0.931 | 64 |
| White Tail Disease | 0.859 | 0.984 | 0.917 | 62 |
| Healthy Fish | 0.901 | 0.970 | 0.934 | 66 |

```
                        precision  recall  f1-score  support
Bacterial Red Disease       0.797   0.823     0.810       62
Aeromoniasis                0.925   0.939     0.932       66
Bacterial Gill Disease      0.964   0.818     0.885       66
EUS Disease                 0.944   0.773     0.850       66
Saprolegniasis              0.935   0.967     0.951       60
Parasitic Disease           0.910   0.953     0.931       64
White Tail Disease          0.859   0.984     0.917       62
Healthy Fish                0.901   0.970     0.934       66

accuracy                                      0.902      512
macro avg                   0.905   0.903     0.901      512
weighted avg                0.905   0.902     0.901      512
```

## Efficiency
- parameters: 4,017,796 total
- inference: 9.183 ms/image (batch 1, cuda, preprocessing excluded)
- training wall time: 25.7 min for 30 epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
