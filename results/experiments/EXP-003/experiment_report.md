# EXP-003 — EXP-003 [P1 fine-tuning recipe]: P1's two-phase transfer learning with Adam [A] - phase 1 frozen backbone, head lr 1e-3; phase 2 unfreeze the last blocks, single lr 1e-5 for every trainable parameter, no third full-unfreeze phase. Total 30 epochs (5 + 25) to match EXP-001's compute. Preprocessing (CLAHE on) and augmentation identical to EXP-001. Adam instead of AdamW follows the paper; weight_decay 0 because P1 reports none.

Split: frozen manifest `split_manifest.csv` (sha256 b7d1fccbb21e…),
train 2384 / val 512. The frozen test split was not read.

## Configuration
- preprocess: {"image_size": 224, "resize_size": 256, "resize_mode": "crop", "clahe": {"clip_limit": 2.0, "tile_grid_size": 8}, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
- augmentation: {"crop_scale": [0.8, 1.0], "crop_ratio": [0.9, 1.1], "horizontal_flip": 0.5, "rotation_degrees": 15.0, "brightness": 0.2, "contrast": 0.2, "saturation": 0.0}
- stages: [{"name": "head", "epochs": 5, "trainable_blocks": 0, "learning_rate": 0.001, "backbone_learning_rate": null}, {"name": "finetune", "epochs": 25, "trainable_blocks": 3, "learning_rate": 1e-05, "backbone_learning_rate": 1e-05}]
- optimizer: {"optimizer": "adam", "weight_decay": 0.0, "momentum": 0.9, "lr_step_size": null, "lr_gamma": 0.1, "amp": true}
- batch size 64, seed 42, device cuda, selection metric val Macro-F1

## Selected checkpoint
Stage **finetune**, epoch 25 (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | 0.8789 |
| macro precision | 0.8798 |
| macro recall | 0.8798 |
| **macro F1** | **0.8783** |
| weighted F1 | 0.8781 |
| val loss | 0.4009 |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Bacterial Red Disease | 0.769 | 0.806 | 0.787 | 62 |
| Aeromoniasis | 0.845 | 0.909 | 0.876 | 66 |
| Bacterial Gill Disease | 0.931 | 0.818 | 0.871 | 66 |
| EUS Disease | 0.877 | 0.758 | 0.813 | 66 |
| Saprolegniasis | 0.903 | 0.933 | 0.918 | 60 |
| Parasitic Disease | 0.875 | 0.875 | 0.875 | 64 |
| White Tail Disease | 0.897 | 0.984 | 0.938 | 62 |
| Healthy Fish | 0.940 | 0.955 | 0.947 | 66 |

```
                        precision  recall  f1-score  support
Bacterial Red Disease       0.769   0.806     0.787       62
Aeromoniasis                0.845   0.909     0.876       66
Bacterial Gill Disease      0.931   0.818     0.871       66
EUS Disease                 0.877   0.758     0.813       66
Saprolegniasis              0.903   0.933     0.918       60
Parasitic Disease           0.875   0.875     0.875       64
White Tail Disease          0.897   0.984     0.938       62
Healthy Fish                0.940   0.955     0.947       66

accuracy                                      0.879      512
macro avg                   0.880   0.880     0.878      512
weighted avg                0.880   0.879     0.878      512
```

## Efficiency
- parameters: 4,017,796 total
- inference: 12.099 ms/image (batch 1, cuda, preprocessing excluded)
- training wall time: 25.1 min for 30 epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
