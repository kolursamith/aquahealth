# Model and training

## Model
- **Backbone**: torchvision `efficientnet_b0` with `EfficientNet_B0_Weights.IMAGENET1K_V1`
  (`efficientnet_b0_rwightman-7f5810bc.pth`, hash-checked download). ImageNet-1K general-purpose weights — not
  fish-trained.
- **Head**: `nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, 8))`, uniform ±1/√1280 init, zero bias.
- **Input**: 3×224×224, ImageNet-normalised. **Output**: 8 logits → softmax.
- **Parameters**: 4,017,796 total (10,248 head-only; 3,165,988 with the last 3 of 9 blocks; all when fully unfrozen).

## Preprocessing (identical for validation, test and inference)
Pillow decode → RGB → CLAHE (OpenCV, LAB L-channel, clip 2.0, tile 8; off in EXP-002) → `Resize(256, bicubic)` →
`CenterCrop(224)` → tensor → `Normalize(mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225))`.
Training adds, after CLAHE: `RandomResizedCrop(224, scale 0.8–1.0, ratio 0.9–1.1)`, `RandomHorizontalFlip(0.5)`,
`RandomRotation(±15°)`, `ColorJitter(brightness 0.2, contrast 0.2[, saturation 0.2 in EXP-005])`.

## Training strategy (`src/finetune.py`, `scripts/run_experiment.py`)
| Stage | Epochs | Trainable | LR head / backbone (baseline) |
|---|---|---|---|
| head | 5 | classifier only (frozen blocks stay in `eval()` so BatchNorm stats do not drift) | 1e-3 / — |
| partial | 10 | last 3 of 9 feature blocks + head | 3e-4 / 3e-5 |
| full | 15 | everything | 3e-4 / 1e-5 |

Cross-entropy loss; AdamW (wd 1e-4) by default, Adam / SGD + StepLR for the paper variants; AMP (float16 + GradScaler)
on CUDA; batch 64; seed 42; each stage starts from the previous stage's best weights; **selection = best validation
Macro-F1 over all epochs of all stages** (no early stopping, no test data). Checkpoints (format v2) store model,
optimizer, scheduler, scaler and RNG state plus `class_names` and `preprocess`. Environment: Colab Tesla T4,
Python 3.13.15, torch 2.14.0+cu130, ~44 s/epoch with CLAHE (CPU-bound), 24–26 min per 30-epoch run.

## Model selection (`results/final_model_selection.md`)
Criteria in order: validation Macro-F1 → weakest-class F1 → ECE → latency → parameters → accuracy.
EXP-005 (0.9335) and EXP-004 (0.9330) misclassify exactly the same number of validation images (34/512), so the
0.0005 gap is below one-image resolution; every tie-break favours EXP-004 (EUS F1 0.850 vs 0.845, ECE 0.0215 vs
0.0218, 10.5 vs 10.9 ms, diseased→Healthy 4 vs 7). **Final model = EXP-004.**

## Final checkpoint
`models/final_model.pth` (git-ignored) = EXP-004 `best_model.pth`, SHA-256
`3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae`, stage `full`, epoch 3, 32.5 MB.

## Validation results (model selection) vs official test
| Split | Accuracy | Macro P | Macro R | Macro-F1 | Weighted F1 | ECE |
|---|---|---|---|---|---|---|
| Validation (512) | 0.9336 | 0.9345 | 0.9347 | 0.9330 | 0.9335 | 0.0215 |
| **Official test (509, once)** | **0.9450** | 0.9470 | 0.9458 | **0.9453** | 0.9450 | 0.0123 |
