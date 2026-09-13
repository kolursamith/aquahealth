# Final model selection (Gate 5)

**Final model selected using validation data. Frozen test set not accessed during selection.**

## 1. Candidate comparison (artifact-verified, validation split, 512 images)

| Experiment | Paper technique | Best stage/epoch | Val acc | Macro P | Macro R | **Macro-F1** | Weighted F1 | Weakest class (F1) | ECE | ms/img (T4) | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | none (repository defaults) | partial/9 | 0.9004 | 0.9018 | 0.901 | **0.9004** | 0.9002 | Bacterial Red Disease (0.8293) | 0.0555 | 9.004 | 4017796 |
| EXP-002 | P1/P4 preprocessing: CLAHE off | partial/10 | 0.918 | 0.9207 | 0.9185 | **0.9182** | 0.918 | EUS Disease (0.8594) | 0.0316 | 8.719 | 4017796 |
| EXP-003 | P1 two-phase Adam fine-tuning | finetune/25 | 0.8789 | 0.8798 | 0.8798 | **0.8783** | 0.8781 | Bacterial Red Disease (0.7874) | 0.0862 | 12.099 | 4017796 |
| EXP-004 | P4 optimiser: SGD 0.9 / lr 0.01 / StepLR | full/3 | 0.9336 | 0.9345 | 0.9347 | **0.933** | 0.9326 | EUS Disease (0.8500) | 0.0215 | 10.483 | 4017796 |
| EXP-005 | P4 augmentation (flip, crop, jitter incl. saturation) | full/15 | 0.9336 | 0.9342 | 0.9343 | **0.9335** | 0.9329 | EUS Disease (0.8455) | 0.0218 | 10.941 | 4017796 |
| EXP-006 | P3 Adam 3e-5 | full/14 | 0.9023 | 0.9047 | 0.9033 | **0.9014** | 0.9012 | Bacterial Red Disease (0.8095) | 0.033 | 9.183 | 4017796 |

Sources: `results/verified_experiment_comparison.csv`, `results/experiments/<ID>/eval_val/metrics.json` (local `src.evaluate --split val` of each `best_model.pth`), `metrics.csv`, `inference_benchmark.json`. Full 22-column table: `results/FINAL_MODEL_COMPARISON.csv`. All six checkpoints share the same architecture and parameter count (4,017,796).

## 2. Selection criteria (in order)

1. Validation Macro-F1 · 2. weakest-class F1 · 3. ECE · 4. inference latency · 5. parameter count · 6. validation accuracy. Published paper numbers were not used.

## 3–4. Selected experiment and why

**FINAL SELECTED EXPERIMENT = EXP-004**  ·  **FINAL MODEL = EfficientNet-B0** (P4-informed SGD momentum 0.9, lr 0.01, StepLR ×0.1 every 5 epochs; CLAHE on; best at stage `full`, epoch 3).

EXP-004 and EXP-005 are the only two candidates above 0.93 and they are statistically indistinguishable on the primary criterion: 0.932986 vs 0.933463 Macro-F1 (Δ = 0.00048) with **exactly the same number of misclassified validation images (34 of 512)** and identical accuracy (0.9336). One validation image is worth 0.2 % accuracy, so a 0.0005 F1 gap is a redistribution of the same errors across classes, not evidence that one model is better; the primary criterion is therefore a tie and the tie-breaks decide.

Every tie-break favours EXP-004. Weakest class: EUS Disease F1 0.8500 vs 0.8455 (and the second-weakest, Bacterial Red, 0.9062 vs 0.9048). Calibration: ECE 0.02146 vs 0.02181. Latency: 10.483 vs 10.941 ms/image on the T4 benchmark. Parameters and accuracy are equal. In addition — and most relevant for a farmer-facing screening tool — EXP-004 makes fewer of the costly errors: 4 diseased validation images predicted *Healthy Fish* versus 7 for EXP-005, and its Healthy-class F1 is 0.955 vs 0.934, so it is less likely to reassure a farmer about a sick fish. EXP-004 also reached its best epoch early (full/3) and then held a plateau, whereas EXP-005's best is its final epoch of a still-rising curve, i.e. an endpoint rather than a converged optimum.

EXP-002 (0.9182, CLAHE off) is a strong third and shows CLAHE does not help this dataset with AdamW; combining CLAHE-off with the P4 recipe was not run and is documented as future work rather than assumed.

## 5–8. Validation metrics of the selected model (EXP-004, `results/experiments/EXP-004/eval_val/metrics.json`)

| Metric | Value |
|---|---|
| Accuracy | 0.9336 (478/512) |
| Macro precision / recall | 0.9345 / 0.9347 |
| **Macro-F1** | **0.9330** |
| Weighted F1 | 0.9326 |
| Per-class F1 | Bacterial Red 0.906 · Aeromoniasis 0.933 · Bacterial Gill 0.953 · EUS 0.850 · Saprolegniasis 0.967 · Parasitic 0.938 · White Tail 0.961 · Healthy 0.955 |
| Weakest class | EUS Disease, F1 0.8500 |
| ECE (10 bins) | 0.0215 |
| Inference latency | 10.483 ms/image, batch 1, Tesla T4 (`inference_benchmark.json`) |
| Parameters | 4,017,796 |

## 9–10. Checkpoint

- `models/final_model.pth` — copied byte-for-byte from `drive_export/experiments/EXP-004/best_model.pth` (Drive `MyDrive/AquaHealth/runs/experiments/EXP-004/best_model.pth`)
- Size: 32,524,991 bytes
- SHA-256: `3045a949ff54f383383cae2130d865eb9b832f1070724fb0495cd8b1ea0c0dae` (identical to the source file)
- Verified with `src.train.load_checkpoint` + `Checkpoint.build_model()`: EfficientNet, 8 outputs, logits shape (1, 8), canonical class list, preprocessing metadata (CLAHE 2.0/8, resize 256, crop 224, ImageNet mean/std), stage full / epoch 3, best_metric 0.9329857; `src.predict.Predictor(models/final_model.pth)` loads it and returns `status=ok`.

## 11. Test status

**OFFICIAL TEST: NOT YET EXECUTED. TEST SPLIT ACCESSED = NO.** The model weights are frozen from this point; the one-shot frozen-test evaluation is the next gate.
