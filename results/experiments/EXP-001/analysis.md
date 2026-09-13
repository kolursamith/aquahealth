# EXP-001 — EfficientNet-B0 baseline: analysis

**Source of every number below:** `config.json`, `metrics.csv`, `inference_benchmark.json`,
`eval_val/metrics.json` (re-evaluation of `best_model.pth` on the frozen validation split),
`training_curves.png`, `confusion_matrix.png`, `confusion_matrix_normalized.png`.
All were produced by `scripts/run_experiment.py` / `src.evaluate` on a Colab Tesla T4
(torch 2.14.0+cu130, Python 3.13.15, seed 42, batch 64, AMP). The PNGs in this folder were
re-rendered locally from `metrics.csv` and `eval_val/metrics.json` with the runner's own
plotting functions; no value was edited. **The frozen test split was not read.**

## 1. Headline result

| Item | Value |
|---|---|
| Best validation Macro-F1 | **0.9004** (stage `partial`, epoch 9 of 10, global epoch 14) |
| Validation accuracy at best | 0.9004 (461 / 512) |
| Macro precision / recall | 0.9018 / 0.9010 |
| Weighted F1 | 0.9002 |
| Validation loss at best | 0.3381 (lowest val loss overall: 0.3203 at partial epoch 10, F1 0.8926) |
| Misclassified validation images | 51 / 512 |
| Parameters | 4,017,796 total; 3,165,988 trainable in the selected stage |
| Inference latency | 9.004 ms / image, batch 1, CUDA (T4) |
| Training time | 1,549.8 s (25.8 min) for 30 epochs, ~44 s / epoch |
| Checkpoint | `MyDrive/AquaHealth/runs/experiments/EXP-001/best_model.pth` (Drive; not in git) |

Re-evaluating the checkpoint reproduces the `metrics.csv` row exactly (F1 0.90042, loss 0.33806),
so the saved `best_model.pth` is the epoch that was selected.

## 2. Convergence

Stage-by-stage (from `metrics.csv`):

| Stage | Epochs | Trainable | LR head / backbone | Train loss | Val loss | Val F1 first → best → last |
|---|---|---|---|---|---|---|
| head | 5 | 10,248 (head only) | 1e-3 / — | 1.567 → 0.642 | 1.273 → 0.777 | 0.624 → **0.772** (ep 5) → 0.772 |
| partial | 10 | 3,165,988 (last 3 blocks + head) | 3e-4 / 3e-5 | 0.856 → 0.122 | 0.846 → 0.320 | 0.756 → **0.900** (ep 9) → 0.893 |
| full | 15 | 4,017,796 (all) | 3e-4 / 1e-5 | 0.299 → 0.065 | 0.449 → 0.367 | 0.857 → **0.871** (ep 15) → 0.871 |

- **Head stage had not converged.** Val F1 was still rising every epoch (0.624, 0.693, 0.725,
  0.744, 0.772) and val loss still falling (1.27 → 0.78) when the 5-epoch budget ended. A
  linear-probe ceiling of ~0.77 is nevertheless well below what the fine-tuned stages reached,
  so the extra epochs would not have changed the final choice.
- **Partial stage did the work.** Val F1 rose monotonically for 9 epochs (0.756 → 0.900), val
  loss fell every epoch (0.846 → 0.320). Epoch 10 dipped slightly in F1 (0.893) while val loss
  hit its minimum (0.320): the curve had flattened, not collapsed. Ten epochs was roughly the
  right budget for this stage at these learning rates.
- **Each stage boundary caused a transient regression.** Head→partial: train loss 0.64 → 0.86,
  val F1 0.772 → 0.756. Partial→full: train loss 0.12 → 0.30, val loss 0.32 → 0.45, val F1
  0.900 → 0.857, despite the full stage starting from the partial-stage best weights. Two things
  change at a boundary: the optimizer is rebuilt (fresh AdamW moments) and, more importantly,
  the newly unfrozen blocks switch from `eval()` to `train()` mode (`src/train.py`, frozen blocks
  keep their BatchNorm statistics fixed), so their BatchNorm running statistics start re-adapting
  to augmented training batches. The size of the drop is therefore largely independent of the
  learning rate (EXP-003 shows the same drop at lr 1e-5). Partial recovered in 2 epochs; full
  never recovered to its starting point (best 0.871 at its last epoch).

## 3. Overfitting

- Final-epoch gap: train accuracy 0.986 vs val accuracy 0.873; train loss 0.065 vs val loss
  0.367. Train loss fell 0.30 → 0.065 across the full stage while val loss stayed flat within
  0.37–0.45 and val F1 oscillated 0.848–0.871 with no trend. That is a **generalisation
  plateau with continued memorisation**, i.e. the full stage overfit relative to the partial
  stage rather than improved on it.
- Val loss did not blow up (no classic divergence), so the model is not badly over-confident on
  errors either (see §7), but the last 15 epochs of compute bought nothing on the
  validation set.
- In the partial stage the gap was also open (train acc 0.978 vs val 0.900 at the best epoch)
  but val metrics were still improving, so early stopping by Macro-F1 correctly kept epoch 9.

## 4. Stage A (head-only) vs Stage B (partial) vs Stage C (full)

- Head-only reached 0.772 F1; unfreezing the last 3 of 9 MBConv blocks added **+0.128 F1**
  (0.772 → 0.900). That is the single largest gain in the run.
- Unfreezing the remaining 6 early blocks at LR 1e-5 gave **−0.029 F1** at its best
  (0.871) and −0.042 at its first epoch, relative to the partial-stage best. With 2,384
  training images, the early ImageNet features are better left frozen (or trained at an even
  smaller LR / for fewer epochs). Selection by val Macro-F1 across stages protected the result.

## 5. Per-class results (validation, best checkpoint)

| Class | P | R | F1 | n |
|---|---|---|---|---|
| Healthy Fish | 0.955 | 0.955 | **0.955** | 66 |
| White Tail Disease | 0.910 | 0.984 | 0.946 | 62 |
| Saprolegniasis | 0.934 | 0.950 | 0.942 | 60 |
| Bacterial Gill Disease | 0.983 | 0.879 | 0.928 | 66 |
| Parasitic Disease | 0.877 | 0.891 | 0.884 | 64 |
| Aeromoniasis | 0.833 | 0.909 | 0.870 | 66 |
| EUS Disease | 0.885 | 0.818 | 0.850 | 66 |
| Bacterial Red Disease | 0.836 | 0.823 | **0.829** | 62 |

- **Strongest:** Healthy Fish (0.955), White Tail (0.946), Saprolegniasis (0.942). Healthy Fish
  is confused only with EUS (2) and Parasitic (1); no diseased image was predicted Healthy
  except 3 EUS images — the clinically most important error direction (missed disease) is rare
  (3 / 446 diseased validation images, 0.7%).
- **Weakest:** Bacterial Red Disease (0.829) and EUS Disease (0.850). Both are low-recall
  classes (0.823, 0.818).

## 6. Confusion patterns (counts, true → predicted)

- Bacterial Red Disease → Aeromoniasis: **5** (the largest single confusion; both are bacterial
  red-lesion presentations). Reverse direction: 2.
- EUS Disease is scattered: → Bacterial Red 3, → Aeromoniasis 3, → White Tail 3, → Healthy 3.
  EUS has no dominant confusion partner; its 12 errors are spread over four classes.
- Bacterial Gill Disease → Bacterial Red 3, → Parasitic 3 (recall 0.879 despite precision 0.983:
  the model under-predicts this class rather than over-predicting it).
- Parasitic Disease → Aeromoniasis 3.
- Aeromoniasis is the most common *false-positive sink* (precision 0.833): it receives 5 + 3 + 3
  + 1 = 12 wrong predictions from other classes.
- White Tail Disease is over-predicted slightly (precision 0.910, recall 0.984).

## 7. Calibration / confidence

- Mean softmax confidence 0.847; correct predictions 0.875, incorrect 0.589.
- ECE (10 bins) = **0.0555**.
- Bin 0.9–1.0: 285 images (56% of val), accuracy **1.000**. Bin 0.8–0.9: 75 images, 0.907.
  Bin 0.7–0.8: 59, 0.814. Below 0.6 confidence: 64 images with accuracy 0.48–0.57.
- The model is **under-confident in the 0.6–0.9 range** (accuracy exceeds confidence) and
  roughly calibrated below 0.6. Consequence for the inference gate (Gate 10): a "HIGH" label at
  >0.80 corresponds to measured accuracy ≥0.907 on validation; "LOW/UNCERTAIN" below 0.50
  corresponds to accuracy ~0.5. The thresholds are consistent with the observed reliability.

## 8. Dataset balance

Train split per class: 278–310 images (Saprolegniasis 278 lowest, Aeromoniasis 310 highest),
val 60–66. The split is effectively balanced (ratio max/min 1.12), so class imbalance is **not**
a driver of the per-class differences; macro-F1 and accuracy are nearly identical (0.9004 both).

## 9. Model size and speed

4.02 M parameters, 9.0 ms / image batch-1 on a T4 (≈110 images/s). Training throughput was
~54 images/s (44 s per epoch of 2,384 images) with 2 dataloader workers, which indicates the
run was CPU/data-bound (JPEG decode + CLAHE + augmentation), not GPU-bound. GPU memory was not
recorded by the runner (NOT MEASURED).

## 10. Preprocessing observations

- CLAHE (clip 2.0, tile 8) was applied to every image in train, val and the benchmark; its
  effect is not isolated by this experiment (a no-CLAHE ablation is needed — see Gate 5).
- The eval path used Resize 256 → CenterCrop 224; train used RandomResizedCrop(0.8–1.0)
  + flip + ±15° rotation + colour jitter 0.2. The gap between train and val accuracy (0.978 vs
  0.900 at the best epoch) suggests the augmentation is not strong enough to prevent
  memorisation at 2.4 k images, but no artefact in the logs points to a preprocessing *bug*
  (no NaN, loss curves smooth, val re-evaluation reproduces training-time metrics).

## 11. Is further tuning justified?

Yes, on evidence, for three specific reasons:

1. The **full-unfreeze stage is harmful as configured** (−0.03 F1). Either drop it, shorten it,
   or lower its learning rates — cheap to test.
2. **Overfitting is visible** (train 0.986 vs val 0.873 at the end): stronger regularisation
   (more augmentation, higher dropout, weight decay) or a differently configured schedule has
   headroom.
3. The two weakest classes (Bacterial Red 0.829, EUS 0.850) and the Bacterial Red ↔
   Aeromoniasis confusion are consistent, interpretable errors, not noise.

Changes that the evidence does **not** motivate: a different backbone family (latency 9 ms and
4 M params are already well inside the deployment budget), or class re-weighting (data is
balanced).

## 12. Recommendation

- **Keep EXP-001 (partial-stage epoch 9, val Macro-F1 0.9004) as the reference baseline.**
- Run the paper-informed single-factor experiments (Gate 5) against it, priority order:
  1. Schedule: stop after the partial stage / shorten the full stage, or reduce full-stage
     backbone LR — directly targets the observed regression.
  2. CLAHE off vs on (the only preprocessing choice with no paper support).
  3. Stronger augmentation (P4-style crop 0.8–1.0 + jitter ±20%) and/or head dropout — targets
     the train/val gap.
  4. Optimiser variants from the papers (P1 Adam 1e-3/1e-5 two-phase; P4 SGD 0.01 + StepLR).
- Do not touch the test split until a single final candidate is chosen (Gate 8/9).

Test set read during this analysis: **NO**.
