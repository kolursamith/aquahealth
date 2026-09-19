# AquaHealth AI — Results Report (validation-only, evidence-tagged)

Status date: 2026-09-12 23:15 IST. Every number is tagged: **MEASURED RESULT** (from a saved
artifact in this repository), **OBSERVED COLAB LOG VALUE** (read from the Colab cell output; the
run's artifacts are on Google Drive and not yet copied into the repo), **PLANNED EXPERIMENT** (not
run), **PUBLISHED PAPER RESULT — NOT AQUAHEALTH RESULT**. There are **no test-set results**.

## 1. Task and data

8-class fish-disease classification (7 diseases + Healthy Fish) on the delivered 3,503-image
dataset, de-duplicated and frozen into 3,405 images: train 2,384 / validation 512 / test 509
(manifest SHA-256 `b7d1fccb…`). Validation is used for every comparison below; the test split has
never been opened.

## 2. Baseline — EXP-001 (MEASURED RESULT)

EfficientNet-B0 (torchvision, ImageNet-1K weights), `Dropout(0.2) → Linear(1280, 8)`, CLAHE on,
staged fine-tuning head (5 ep, 1e-3) → last 3 blocks (10 ep, 3e-4/3e-5) → all (15 ep, 3e-4/1e-5),
AdamW wd 1e-4, batch 64, AMP, seed 42, Colab Tesla T4.

| Metric (validation, 512 images) | Value |
|---|---|
| Macro-F1 | **0.9004** |
| Accuracy | 0.9004 (461/512) |
| Macro precision / recall | 0.9018 / 0.9010 |
| Weighted F1 | 0.9002 |
| Best epoch | stage `partial`, epoch 9 of 10 (global 14 of 30) |
| Training time | 25.8 min (30 epochs) |
| Parameters | 4,017,796 |
| Inference | 9.004 ms / image (batch 1, T4) |
| ECE | 0.0555 |
| Weakest / strongest class F1 | Bacterial Red Disease 0.829 / Healthy Fish 0.955 |

Source: `results/experiments/EXP-001/{metrics.csv, eval_val/metrics.json, inference_benchmark.json}`.
Full analysis: `results/experiments/EXP-001/analysis.md`. Key finding: the third (full-unfreeze)
stage never beat the partial stage (best 0.871) while train accuracy reached 0.986 — overfitting.

## 3. Paper-informed single-factor experiments

Each changes exactly one thing relative to EXP-001 (same data, seed, batch, epochs budget, selection
rule). Architecture is identical in all of them (EfficientNet-B0 + 8-way head).

| ID | Paper technique | Change | Best val Macro-F1 | Best stage/epoch | Time | Status / evidence class |
|---|---|---|---|---|---|---|
| EXP-001 | — (baseline) | — | **0.9004** | partial / 9 | 25.8 min | MEASURED RESULT |
| EXP-002 | P1/P4 preprocessing | CLAHE off | NOT RECORDED | — | — | ran 22:28–≤22:43 IST; outcome UNVERIFIED (possibly failed) |
| EXP-003 | P1 two-phase Adam | head 1e-3 → last 3 blocks @ 1e-5 for 25 ep, no full stage | NOT RECORDED (run 1 reached 0.840 at fine-tune epoch 13/25 before the frontend disconnect; its final artifacts were overwritten) | — | — | run 3 RUNNING since 22:43 IST |
| EXP-004 | P4 optimiser | SGD momentum 0.9, lr 0.01, StepLR ×0.1 / 5 ep | 0.9330 | full / 3 | 24.7 min | COMPLETED — OBSERVED COLAB LOG VALUE (artifacts on Drive) |
| EXP-005 | P4 augmentation | flip 0.5, crop 0.8–1.0, jitter ±0.2 incl. saturation, no rotation | 0.9335 | full / 15 | 23.6 min | COMPLETED — OBSERVED COLAB LOG VALUE (artifacts on Drive) |
| EXP-006 | P3 training | Adam 3e-5 everywhere | NOT RECORDED (last observed partial ep 5: 0.766) | — | — | COMPLETED on Colab (inferred); log not observed |

Provisional reading (to be confirmed from `metrics.csv` on Drive): both P4-derived changes improve
the baseline by about +0.033 Macro-F1, and both do so in the *full* stage, unlike EXP-001 — i.e.
SGD with step decay and stronger colour augmentation let the full-unfreeze stage generalise.
EXP-005 was still improving at its last epoch. No claim is made about EXP-002/003/006 until their
artifacts are read.

## 4. Architecture differences between experiments

None. All six experiments use the same network; the differences are preprocessing (EXP-002),
optimiser / learning-rate schedule (EXP-003, EXP-004, EXP-006) and augmentation (EXP-005). The
paper techniques that *would* change the architecture — P3's BN-Dense128-Dropout head, P4's SLCAM
attention module, P2's three-backbone fusion + SVM, P5's DINOv2 anomaly detector — are PLANNED /
NOT EXECUTED (P2 and P5 deliberately not pursued).

## 5. Paper techniques — identified vs executed

18 techniques identified (`results/paper_technique_matrix.csv`); 5 have an AquaHealth experiment
(EXP-002…006); 2 of those have completed runs with observed results (EXP-004, EXP-005); the rest
are not executed. Published paper numbers (P1 92.92 %, P2 99.59 %, P3 98.14 %, P4 98.05 %, P5 AUROC
0.93) are **PUBLISHED PAPER RESULTS — NOT AQUAHEALTH RESULTS**, obtained on different datasets or
on leaky splits, and are not comparable to the validation numbers above.

## 6. YOLO status

YOLO12 training: NOT EXECUTED — the dataset contains zero bounding-box annotations, so supervised
detector training is impossible without fabricating labels. A pretrained-detector crop stage was
assessed and judged not feasible (no fish class in COCO weights, lesion close-ups, 128-px images,
no way to validate crops). See `results/yolo_annotation_audit.md`.

## 7. Current best validation result

- Verified in the repository: **EXP-001, 0.9004** (CURRENT BEST VALIDATION CANDIDATE with artifacts).
- Provisional, pending artifact copy: **EXP-005, 0.9335** and **EXP-004, 0.9330**.
- FINAL MODEL: **NOT YET SELECTED**. OFFICIAL TEST: **NOT YET EXECUTED**.

## 8. Limitations

- Validation set is 512 images: one image ≈ 0.2 % accuracy; differences below ~0.01 Macro-F1 are
  within noise for a single seed. All runs use seed 42 only.
- Resolution correlates with class (EUS/Healthy all 640 px; Aeromoniasis mostly 128 px) — a
  shortcut-learning risk not yet probed.
- Many images are pre-augmented lesion crops from a Kaggle set; real-farm generalisation is unknown.
- Gate 1 numbers from Colab are not preserved as an artifact; EXP-004/005/006 results are on Drive
  only; EXP-003 run 1's artifacts were overwritten.
- Combination experiments, final selection, the one-shot test evaluation and the inference audit
  with a real checkpoint remain to be done.
