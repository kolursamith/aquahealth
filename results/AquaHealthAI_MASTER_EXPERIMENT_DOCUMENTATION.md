# AquaHealth AI — Master Experiment & Training Documentation (evidence audit)

Audit date: 2026-09-12, ~23:00 IST. Branch `feature/real-data-training`, HEAD `598273c`.
Author of the audit: Claude (Opus 5) working in the repository; every number below is tagged with its
source and its status class:

- **VALIDATION RESULT** — measured on the frozen validation split (512 images) by repository code.
- **OFFICIAL TEST RESULT** — measured on the frozen test split. **None exist yet.**
- **TECHNICAL SMOKE-TEST VALUE — NOT PERFORMANCE RESULT** — Gate 1 numbers.
- **PUBLISHED PAPER RESULT — NOT AQUAHEALTH RESULT** — numbers quoted from the five PDFs.
- **OBSERVED COLAB LOG VALUE — ARTIFACT NOT YET COPIED LOCALLY** — a number read from a Colab cell
  output during the session whose `metrics.csv` is on Google Drive but has not been copied into
  the repository. Treated as *provisional* until the artifact is copied.
- **USER-REPORTED — NEEDS ARTIFACT VERIFICATION** — a number that exists only in conversation.
- NOT EXECUTED / INTERRUPTED / QUEUED / RUNNING / COMPLETED / FAILED / NOT RECORDED — as defined
  in the audit brief.

Companion files produced by this audit:

| File | Content |
|---|---|
| `results/AquaHealthAI_MASTER_RESULTS.csv` | one row per experiment × epoch (validation), plus re-evaluation rows |
| `results/AquaHealthAI_METRIC_PROVENANCE.csv` | every important number with its source file / line |
| `results/AquaHealthAI_RESULTS_REPORT.md` | presentation-level results report |
| `results/AquaHealthAI_EXPERIMENT_TIMELINE.md` | timeline with actual timestamps |
| `results/paper_technique_matrix.csv` / `.md` | five-paper technique matrix (Gate 4), now with execution status |

---

## 1. Executive summary

- Problem: 8-class single-image fish-disease classification (7 diseases + Healthy Fish) with a
  confidence-based risk level, served through Streamlit.
- Data: the delivered Kaggle-derived "New Dataset" (3,503 files) was audited, de-duplicated and
  frozen into a 70/15/15 group-aware split of **3,405 images** (train 2,384 / val 512 / test 509),
  manifest SHA-256 `b7d1fccb…d708021d43`. The test split has **never been opened** by any training,
  selection or evaluation run.
- Model: torchvision EfficientNet-B0 (ImageNet-1K weights) with a `Dropout(0.2) → Linear(1280, 8)`
  head; 4,017,796 parameters.
- Baseline EXP-001 (Colab T4, 30 epochs, 25.8 min): **best validation Macro-F1 0.9004** at stage
  `partial`, epoch 9 (VALIDATION RESULT, `results/experiments/EXP-001/metrics.csv` line 15,
  reproduced by re-evaluating `best_model.pth`: 0.9004223).
- Paper-informed single-factor experiments: EXP-004 (P4 SGD+StepLR) and EXP-005 (P4 augmentation)
  **completed** on Colab with observed log values 0.9330 and 0.9335 (OBSERVED COLAB LOG VALUES;
  artifacts on Drive, not yet copied into the repo). EXP-006 completed on Colab (final value not
  observed). EXP-002 ran 22:28–≤22:43 (too short; status UNVERIFIED) and EXP-003 run 3 was **running** at audit
  time; EXP-003 run 1 probably completed in the background but its artifacts were overwritten.
- "0.904" is **not** an EXP-001 number. It is EXP-004's full-stage epoch-2 validation accuracy /
  Macro-F1 as printed at 3 decimals in the Colab log — not that run's best epoch. See §34.
- YOLO12: **NOT EXECUTED** and not executable — zero bounding-box annotations exist
  (`results/yolo_annotation_audit.md`).
- Final model selection (Gate 8), the one-shot frozen test evaluation (Gate 9) and the inference
  audit (Gate 10) have **not** happened. There is no `models/final_model.pth`.

## 2. Problem statement

Given one RGB photograph of a fish (or a fish lesion), predict which of eight classes it belongs to
and return a calibrated confidence with a three-level risk label for aquaculture operators. Source:
`AI_TECHTAHON_DOCUMENTS/PRD_AquaHealthAI_Final.md` ("Single-fish image → disease classification →
risk alert"; fish detection on pond images, water-quality prediction, GAN augmentation and
production ensembling are explicitly out of scope).

## 3. Project scope (as executed)

In scope and executed: data audit + frozen split; preprocessing (CLAHE optional); EfficientNet-B0
transfer learning with staged unfreezing; config-driven experiment runner; validation-only model
comparison; prediction API + risk engine + Streamlit app skeleton; five-paper technique matrix;
YOLO annotation audit. Not executed: Gate 6 combination experiments, Gate 8 selection, Gate 9 test
evaluation, Gate 10 inference audit with a real checkpoint, final ML audit.

## 4. Dataset

| Item | Value | Source |
|---|---|---|
| Delivered archive | `AI_TECHTAHON_DOCUMENTS/DATASET.zip`, 106,371,554 bytes, 3,504 zip entries (3,503 images + `test.csv`) | `unzip -l` |
| Extracted folder | `AI_TECHTAHON_DOCUMENTS/New Dataset/` → `train_split/<8 class folders>/`, flat `test_split/`, `test.csv` (`filename,label`) | filesystem |
| Repository reference | `data/original/aquahealth` is a **symlink** to the extracted folder (read-only use; nothing under it is modified or committed) | `ls -la data/original` |
| Colab copy | `DATASET.zip` in `MyDrive/AquaHealth/`, unzipped to `/content/aquahealth_data/New Dataset` each session | notebook cell 9 output |
| Origin (per P1) | Kaggle `irfanulhuda/fish-disease-detection-dataset` | paper P1 text |
| Files | 3,453 `.jpg`, 30 `.jpeg`, 20 `.png`, 1 `.csv`, 2 `.DS_Store` | `find` |
| Resolutions | 640×640: 2,099 · 128×128: 949 · 224×224: 427 · other: 4 | `results/data_audit_report.md` |

Resolution is strongly class-correlated (EUS and Healthy are 100 % 640×640; Aeromoniasis is 81 %
128×128) — a shortcut-learning risk recorded in the audit report. Many images are pre-augmented
(rotated with borders, mirrored copies) and many are lesion close-ups rather than whole fish
(contact-sheet inspection of 16 random *training* images, `results/yolo_annotation_audit.md` §3).

## 5. Dataset audit (executed by `scripts/build_split_manifest.py`, outputs committed)

| Quantity | Value | Evidence |
|---|---|---|
| Files discovered | **3,503** | `results/data_audit_report.md` line "files discovered: 3503"; `results/data_audit.csv` has 3,503 rows |
| Corrupt / unreadable | **0** | same report |
| Unmapped class folders | 0 | same |
| Exact-duplicate groups (SHA-256) | **68** groups, 142 images involved → 74 duplicate copies dropped | same; `data_audit.csv`: 74 rows `status=ok` with empty `split`, all with `exact_dup_group` |
| Near-duplicate groups (dHash, distinct bytes) | **334** | same |
| Near-dup groups spanning delivered train/test | 59 (the delivered split leaks) | same |
| Label-conflict near-dup groups | **8** groups, **24** images excluded (`status=label_conflict`) | same; `data_audit.csv` |
| Accounting | 3,503 − 74 duplicate copies − 24 label conflicts = **3,405** manifest rows | verified by counting `data_audit.csv` |
| YOLO / bounding-box annotations | none found | same report; `results/yolo_annotation_audit.md` |

The audit brief's "valid images 3,479" = 3,503 − 24 label-conflict images (the report's own
definition); it is *not* the manifest size.

## 6. Frozen split

- Method: per-class 70/15/15 largest-remainder quotas, seed 42, exact duplicates collapsed to one
  image, dHash near-duplicate groups kept inside one split (`src/manifest.py::group_aware_stratified_split`).
- Manifest: `data/split_manifest.csv` (3,405 rows + header = 3,406 lines), columns
  `filepath,label,class_name,split`; digest file `data/split_manifest.sha256` =
  `b7d1fccbb21e73a847e078aed921c02339505bc735199d311103f4d708021d43`.
- The digest is re-verified at every run (`verify_manifest_digest`); EXP-001's `config.json`
  records the same `manifest_sha256`. `tests/test_frozen_manifest.py` (30 tests, pass on Colab
  2026-09-12 and locally) pins counts, canonical labels and disjointness.

| Class | train | val | test | total |
|---|---|---|---|---|
| Bacterial Red Disease | 291 | 62 | 62 | 415 |
| Aeromoniasis | 310 | 66 | 66 | 442 |
| Bacterial Gill Disease | 307 | 66 | 66 | 439 |
| EUS Disease | 307 | 66 | 66 | 439 |
| Saprolegniasis | 278 | 60 | 59 | 397 |
| Parasitic Disease | 297 | 64 | 63 | 424 |
| White Tail Disease | 286 | 62 | 61 | 409 |
| Healthy Fish | 308 | 66 | 66 | 440 |
| **total** | **2,384** | **512** | **509** | **3,405** |

(Source: `results/data_audit_report.md` "Frozen split" table; re-counted from
`data/split_manifest.csv` during this audit — identical.)

## 7. Class mapping (canonical, `src/manifest.py::CANONICAL_CLASSES`)

| Label | Canonical class | Source folder |
|---|---|---|
| 0 | Bacterial Red Disease | `Bacterial Red disease` |
| 1 | Aeromoniasis | `Bacterial diseases - Aeromoniasis` |
| 2 | Bacterial Gill Disease | `Bacterial gill disease` |
| 3 | EUS Disease | `EUS` |
| 4 | Saprolegniasis | `Fungal diseases Saprolegniasis` |
| 5 | Parasitic Disease | `Parasitic diseases` |
| 6 | White Tail Disease | `Viral diseases White tail disease` |
| 7 | Healthy Fish | `Healthy Fish` |

`src/config.py::CLASS_NAMES` was a placeholder list from the brief until commit `598273c`, when it
was aligned to this order (a test now asserts equality). Checkpoints carry their own `class_names`.

## 8. Preprocessing (verified in `src/preprocessing.py`, `src/dataset.py`)

| Step | Actual implementation |
|---|---|
| Image loading | Pillow `Image.open(path).convert("RGB")` (`src/dataset.py` line 96) |
| CLAHE | OpenCV `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))` applied to the **L channel of LAB** (`cv2.COLOR_RGB2LAB` → apply → `COLOR_LAB2RGB`); PIL → PIL callable; optional (on in EXP-001/003/004/005/006, off in EXP-002) |
| Resize | `v2.Resize(256, interpolation=BICUBIC, antialias=True)` (shorter side) |
| Crop | `v2.CenterCrop(224)` |
| Tensor | `v2.PILToTensor()` → `v2.ToDtype(float32, scale=True)` |
| Normalisation | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| Output | float32 tensor (3, 224, 224) |
| Validation / test / inference | **identical pipeline** (`build_eval_transform`); `src/validation.py` refuses any transform containing Random*/ColorJitter stages for val/test |
| Seed | `set_seed(42)` seeds `random`, `numpy`, `torch`; DataLoader shuffle uses `torch.Generator().manual_seed(seed)` |

The pipeline in the audit brief matches the code. One nuance: CLAHE runs *before* the geometric
stage in both train and eval, so the augmentation crops a CLAHE-enhanced image.

## 9. Augmentation (train only, `src/augmentation.py`)

EXP-001 (repository default `AugmentConfig`): CLAHE (if on) → `RandomResizedCrop(224, scale=(0.8, 1.0),
ratio=(0.9, 1.1), BICUBIC)` → `RandomHorizontalFlip(0.5)` → `RandomRotation(±15°, BILINEAR)` →
`ColorJitter(brightness=0.2, contrast=0.2)` → tensor → ImageNet normalisation.
EXP-005 (P4 recipe): rotation 0°, `saturation=0.2` added (new optional field, default 0 = unchanged;
`tests/test_augmentation.py` covers it). No augmentation on val/test.

## 10. EfficientNet-B0 architecture (verified in `src/model.py`, `results/smoke_local.json`, EXP-001 artifacts)

| Item | Value |
|---|---|
| Framework / library | PyTorch 2.14.0 / torchvision 0.29.0 (`torchvision.models.efficientnet_b0`) |
| Pretrained weights | `EfficientNet_B0_Weights.IMAGENET1K_V1` → `efficientnet_b0_rwightman-7f5810bc.pth` (torchvision verifies the `7f5810bc` hash prefix on download) |
| Input | 3×224×224, ImageNet-normalised |
| Backbone | `model.features` = 9 blocks (stem + 7 MBConv stages + head conv), unchanged |
| Classifier head | `nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 8))`; weight init uniform ±1/√1280, bias 0 |
| Output | logits (N, 8); softmax in `src/model.py::logits_to_probabilities` |
| Total parameters | **4,017,796** (`results/smoke_local.json` model.total_parameters; EXP-001 `inference_benchmark.json`) |
| Trainable, head only (stage A) | **10,248** = 1280×8 + 8 (`smoke_local.json`; EXP-001 `metrics.csv` lines 2–6) |
| Trainable, last 3 blocks + head (stage B) | 3,165,988 (EXP-001 `metrics.csv` lines 7–16) |
| Trainable, all (stage C) | 4,017,796 (lines 17–31) |
| Activation | SiLU inside MBConv (torchvision default); none after the head |

## 11. Transfer learning

Backbone weights come from ImageNet-1K; only the head is new. Freezing is by feature block
(`set_trainable_blocks(model, last_n)`); frozen blocks are also kept in `eval()` mode so their
BatchNorm statistics do not drift (`src/train.py` line 158). Each stage starts from the previous
stage's best checkpoint (`src/finetune.py::run_schedule`).

## 12. Training strategy (Layer 9 staged schedule, `src/finetune.py::DEFAULT_SCHEDULE`)

| Stage | Epochs | Trainable | LR head | LR backbone |
|---|---|---|---|---|
| head (A) | 5 | head only | 1e-3 | — |
| partial (B) | 10 | last 3 of 9 blocks + head | 3e-4 | 3e-5 |
| full (C) | 15 | all | 3e-4 | 1e-5 |

Optimizer AdamW (wd 1e-4) by default; Adam / SGD(momentum) and per-stage StepLR are options added
for the paper experiments. Loss: cross-entropy. AMP: float16 + GradScaler on CUDA (`--amp`).
Gradient accumulation: none. Gradient clipping: none. Warmup: none. Early stopping: none — the
selection rule is "best validation Macro-F1 across all epochs of all stages" (`--selection-metric
f1_macro`), and `best_model.pth` is that epoch. Checkpoints (`checkpoints/<stage>/best.pt|last.pt`)
store model, optimizer, scheduler, scaler and RNG state (format v2).

## 13. Gate 1 (real-data smoke test, `scripts/smoke_train.py`)

Two executions exist:

**(a) Local, Apple MPS — artifact `results/smoke_local.json` (COMPLETED, PASS).** torch 2.14.0,
device mps, batch 32, AMP off, manifest rows 3,405, splits 2,384/512/509, model 4,017,796 params /
10,248 trainable, batch (32, 3, 224, 224) float32, initial loss **2.0953**, 17/17 checks OK.

**(b) Colab Tesla T4 — no artifact preserved.** The cell output was observed during the session
(CUDA True, Tesla T4 14.6 GB, torch 2.14.0+cu130, Python 3.13.15, batch 64, logits (64, 8), initial
loss 2.0894, post-update loss 1.9826, gradients finite YES, parameters changed YES, peak allocated
0.38 GB, SMOKE TEST: PASS) but the notebook cell was run without `--json`, and the output was lost
when the VM was recycled. **Status: USER/SESSION-REPORTED — NEEDS ARTIFACT VERIFICATION.** A second
smoke run on the rebuilt VM (20:29 IST) printed `SMOKE TEST: PASS` (observed in the page text before
the reload) — same caveat.

**THE SMOKE-TEST LOSS IS NOT A PERFORMANCE RESULT.** ln(8) = 2.079; a value near it only shows the
head is initialised sanely. Repository changes for Gate 1: `scripts/smoke_train.py` committed in
`9828d05` (never reads the test split; `report["test_split_read"] = False`); the notebook install-cell
fix was committed later in `9447bb4`.

## 14. EXP-001 — EfficientNet-B0 baseline (COMPLETED)

| Item | Value | Source |
|---|---|---|
| Objective | measure the existing Layer 9 pipeline on the real frozen split as the reference for every paper-informed change | `configs/exp001_efficientnet_b0_baseline.json` description |
| Hypothesis | staged fine-tuning of an ImageNet EfficientNet-B0 reaches a usable macro-F1 on 2.4 k images; later stages may overfit | analysis.md |
| Data | train 2,384 / val 512 (manifest sha256 b7d1…), test never read | `config.json` |
| Model | §10 | |
| Preprocessing / augmentation | §8 / §9, CLAHE on | `config.json` |
| Stages / LRs / epochs | §12 (5 + 10 + 15 = 30 epochs) | `config.json` |
| Optimizer | AdamW, wd 1e-4, no scheduler, AMP on | `config.json` |
| Batch / workers / seed | 64 / 2 / 42 | `config.json` (`--batch-size 64` overrode the file's 32) |
| Environment | Colab, Python 3.13.15, torch 2.14.0+cu130, Tesla T4, started 2026-09-12T11:37:37 UTC | `config.json` |
| Checkpoint strategy | per-stage `best.pt`/`last.pt`; run-level `best_model.pth` = best val Macro-F1 | runner |
| **Best val Macro-F1** | **0.9004223** (0.9004) | `metrics.csv` line 15; `eval_val/metrics.json` summary.f1_macro |
| Best stage / epoch | `partial`, epoch 9 (global epoch 14) | same |
| Val accuracy / macro P / macro R / weighted F1 / loss at best | 0.9004 (461/512) / 0.9018 / 0.9010 / 0.9002 / 0.3381 | `eval_val/metrics.json` |
| Training time | 1,549.8 s = **25.8 min** for 30 epochs (~44 s/epoch) | `inference_benchmark.json` train_seconds |
| Inference benchmark | **9.004 ms/image**, batch 1, CUDA T4, 100 images | `inference_benchmark.json` |
| Parameters | 4,017,796 total; 3,165,988 trainable at the selected stage | same |
| Checkpoint path | `MyDrive/AquaHealth/runs/experiments/EXP-001/best_model.pth` (Drive; not in git; not yet downloaded) | `eval_val/metrics.json` checkpoint |
| Misclassified (val) | 51 / 512 | `eval_val/metrics.json` |
| ECE (10 bins) | 0.0555; mean confidence 0.847 (correct 0.875, wrong 0.589) | same |

Per-class validation metrics (best checkpoint): Bacterial Red 0.829 F1 (P .836 R .823, n 62) ·
Aeromoniasis 0.870 (.833/.909, 66) · Bacterial Gill 0.928 (.983/.879, 66) · EUS 0.850 (.885/.818, 66)
· Saprolegniasis 0.942 (.934/.950, 60) · Parasitic 0.884 (.877/.891, 64) · White Tail 0.946
(.910/.984, 62) · Healthy Fish 0.955 (.955/.955, 66). Confusion matrix rows (true → predicted counts):
see `eval_val/metrics.json` "confusion"; largest confusion Bacterial Red → Aeromoniasis (5); only 3
diseased images (all EUS) were predicted Healthy.

Convergence / overfitting (from `metrics.csv`): head stage still rising at epoch 5 (0.772); partial
stage rose monotonically to 0.9004 at epoch 9 then 0.893; full stage never exceeded 0.871 while train
loss fell 0.30 → 0.065 and train accuracy reached 0.986 vs val 0.873 → generalisation plateau with
memorisation. Each stage boundary caused a transient drop (BatchNorm re-adaptation of newly
unfrozen blocks + fresh optimizer). Full analysis: `results/experiments/EXP-001/analysis.md`.

Was any EXP-001 epoch higher than 0.9004? **No.** Second best: partial epoch 10, 0.8926; best full
epoch: 15, 0.8706. Lowest val loss was partial epoch 10 (0.3203), not the selected epoch.

## 15. EXP-001 epoch-by-epoch (VALIDATION RESULTS, `results/experiments/EXP-001/metrics.csv`)

| stage | ep | global | train loss | train acc | val loss | val acc | val P | val R | **val F1** | lr head | lr bb | s | trainable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| head | 1 | 1 | 1.5670 | 0.4786 | 1.2734 | 0.6309 | 0.6492 | 0.6278 | 0.6240 | 0.001 | — | 44.6 | 10,248 |
| head | 2 | 2 | 1.0545 | 0.6925 | 1.0538 | 0.6973 | 0.7003 | 0.6963 | 0.6935 | 0.001 | — | 41.8 | 10,248 |
| head | 3 | 3 | 0.8378 | 0.7739 | 0.9233 | 0.7266 | 0.7296 | 0.7261 | 0.7255 | 0.001 | — | 42.3 | 10,248 |
| head | 4 | 4 | 0.7229 | 0.8142 | 0.8391 | 0.7461 | 0.7503 | 0.7452 | 0.7445 | 0.001 | — | 42.4 | 10,248 |
| head | 5 | 5 | 0.6424 | 0.8331 | 0.7765 | 0.7734 | 0.7764 | 0.7735 | 0.7724 | 0.001 | — | 42.8 | 10,248 |
| partial | 1 | 6 | 0.8561 | 0.7777 | 0.8460 | 0.7598 | 0.7616 | 0.7591 | 0.7562 | 0.0003 | 3e-05 | 44.5 | 3,165,988 |
| partial | 2 | 7 | 0.6033 | 0.8637 | 0.7043 | 0.8047 | 0.8052 | 0.8048 | 0.8025 | 0.0003 | 3e-05 | 42.4 | 3,165,988 |
| partial | 3 | 8 | 0.4706 | 0.8909 | 0.5979 | 0.8359 | 0.8379 | 0.8361 | 0.8352 | 0.0003 | 3e-05 | 44.6 | 3,165,988 |
| partial | 4 | 9 | 0.3754 | 0.9140 | 0.5220 | 0.8496 | 0.8529 | 0.8501 | 0.8494 | 0.0003 | 3e-05 | 45.2 | 3,165,988 |
| partial | 5 | 10 | 0.2939 | 0.9396 | 0.4576 | 0.8633 | 0.8648 | 0.8639 | 0.8632 | 0.0003 | 3e-05 | 46.1 | 3,165,988 |
| partial | 6 | 11 | 0.2373 | 0.9568 | 0.4195 | 0.8730 | 0.8758 | 0.8734 | 0.8729 | 0.0003 | 3e-05 | 45.6 | 3,165,988 |
| partial | 7 | 12 | 0.2041 | 0.9602 | 0.3884 | 0.8848 | 0.8860 | 0.8853 | 0.8841 | 0.0003 | 3e-05 | 44.3 | 3,165,988 |
| partial | 8 | 13 | 0.1681 | 0.9702 | 0.3655 | 0.8848 | 0.8868 | 0.8855 | 0.8844 | 0.0003 | 3e-05 | 45.6 | 3,165,988 |
| partial | 9 | 14 | 0.1408 | 0.9782 | 0.3381 | 0.9004 | 0.9018 | 0.9010 | 0.9004 | 0.0003 | 3e-05 | 45.8 | 3,165,988 |
| partial | 10 | 15 | 0.1218 | 0.9773 | 0.3203 | 0.8926 | 0.8947 | 0.8934 | 0.8926 | 0.0003 | 3e-05 | 42.7 | 3,165,988 |
| full | 1 | 16 | 0.2987 | 0.9098 | 0.4491 | 0.8594 | 0.8640 | 0.8605 | 0.8573 | 0.0003 | 1e-05 | 48.0 | 4,017,796 |
| full | 2 | 17 | 0.2355 | 0.9388 | 0.4450 | 0.8516 | 0.8575 | 0.8526 | 0.8499 | 0.0003 | 1e-05 | 41.4 | 4,017,796 |
| full | 3 | 18 | 0.2091 | 0.9425 | 0.4066 | 0.8672 | 0.8697 | 0.8684 | 0.8660 | 0.0003 | 1e-05 | 43.7 | 4,017,796 |
| full | 4 | 19 | 0.1835 | 0.9522 | 0.4283 | 0.8594 | 0.8645 | 0.8606 | 0.8575 | 0.0003 | 1e-05 | 45.6 | 4,017,796 |
| full | 5 | 20 | 0.1597 | 0.9614 | 0.3957 | 0.8652 | 0.8685 | 0.8661 | 0.8636 | 0.0003 | 1e-05 | 43.3 | 4,017,796 |
| full | 6 | 21 | 0.1464 | 0.9648 | 0.3862 | 0.8672 | 0.8690 | 0.8679 | 0.8648 | 0.0003 | 1e-05 | 43.0 | 4,017,796 |
| full | 7 | 22 | 0.1293 | 0.9685 | 0.4134 | 0.8516 | 0.8566 | 0.8525 | 0.8484 | 0.0003 | 1e-05 | 43.9 | 4,017,796 |
| full | 8 | 23 | 0.1160 | 0.9694 | 0.4084 | 0.8535 | 0.8615 | 0.8545 | 0.8516 | 0.0003 | 1e-05 | 44.2 | 4,017,796 |
| full | 9 | 24 | 0.1104 | 0.9786 | 0.3751 | 0.8711 | 0.8752 | 0.8719 | 0.8697 | 0.0003 | 1e-05 | 43.9 | 4,017,796 |
| full | 10 | 25 | 0.1024 | 0.9773 | 0.3794 | 0.8613 | 0.8654 | 0.8622 | 0.8591 | 0.0003 | 1e-05 | 45.2 | 4,017,796 |
| full | 11 | 26 | 0.0926 | 0.9773 | 0.3827 | 0.8613 | 0.8674 | 0.8620 | 0.8590 | 0.0003 | 1e-05 | 44.5 | 4,017,796 |
| full | 12 | 27 | 0.0826 | 0.9803 | 0.3929 | 0.8535 | 0.8594 | 0.8544 | 0.8504 | 0.0003 | 1e-05 | 44.1 | 4,017,796 |
| full | 13 | 28 | 0.0874 | 0.9786 | 0.3803 | 0.8633 | 0.8681 | 0.8638 | 0.8608 | 0.0003 | 1e-05 | 44.8 | 4,017,796 |
| full | 14 | 29 | 0.0735 | 0.9849 | 0.3820 | 0.8613 | 0.8679 | 0.8620 | 0.8585 | 0.0003 | 1e-05 | 44.6 | 4,017,796 |
| full | 15 | 30 | 0.0654 | 0.9857 | 0.3669 | 0.8730 | 0.8763 | 0.8735 | 0.8706 | 0.0003 | 1e-05 | 44.1 | 4,017,796 |

## 16. Experiment inventory (all experiments that exist as a config, a Drive directory or a log)

Every experiment uses the same architecture (§10), the same frozen split, batch 64, 2 workers,
seed 42, AMP, selection by validation Macro-F1, and **never reads the test split**
(`scripts/run_experiment.py` builds only `train` and `val` datasets). Times are IST unless marked UTC
(Colab clock = UTC = IST − 5:30).

| ID | Config / what changed vs EXP-001 | Paper | Status at audit | Start → end | Epochs | Best val Macro-F1 | Best stage/epoch | Time | Latency | Artifacts |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | repository defaults, CLAHE on, AdamW | none [C] | **COMPLETED** | 17:07 → ~17:34 (11:37→12:03 UTC) | 30/30 | **0.9004** (VALIDATION RESULT) | partial / 9 | 25.8 min | 9.004 ms | in repo `results/experiments/EXP-001/` + Drive |
| EXP-002 | CLAHE **off** | P1/P4 preprocessing | **status UNVERIFIED**: started 22:28 (Drive `config.json` + `checkpoints/`); the next queued cell (EXP-003 run 3) started at 22:43, i.e. only ~15 min later, which is shorter than a 30-epoch run (~24 min) → EXP-002 either FAILED or exited early; its Drive folder must be inspected (no `metrics.csv` was visible at 22:32) | 22:28 → ≤ 22:43 | NOT RECORDED | NOT RECORDED | — | — | — | Drive only |
| EXP-003 (run 1) | P1 two-phase Adam: head 1e-3 → last-3-blocks 1e-5 for 25 ep, no full stage, wd 0 | P1 | **INTERRUPTED from the frontend at 18:16; most likely COMPLETED in the background** (at 21:13 the runner refused the directory with "already holds a completed run", a message that requires `metrics.csv`, which the runner writes only after the last epoch) — **its artifacts were then deleted by the re-run's `rm -rf` at ~22:43**, so its final metrics were never captured: NOT RECORDED | 18:02 → est. 18:26 (12:32→~12:56 UTC) | last observed 5 + 13/25; probably 30/30 | last observed 0.840 at finetune ep 13 (OBSERVED COLAB LOG VALUE); final value NOT RECORDED | — | — | — | overwritten |
| EXP-003 (run 2) | same | P1 | **FAILED to start** (exit 2: directory already held the interrupted run) | 21:13 | 0 | — | — | — | — | log line only |
| EXP-003 (run 3) | same | P1 | **RUNNING** (Drive `EXP-003` folder re-created 22:43 IST by the queued cell's `rm -rf` + fresh start) | 22:43 → est. 23:10 | in progress | NOT RECORDED | — | — | — | Drive (in progress) |
| EXP-004 | SGD momentum 0.9, lr 0.01 head+backbone, StepLR ×0.1 every 5 epochs per stage, wd 1e-4 | P4 optimiser | **COMPLETED** on Colab (artifacts on Drive, **not yet copied locally**) | 21:13 → 21:38 (15:43→16:08 UTC) | 30/30 | 0.9330 (OBSERVED COLAB LOG VALUE) | full / 3 | 24.7 min (log) | NOT RECORDED locally | Drive: config, metrics.csv, benchmark, report, curves, CM, best_model.pth |
| EXP-005 | P4 augmentation: flip 0.5, crop 0.8–1.0, jitter ±0.2 b/c/**saturation**, rotation 0 | P4 augmentation | **COMPLETED** on Colab (Drive, not yet copied) | 21:38 → 22:02 (16:08→16:32 UTC) | 30/30 | 0.9335 (OBSERVED COLAB LOG VALUE; still rising at the last epoch) | full / 15 | 23.6 min (log) | NOT RECORDED locally | Drive |
| EXP-006 | Adam lr 3e-5 for every parameter in every stage, wd 0 | P3 training | **COMPLETED** on Colab (inferred: EXP-002 could only start after it exited; Drive folder created 22:02) — completion log **not observed** | 22:02 → ~22:28 | 30/30 (inferred) | NOT RECORDED (last observed: partial ep 5/10, 0.766) | NOT RECORDED | NOT RECORDED | NOT RECORDED | Drive |
| EXP-001 (Colab "section 5" cell re-run on the rebuilt VM, 20:3x) | — | — | not a new run: the runner refuses an existing directory without `--resume` (exit 2) | — | 0 | — | — | — | — | — |

Values marked OBSERVED COLAB LOG VALUE were read from `src.train`/`experiment` log lines in the
Colab cell output during the session (e.g. `experiment: EXP-004 done: val macro-F1 0.9330 (stage
full, epoch 3) in 24.7 min`). They must be confirmed by copying each run's `metrics.csv` /
`inference_benchmark.json` from Drive into `results/experiments/<ID>/` before they are used in any
selection or report (this is the next step in §32).

Combination experiments (Gate 6), the hybrid HYB-YOLO-EFF-001 (Gate 7B) and any LSTM/Transformer
experiment: **NOT EXECUTED** (no configs, no directories).

## 17. The five research papers (`AI_TECHTAHON_DOCUMENTS/Research papers/`)

| # | Title | Authors | Year / venue | Dataset (size, classes) | Backbone | Attention / Transformer / LSTM / YOLO / fusion | Optimizer, LR, scheduler, fine-tuning | Published result (PUBLISHED PAPER RESULT — NOT AQUAHEALTH RESULT) | Reproducible on AquaHealth? |
|---|---|---|---|---|---|---|---|---|---|
| P1 | Efficient Fish Disease Classification Using Fine-Tuned MobileNetV3Large for Mobile-Based Aquaculture Diagnosis | Dela Fifi Lusiana, Ellya Helmud, Rahmat Sulaiman | 2026, Sinkron 10(3) | Kaggle irfanulhuda fish-disease (2,400 images, 8 classes — the AquaHealth source data) | MobileNetV3Large (ImageNet) | none / none / none / none / none | Adam, categorical CE; phase 1 head lr 1e-3, phase 2 last 70 layers lr 1e-5; rotation + flip | test acc 92.92 % on their 240-image test split | protocol yes (EXP-003); their number is not comparable (their split leaks near-duplicates) |
| P2 | Empirical Evaluation of Deep Learning Techniques for Fish Disease Detection in Aquaculture Systems: A Transfer Learning and Fusion-Based Approach | Biswas, Muduli, Islam, Kanade, Zamani, Kanade, Parveen | 2024, IEEE Access | three datasets; own dataset-3 ~2,450 images, 7 classes (no Healthy) | VGG16 / MobileNetV2 / InceptionV3 | feature fusion of three CNNs + SVM; no attention/transformer/LSTM/YOLO | Adam vs RMSProp/Adadelta/SGD comparison; augmentation before splitting | VGG16 88.82 % acc; fusion+SVM 99.59 % | backbone comparison yes; fusion+SVM not pursued (3× inference cost, SVM head); augment-before-split not replicated |
| P3 | Comparison of EfficientNetB1 Model Effectiveness in Identifying Fish Diseases in South Asian Fish Diseases and Salmon Fish Diseases | Rahmanda Afridiansyah, De Rosal Ignatius Moses Setiadi | 2024, JAIC 8(2) | South-Asian 700 images 7 classes; SalmonScan 243 images 2 classes | EfficientNetB1 (ImageNet), BN→Dense128→Dropout→Dense head | none | Adam 3e-5, batch 64/32, 25 epochs, EarlyStopping(5); 150 px, no augmentation | 98.14 % (80/20 validation split) / 99.18 % | LR recipe yes (EXP-006); head variant NOT EXECUTED; 150 px not adopted |
| P4 | SLCAM-AquaNet: an attention-enhanced lightweight deep learning model for accurate classification of aquaculture disease | A. Athiraja, A. B. Gurulakshmi, M. Gurupriya, S. Nagarajan | 2026, Discover Applied Sciences 8:382 | own 5,260-image multi-species set, 5 classes | ResNet18/50, MobileNetV2 | SLCAM spatial-location-channel attention; no transformer/LSTM/YOLO | SGD momentum 0.9, lr 0.01, ×0.1 every 20 of 100 epochs, batch 32; flip 0.5, crop 0.8–1.0, jitter ±20 % | ResNet50+SLCAM 98.05 % acc / 98.2 F1 | optimiser and augmentation yes (EXP-004, EXP-005); SLCAM module NOT EXECUTED (needs new module code) |
| P5 | DINO-Patch: Unsupervised Fish Disease Detection via DINOv2 with Leave-One-Fish-Out Evaluation | Xingmou Liu, Ruiming Zhu, Liming Shi, Yihan Dong, Yao Xiao, Yuhua Liao | 2025 (preprint text; year from citation dates) | MatsyaDx-BD (own, specimen-level) + SalmonScan | DINOv2 ViT-S/14 frozen | Vision Transformer features + PCA foreground mask + k-NN memory bank; no YOLO (mentioned only as the supervised alternative), no LSTM | training-free | mean AUROC > 0.930 | not applicable to 8-class supervised classification; no specimen IDs for LOFO |

No paper provides bounding boxes, uses CLAHE, or builds a sequence for an LSTM. The five "hybrid
categories" named in the brief (CNN+ViT+LSTM, YOLO+EfficientNet, CNN+BiLSTM, ResNet+Attention,
YOLO+Transformer) match the papers only partially (P4 = ResNet+Attention; P5 = ViT only).

## 18. Paper technique matrix

`results/paper_technique_matrix.csv` (18 techniques) and `.md` carry the [A]/[B]/[C]/[D] origin
labels. **PAPER TECHNIQUES IDENTIFIED: 18. PAPER TECHNIQUES ACTUALLY EXECUTED (with a completed or
running AquaHealth experiment): 5** —

| Technique | Experiment | Execution status |
|---|---|---|
| P1 preprocessing without CLAHE (resize + ImageNet norm) | EXP-002 | ran; outcome UNVERIFIED (possibly FAILED) |
| P1 two-phase fine-tuning with Adam (1e-3 → 1e-5) | EXP-003 | run 1 probably completed but artifacts lost; run 3 RUNNING |
| P4 SGD momentum 0.9, lr 0.01, step decay | EXP-004 | COMPLETED (Drive) |
| P4 augmentation (flip, crop 0.8–1.0, jitter ±20 % incl. saturation) | EXP-005 | COMPLETED (Drive) |
| P3 Adam lr 3e-5 (+ best-checkpoint selection in place of EarlyStopping) | EXP-006 | COMPLETED (Drive, final value not observed) |

Identified but NOT EXECUTED: P1 rotation+flip-only augmentation; P1 MobileNetV3Large backbone; P2
fusion+SVM; P2 shift augmentation; P2 RMSProp/Adadelta; P3 BN-Dense128-Dropout head; P3 150 px input;
P3 EfficientNetB1; P4 SLCAM attention module; P4 acquisition-run leakage control (our near-dup
grouping is the analogue, [C]); P5 DINOv2 anomaly detection; P5 LOFO evaluation.

## 19. Paper experiments (EXP-002 … EXP-006) — what each actually is

Identified from each `configs/exp00N_*.json` description (not from the number):

| ID | Paper | Exact change vs EXP-001 (everything else identical) | Completed? | Interrupted? | Test untouched? |
|---|---|---|---|---|---|
| EXP-002 `exp002_clahe_off.json` | P1/P4 | `preprocess.clahe = null` | UNVERIFIED (see §16) | unknown | YES (runner never builds the test split) |
| EXP-003 `exp003_p1_two_phase_adam.json` | P1 | optimizer `adam`, wd 0; stages: head 5 ep lr 1e-3 → `finetune` 25 ep, last 3 blocks, lr 1e-5 head and backbone; no full stage | run 1 probably yes (artifacts lost); run 3 in progress | run 1 frontend-interrupted at finetune epoch 13/25 (val F1 0.840 observed) | YES |
| EXP-004 `exp004_p4_sgd_steplr.json` | P4 | optimizer `sgd`, momentum 0.9, lr 0.01 for head and backbone in all three stages, `lr_step_size 5`, gamma 0.1, wd 1e-4 | COMPLETED (log: 0.9330 full/3, 24.7 min) | no | YES |
| EXP-005 `exp005_p4_augmentation.json` | P4 | `rotation_degrees 0`, `saturation 0.2` (brightness/contrast 0.2, flip 0.5, crop 0.8–1.0 as before) | COMPLETED (log: 0.9335 full/15, 23.6 min) | no | YES |
| EXP-006 `exp006_p3_adam_3e5.json` | P3 | optimizer `adam`, wd 0, lr 3e-5 for head and backbone in all stages | COMPLETED (inferred), final value NOT RECORDED | no | YES |

Training-time observation for EXP-003 run 1 (log, not an artifact): the same stage-boundary drop as
EXP-001 (0.772 → 0.686) appeared even at lr 1e-5, supporting the BatchNorm-mode explanation in
`results/experiments/EXP-001/analysis.md`.

## 20. Interrupted / failed runs

1. **EXP-003 run 1 — frontend INTERRUPTED at 18:16 IST; background completion likely, artifacts lost.**
   Colab log observed: started 12:32:40 UTC, head stage 5/5 (val F1 0.772), finetune epochs 1–13 of
   25 (0.686 → 0.840, still rising ~+0.01/epoch). The Colab *frontend* disconnected at ~18:16 IST;
   the kernel evidently kept running, because at 21:13 IST (new VM) the runner refused
   `EXP-003` with "already holds a completed run" — `scripts/run_experiment.py` line 214 only refuses
   when `metrics.csv` exists, and `metrics.csv` is written once, after the final stage (line 328).
   So run 1 most probably finished (~18:26 IST) before the old VM was recycled (the new VM's uptime
   was 13 min at 20:28 IST). Whether EXP-004 also started on the old VM is unknown (it did not refuse
   on the new VM, so no `metrics.csv` existed for it). The queued re-run deleted `EXP-003/` at 22:43
   IST (`rm -rf` in the launcher cell) before anything was copied, so **run 1's final metrics are
   NOT RECORDED and cannot be recovered**. Its observed 0.840 (epoch 13/25) must not be combined with
   run 3.
2. **EXP-003 run 2 — FAILED to start** (21:13 IST): the launcher line's `pkill -f
   "[s]cripts/run_experiment"` matched the shell that contained the very same command text and
   killed it before `rm -rf`; the runner then refused the existing directory (`already holds a
   completed or partial run; pass --resume`, exit 2).
3. **EXP-002 attempt 1 — never started** on the first VM for the same `pkill` reason (interrupted by
   me within 2 min while diagnosing output buffering); attempt 2 is the run in progress.
4. **Colab page reload at ~22:20 IST** (Browser pane resize) dropped the visible outputs of the
   running cell; the kernel kept executing (EXP-006 finished, EXP-002 started from the server-side
   queue at 22:28). The notebook shown after reload is the GitHub copy (`9447bb4` notebook), so the
   ad-hoc launcher cells no longer appear in it.
5. First Drive mount on the rebuilt VM timed out (`ValueError: mount failed`, 2 min); the second
   attempt, authorised by the user in the browser, succeeded.

## 21. YOLO12 investigation

**YOLO12 TRAINING: NOT YET EXECUTED** — and not executable with this data.

Annotation audit (`results/yolo_annotation_audit.md`): the delivered archive contains 3,503 images
and one CSV of image-level labels; **0 bounding-box files, 0 boxes, no label format, no class-ID
scheme, no images with annotations**. Disease-classification folder labels are not detection
annotations. The PRD marks fish detection out of scope. A pretrained detector in inference-only mode
was **assessed, not tested**: COCO weights have no fish class; the images are largely lesion
close-ups at 128–640 px; crop quality could not be validated without boxes → HYB-YOLO-EFF-001
NOT FEASIBLE / NOT EXECUTED. No `ultralytics` dependency was added. The pre-real-data decision record
is `docs/layers/layer-12-yolo-decision.md`.

## 22. Current best validation candidate

**CURRENT BEST VALIDATION CANDIDATE (verified artifact in the repository): EXP-001**, val Macro-F1
0.9004, accuracy 0.9004, macro precision 0.9018, macro recall 0.9010, 4,017,796 parameters,
9.004 ms/image (T4), checkpoint `MyDrive/AquaHealth/runs/experiments/EXP-001/best_model.pth`.

**Provisional (OBSERVED COLAB LOG VALUES, artifacts on Drive, not yet copied):** EXP-005 0.9335
(full/15) and EXP-004 0.9330 (full/3) both exceed EXP-001 by ~+0.033. If their `metrics.csv` files
confirm these numbers, EXP-005 becomes the current best validation candidate. Neither is the FINAL
MODEL: Gate 8 selection has not been performed.

## 23. Final model selection protocol (Gate 8 — NOT EXECUTED)

1. Copy every completed run's artifacts from Drive into `results/experiments/<ID>/` and run
   `python -m src.evaluate --split val` for per-class metrics and ECE.
2. `python scripts/compare_experiments.py --ids EXP-001 … --baseline EXP-001 --out
   results/paper_experiments_comparison.csv` (validation only).
3. Optionally ≤ 2 combination experiments (Gate 6) — NOT EXECUTED.
4. Choose SELECTED_FINAL_CANDIDATE by validation Macro-F1 (tie-breaks: weakest-class F1, ECE,
   latency) → `results/final_model_selection.md`. No test data at any step.

## 24. Frozen test protocol (Gate 9 — NOT EXECUTED)

`python -m src.evaluate --checkpoint <selected best_model.pth> --split test --out-dir results/final`
exactly once; write `final_config.json`, `final_metrics.json/.csv`, `final_confusion_matrix.png`,
`final_report.md`, `final_benchmark.json`; copy the checkpoint to `models/final_model.pth`; record
"TEST SET USED: YES — once" and "MODEL CHANGED AFTER TEST: NO". **OFFICIAL FINAL TEST: NOT YET
EXECUTED.** No test score exists and none is estimated here.

## 25. Inference pipeline (`src/predict.py`, verified by `tests/test_prediction.py`)

`Predictor(checkpoint)` rebuilds the model and the eval transform from the checkpoint's stored
`preprocess` and `class_names`; `predict(image)` accepts PIL / numpy uint8 / bytes / path, returns
`PredictionResult` → dict with `predicted_class, confidence, risk, message` (+ `healthy`,
`ranked_predictions, model_version, preprocessing_version, api_version, status, warnings, error,
device`). Invalid input returns `status="error"` instead of raising. Not yet exercised with a real
trained checkpoint locally (no checkpoint has been downloaded from Drive) — Gate 10 audit
NOT EXECUTED.

## 26. Risk engine (`src/risk_engine.py`)

`get_risk_level(confidence)`: < 0.50 → LOW, 0.50–0.80 inclusive → MODERATE, > 0.80 → HIGH
(`RISK_THRESHOLD_MODERATE = 0.50`, `RISK_THRESHOLD_HIGH = 0.80` in `src/config.py`;
`tests/test_risk_engine.py`). Messages: disease wording ("…disease indication") vs Healthy Fish
wording ("no disease detected (Healthy Fish)"); LOW is "result is uncertain" for both. Calibration
evidence from EXP-001 validation: the > 0.9 confidence bin (285 images) had accuracy 1.000; 0.8–0.9
had 0.907; < 0.6 bins ≈ 0.5.

## 27. Streamlit integration (`app/main.py`, `app/components/`)

`select_predictor()` uses the real `src.predict` predictor when `models/final_model.pth` loads,
otherwise the mock predictor (fixed Aeromoniasis 0.87 response, labelled as mock). Result card shows
class + ranked probabilities; risk card shows the risk colour, with a green "HEALTHY — no disease
detected" variant. Because no final checkpoint exists in the repo, the app currently runs in mock
mode.

## 28. Technical stack — documented vs executed

| Technology | Actual role in this project | Executed version | Documented (PRD) |
|---|---|---|---|
| Python | all code | 3.11 (project pin, local `.venv`); **3.13.15 on Colab** (mismatch reported by `verify_environment.py`, all tests pass) | 3.11 |
| PyTorch | model, training loop, AMP, metrics | 2.14.0 (local, MPS) / 2.14.0+cu130 (Colab) | PyTorch |
| torchvision | EfficientNet-B0 + IMAGENET1K_V1 weights, `transforms.v2` pipelines | 0.29.0 / 0.29.0+cu130 | **PRD says `timm`** — DISCREPANCY: no timm anywhere in the code |
| OpenCV (headless) | CLAHE only | 5.0.0.93 | — |
| Pillow | image decoding | 12.3.0 | — |
| Augmentation | torchvision v2 transforms | — | **PRD says Albumentations** — DISCREPANCY: not used |
| NumPy | dHash, array handling | 2.4.6 | NumPy |
| Pandas | **not used** (csv module) | — | PRD lists Pandas — DISCREPANCY |
| scikit-learn | **not used**; metrics implemented in `src/metrics.py` with torch | — | PRD lists scikit-learn — DISCREPANCY |
| Matplotlib | training curves / confusion plots (`requirements/experiments.txt`) | 3.11.2 | — |
| Streamlit | demo UI | 1.63.0 (pinned; app runs in mock mode) | Streamlit |
| pytest / ruff / mypy / black | 436 tests, lint | 9.1.1 / 0.16.7 / 2.3.1 / 26.5.1 | — |
| Git / GitHub | `kolursamith/aquahealth`, branch `feature/real-data-training` (PR #2 open) | — | — |
| Google Colab | Tesla T4 15 GB, CUDA 13.0 driver 580.82.07 | — | — |
| Google Drive | `MyDrive/AquaHealth/{DATASET.zip, runs/experiments/…}` — checkpoints and run artifacts | — | — |
| AMP | `torch.autocast(float16)` + `GradScaler` on CUDA | on for all Colab runs | — |

## 29. Repository / code architecture (actual files)

| File | Purpose | Inputs → outputs | Pipeline position |
|---|---|---|---|
| `src/config.py` | constants: paths, `IMAGE_SIZE 224`, `SEED 42`, risk thresholds, canonical `CLASS_NAMES`, `CHECKPOINT_PATH models/final_model.pth` | — | shared |
| `src/manifest.py` | canonical classes, manifest read/write/digest, group-aware split, `ManifestDataset`, `build_split_dataset` | `data/split_manifest.csv` → datasets | data |
| `src/dataset.py` | Pillow decoding, `Sample`, `build_dataloader` (seeded shuffle, workers, pin memory) | files → tensors | data |
| `src/preprocessing.py` | `CLAHE`, `PreprocessConfig`, `build_eval_transform` | PIL → (3,224,224) float | data |
| `src/augmentation.py` | `AugmentConfig`, `build_train_transform` | PIL → tensor (train only) | data |
| `src/validation.py` | refuses random transforms on eval; stage-name introspection | transforms → checks | data |
| `src/model.py` | `build_efficientnet_b0`, `build_classifier`, `set_trainable_blocks`, `summarize`, softmax helper | — | model |
| `src/train.py` | `TrainConfig`, optimizer/scheduler/scaler builders, `train_one_epoch`, `fit`, checkpoint v2 save/load, `Checkpoint` | loaders → checkpoints, `EpochStats` | training |
| `src/finetune.py` | `Stage`, `OptimizerSettings`, `run_schedule` (staged unfreezing, carry best weights) | stages → per-stage checkpoints | training |
| `src/metrics.py` | accuracy / macro P-R-F1 / confusion (torch) | logits, labels → metrics | eval |
| `src/evaluate.py` | `evaluate_checkpoint`; CLI `--split val|test`; ECE, per-class, confusion, predictions CSV | checkpoint + split → `metrics.json` etc. | eval |
| `src/predict.py`, `src/risk_engine.py` | prediction API + risk levels/messages | image → result dict | serving |
| `src/device.py`, `src/environment.py`, `src/utils.py` | device resolution, environment report, seeding/logging | — | shared |
| `scripts/build_split_manifest.py` | audit + frozen split (refuses to overwrite) | dataset → manifest, audit CSV/MD | data (once) |
| `scripts/smoke_train.py` | Gate 1 one-batch smoke test, never opens test | manifest → report | pre-training |
| `scripts/run_experiment.py` | config-driven experiment runner → `config.json, metrics.csv, curves, CM, best_model.pth, inference_benchmark.json, experiment_report.md, checkpoints/` | config → run dir | experiments |
| `scripts/compare_experiments.py` | tabulate saved runs (validation only) | run dirs → comparison CSV | selection |
| `scripts/verify_environment.py`, `scripts/audit_dataset.py`, `scripts/create_split.py` | environment gate; earlier dataset tools | — | support |
| `configs/exp00*.json` | the six experiment definitions | — | experiments |
| `notebooks/aquahealth_real_training.ipynb` | Colab orchestration (clone, install, Drive mount, unzip, tests, smoke, EXP-001) | — | Colab |
| `app/main.py`, `app/components/{result_card,risk_card}.py`, `app/mock_prediction.py` | Streamlit UI | image → cards | serving |
| `tests/` (24 files, 436 tests) | unit/integration tests incl. frozen-manifest guard, runner, prediction contract | — | quality |
| `results/` | committed evidence: data audit, EXP-001 record + analysis, paper matrix, YOLO audit, this audit | — | records |
| `models/` | `README.md` only — **no checkpoint in the repo** | — | serving |
| `docs/layers/*.md` | foundation-build layer records (0–12) | — | docs |

## 30. Reproducibility

- Environment pins: `requirements/base.txt` (torch 2.14.0, torchvision 0.29.0, numpy 2.4.6,
  pillow 12.3.0, opencv-python-headless 5.0.0.93), `experiments.txt` (+ matplotlib 3.11.2),
  `app.txt` (streamlit 1.63.0), `dev.txt` (pytest, ruff, mypy, black).
- Determinism: seed 42 everywhere; the head stage of EXP-003 reproduced EXP-001's head stage
  epoch-for-epoch to 4 decimals (loss 1.5670 / F1 0.624 → 0.772) — same data order, same init.
- Every run records `config.json` with environment, manifest digest, image counts and class names;
  `best_model.pth` embeds preprocessing and class names.
- Not reproducible from the repo alone: the trained weights (Drive only) and the Colab Gate 1 output.

## 31. Current status (23:00 IST, 2026-09-12)

COMPLETED: foundation (Layers 0–11), data audit, frozen split, Colab environment, Gate 1 (local
artifact; Colab run observed), EXP-001 (record + analysis in repo), Gate 4 paper matrix, Gate 7A
annotation audit, Gate 10 code preparation (healthy-fish wording), EXP-004/EXP-005/EXP-006 training
on Colab (Drive artifacts).  RUNNING: EXP-003 (run 3, since 22:43).  UNVERIFIED: EXP-002 (ran
22:28–≤22:43, too short for 30 epochs).  NOT EXECUTED: Gate 6, Gate 7B, Gate 8, Gate 9, Gate 10
audit, final ML audit.  The Colab frontend has been stuck in "Connecting… / Resuming execution"
since the ~22:20 page reload, which is why the Drive folders could not be listed from the kernel.

## 32. Remaining work (exact order)

1. Reconnect the Colab frontend; let EXP-003 (run 3) finish; inspect EXP-002's folder (did it fail?). Do not start new runs.
2. Export every run's text artifacts from Drive to `results/experiments/<ID>/` (config.json,
   metrics.csv, inference_benchmark.json, experiment_report.md, plots) and run
   `src.evaluate --split val` per checkpoint → `eval_val/metrics.json`.
3. `scripts/compare_experiments.py` → `results/paper_experiments_comparison.csv`; replace every
   OBSERVED COLAB LOG VALUE in this audit with the artifact value.
4. Gate 6 (optional, ≤ 2 combinations), Gate 8 selection, download the selected `best_model.pth`
   to `models/final_model.pth`, Gate 9 one-shot test, Gate 10 inference audit, final audit files.

## 33. Exact experiment timeline

See `results/AquaHealthAI_EXPERIMENT_TIMELINE.md` (same evidence, chronological).

## 34. Metric provenance — where every number came from (full table: `results/AquaHealthAI_METRIC_PROVENANCE.csv`)

| Number | Meaning | Source | Class |
|---|---|---|---|
| **0.9004** | EXP-001 best val Macro-F1 (0.9004223) — and, coincidentally, EXP-001 val accuracy 0.900390625 = 461/512 | `results/experiments/EXP-001/metrics.csv` line 15; `eval_val/metrics.json` | VALIDATION RESULT |
| **0.904** | **NOT an EXP-001 value.** Found once: EXP-004, stage full, epoch 2 — `val loss 0.2991 acc 0.904 f1_macro 0.904` (3-decimal log rounding) in the Colab cell output; not EXP-004's best epoch (0.9330 at full/3). Not present in any repository file. | Colab log line (session), Drive `EXP-004/metrics.csv` line 18 (to be copied) | OBSERVED COLAB LOG VALUE |
| 0.900 | EXP-001 val accuracy 0.9004 rounded | as 0.9004 | VALIDATION RESULT |
| 2.0894 / 1.9826 | Colab Gate 1 initial / post-update loss, batch 64 | Colab cell output, not preserved | TECHNICAL SMOKE-TEST VALUE; USER/SESSION-REPORTED — NEEDS ARTIFACT VERIFICATION |
| 2.0953 | local Gate 1 initial loss, batch 32, MPS | `results/smoke_local.json` step.loss | TECHNICAL SMOKE-TEST VALUE |
| 0.38 GB | Colab Gate 1 peak allocated GPU memory | Colab cell output, not preserved | TECHNICAL — NEEDS ARTIFACT VERIFICATION |
| 25.8 min | EXP-001 training time = 1,549.8 s | `results/experiments/EXP-001/inference_benchmark.json` train_seconds | measured |
| 9.0 ms/image | 9.004 ms batch-1 CUDA | same file, ms_per_image_batch1 | measured (not a performance metric) |
| 4,017,796 | total parameters | `smoke_local.json`; EXP-001 `inference_benchmark.json` | verified |
| 10,248 | head-only trainable parameters | `smoke_local.json`; EXP-001 `metrics.csv` lines 2–6 | verified |
| 2,384 / 512 / 509 | split sizes | `data/split_manifest.csv` counts; `results/data_audit_report.md` | verified |
| 3,405 | manifest rows | `data/split_manifest.csv` | verified |
| 3,503 | source files | `results/data_audit.csv` rows; report | verified |
| 0.9330, 24.7 min | EXP-004 best val F1 (full/3), time | Colab log line `EXP-004 done: …` | OBSERVED COLAB LOG VALUE |
| 0.9335, 23.6 min | EXP-005 best val F1 (full/15), time | Colab log line `EXP-005 done: …` | OBSERVED COLAB LOG VALUE |
| 0.840 | EXP-003 run 1 val F1 at finetune epoch 13/25 (interrupted) | Colab log | OBSERVED COLAB LOG VALUE, partial run |
| 92.92 %, 99.59 %, 98.14 %, 98.05 %, AUROC 0.930 | P1, P2, P3, P4, P5 headline results | the PDFs | PUBLISHED PAPER RESULT — NOT AQUAHEALTH RESULT |

## 35. Discrepancies and caveats

1. **0.9004 vs 0.904** — different experiments (§34); the audit brief's "known result" for EXP-001
   (0.9004, partial, epoch 9, 25.8 min, ~9.0 ms) is fully confirmed by artifacts.
2. **PRD stack vs executed stack** — timm, Albumentations, scikit-learn, Pandas are documented in
   the PRD but not used; torchvision, torchvision.transforms.v2, torch-based metrics and the csv
   module are used instead.
3. **Python 3.13.15 on Colab vs 3.11 pin** — `verify_environment.py` reports the mismatch; the
   pinned packages installed and all tests passed anyway.
4. **Colab Gate 1 numbers have no artifact** (run without `--json`; output lost with the VM).
5. **EXP-004/005/006 results exist only on Drive and in observed logs** until copied into the repo.
6. **EXP-006 completion is inferred** (the next queued cell started), not observed.
7. `results/README.md` describes an older `results/<run-name>/` layout produced by `src.evaluate`;
   experiment records now live under `results/experiments/<ID>/` (runner layout).
8. `docs/layers/layer-05-dataset.md` still shows placeholder class names from the synthetic fixture;
   the canonical names are in `src/manifest.py` and (since `598273c`) `src/config.py`.
9. `data_audit_report.md` says "valid images: 3479" — this excludes the 24 label-conflict images but
   not the 74 exact-duplicate copies; the manifest has 3,405.
10. The Colab notebook committed in the repo is `9447bb4`'s version; the ad-hoc launcher cells used
    for EXP-002…006 were typed into the live session and are not in the repository. The exact
    commands are recorded in `results/AquaHealthAI_EXPERIMENT_TIMELINE.md`.
11. Local git tree is clean at `598273c`; `598273c` is **not yet pushed** (origin is at `9447bb4`)
    — the Colab clone used `9447bb4`, which contains everything the experiments needed.

## 36. Viva / judge explanation (two minutes)

"We took the 3,503 delivered fish images, found 68 exact-duplicate groups and 334 near-duplicate
groups — 59 of them straddling the vendor's own train/test folders — so we rebuilt a leakage-free
70/15/15 split of 3,405 images and froze it with a SHA-256 manifest that every run verifies. The
test split (509 images) has never been opened. Our model is torchvision's EfficientNet-B0 with
ImageNet weights and an 8-way head, trained on a Colab T4 in three unfreezing stages with optional
CLAHE. The baseline reached 0.9004 validation Macro-F1 in 26 minutes at 9 ms per image. We then ran
one-factor experiments taken from the five reference papers — no-CLAHE, P1's two-phase Adam
schedule, P4's SGD + step decay, P4's augmentation, P3's Adam 3e-5 — and the P4 optimiser and
augmentation runs reached about 0.93 on validation (artifacts on Drive, being copied in). YOLO was
investigated and rejected because the data has no bounding boxes at all. The final model has not
been selected and the test set has not been evaluated; that is the next, one-shot step."
