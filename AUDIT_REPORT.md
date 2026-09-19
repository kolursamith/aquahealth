# AquaHealth AI — Repository Audit Report

**Audit date:** 2026-09-18 · **Branch:** `release/final-completion` (HEAD `79a216d`) · **Audit type:** read-only inventory of what is *implemented* and what was *actually executed*. Nothing in the repository was modified, installed, trained or re-run for this audit; the only file created is this report.

**Evidence standard used throughout:** a technique counts as "executed" only if the repository contains generated artifacts (metrics files, checkpoints, per-image predictions, resolved config with environment stamps) that could only exist if the code ran. Docstrings, README statements, PRD text and notebook cells with cleared outputs are treated as *claims*, not evidence.

---

## 0. Executive summary (one paragraph)

The repository is a **single-model EfficientNet-B0 transfer-learning project** with a genuinely rigorous data pipeline (SHA-256 + perceptual-hash de-duplication, group-aware frozen 70/15/15 split, one-shot test evaluation) and six real Colab/Tesla-T4 experiment runs whose artifacts are complete and mutually consistent. CLAHE and standard (torchvision v2) augmentation are implemented, wired, and were executed. **However: none of the five hybrid architectures exist in code, 10-fold cross-validation does not exist in any form, GAN augmentation does not exist in any form, and only one dataset is present (referenced by a symlink that is currently broken).** The project's own documentation is honest about this — it explicitly states YOLO, LSTM/Transformer and cGAN were not built. The teacher's new methodology is therefore largely *unimplemented*, not *misreported*.

---

## 1. Project overview

| Item | Fact (from repository) |
|---|---|
| Task | 8-class single-fish disease image classification (7 diseases + Healthy Fish) |
| Model actually trained | torchvision `efficientnet_b0` (ImageNet-1K V1 weights) + `Dropout(0.2) → Linear(1280, 8)`; 4,017,796 parameters — [src/model.py:50](src/model.py:50) |
| Training engine | `src/train.py` (`fit`) + `src/finetune.py` (`run_schedule`, 3-stage head→partial→full) driven by `scripts/run_experiment.py` with JSON configs in `configs/` |
| Experiments executed | EXP-001 … EXP-006 (six runs, 30 epochs each, Colab Tesla T4, torch 2.14.0+cu130, Python 3.13.15) — `results/experiments/EXP-00x/config.json` `environment` block |
| Final model | EXP-004 (`models/final_model.pth`, SHA-256 `3045a949…`, byte-identical to `drive_export/experiments/EXP-004/best_model.pth` — verified in this audit) |
| Official test | one run, 509 images, accuracy 0.9450 / Macro-F1 0.9453 — `results/final_test/` |
| App | Streamlit UI (`app/`) over `src/predict.Predictor`, with Grad-CAM (`src/explainability.py`) |
| Tests | 469 collected (`pytest --collect-only`, not executed for this audit) |
| Dependencies | torch, torchvision, numpy, pillow, opencv-python-headless, matplotlib, streamlit. **No** `ultralytics`, `timm`, `scikit-learn`, `albumentations` — `requirements/*.txt` |

---

## PART 1 — Complete repository inventory

Legend — **Impl?** = contains executable implementation · **Used?** = evidence it was actually executed/consumed · Stage = teacher-workflow stage.

### 1.1 `src/` (library code)

| Path | Type | Purpose | Pipeline stage | Impl? | Used? | Evidence |
|---|---|---|---|---|---|---|
| `src/__init__.py` | py | package marker | — | no | — | empty |
| `src/config.py` | py | constants: 8 class names, IMAGE_SIZE 224, SEED 42, paths, BATCH 32, LR 3e-4, WD 1e-4 | config | yes | yes | imported by every module |
| `src/dataset.py` | py | `ImageFolderDataset`, `build_dataloader`, Pillow RGB decode | Load dataset | yes | partly | `load_image`/`build_dataloader` used by the manifest path; `ImageFolderDataset` only used by tests/CLI fallbacks (`data/train/` is empty) |
| `src/manifest.py` | py | frozen split manifest I/O, SHA-256 digest guard, `perceptual_dhash`, `group_aware_stratified_split`, `ManifestDataset`, `build_split_dataset` | Leakage checks, split | yes | **yes** | `data/split_manifest.csv` + `.sha256` exist; every experiment `config.json` records `manifest_sha256 b7d1fccb…` |
| `src/preprocessing.py` | py | `CLAHEConfig`, `CLAHE` (cv2, LAB-L), `PreprocessConfig`, `build_eval_transform`, `denormalize` | Preprocessing, CLAHE | yes | **yes** | every checkpoint stores `preprocess.clahe={clip 2.0, tile 8}`; `results/final_test/metrics.json` `preprocess` block |
| `src/augmentation.py` | py | `AugmentConfig`, `build_train_transform` (train-only) | Standard augmentation | yes | **yes** | `results/experiments/*/config.json` `augment` block; `smoke_local.json` check "train pipeline contains augmentation" |
| `src/model.py` | py | EfficientNet-B0 builder, head swap, block freezing | Model | yes | **yes** | checkpoints rebuild via `Checkpoint.build_model()` |
| `src/train.py` | py | `TrainConfig`, `fit` (epoch loop, AMP, best-by-val-F1 checkpointing, resume), `save/load_checkpoint`, CLI | Initial training | yes | **yes** | `metrics.csv` per experiment (30 rows each) |
| `src/finetune.py` | py | `Stage`, `run_schedule` (head→partial→full) | Training / correction (unfreezing) | yes | **yes** | `checkpoints/{head,partial,full}/` in `drive_export/experiments/*` |
| `src/validation.py` | py | `run_validation` (eval mode, no grad, refuses random transforms) | Validation | yes | **yes** | `val_*` columns in every `metrics.csv` |
| `src/metrics.py` | py | accuracy / macro P-R-F1 / weighted F1 / confusion (pure torch, no sklearn) | Evaluation | yes | yes | all metrics files |
| `src/evaluate.py` | py | checkpoint-driven evaluation, exports metrics/predictions/misclassified/confusion/ECE; warns on `test` | Final test evaluation | yes | **yes** | `results/experiments/*/eval_val/*`, `results/final_test/*` |
| `src/predict.py` | py | `Predictor` (rebuilds eval transform + model from checkpoint), risk + quality integration | Inference | yes | yes | `results/final_backend_integration.md`, app wiring |
| `src/explainability.py` | py | Grad-CAM on `features[-1]`; re-applies CLAHE for overlay geometry | Inference (XAI) | yes | yes | `results/final_ui_ux_integration.md` |
| `src/image_quality.py` | py | dark/blur advisory | Inference | yes | yes | app + tests |
| `src/risk_engine.py` | py | confidence thresholds → LOW/MODERATE/HIGH/HEALTHY | Inference | yes | yes | app + tests |
| `src/device.py`, `src/environment.py`, `src/utils.py` | py | device resolution, env report, seeding/logging | infra | yes | yes | `environment` block in configs |

### 1.2 `scripts/`

| Path | Purpose | Stage | Impl? | Used? | Evidence |
|---|---|---|---|---|---|
| `scripts/build_split_manifest.py` | audits real dataset (corrupt, SHA-256 exact dups, dHash near-dups, label conflicts), writes frozen manifest + audit report; refuses to overwrite | Leakage checks, split | yes | **yes** | `results/data_audit.csv` (3,503 rows), `results/data_audit_report.md`, `data/split_manifest.csv` |
| `scripts/run_experiment.py` | config-driven training on manifest train/val, writes metrics/curves/confusion/best model; never reads `test` | Training | yes | **yes** | six `results/experiments/EXP-00x/` folders |
| `scripts/smoke_train.py` | one-batch GPU smoke test + pipeline assertions | QA | yes | yes | `results/smoke_local.json` |
| `scripts/compare_experiments.py` | tabulates experiment folders | reporting | yes | yes | `results/verified_experiment_comparison.csv` |
| `scripts/verify_environment.py` | env gate | infra | yes | yes (CI) | `.github/workflows/ci.yml` |
| `scripts/create_split.py` | **older** folder-copy stratified split into `data/train,val,test` | split (legacy) | yes | **no** | `data/train`, `data/val`, `data/test` contain only `.gitkeep`; superseded by manifest |
| `scripts/audit_dataset.py` | **older** folder audit (SHA-256 exact dups only) | audit (legacy) | yes | **no evidence** | no output file from it exists; superseded by `build_split_manifest.py` |

### 1.3 `configs/`

| Path | Purpose | Used? | Evidence |
|---|---|---|---|
| `exp001_efficientnet_b0_baseline.json` | CLAHE on, AdamW, 3 stages | yes | `results/experiments/EXP-001/config.json` (resolved copy, started 2026-09-12T11:37:37 UTC) |
| `exp002_clahe_off.json` | single-factor: CLAHE off | yes | EXP-002 artifacts |
| `exp003_p1_two_phase_adam.json` | Adam, 2 stages | yes | EXP-003 artifacts |
| `exp004_p4_sgd_steplr.json` | SGD 0.9 / 0.01 / StepLR(5, 0.1) — **final model** | yes | EXP-004 artifacts |
| `exp005_p4_augmentation.json` | no rotation, +saturation jitter | yes | EXP-005 artifacts |
| `exp006_p3_adam_3e5.json` | Adam 3e-5 everywhere | yes | EXP-006 artifacts |

### 1.4 `data/`

| Path | Type | Purpose | Used? | Evidence / current state |
|---|---|---|---|---|
| `data/README.md` | md | instructions | — | **STALE**: says "2,400 images", lists classes `Columnaris, Dropsy, Fin_Rot, White_Spot` that do not exist in the real dataset |
| `data/original/.gitkeep`, `data/train/.gitkeep`, `data/val/.gitkeep`, `data/test/.gitkeep` | placeholder | — | — | split folders are **empty**; the manifest path is used instead |
| `data/original/aquahealth` | **symlink** → `/Users/sogo/genai/TEMP/Projects/AI_TECHTAHON_DOCUMENTS/New Dataset` | dataset reference | was | **BROKEN** — target no longer exists (the external folder is now named `…/Dataset`, and its `test_split/` folder is now `test_split&validation/`). All 3,405 manifest files resolve after those two renames (verified in this audit). Until fixed, `build_split_dataset()` and `tests/test_frozen_manifest.py::test_frozen_manifest_files_exist_locally` will fail locally. |
| `data/split_manifest.csv` | csv | 3,405 rows: `filepath,label,class_name,split` | **yes** | SHA-256 recomputed = `b7d1fccb…21d43`, matches `.sha256` and every experiment config |
| `data/split_manifest.sha256` | txt | immutability digest | yes | verified |

### 1.5 `models/`, `notebooks/`, `drive_export/`

| Path | Purpose | Used? | Evidence |
|---|---|---|---|
| `models/final_model.pth` (git-ignored, 32.5 MB) | final checkpoint | yes | SHA-256 `3045a949…` = `drive_export/experiments/EXP-004/best_model.pth`; used by `results/final_test/` |
| `models/README.md` | doc | — | slightly stale (says produced by `src/train.py`; actually by `scripts/run_experiment.py`) |
| `notebooks/aquahealth_real_training.ipynb` | Colab orchestration (mount Drive, unzip, pytest manifest, smoke test, `RUN_FULL_TRAINING=False` switch for EXP-001) | yes (indirect) | **all 12 code cells have `execution_count: null` and zero outputs** (outputs cleared per `notebooks/README.md`). Execution is evidenced only by the artifacts it produced on Drive (`/content/…` paths, Tesla T4 in configs). The notebook only names EXP-001; EXP-002…006 were launched with the same runner (config `manifest: /content/aquahealth/data/split_manifest.csv`). |
| `drive_export/experiments/EXP-00{1..6}/` (git-ignored) | raw Colab outputs: `best_model.pth`, `checkpoints/<stage>/{best,last}.pt`, `metrics.csv`, `config.json`, PNGs | yes | local copy of `MyDrive/AquaHealth/runs/experiments/` |
| `drive_export/experiments.zip` | archive of the above | — | — |

### 1.6 `results/`

| Path | Purpose | Generated by code? | Evidence |
|---|---|---|---|
| `results/data_audit.csv` (3,503 rows) / `data_audit_report.md` | dataset audit + split report | **yes** | columns `sha256,dhash,status,exact_dup_group,near_dup_group,split` |
| `results/experiments/EXP-00x/{config.json,metrics.csv,inference_benchmark.json,experiment_report.md,training_curves.png,confusion_matrix*.png}` ×6 | per-run training record | **yes** | `started` timestamps 2026-09-12 11:37–17:13 UTC, `cuda_device_name: Tesla T4` |
| `results/experiments/EXP-00x/eval_val/{metrics.json,predictions.csv,misclassified.csv,confusion_matrix*.csv,classification_report.txt}` ×6 | local re-evaluation of each `best_model.pth` on `val` (Apple MPS) | **yes** | `data_root` = local repo; `checkpoint` = `drive_export/…` |
| `results/experiments/EXP-001/analysis.md` | hand-written overfitting/convergence analysis of EXP-001 | analysis (human) | quotes `metrics.csv` values (verified against CSV) |
| `results/final_test/*` (metrics.json, predictions.csv 509 rows, misclassified.csv 28 rows, confusion CSV/PNG, classification_report, final_test_report.md, inference_benchmark.json) | **one-shot official test** | **yes** | `checkpoint: models/final_model.pth`, `samples: 509` |
| `results/final_model_selection.{json,md}` | selection rationale EXP-004 vs EXP-005 | human + numbers | numbers match `eval_val/metrics.json` |
| `results/verified_experiment_comparison.{csv,md}`, `FINAL_MODEL_COMPARISON.csv`, `AquaHealthAI_MASTER_RESULTS.csv`, `AquaHealthAI_METRIC_PROVENANCE.csv` | comparison tables | script + human | consistent with per-experiment artifacts |
| `results/paper_technique_matrix.{csv,md}`, `final_paper_technique_matrix.{csv,md}`, `paper_reproducibility.md` | 5 research papers → 18 techniques, 5 executed | human | states LSTM/YOLO/cGAN/attention **not executed** |
| `results/YOLO12_AUDIT.md`, `yolo_annotation_audit.md`, `docs/layers/layer-12-yolo-decision.md` | YOLO feasibility decision | human | conclude: **0 bounding boxes; YOLO not implemented** |
| `results/AquaHealthAI_MASTER_EXPERIMENT_DOCUMENTATION.md` (624 lines), `AquaHealthAI_RESULTS_REPORT.md`, `AquaHealthAI_EXPERIMENT_TIMELINE.md`, `FINAL_PROJECT_REPORT.md` | narrative reports | human | consistent with artifacts |
| `results/final_backend_integration.md`, `final_end_to_end_qa.md`, `final_ui_ux_integration.md`, `final_release_audit.md` | QA records | human | — |
| `results/smoke_local.json` | pipeline assertions | yes | 20 checks all `ok: true` |
| `results/README.md` | doc | — | stale (refers to `data/test/` folder; actual is manifest split) |

### 1.7 `docs/`, `app/`, `tests/`, misc

| Path | Purpose | Notes |
|---|---|---|
| `docs/{PROJECT_OVERVIEW,ARCHITECTURE,DATASET,MODEL_AND_TRAINING,EXPERIMENTS,EVALUATION,EXPLAINABILITY,FRONTEND,DEPLOYMENT}.md` | project docs | accurate to code (checked §15); explicitly list YOLO/LSTM/Transformer/cGAN as out of scope |
| `docs/architecture/{system_architecture,training_pipeline,inference_pipeline}.md` | diagrams | EfficientNet-only |
| `docs/layers/layer-00…11-*.md` | per-layer build records | accurate; `layer-05-dataset.md` still shows synthetic placeholder class names |
| `docs/layers/layer-12-yolo-decision.md` | YOLO deferral record | "No YOLO code, dependency or weights are added" |
| `docs/research/README.md`, `docs/team/student_{1..4}.md` | pointers / roles | — |
| `app/main.py`, `app/theme.py`, `app/state.py`, `app/components/{upload,result_card,risk_card,dashboard,how_it_works}.py` | Streamlit UI | inference only; loads `models/final_model.pth` or `$AQUAHEALTH_CHECKPOINT` |
| `tests/test_*.py` (27 files, 469 tests) | unit/integration tests on synthetic fixtures + frozen-manifest guards | `test_frozen_manifest.py` asserts digest `b7d1fccb…`, 70/15/15 ratios, canonical labels, files exist |
| `.github/workflows/{ci,python-checks}.yml` | CI: pytest, ruff, black, mypy, env verify | **no training in CI** |
| `pyproject.toml`, `requirements/{base,app,experiments,dev}.txt`, `.python-version`, `.streamlit/config.toml`, `.claude/launch.json`, `LICENSE`, `README.md`, `.gitignore` | config | dependency list has no ultralytics/timm/sklearn/albumentations |
| `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/` | tooling caches | ignored |

---

## PART 2 — Actual preprocessing code (incl. CLAHE)

### 2.1 Location

| Item | Value |
|---|---|
| File | [src/preprocessing.py](src/preprocessing.py) |
| CLAHE class | `CLAHE` — lines 79–118; `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` at lines 103–106; applied at lines 109–115 |
| Config | `CLAHEConfig(clip_limit=2.0, tile_grid_size=8)` lines 46–55; `PreprocessConfig(image_size=224, resize_size=256, resize_mode="crop", clahe=None, mean=ImageNet, std=ImageNet)` lines 58–76 |
| Geometric stage | `resize_transforms()` lines 121–134: `v2.Resize(256, bicubic, antialias)` → `v2.CenterCrop(224)` (or squash mode) |
| Tensor stage | `tensor_transforms()` lines 137–143: `PILToTensor → ToDtype(float32, scale=True) → Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` |
| Canonical eval pipeline | `build_eval_transform()` lines 146–153 |

### 2.2 Exact code (CLAHE core)

```python
# src/preprocessing.py:101-115
def _handle(self) -> cv2.CLAHE:
    if self._clahe is None:
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=(self.config.tile_grid_size, self.config.tile_grid_size),
        )
    return self._clahe

def __call__(self, image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        raise ValueError(f"CLAHE expects an RGB image, got mode {image.mode!r}")
    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = self._handle().apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
```

```python
# src/preprocessing.py:146-153
def build_eval_transform(config: PreprocessConfig = PreprocessConfig()) -> v2.Compose:
    stages: list = []
    if config.clahe is not None:
        stages.append(CLAHE(config.clahe))
    stages += resize_transforms(config)
    stages += tensor_transforms(config)
    return v2.Compose(stages)
```

### 2.3 What it does
PIL RGB → (optional) CLAHE on the L channel of LAB (chroma untouched) → Resize shorter side to 256 (bicubic) → CenterCrop 224 → uint8→float32 [0,1] → ImageNet mean/std normalisation → `(3,224,224)` float tensor. **Default in the library is CLAHE OFF** (`clahe=None`); it is switched ON by the experiment configs (`"clahe": {"clip_limit": 2.0, "tile_grid_size": 8}`) and is thereby stored in every checkpoint.

### 2.4 Where it is used (A–E classification)

| Use | Path | Level |
|---|---|---|
| Training (train split) | `scripts/run_experiment.py:227-229` → `build_train_transform(preprocess, augment)` which prepends `CLAHE` when configured ([src/augmentation.py:52-53](src/augmentation.py:52)) | **E** — executed (EXP-001, 003–006 CLAHE on; EXP-002 off) |
| Validation | `scripts/run_experiment.py:230-232` → `build_eval_transform(preprocess)`; `src/train.py:646`; `src/finetune.py:318` | **E** — `val_*` columns in all `metrics.csv` |
| Test | `src/evaluate.py:251-253, 271` → `build_eval_transform(checkpoint.preprocess)` | **E** — `results/final_test/metrics.json` `preprocess.clahe = {2.0, 8}` |
| Inference (app) | `src/predict.py:221` → `build_eval_transform(self.checkpoint.preprocess)` | **E** — QA records; `results/final_test/inference_benchmark.json` "full_predict_incl_preprocess" |
| Grad-CAM overlay | `src/explainability.py:60` re-applies `CLAHE` for geometry | C/E |
| Notebook | not directly; the notebook shells out to `scripts/run_experiment.py` | D (cleared outputs) |

**Conclusion:** CLAHE is *implemented, imported, called, executed on Colab, and has generated results* (A→E all satisfied). It is applied identically to train, val, test and inference (train additionally has augmentation after CLAHE). The CLAHE-off ablation (EXP-002, val Macro-F1 0.9182) actually **beat** CLAHE-on baseline (EXP-001, 0.9004) under AdamW; the final model (EXP-004) nevertheless has CLAHE on because the combination "SGD+StepLR + CLAHE off" was never run.

---

## PART 3 — Actual data augmentation

| Item | Value |
|---|---|
| File | [src/augmentation.py](src/augmentation.py) |
| Function | `build_train_transform(config: PreprocessConfig, augment: AugmentConfig)` lines 47–71 |
| Library | `torchvision.transforms.v2` — **not** Albumentations (PRD says Albumentations; not installed) |
| Connected to Dataset/DataLoader at | `scripts/run_experiment.py:227-229` (`build_split_dataset("train", transform=build_train_transform(...))`); also `src/train.py:619`, `src/finetune.py:297`, `scripts/smoke_train.py:93` |

Exact operations and defaults (`AugmentConfig`, lines 28–44):

| Order | Transform | Parameters (default / EXP-005) | Probability |
|---|---|---|---|
| 0 | `CLAHE` (only if `config.clahe` set) | clip 2.0, tile 8 | deterministic |
| 1 | `v2.RandomResizedCrop(224)` | scale (0.8, 1.0), ratio (0.9, 1.1), bicubic | always |
| 2 | `v2.RandomHorizontalFlip` | p = 0.5 | 0.5 |
| 3 | `v2.RandomRotation` | ±15° (EXP-005: 0°) | always (angle sampled) |
| 4 | `v2.ColorJitter` | brightness 0.2, contrast 0.2, saturation 0.0 (EXP-005: 0.2) | always |
| 5 | `PILToTensor → ToDtype(float32) → Normalize(ImageNet)` | identical to eval | — |

**Train-only?** Yes. Val/test use `build_eval_transform` (no random ops). Enforced at runtime by `src/validation.py:69-76` `assert_deterministic_loader`, which raises if any `Random*` transform is found in the validation loader. Confirmed by `results/smoke_local.json` checks "val/test pipelines contain NO augmentation".

**No vertical flip, no GaussianBlur, no ShiftScaleRotate, no MixUp/CutMix, no GAN.**

> **Answer:** *The existing project actually used torchvision-v2 standard augmentation on the training split only — RandomResizedCrop(224, scale 0.8–1.0, ratio 0.9–1.1) + RandomHorizontalFlip(0.5) + RandomRotation(±15°) + ColorJitter(brightness 0.2, contrast 0.2), applied after CLAHE and before ImageNet normalisation. EXP-005 varied this to no rotation + saturation jitter 0.2. Validation, test and inference are un-augmented.*

---

## PART 4 — GAN / cGAN audit

Search performed over every `.py`, `.ipynb`, `.json`, `.yaml/.yml`, `.toml`, `.txt`, `.md` outside `.venv`/`.git` for: `GAN`, `cGAN`, `DCGAN`, `conditional gan`, `generator`, `discriminator`, `latent`, `synthetic image`, `fake image`, `adversarial`.

| Hit | Context | Relevance |
|---|---|---|
| `src/dataset.py:156,165`, `src/train.py:346-406` | `torch.Generator` for reproducible shuffling / RNG state | **not a GAN** |
| `tests/conftest.py:1` | "synthetic image folders" = coloured-noise unit-test fixtures | **not a GAN** |
| `docs/PROJECT_OVERVIEW.md:23`, `results/final_paper_technique_matrix.md:73`, `results/AquaHealthAI_MASTER_EXPERIMENT_DOCUMENTATION.md:60` | state that cGAN augmentation is **out of scope / has no support** | documentation only |
| External PRD §3, §11 | "cGAN: DROPPED entirely … This is a final decision" | documentation only |

1. Implemented? **No.** 2. Trained? **No.** 3. Generated images? **No.** 4. Saved? **No.** 5. Added to training set? **No.** 6. Used during model training? **No.**

**STATUS: NOT IMPLEMENTED.**
**"No evidence found that GAN augmentation was actually used."**

---

## PART 5 — 10-fold cross-validation audit

Search over all code and docs for `KFold`, `StratifiedKFold`, `cross_validate`, `cross_val_score`, `n_splits`, `10-fold`, `k-fold`, `fold`, `sklearn.model_selection`: **zero hits in code**; the only `sklearn` mention is a docstring in `src/metrics.py:7` noting metric equivalence (scikit-learn is not a dependency).

What exists instead: one fixed, seeded, group-aware stratified **hold-out** split (70/15/15) in `src/manifest.py::group_aware_stratified_split` (lines 175–210), consumed by every experiment. Each experiment trains **once** on `train` and selects the best epoch on `val`.

**STATUS: NOT IMPLEMENTED** (option 1). No fold generation, no per-fold retraining, no per-fold metric aggregation exists. The test split *is* untouched (which is a prerequisite that will carry over), but a hold-out split is **not** cross-validation.

---

## PART 6 — Dataset audit

### 6.1 Datasets present

| Property | Value | Source |
|---|---|---|
| Datasets in repository | **ONE** (referenced, not copied): `data/original/aquahealth` → external folder | `.gitignore`, symlink |
| Claimed origin | Kaggle "irfanulhuda/fish-disease-detection-dataset" (as cited by paper P1) — *stated in `docs/DATASET.md`; not independently verified in this audit* | docs |
| Delivered form | `AI_TECHTAHON_DOCUMENTS/DATASET.zip` (106 MB, 3,504 entries) | `results/YOLO12_AUDIT.md` |
| Current external location | `/Users/sogo/genai/TEMP/Projects/AI_TECHTAHON_DOCUMENTS/Dataset/{train_split/<8 folders>/, test_split&validation/ (flat), test.csv}` — 3,503 images (3,453 jpg · 30 jpeg · 20 png) | `find` in this audit |
| Symlink state | **broken** (points to renamed `New Dataset`) | `readlink` |
| Labels | folder-based for `train_split`; for the flat `test_split` folder: `test.csv` (`filename,label`) with filename-prefix fallback | `scripts/build_split_manifest.py:65-70, 148` |
| Bounding boxes / annotations | **none** | `results/data_audit_report.md`, `YOLO12_AUDIT.md` |

**Additional datasets: none present.** The teacher's multi-dataset requirement is unmet; any additional dataset(s) must be sourced, licence-checked, audited with the same de-dup/leakage procedure, and have their class names mapped onto (or extending) the canonical 8-class list. No candidate dataset names are asserted here because none has been verified.

### 6.2 Per-class counts (valid images, before de-dup) — `results/data_audit_report.md`

| Class (label) | train_split | test_split | total |
|---|---|---|---|
| Bacterial Red Disease (0) | 397 | 28 | 425 |
| Aeromoniasis (1) | 397 | 49 | 446 |
| Bacterial Gill Disease (2) | 397 | 44 | 441 |
| EUS Disease (3) | 395 | 53 | 448 |
| Saprolegniasis (4) | 397 | 19 | 416 |
| Parasitic Disease (5) | 399 | 37 | 436 |
| White Tail Disease (6) | 398 | 22 | 420 |
| Healthy Fish (7) | 400 | 47 | 447 |
| **total** | 3,180 | 299 | **3,479** valid (+24 label-conflict = 3,503) |

### 6.3 Quality facts

| Check | Result | Evidence |
|---|---|---|
| Corrupt / unreadable | 0 | `status` column of `data_audit.csv` |
| Exact duplicates (SHA-256) | 68 groups, 142 files → 74 copies dropped | `exact_dup_group` column; verified: no SHA-256 appears in more than one final split |
| Near duplicates (8×8 dHash, exact-hash match) | 334 groups; 59 straddled the vendor's own train/test folders (vendor split leaks) | `near_dup_group` column |
| Label conflicts (near-dup group with ≥2 labels) | 8 groups, 24 images **excluded** | `status == label_conflict` (24 rows) |
| Dimensions | 640×640: 2,099 · 128×128: 949 · 224×224: 427 · other: 4 | report |
| **Resolution correlates with class** (shortcut risk) | EUS and Healthy Fish are 100 % 640 px; Aeromoniasis 81 % 128 px | report table "Resolution by class" — *audited, not mitigated* |
| Duplicates across final train/val/test | **0** exact, **0** near-dup groups spanning splits (re-verified in this audit from `data_audit.csv`) | — |

### 6.4 Duplicate-checking code

| File | Method | Removed or only detected? |
|---|---|---|
| `src/manifest.py:79-90` | `file_sha256` (hashlib), `perceptual_dhash` (own 8×8 difference hash, no `imagehash` lib) | — |
| `scripts/build_split_manifest.py:167-196` | `assign_duplicate_groups`, `flag_label_conflicts`, `build_split` | **Removed from the split**: exact dups collapsed to one representative (`kept[r.sha256]`), near-dup groups forced into one split, conflicting groups dropped |
| `scripts/audit_dataset.py` (legacy) | SHA-256 only | detect-only; no evidence of use on real data |

**Target/data leakage:** filenames encode the class (`Bacterial Red disease_…_12.jpg`) but the dataset reads pixels only, so no feature leakage; the manifest guards duplicate leakage; resolution-class correlation remains an *unmitigated* potential shortcut (documented in `docs/PROJECT_OVERVIEW.md` Limitations).

---

## PART 7 — Current train/val/test split

| Item | DOCUMENTATION CLAIM | ACTUAL CODE | ACTUAL GENERATED FILE |
|---|---|---|---|
| Ratio | README/docs: 70/15/15; **PRD: 1,680/360/360 (210/45/45 per class) of 2,400** | `group_aware_stratified_split(ratios=(0.70,0.15,0.15))` per class with largest-remainder quotas, groups assigned largest-first to most-deficient split — `src/manifest.py:175-210` | 2,384 / 512 / 509 = 70.0 / 15.0 / 14.9 % |
| Seed | 42 | `SEED = 42` (`src/config.py:24`), `random.Random(seed)` | report: "seed: 42" |
| Stratification | per class | per class, group-aware | per-class counts 278–310 / 60–66 / 59–66 |
| Samples | 3,405 | — | 3,405 rows (recounted) |
| Saved manifest | yes | `write_manifest` + `.sha256` | `data/split_manifest.csv` sha `b7d1fccb…` |
| Reused | yes | `build_split_dataset` verifies digest on every call (`src/manifest.py:278`) | all six `config.json` carry the same `manifest_sha256` |
| Test frozen | yes | runner builds only train/val (`run_experiment.py:227-232`); `evaluate.py` warns on `test`; `build_split_manifest.py` refuses to overwrite | test evaluated once: `results/final_test/` (`TEST RUN COUNT = 1`) |

Class distribution of the manifest (train/val/test): BRD 291/62/62 · Aero 310/66/66 · BGD 307/66/66 · EUS 307/66/66 · Sapro 278/60/59 · Para 297/64/63 · WTD 286/62/61 · Healthy 308/66/66.

**Note:** the manifest was built over the *union* of the vendor's train_split and test_split folders (vendor test images: 200 train / 49 val / 50 test in the new split) because the vendor split was found to leak.

---

## PART 8 — Model audit

### 8.1 Only model that exists

```
MODEL:              EfficientNet-B0 classifier
ARCHITECTURE:       torchvision efficientnet_b0 (features: 9 blocks) + Sequential(Dropout(0.2), Linear(1280, 8))
BACKBONE:           EfficientNet-B0, EfficientNet_B0_Weights.IMAGENET1K_V1
HYBRID COMPONENT:   none
FILE:               src/model.py (build_classifier lines 50-84)
TRAINING NOTEBOOK:  notebooks/aquahealth_real_training.ipynb → scripts/run_experiment.py (outputs cleared)
WEIGHTS/CHECKPOINT: models/final_model.pth (EXP-004); drive_export/experiments/EXP-00{1..6}/best_model.pth
ACTUALLY TRAINED?:  YES — six runs on Colab Tesla T4, 30 epochs each
RESULTS AVAILABLE?: YES — validation for all six; official test for EXP-004
```

### 8.2 The five required hybrids

| # | Hybrid | Any code? | Any dependency? | Any weights/results? | Status |
|---|---|---|---|---|---|
| 1 | CNN + Vision Transformer + LSTM | **no** (`grep -i "vit\|transformer\|lstm"` → 0 code hits) | no timm / no ViT | no | **NOT IMPLEMENTED** |
| 2 | YOLO + EfficientNet | **no** (only a string in `build_split_manifest.py:254` saying no YOLO annotations were found) | no ultralytics | no; `results/experiments/HYB-YOLO-EFF-001/` explicitly *not created* | **NOT IMPLEMENTED** — deferred by `docs/layers/layer-12-yolo-decision.md`, rejected by `results/YOLO12_AUDIT.md` |
| 3 | CNN + BiLSTM | **no** (`nn.LSTM` absent) | — | no | **NOT IMPLEMENTED** |
| 4 | ResNet + Attention | **no** (`resnet`, `MultiheadAttention`, SLCAM absent; the word "attention" appears once, in a Grad-CAM UI docstring) | — | no | **NOT IMPLEMENTED** (paper P4's SLCAM listed as "not executed") |
| 5 | YOLO + Transformer | **no** | — | no | **NOT IMPLEMENTED** |

---

## PART 9 — YOLO + EfficientNet audit

**Finding: there is no YOLO of any version anywhere in the repository — no code, no import, no dependency, no weights, no detections, no crops, no experiment folder.** The statement "current work involved YOLO + EfficientNet" is not supported by the repository; the repository's own records say the opposite:

- `docs/layers/layer-12-yolo-decision.md`: "No YOLO code, dependency (`ultralytics` or otherwise), or weights are added."
- `results/YOLO12_AUDIT.md` §5: "YOLO12 SUPERVISED TRAINING: NOT SCIENTIFICALLY REPRODUCIBLE WITH CURRENT DATA (0 bounding-box files…). PRETRAINED DETECTOR INFERENCE-ONLY EXPERIMENT: NOT EXECUTED … No `ultralytics` dependency was added."
- `results/final_paper_technique_matrix.md:73`: "YOLO detection … have no support".

| Question | Answer |
|---|---|
| YOLO version | none |
| YOLO role (detection / feature extraction / preprocessing) | none — a *plan* exists (`layer-12-yolo-decision.md`: detector → highest box → 10 % margin crop → existing `Predictor`) but it was never coded |
| EfficientNet version | B0 (torchvision) |
| Connection YOLO→EfficientNet, tensor dims | n/a (EfficientNet receives `(N,3,224,224)` directly from the transform) |
| Genuinely hybrid? | **No — single model** |
| Training procedure of what exists | 3 stages: head 5 ep (blocks 0/9) → partial 10 ep (last 3 blocks) → full 15 ep (all 9); each stage a fresh optimiser, starts from previous stage's best |
| Loss | `F.cross_entropy` (`src/train.py`), no class weights, no label smoothing |
| Optimiser / LR (final EXP-004) | SGD momentum 0.9, lr 0.01 head & backbone, weight decay 1e-4, StepLR(step 5, γ 0.1) per stage |
| Batch / epochs | 64 (config says 32; runs overrode to 64 via `--batch-size`), 30 total epochs, AMP on |
| Augmentation | Part 3 defaults (rotation ±15°, no saturation) |
| Checkpoint | `models/final_model.pth` = stage `full`, epoch 3 (global epoch 18) |
| Validation (512) | acc 0.9336 · macro P 0.9345 · R 0.9347 · **F1 0.9330** · ECE 0.0215 · 34 misclassified |
| Test (509, once) | acc 0.9450 · macro P 0.9470 · R 0.9458 · **F1 0.9453** · ECE 0.0123 · 28 misclassified |

Architecture **as it actually exists**:

```
INPUT (PIL RGB, any size)
 ↓
CLAHE (LAB-L, clip 2.0, tile 8×8)
 ↓
Resize(256, bicubic) → CenterCrop(224)          [train: RandomResizedCrop/Flip/Rotation/ColorJitter instead]
 ↓
ToTensor float32 → Normalize(ImageNet)   → (3, 224, 224)
 ↓
EfficientNet-B0 features (9 MBConv blocks, ImageNet-1K pretrained) → (1280, 7, 7)
 ↓
AdaptiveAvgPool → (1280)
 ↓
Dropout(0.2) → Linear(1280, 8)
 ↓
softmax → 8 class probabilities → risk engine → CLASSIFICATION OUTPUT
```
**No YOLO stage exists between INPUT and EfficientNet.**

---

## PART 10 — Overfitting / underfitting analysis

### 10.1 What is recorded
Every experiment's `metrics.csv` has per-epoch `train_loss, train_accuracy, val_loss, val_accuracy, val_precision_macro, val_recall_macro, val_f1_macro, lr_head, lr_backbone`; `training_curves.png` (loss + F1/acc vs epoch with stage boundaries) was generated by `scripts/run_experiment.py::plot_curves` for all six runs (verified visually for EXP-004).

| Exp | Best global epoch | Last-epoch train acc | Last-epoch val acc | Gap | Min val loss (epoch) | Interpretation (from data) |
|---|---|---|---|---|---|---|
| 001 | 14 | 0.986 | 0.873 | 0.113 | 0.320 (15) | full stage overfits: val F1 falls 0.900→0.871 while train loss 0.30→0.065 |
| 002 | 15 | 0.990 | 0.904 | 0.085 | 0.266 (15) | same pattern, milder |
| 003 | 30 | 0.964 | 0.879 | 0.085 | 0.401 (30) | **under-trained**: still rising at epoch 30 (lr 1e-5 too low) |
| 004 | 18 | 0.995 | 0.910 | 0.085 | 0.216 (18) | best at full-ep 3; val loss then rises 0.216→0.26 and plateaus (StepLR decays lr) |
| 005 | 30 | 0.990 | 0.934 | 0.057 | 0.208 (30) | still improving at end; smallest gap (stronger augmentation) |
| 006 | 29 | 0.992 | 0.900 | 0.091 | 0.300 (30) | plateau |

### 10.2 Was analysis actually performed?
**Yes, for EXP-001** — `results/experiments/EXP-001/analysis.md` §2–§4, §11 is a genuine curve analysis (head stage not converged; partial stage did the work; full stage = "generalisation plateau with continued memorisation"; train 0.986 vs val 0.873). The numbers quoted match `metrics.csv`. For EXP-002…006 there is a comparison table but no dedicated overfitting write-up; `docs/EXPERIMENTS.md` notes "the full-unfreeze stage … hurt EXP-001" and that P4 optimiser/augmentation "make the full-unfreeze stage useful".

### 10.3 Regularisation mechanisms present in code

| Mechanism | Present? | Where |
|---|---|---|
| Dropout | yes, 0.2 in head (stock torchvision) | `src/model.py:28,80` |
| Weight decay | yes: 1e-4 (AdamW/SGD runs), 0 (Adam runs EXP-003/006) | configs |
| LR scheduler | StepLR only in EXP-004 | `src/train.py::build_scheduler` |
| Early stopping | **NO** — `fit()` always runs all epochs; best-epoch checkpoint selection by val Macro-F1 is used instead | `src/train.py:440-540`; `docs/MODEL_AND_TRAINING.md` "no early stopping" |
| Class weighting / label smoothing / MixUp | no | — |
| Frozen-BN for frozen blocks | yes | `src/train.py::set_train_mode` |

---

## PART 11 — Correction techniques

The repository frames its follow-ups as *paper-informed single-factor experiments* rather than as an explicit "detect overfitting → correct → retrain" loop; but `analysis.md` §12 does recommend corrections, two of which were then run. Only these have real before/after evidence:

| TECHNIQUE | WHY APPLIED | BEFORE (EXP-001 val Macro-F1) | AFTER (val Macro-F1) | EVIDENCE |
|---|---|---|---|---|
| Stronger augmentation (no rotation, crop 0.8–1.0, jitter ±0.2 incl. saturation) | `analysis.md` §11(2)/§12(3): "targets the train/val gap" | 0.9004 (gap 0.113) | **0.9335** (EXP-005, gap 0.057) | `results/experiments/EXP-005/` |
| Optimiser change + LR reduction schedule (SGD 0.9, lr 0.01, StepLR ×0.1 / 5 ep) | §12(4) optimiser variants; also addresses full-stage regression | 0.9004 | **0.9330** (EXP-004) | `results/experiments/EXP-004/` — selected as final |
| Remove CLAHE | §12(2) | 0.9004 | 0.9182 (EXP-002) | `results/experiments/EXP-002/` |
| Two-phase Adam (P1) | paper recipe | 0.9004 | 0.8783 (EXP-003, worse/under-trained) | `results/experiments/EXP-003/` |
| Adam 3e-5 (P3) | paper recipe | 0.9004 | 0.9014 (EXP-006) | `results/experiments/EXP-006/` |
| Staged unfreezing (head → partial → full) | design choice from the start, not a correction | — | — | all runs |
| Best-epoch selection on val Macro-F1 (early-stopping substitute) | design choice | — | — | `fit()` |
| **Not applied:** §12(1) "drop/shorten the full stage", higher dropout, combined EXP-004+EXP-005 (+CLAHE off), class weighting, label smoothing, GAN augmentation | — | — | — | `docs/EXPERIMENTS.md`: "Combinations … were not run" |

No metric above is invented; all are re-evaluated checkpoints (`eval_val/metrics.json`).

---

## PART 12 — Final evaluation

### VALIDATION RESULTS (model selection, 512 images, `results/experiments/*/eval_val/metrics.json`)
| Exp | Acc | Macro P | Macro R | Macro F1 | Weighted F1 | ECE | Misclassified |
|---|---|---|---|---|---|---|---|
| 001 | 0.9004 | 0.9018 | 0.9010 | 0.9004 | 0.9002 | 0.0555 | 51 |
| 002 | 0.9180 | 0.9207 | 0.9185 | 0.9182 | 0.9180 | 0.0316 | 42 |
| 003 | 0.8789 | 0.8798 | 0.8798 | 0.8783 | 0.8781 | 0.0862 | 62 |
| **004** | 0.9336 | 0.9345 | 0.9347 | **0.9330** | 0.9326 | 0.0215 | 34 |
| 005 | 0.9336 | 0.9342 | 0.9343 | 0.9335 | 0.9329 | 0.0218 | 34 |
| 006 | 0.9023 | 0.9047 | 0.9033 | 0.9014 | 0.9012 | 0.0330 | 50 |

### CROSS-VALIDATION RESULTS
**None — no cross-validation was performed.**

### FINAL TEST RESULT (`results/final_test/`, checkpoint `models/final_model.pth` = EXP-004 full/epoch 3, 509 images, run once on 2026-09-13 15:19 IST on Apple MPS with `python -m src.evaluate --split test`)

| Metric | Value |
|---|---|
| Accuracy | **0.9450** (481 / 509) |
| Macro precision / recall | 0.9470 / 0.9458 |
| **Macro F1** | **0.9453** |
| Weighted F1 | 0.9450 |
| Loss | 0.1916 |
| ECE (10 bins) | 0.0123 |
| Inference | 13.52 ms/img (MPS, model only), 16.71 ms incl. preprocessing; 10.48 ms on T4 |

Per-class (P / R / F1 / n): BRD 0.983/0.935/0.959/62 · Aero 0.940/0.955/0.947/66 · BGD 0.969/0.939/0.954/66 · EUS 0.952/0.894/0.922/66 · Sapro 0.967/1.000/0.983/59 · Para 0.966/0.889/0.926/63 · WTD 0.859/1.000/0.924/61 · Healthy 0.940/0.955/0.947/66.

Confusion matrix (rows = true, cols = predicted, order 0–7):
```
[58  1  0  1  0  0  2  0]
[ 0 63  1  0  0  0  2  0]
[ 0  0 62  1  0  1  1  1]
[ 1  3  0 59  0  0  1  2]
[ 0  0  0  0 59  0  0  0]
[ 0  0  1  1  2 56  2  1]
[ 0  0  0  0  0  0 61  0]
[ 0  0  0  0  0  1  2 63]
```

---

## PART 13 — Notebook / Colab execution evidence

| Item | `notebooks/aquahealth_real_training.ipynb` |
|---|---|
| Purpose | Colab orchestration: nvidia-smi, pip pins, mount Drive, unzip `DATASET.zip` to `/content/aquahealth_data/New Dataset`, symlink `data/original/aquahealth`, run manifest tests, dry-run audit, GPU smoke test, (switch-gated) `scripts/run_experiment.py --id EXP-001`, commented-out `src.evaluate --split val` |
| Dataset used | `DATASET.zip` from `MyDrive/AquaHealth/` |
| Preprocessing / augmentation / model / epochs | inherited from `configs/exp001…json` via the runner (CLAHE on; Part-3 augmentation; EfficientNet-B0; 5+10+15 epochs) |
| Cells executed? | **Cannot be shown from the notebook**: 12/12 code cells have `execution_count: null` and 0 outputs (outputs deliberately cleared, see `notebooks/README.md`). `RUN_FULL_TRAINING = False` is the committed state. |
| Outputs/results exist? | **Yes, externally**: `drive_export/experiments/EXP-00{1..6}/` and `results/experiments/…` carry Colab paths (`/content/drive/MyDrive/AquaHealth/runs/experiments/EXP-001/best_model.pth`), `Linux-6.6.122+-x86_64`, `Tesla T4`, `torch 2.14.0+cu130`, `started` stamps between 11:37 and 17:13 UTC on 2026-09-12 |
| Final metrics | see Part 12 |
| Checkpoints generated | 6 × `best_model.pth` + per-stage `best.pt`/`last.pt` (in `drive_export/`, git-ignored) |
| Other notebooks | none |

**Verdict:** the six experiments were actually run on Colab (artifact-proven), but the notebook itself carries no execution trace, and only EXP-001 is scripted in it; EXP-002…006 were launched with the same runner from cells that were not committed.

---

## PART 14 — Truth table

| Method | Exists in code? | Actually executed? | Evidence/result? | Current status |
|---|---|---|---|---|
| Dataset preprocessing | Yes — `src/preprocessing.py` (Resize 256 → CenterCrop 224 → ImageNet norm) | Yes | every checkpoint `preprocess`; `final_test/metrics.json` | **DONE** |
| Duplicate removal | Yes — SHA-256 + dHash, `scripts/build_split_manifest.py` | Yes | `results/data_audit.csv` (68 exact groups collapsed, 8 conflict groups excluded) | **DONE** (exact-hash dHash only; no Hamming-distance near-dup search) |
| Data leakage check | Partial — duplicate leakage across splits (yes); vendor split leak detected (59 groups); resolution/class shortcut audited | Yes | `data_audit_report.md`; 0 dup groups span final splits (re-verified) | **PARTIAL** — no explicit "target leakage"/"suspicious feature" test beyond resolution table; shortcut not mitigated |
| CLAHE | Yes — `src/preprocessing.py::CLAHE` (cv2, LAB-L, 2.0/8×8) | Yes | on in EXP-001,003–006 and final model; off-ablation EXP-002 | **DONE** (ablation shows it is not clearly beneficial) |
| Standard augmentation | Yes — `src/augmentation.py` (torchvision v2, train-only) | Yes | `config.json` augment blocks; EXP-005 variant | **DONE** |
| GAN augmentation | **No** | No | none | **NOT IMPLEMENTED** |
| Train/Val/Test split | Yes — frozen group-aware 70/15/15 manifest | Yes | `data/split_manifest.csv` sha `b7d1fccb…`, 2,384/512/509 | **DONE** (symlink currently broken locally) |
| 10-fold CV | **No** | No | none | **NOT IMPLEMENTED** |
| CNN + ViT + LSTM | **No** | No | none | **NOT IMPLEMENTED** |
| YOLO + EfficientNet | **No** (decision record only) | No | `YOLO12_AUDIT.md`: not created | **NOT IMPLEMENTED** |
| CNN + BiLSTM | **No** | No | none | **NOT IMPLEMENTED** |
| ResNet + Attention | **No** | No | none | **NOT IMPLEMENTED** |
| YOLO + Transformer | **No** | No | none | **NOT IMPLEMENTED** |
| Overfitting analysis | Yes — per-epoch train/val metrics + curves generated by runner | Yes | `metrics.csv`, `training_curves.png` ×6, `EXP-001/analysis.md` | **DONE for EXP-001; tabular only for others** |
| Correction/retraining | Partial — augmentation / optimiser+StepLR / CLAHE-off re-runs | Yes (EXP-002/004/005) | 0.9004 → 0.9335 / 0.9330 | **PARTIAL** (no early stopping; combinations not run; not framed as an explicit correction loop) |
| Final test evaluation | Yes — `src/evaluate.py --split test` | Yes, once | `results/final_test/` (509 images, acc 0.9450, F1 0.9453) | **DONE** (single model only) |

---

## PART 15 — Documentation vs real implementation

| Topic | Documentation says | Actual repository says | Difference |
|---|---|---|---|
| Dataset size | PRD: 2,400 images, 300/class; `data/README.md`: 2,400, classes incl. Columnaris/Dropsy/Fin_Rot/White_Spot | 3,503 files → 3,405 after de-dup; classes are BRD/Aero/BGD/EUS/Sapro/Para/WTD/Healthy | PRD + `data/README.md` are stale; README.md/docs/DATASET.md are correct |
| Number of datasets | (teacher requires multiple) | exactly one | additional datasets missing |
| Preprocessing | PRD: "Resize → CLAHE → Normalize"; README: "CLAHE → Resize 256 → CenterCrop 224 → ImageNet norm" | CLAHE → Resize(256) → CenterCrop(224) → Normalize (`src/preprocessing.py`) | order in PRD differs (CLAHE is applied *before* resize in code); README correct |
| CLAHE | PRD: "ON by default" | library default OFF (`clahe=None`); turned ON by configs; final model ON | consistent in effect; EXP-002 shows OFF was better under AdamW |
| Augmentation | PRD: "Albumentations (flip, rotate, brightness/contrast, crop — train only)" | torchvision v2 `RandomResizedCrop/HFlip/Rotation/ColorJitter`, train only; Albumentations not installed | library differs; content matches |
| Duplicate checking | PRD: "verify (counts, corrupt files, duplicates)" | SHA-256 exact + dHash near-dup + label-conflict exclusion, group-aware split | actual is *more* thorough than documented |
| Split | PRD: 1,680/360/360 (210/45/45 per class); README/docs: 2,384/512/509 | 2,384/512/509, seed 42, group-aware | PRD stale; docs correct |
| 10-fold CV | not claimed anywhere | not implemented | consistent — simply absent |
| GAN | PRD: "cGAN: DROPPED entirely"; docs: out of scope | not implemented | consistent — absent |
| Early stopping | PRD: "early stopping on val Macro-F1" | no early stopping; fixed 30 epochs + best-epoch checkpoint by val Macro-F1 (`docs/MODEL_AND_TRAINING.md` states this correctly) | PRD inaccurate; docs accurate |
| EfficientNet | PRD: `timm` EfficientNet-B0 | torchvision `efficientnet_b0` IMAGENET1K_V1 | library differs |
| YOLO + EfficientNet | `layer-12-yolo-decision.md`: deferred; `YOLO12_AUDIT.md`: not feasible, not created | no YOLO code/dependency/weights | consistent — **not implemented**; any external claim of YOLO work is unsupported |
| Metrics library | PRD: scikit-learn | pure-torch `src/metrics.py`; sklearn not installed | library differs |
| Batch size | `src/config.py` 32; configs 32 | runs used 64 (`--batch-size` override, recorded in resolved `config.json`) | resolved config is authoritative |
| Evaluation | docs: val for selection, test once | `run_experiment.py` never builds `test`; `final_test_report.md` "TEST RUN COUNT = 1" | consistent |
| Final metrics | README: val F1 0.9330 / test acc 0.9450, F1 0.9453 | `eval_val/metrics.json` 0.93299; `final_test/metrics.json` 0.94499 / 0.94529 | consistent |
| Notebook | README: "Colab training notebook" | notebook has cleared outputs, `RUN_FULL_TRAINING=False`, only EXP-001 scripted | execution proven by artifacts, not by the notebook |
| `results/README.md`, `models/README.md` | refer to `data/test/` folder and `src/train.py` producing the final model | manifest-based split; `scripts/run_experiment.py` produced it | minor staleness |
| Test count | README: 469 | `pytest --collect-only`: 469 | consistent |

---

## PART 16 — Summary sections

### 16.1 What has actually been completed
1. Dataset audit with corrupt check, exact + perceptual duplicate detection, label-conflict exclusion, vendor-split leak detection (`scripts/build_split_manifest.py`, `results/data_audit*.{csv,md}`).
2. Frozen, seeded, group-aware, stratified 70/15/15 split with SHA-256 immutability guard (`data/split_manifest.csv`).
3. Canonical preprocessing with optional CLAHE (`src/preprocessing.py`), identical across val/test/inference.
4. Train-only standard augmentation (`src/augmentation.py`) with a runtime guard against augmenting validation.
5. EfficientNet-B0 staged transfer learning engine with exact-resume checkpoints (`src/train.py`, `src/finetune.py`).
6. Six controlled single-factor experiments on Colab T4 with complete artifacts and local re-verification.
7. Per-epoch train/val metrics + learning curves for all runs; written overfitting analysis for EXP-001; corrective re-runs (EXP-004, EXP-005) with measured improvement.
8. Evidence-based final model selection on validation only; one-shot official test evaluation (509 images).
9. Inference API, risk engine, image-quality screening, Grad-CAM, Streamlit UI, 469 tests, CI lint/type/test gates.

### 16.2 What has NOT been completed
1. **Any of the five hybrid architectures** (CNN+ViT+LSTM, YOLO+EfficientNet, CNN+BiLSTM, ResNet+Attention, YOLO+Transformer) — no code, no dependencies, no results.
2. **10-fold cross-validation** — absent.
3. **GAN augmentation on training data** — absent.
4. **Multiple datasets** — only one; symlink to it is currently broken.
5. Explicit **target-leakage / suspicious-feature** tests (beyond the resolution-vs-class table); the resolution-class shortcut is unmitigated.
6. **Early stopping** (best-epoch selection is used instead); combination experiments (EXP-004 + EXP-005 ± CLAHE off) recommended by the project's own analysis were not run.
7. Overfitting write-ups for EXP-002…006 (numbers exist; analysis text does not).
8. Notebook execution provenance (outputs cleared; only EXP-001 scripted).

### 16.3 What must be implemented next to satisfy the teacher's methodology
| Teacher stage | Gap | Minimum work (design, not yet approved) |
|---|---|---|
| Load multiple datasets | one dataset | source ≥1 additional fish-disease image dataset; extend `SOURCE_FOLDER_TO_CLASS` mapping or add per-dataset class maps; re-run the audit per dataset and *across* datasets (cross-dataset duplicates) |
| Fix data reference | broken symlink; `test_split` renamed | repoint `data/original/aquahealth` (or add a path alias); the manifest expects `…/test_split/` |
| Leakage checks | duplicate leakage only | add explicit target-leakage test (filename/metadata never read as features — document), near-duplicate search with Hamming distance ≤ k, cross-dataset dedup, class-vs-resolution shortcut test (e.g., train a resolution-only classifier as a control) |
| 10-fold CV | absent | implement `StratifiedGroupKFold`-style folds over `train ∪ val` (groups = near-dup groups) keeping `test` frozen; per-fold retrain; aggregate mean ± std; new runner (`scripts/run_cv.py`) writing `results/cv/<model>/fold_k/` |
| GAN augmentation (train only) | absent | implement + train a conditional GAN (e.g., cGAN/DCGAN class-conditional) on **training folds only**; save synthetic images under a separate root; a flag to include them; measure FID/visual sanity; ablation with vs without |
| 5 hybrid models | absent | new `src/models/` modules: (1) CNN backbone → patch tokens → ViT encoder → LSTM over tokens; (2) YOLO detector crop → EfficientNet (requires boxes or an open-vocabulary detector + validation of crops); (3) CNN feature map → sequence → BiLSTM; (4) ResNet + attention module (SE/CBAM/SLCAM); (5) YOLO crop → ViT. Each needs: config, training via the existing engine (generalise `EfficientNet`-typed signatures in `src/train.py`, `src/model.py`), CV results, curves, test result |
| Overfitting detection → correction → retrain | partially done | formalise: per-fold curve analysis, an explicit detection rule (gap/val-loss trend), a documented correction per model, and retraining evidence |
| Final test | done for one model | repeat once per final hybrid model on the same frozen test split |

---

## PART 17 — WHAT I CAN TELL MY TEACHER TODAY

**Q1. What preprocessing did you use?**
Pillow RGB decode → CLAHE (OpenCV, on the L channel of LAB, clip 2.0, 8×8 tiles) → Resize shorter side to 256 (bicubic) → CenterCrop 224 → float32 → ImageNet mean/std normalisation. Implemented in `src/preprocessing.py`, stored in each checkpoint, and applied identically to validation, test and the app.

**Q2. Where is CLAHE implemented?**
`src/preprocessing.py`, class `CLAHE` (lines 79–118), using `cv2.createCLAHE` at lines 103–106 and `cv2.COLOR_RGB2LAB` at line 113. It was executed in 5 of 6 experiments and in the final model; the CLAHE-off ablation (EXP-002) scored slightly higher than the CLAHE-on baseline, so its benefit is not established.

**Q3. What augmentation did you use?**
torchvision-v2 standard augmentation in `src/augmentation.py`: RandomResizedCrop(224, scale 0.8–1.0), RandomHorizontalFlip(0.5), RandomRotation(±15°), ColorJitter(brightness 0.2, contrast 0.2); EXP-005 used no rotation + saturation 0.2. No GAN.

**Q4. Was augmentation applied only to training data?**
Yes. Validation/test/inference use `build_eval_transform` with no random ops, and `src/validation.py` raises an error if a random transform reaches the validation loader.

**Q5. Did you remove duplicate images?**
Yes. SHA-256 exact duplicates (68 groups) were collapsed to one image, perceptual-hash near-duplicates (334 groups) were kept inside a single split, and 8 near-duplicate groups with conflicting labels (24 images) were excluded. Evidence: `results/data_audit.csv`, `results/data_audit_report.md`.

**Q6. Did you perform data leakage checks?**
Partially. Duplicate leakage across train/val/test was checked and eliminated (0 groups span splits), and the vendor's own train/test split was found to leak (59 groups). A class-vs-resolution shortcut was detected but not mitigated. No separate target-leakage/suspicious-feature test was written.

**Q7. Did you use 10-fold cross-validation?**
Not yet implemented/executed. We used a single frozen 70/15/15 hold-out split (seed 42).

**Q8. Did you use GAN augmentation?**
Not yet implemented/executed. There is no GAN code in the repository.

**Q9. Which hybrid models have actually been trained?**
None of the five. Only a single EfficientNet-B0 classifier was trained (six configurations; final EXP-004: val Macro-F1 0.9330, test Macro-F1 0.9453 on 509 images). YOLO + EfficientNet was investigated and rejected because the dataset has zero bounding-box annotations; no YOLO code exists.

**Q10. What still needs to be done?**
Add at least one more dataset and re-audit; fix the dataset symlink; implement 10-fold (group-aware, stratified) CV with the test split still frozen; implement and train a conditional GAN on training folds only and ablate it; implement and train all five hybrid architectures under the same CV protocol; formalise the overfitting-detection → correction → retrain loop per model; evaluate each final model once on the frozen test split.

---

## Appendix — Answers A–G

- **A. CLAHE code:** `src/preprocessing.py` class `CLAHE` lines 79–118 (`cv2.createCLAHE` 103–106; LAB conversion 113–115); enabled via `PreprocessConfig.clahe` and `build_eval_transform` lines 146–153.
- **B. Augmentation code:** `src/augmentation.py` `build_train_transform` lines 47–71 (`AugmentConfig` lines 28–44); wired at `scripts/run_experiment.py:227-229`.
- **C. GAN used?** **No — NOT IMPLEMENTED.** No evidence found that GAN augmentation was actually used.
- **D. 10-fold CV used?** **No — NOT IMPLEMENTED.** Single hold-out split only.
- **E. Datasets present:** one — the delivered fish-disease image set (3,503 files, 8 classes) referenced via the (currently broken) symlink `data/original/aquahealth`; the committed manifest `data/split_manifest.csv` lists 3,405 of them. No other dataset exists in the repository.
- **F. Hybrids actually trained:** **none of the five.** Only EfficientNet-B0 (single model) was trained.
- **G. Missing for the teacher's workflow:** multiple datasets; symlink repair; explicit target/suspicious-feature leakage tests and shortcut mitigation; 10-fold CV runner; GAN (train-only) augmentation pipeline; five hybrid architectures (code, training, CV, test); early-stopping or an explicit overfitting-detection rule with documented corrections and retraining per model; per-model one-shot test evaluation; per-experiment overfitting write-ups for EXP-002…006; notebook execution provenance.
