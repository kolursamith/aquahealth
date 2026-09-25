# Phase 12 — migration from Google Colab to the college lab GPU

Audit date: 2026-09-25 on `develop`. Nothing was trained, regenerated or re-split for this
document. It records what the repository holds, where the large artefacts are, and how the lab
machine continues the 100-experiment matrix **without retraining anything that is already done**.

## 0. Read this first — where the experiment state lives

| source | what it says | status |
|---|---|---|
| `results/v2/experiment_matrix.csv` (git) | 100 rows: **1 COMPLETED**, 99 PENDING, 0 RUNNING, 0 INTERRUPTED, 0 FAILED | verified |
| `results/v2/experiments/` (git) | one run: `cnn_vit_lstm_fold01_without_gan` (records only; no checkpoints) | verified |
| `MyDrive/AquaHealth/results_v2_datasetv3/` (Google Drive) | **the live registry.** The Phase 12 notebook (section 5) moves `results/v2` onto this Drive folder, so every `status.json`, `best.pt`, `latest.pt` and the refreshed `experiment_matrix.csv` written on Colab is here | **not visible from the audit machine** |

The project reports **15 completed experiments**. Only **1** is recorded in git, and the local
Google Drive mount (`samithkolur@gmail.com`) does not show `results_v2_datasetv3/`,
`gan_datasetv3/`, the GAN archive or the dataset bundle, so the other 14 could not be checked.
Until `results_v2_datasetv3/` has been copied onto the lab machine (section 4), the batch driver
cannot know about them, and it would train them again.

**Rule: do not start any batch on the lab machine until section 4 has been done and
`grep -c COMPLETED results/v2/experiment_matrix.csv` shows the expected count.**

## 1. Frozen configuration (do not change)

| item | value | where |
|---|---|---|
| dataset configuration | **v3**: four sources (`current_freshwater`, `kaptai`, `roboflow`, `paper_dataset`); MatsyaDx-BD/Mendeley excluded | `data/audit/DATASET_CONFIG_V3.md`, `src/multi_dataset.py` (`DATASET_CONFIG_VERSION = "v3"`) |
| clean corpus | 3,805 included images | `data/audit/clean_manifest.csv` |
| frozen final test | 761 images, **never read by training** | `data/audit/split_v3/final_test.csv` — sha256 `baa330347a89305212b08ac4c79a1bb6a98cdd7447172f4aa3aea113c16cb7a2` |
| development set | 3,044 images | `data/audit/split_v3/development.csv` — sha256 `68b5f8f7cd60f0fda6ecad2241f27a1fa2af4c0c35ceca9fcfc5ba4d940e6599` |
| CV | 10 group-aware folds, class/source-stratified | `data/audit/cv_v3/folds.csv` (sha256 `5cac023f…aa4a7f`) + `fold_XX_{train,validation}.csv`; all 21 digests in `data/audit/cv_v3/folds.sha256` |
| seed | **42** (split, folds, GAN, training) | `configs/cv_v2/default.json` run records, `configs/gan_v2/default.json` |
| classes (8) | Bacterial Red Disease, Aeromoniasis, Bacterial Gill Disease, EUS Disease, Saprolegniasis, Parasitic Disease, White Tail Disease, Healthy Fish | `data/audit/label_mapping.csv`, `src/multi_dataset.py` |
| preprocessing | CLAHE on LAB-L (clip 2.0, tile 8×8) → resize 256 → crop 224 → ImageNet mean/std | `configs/preprocess_v2_clahe.json` (sha256 `4c35039a…99ab7`), `data/audit/preprocessing_config.json` |
| augmentation | RandomResizedCrop scale 0.8–1.0 / ratio 0.9–1.1, h-flip 0.5, rotation 15°, brightness 0.2, contrast 0.2 | `configs/cv_v2/default.json` → `augment` |
| training | 30 epochs, batch 32, AdamW 3e-4 (backbone 3e-5), wd 1e-4, cosine, head 3 epochs then full, early stop 8 on val Macro-F1, AMP, pretrained | `configs/cv_v2/default.json` (the only full-run config; `configs/cv_v3/` does not exist) |
| GAN | cDCGAN, 64×64, z=100, Adam 2e-4 (0.5, 0.999), batch 64, 30 epochs, seed 42; one generator per fold on that fold's training rows only | `configs/gan_v2/default.json`, `docs/GAN_AUGMENTATION.md` §8 |
| matrix | 5 models (`cnn_vit_lstm`, `yolo_efficientnet`, `cnn_bilstm`, `resnet_attention`, `yolo_transformer`) × 10 folds × 2 arms (`without_gan`, `with_gan`) = 100 | `results/v2/experiment_matrix.csv` |
| environment of the recorded run | Python 3.13, torch 2.14.0+cu130, torchvision 0.29.0+cu130, Tesla T4, CUDA 13.0 | `results/v2/experiments/cnn_vit_lstm_fold01_without_gan/config.json` |

Experiment id = `<model>_fold<XX>_<arm>`.

## 2. Large artefacts — never in git

| artefact | location (Google Drive, `MyDrive/AquaHealth/`) | size | how it is verified | goes to (repo-relative) |
|---|---|---|---|---|
| dataset bundle (3,805 images) | `aquahealth_bundle_v3.tar` (or rebuild: `python scripts/build_colab_bundle.py --out <dir>/aquahealth_bundle --tar` on the machine that has the raw delivery) | ~100 MB | `bundle.sha256` pins the digests of `clean_manifest.csv`, `development.csv`, `final_test.csv`, `folds.csv`; `scripts/colab_dataset.py verify --hash all` re-hashes all 3,805 images against the manifest | anywhere; `AQUAHEALTH_COLAB_DATASET_ROOT` points at it, `data/raw/<key>` become symlinks |
| raw delivery (alternative) | the five delivered folders in `MyDrive/AquaHealth/` | ~12 GB | `scripts/verify_drive_delivery.py` | outside the repo |
| **official GAN archive** | `aquahealth_gan_layer3_a9bc3c8.tar.gz` (records + 5,004 PNGs + 10 generators + per-fold manifests) | 689.6 MB | no archive-level checksum was recorded. Every file inside is pinned by committed digests: `results/v2/gan/registry.csv` (`generator_sha256`, `synthetic_manifest_sha256`, `with_gan_manifest_sha256` per fold) and `results/v2/gan/GAN_MANIFEST.csv` (`sha256` of each of the 5,004 PNGs); `scripts/verify_gan_outputs.py --images all` must print `RESULT: VERIFIED` (229/229 on Colab) | extracted **at the repository root** → `data/gan/fold_XX/{generator.pt, synthetic_manifest.csv, fold_XX_train_gan.csv, train/<class>/*.png}` |
| live GAN folder | `gan_datasetv3/` (same content as the archive, linked to `data/gan` by the notebook) | — | same as above | `data/gan/` |
| **experiment registry + checkpoints** | `results_v2_datasetv3/` → `experiment_matrix.csv`, `experiments/<id>/{config.json, history.csv, metrics.json, confusion_matrix.csv, run_summary.json, status.json, best.pt, latest.pt, logs/, fit_analysis.*}`, `batches/`, `summaries/` | per-run `.pt` sizes could not be measured here | `status.json`; `scripts/verify_resume_checkpoint.py` for any `latest.pt` that is not COMPLETED | `results/v2/` |

How the code finds them: every manifest path is repo-relative (`data/raw/<key>/…`,
`data/gan/fold_XX/…`); the runner writes to `results/v2/experiments/<id>/`. Nothing is found
through an absolute path. The `/content/aquahealth/...` strings inside run records are
provenance only.

Checkpoints: `best.pt` = best validation Macro-F1 epoch; `latest.pt` = last finished epoch with
optimizer / scheduler / AMP scaler / RNG state. Both are **required**. `latest.pt` is what a
resume loads, and a run counts as COMPLETED (`src/cv_runner.py::is_completed`) only when
`status.json` says COMPLETED **and** `best.pt` + `latest.pt` + the five record files exist.

## 3. Lab machine — install

```bash
git clone --branch develop https://github.com/kolursamith/aquahealth.git
cd aquahealth
python3 -m venv .venv && source .venv/bin/activate          # Python >= 3.11
# torch/torchvision are pinned (2.14.0 / 0.29.0). Install the CUDA build that matches the lab
# driver from download.pytorch.org (Colab used +cu130), then the rest of the pins:
pip install -r requirements/experiments.txt
python scripts/gpu_smoke.py --require-cuda
python scripts/colab_preflight.py --env-only --require-cuda
```
`pretrained: true` downloads the torchvision ImageNet weights on first use, so the machine needs
internet access (or a pre-filled `~/.cache/torch`).

## 4. Lab machine — restore state (before any training)

1. **Registry + checkpoints.** Copy the whole Drive folder `MyDrive/AquaHealth/results_v2_datasetv3/`
   onto `results/v2/` so that it replaces the repository copy (use rclone, Drive for desktop, or
   download a zip; keep every `.pt`). Do not merge it by hand, and do not delete anything.
   ```bash
   grep -c COMPLETED results/v2/experiment_matrix.csv       # must equal the known count (15)
   python scripts/build_experiment_matrix.py --refresh        # statuses from status.json files
   ```
2. **Dataset.**
   ```bash
   export AQUAHEALTH_COLAB_DATASET_ROOT=/path/to/aquahealth_bundle
   python scripts/colab_dataset.py info
   python scripts/colab_dataset.py attach
   python scripts/colab_dataset.py verify --hash all          # exit 1 = STOP; never re-split
   ```
3. **GAN.** Put the archive anywhere and extract it at the repository root:
   ```bash
   tar -xzf /path/to/aquahealth_gan_layer3_a9bc3c8.tar.gz -C .
   python scripts/verify_gan_outputs.py --images all          # must print RESULT: VERIFIED
   ```
   Never run `run_gan_fold.py` / `run_gan_all_folds.py`: that would create a different GAN.

## 5. Read the registry, resume, continue

Status per experiment: `results/v2/experiments/<id>/status.json` (PENDING, RUNNING, COMPLETED,
FAILED, INTERRUPTED); summary: `results/v2/experiment_matrix.csv`.

```bash
cut -d, -f7 results/v2/experiment_matrix.csv | sort | uniq -c
```

**Next experiment.** The Colab batch order (notebook section 9) is models
`cnn_vit_lstm,yolo_efficientnet,cnn_bilstm,resnet_attention,yolo_transformer`, folds 1–10, one arm
per batch (`with_gan` was the active arm). The next run is the first id in that order that is not
COMPLETED in the **restored** registry:
```bash
python - <<'EOF'
import csv
st = {r["experiment_id"]: r["status"] for r in csv.DictReader(open("results/v2/experiment_matrix.csv"))}
order = "cnn_vit_lstm,yolo_efficientnet,cnn_bilstm,resnet_attention,yolo_transformer".split(",")
for arm in ("with_gan", "without_gan"):
    nxt = next((f"{m}_fold{f:02d}_{arm}" for m in order for f in range(1, 11)
                if st.get(f"{m}_fold{f:02d}_{arm}") != "COMPLETED"), None)
    print(arm, "next:", nxt, st.get(nxt))
EOF
```
The git copy alone gives `cnn_vit_lstm_fold01_with_gan` (for `with_gan`) because it only knows
fold 01 WITHOUT-GAN. That is **not** the real next run until step 4.1 has been done.

**Interrupted run** (a `latest.pt` exists, status not COMPLETED): the batch driver runs
`scripts/verify_resume_checkpoint.py` first (model / fold / arm / seed / dataset v3 / manifest,
GAN and preprocessing digests / optimizer, scheduler, scaler, RNG / CUDA) and resumes with
`--resume`, or it **stops the batch** on a mismatch. The GPU model is not pinned, so a T4
checkpoint can be resumed on the lab GPU. Single run by hand:
```bash
python scripts/verify_resume_checkpoint.py --model M --fold F --data-arm A \
    --config configs/cv_v2/default.json --require-cuda
python scripts/run_cv_experiment.py --model M --fold F --data-arm A \
    --config configs/cv_v2/default.json --require-cuda --resume
```

**Continue the matrix** (the same command the notebook runs; it skips COMPLETED runs, re-checks
every gate before each run, and writes one line per experiment to
`results/v2/batches/batch_<arm>_<timestamp>.jsonl`):
```bash
python scripts/run_cv_batch.py \
    --models cnn_vit_lstm,yolo_efficientnet,cnn_bilstm,resnet_attention,yolo_transformer \
    --folds 1-10 --data-arm with_gan --config configs/cv_v2/default.json --require-cuda --report
```

## 6. How completed runs are protected

- `run_cv_experiment.py` refuses a COMPLETED experiment (`FileExistsError`) and refuses to start
  over an existing `latest.pt` without `--resume`.
- Since this migration, a run whose `status.json` says COMPLETED is also refused, and the batch
  driver skips it (`skipped_completed_outputs_missing`), when its checkpoints are missing. Before
  this change, a fresh clone, which has the committed `status.json` but not the git-ignored `.pt`
  files, would have **retrained** `cnn_vit_lstm_fold01_without_gan`. Covered by
  `tests/test_cv_runner.py`.
- This guard only protects runs whose `status.json` is present. The 14 runs that are only on
  Drive are protected once step 4.1 is done.

## 7. After runs on the lab machine

- Commit the records only (whitelisted by `.gitignore`): `experiment_matrix.csv` and, per run,
  `config.json`, `history.csv`, `metrics.json`, `confusion_matrix.csv`, `run_summary.json`,
  `status.json`, `fit_analysis.md`. Never commit `.pt`, images, `data/gan/`, archives
  (`*.tar`, `*.tar.gz`, `*.zip`) or `data/raw/`.
- Copy `results/v2/` (with its `.pt` files) back to `MyDrive/AquaHealth/results_v2_datasetv3/` (or
  to the lab's storage) so there is one authoritative registry. Never train from two machines at
  once.
- Commit on `develop` only.
