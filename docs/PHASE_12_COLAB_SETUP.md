# Phase 12 — Google Colab setup for the CV experiments

The local machine develops code and manifests; **every experiment runs on Colab CUDA**
through `notebooks/AquaHealthAI_Phase12_Colab.ipynb`, which calls the repository
scripts (nothing is implemented inside the notebook).

## Start
1. Open the notebook in Colab, set *Runtime → Change runtime type → GPU* (T4 or better).
2. Section 1 clones the repository (`REPO_URL`, `BRANCH = develop` at the top of the cell) into
   `/content/aquahealth` and installs `requirements/experiments.txt` only (torch 2.14.0,
   torchvision 0.29.0, opencv-headless, numpy, pillow, matplotlib — the existing pins; no new dependency).
3. Section 2 runs the GPU smoke test and the environment pre-flight; the notebook stops there without a GPU.

## Dataset access (Layer 2): `AQUAHEALTH_COLAB_DATASET_ROOT`
The local machine keeps the raw delivery (`/…/AI_TECHTAHON_DOCUMENTS/Dataset`, outside git); the
clean manifest `data/audit/clean_manifest.csv` is the authoritative description of the dataset. Colab
gets the images through one configurable root, the environment variable
**`AQUAHEALTH_COLAB_DATASET_ROOT`** — never a hard-coded path and not necessarily Google Drive:

1. **Clean bundle (recommended, ~1 GB).** Locally:
   `python scripts/build_colab_bundle.py --out /some/dir/aquahealth_bundle --tar`
   copies *exactly* the manifest's 3,805 included images (development + frozen final test; dataset configuration v3 excludes MatsyaDx-BD/Mendeley entirely; excluded
   duplicates / unresolved labels are not copied) to `<out>/data/raw/<key>/…` — the same relative
   paths the committed manifests use — and writes `bundle_manifest.csv` (image_id, filepath, sha256,
   source, class, label, group_id, split, fold) plus `bundle.sha256`, which pins the bundle to the
   digests of `clean_manifest.csv`, `development.csv`, `final_test.csv` and `folds.csv`. Nothing is
   sampled at random. Upload the tar wherever you like (Drive, bucket, direct upload), extract it on
   the VM and point the variable at the extracted folder. (`--hardlink` builds it instantly on the
   same volume; such a bundle shares inodes with the raw files — never edit its files.)
2. **Full delivery drop.** Point the variable at the delivered `Dataset/` folder; `attach` delegates
   to `scripts/link_raw_datasets.py --source`.

On Colab (notebook sections 3–4):
```bash
export AQUAHEALTH_COLAB_DATASET_ROOT=/content/aquahealth_data/aquahealth_bundle
python scripts/colab_dataset.py info      # layout: bundle | delivery; fails (exit 3) if unset/missing
python scripts/colab_dataset.py attach    # data/raw/<key> -> root (symlinks only)
python scripts/colab_dataset.py verify --hash all   # exit 1 = STOP, never regenerate on Colab
```
`verify` checks: manifest present; split and fold digests; bundle pins equal the runtime's manifest
digests (manifest identity); bundle rows == included rows with identical filepath/sha256/class/label/
group_id/split/fold/source; every image exists; count == 3,805; image SHA-256 == manifest (all, or a
seeded sample with `--hash sample`); labels canonical; development rows carry a fold and test rows none;
no group in two folds or across development/final_test. Verified locally against the real bundle:
18/18 checks, 3,805/3,805 hashes.

Runtime pre-flight without a fold: `python scripts/colab_preflight.py --env-only --require-cuda`
(Python, platform, torch/torchvision/numpy/pillow/opencv/matplotlib versions and pins, CUDA/GPU/VRAM,
repository digests; scikit-learn/timm/transformers are reported as *not required* — the code base does
not import them). GPU smoke: `python scripts/gpu_smoke.py --require-cuda` (tensor to GPU, forward,
backward, optimizer step of a tiny network; peak memory and runtime; exit 2 without CUDA).

## How manifests locate images
Every manifest row has `filepath` relative to the repository root (`data/raw/<key>/…`), so the same
CSVs work locally and on Colab once `data/raw/` links exist. `scripts/colab_preflight.py` checks that
every image of the fold's training and validation manifests exists before anything runs.

## GAN data (Layer 3)
WITH-GAN needs `data/gan/fold_XX/fold_XX_train_gan.csv` (+ the PNGs under `data/gan/fold_XX/train/`),
produced on Colab CUDA only — `notebooks/AquaHealthAI_Layer3_GAN_Colab.ipynb` runs the GAN smoke test
(`scripts/run_gan_fold.py --smoke --require-cuda`), then all ten folds
(`scripts/run_gan_all_folds.py --config configs/gan_v2/default.json --require-cuda`), then
`scripts/verify_gan_outputs.py`; see `docs/GAN_AUGMENTATION.md`. Section 5 links `data/gan` and
`results/v2` to `MyDrive/AquaHealth/{gan,results_v2}` so they persist across sessions. Synthetic
images and generators are never committed; `results/v2/gan/{registry.csv,GAN_MANIFEST.csv,
verification.json,generation_summary.json}` are.

## CUDA verification
`scripts/colab_preflight.py --fold F --data-arm A --require-cuda` prints a JSON report: GPU name, CUDA
version, memory, torch/torchvision vs the pinned versions, git commit, frozen-split and fold digests,
matrix presence, per-fold image availability, GAN manifest (WITH-GAN), final-test SHA-256, and the
isolation numbers. Exit 2 = no CUDA, 1 = a check failed. The runner itself takes `--require-cuda` and
aborts rather than falling back to CPU/MPS.

## Launch one experiment
```bash
python scripts/run_cv_experiment.py --model cnn_vit_lstm --fold 1 --data-arm without_gan \
    --config configs/cv_v2/default.json --require-cuda
```
`--model` ∈ {efficientnet_b0, cnn_vit_lstm, yolo_efficientnet, cnn_bilstm, resnet_attention,
yolo_transformer}; `--data-arm` ∈ {without_gan, with_gan}; `--fold` 1–10. The experiment id is
`<model>_fold<XX>_<arm>`.

## Where results are saved
`results/v2/experiments/<experiment_id>/`: `config.json` (full provenance), `history.csv` (per-epoch train
and validation loss/accuracy/precision/recall/Macro-F1, LRs, stage), `metrics.json` + `confusion_matrix.csv`
(best epoch: per-class P/R/F1, confusion, classification report), `best.pt`, `latest.pt`, `run_summary.json`,
`status.json`, `logs/train.log`. With the section-4 links these live on Drive. The matrix statuses are
refreshed with `scripts/build_experiment_matrix.py --refresh`.

## Resume after an interruption
Re-run sections 1–5, then `scripts/run_cv_experiment.py … --resume`. The runner loads `latest.pt`
(model, optimizer, scheduler, scaler, RNG, history, best epoch), verifies that model/fold/arm/seed and
the configuration hash match, and continues at the next epoch; the stage (head/full) is recomputed from
the epoch. A `COMPLETED` experiment (status + all output files present) is refused; an existing
checkpoint without `--resume` is refused. Status values: PENDING, RUNNING, COMPLETED, FAILED, INTERRUPTED.

## Reproducibility limits
Seeds are set for Python, NumPy, torch and the DataLoader shuffle generator and recorded with the
configuration hash and every manifest SHA-256. Bit-exact repeatability is **not** guaranteed on CUDA
(cuDNN autotuning, atomic reductions) or under AMP; the same seed reproduces initialisation and data order.
