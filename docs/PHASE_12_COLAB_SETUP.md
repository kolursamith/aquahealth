# Phase 12 — Google Colab setup for the CV experiments

The local machine develops code and manifests; **every experiment runs on Colab CUDA**
through `notebooks/AquaHealthAI_Phase12_Colab.ipynb`, which calls the repository
scripts (nothing is implemented inside the notebook).

## Start
1. Open the notebook in Colab, set *Runtime → Change runtime type → GPU* (T4 or better).
2. Section 1 clones the repository (`REPO_URL`, `BRANCH` at the top of the cell) into
   `/content/aquahealth` and installs `requirements/experiments.txt` only (torch 2.14.0,
   torchvision 0.29.0, opencv-headless, numpy, pillow, matplotlib — the existing pins; no new dependency).
3. Section 2 asserts CUDA; the notebook stops there without a GPU.

## Dataset mounting
- Upload the delivered `Dataset/` folder once to Google Drive: `MyDrive/AquaHealth/Dataset/` with the
  five deliveries exactly as they were delivered (`train_split/`, `test_split&validation/`, `test.csv`,
  `Fresh Water Fish Dataset/`, `Fish Disease.v1i.folder/`, `MatsyaDx-BD …/MatsyaDx-BD/`,
  `SalmonScan …/SalmonScan/`). Nothing is renamed; the `.7z`/`.zip` originals may stay next to the folders.
- Section 3 mounts Drive, copies the folder to the VM disk (`/content/aquahealth_data/Dataset`, ~5.6 GB;
  faster epochs, redone after a runtime reset) and runs `scripts/link_raw_datasets.py --source …`, which
  recreates `data/raw/<key>` exactly as on the local machine.
- Images are never in git. Manifests are: `data/audit/split_v2/`, `data/audit/cv_v2/` (with `.sha256`
  guards) and `results/v2/experiment_matrix.csv`.

## How manifests locate images
Every manifest row has `filepath` relative to the repository root (`data/raw/<key>/…`), so the same
CSVs work locally and on Colab once `data/raw/` links exist. `scripts/colab_preflight.py` checks that
every image of the fold's training and validation manifests exists before anything runs.

## GAN data
WITH-GAN needs `data/gan/fold_XX/fold_XX_train_gan.csv` (+ the PNGs under `data/gan/fold_XX/train/`),
produced on Colab by `scripts/run_gan_fold.py --fold XX --device cuda` (section 6). Section 4 links
`data/gan` and `results/v2` to `MyDrive/AquaHealth/{gan,results_v2}` so they persist across sessions.
Synthetic images are never committed.

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
