#!/usr/bin/env python
"""Phase 10 — train the fold-specific cDCGAN on ONE fold's training rows and
generate the synthetic training images for that fold.

    python scripts/run_gan_fold.py --fold 1 --config configs/gan_v2/default.json --require-cuda
    python scripts/run_gan_fold.py --fold 1 --config configs/gan_v2/smoke.json --smoke \
        --require-cuda
    python scripts/run_gan_fold.py --fold 1 --epochs 30 [--image-size 64] [--target-per-class N]

`--config` supplies every GANConfig field; explicit CLI flags override it. `--smoke`
writes under data/gan/smoke/ and results/v2/smoke/ so a real fold directory and the
real registry are never touched by an infrastructure check. Real GAN training runs on
Colab CUDA only (Layer 3 rule): pass `--require-cuda` and the script exits 2 rather than
training on CPU/MPS. All ten folds: scripts/run_gan_all_folds.py.

Reads data/audit/cv_v3/fold_XX_train.csv (through src.split_v2.read_folds, digest
verified) and data/audit/split_v3/final_test.csv ONLY to build the forbidden-id
set; every training image is checked against that set before the GAN sees it.

Writes, all under data/gan/fold_XX/ (git-ignored):

    generator.pt, gan_run.json                 checkpoint + full run record
    train/<class>/synthetic_NNNNN.png          the synthetic images (train directory only)
    synthetic_manifest.csv                     provenance of every synthetic image
    fold_XX_train_gan.csv                      WITH-GAN training list (real + synthetic)
                                               — the WITHOUT-GAN arm is fold_XX_train.csv

and one line per fold in the tracked registry results/v2/gan/registry.csv
(architecture, latent dim, resolution, epochs, batch size, optimizer, learning
rate, seed, fold, generated count, source training fold + digests).

No classifier is trained here. The validation fold and the frozen test set are
never read as images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.device import resolve_device  # noqa: E402
from src.gan_augmentation import (  # noqa: E402
    GAN_DIR_NAME,
    FoldTrainingImages,
    GANConfig,
    augmented_training_manifest,
    device_facts,
    forbidden_ids_for_fold,
    generate,
    load_gan_config,
    plan_synthetic_counts,
    registry_row,
    train_gan,
    update_registry,
    validate_synthetic_rows,
    write_augmented_manifest,
    write_synthetic_manifest,
)
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402
from src.split_v2 import (  # noqa: E402
    CV_DIR_NAME,
    FINAL_TEST,
    SPLIT_DIR_NAME,
    fold_members,
    read_folds,
    read_split,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("gan_fold")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=None, help="configs/gan_v2/<name>.json")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="infrastructure check: outputs under data/gan/smoke/ and results/v2/smoke/",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--results-dir", type=Path, default=None, help="default: <repo-root>/results"
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--target-per-class", type=int, default=None)
    parser.add_argument(
        "--synthetic-per-class",
        type=str,
        default=None,
        help="JSON, e.g. '{\"Aeromoniasis\": 100}'; overrides --target-per-class",
    )
    parser.add_argument("--max-synthetic-ratio", type=float, default=None)
    parser.add_argument("--max-train-images", type=int, default=None, help="smoke runs only")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="real runs: abort (exit 2) unless the selected device is CUDA — never CPU/MPS",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="decode images every epoch instead of caching the resized tensors per fold",
    )
    args = parser.parse_args(argv)

    audit_dir = Path(args.data_dir) / AUDIT_DIR_NAME
    gan_dir = Path(args.data_dir) / GAN_DIR_NAME / ("smoke" if args.smoke else "")
    results_dir = (
        Path(args.results_dir) if args.results_dir is not None else Path(args.repo_root) / "results"
    )
    registry = results_dir / "v2" / ("smoke" if args.smoke else "gan") / "registry.csv"
    out_dir = gan_dir / f"fold_{args.fold:02d}"
    if (out_dir / "generator.pt").exists():
        logger.error("%s already holds a run; use a new fold or remove it deliberately", out_dir)
        return 2
    overrides = dict(
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        target_per_class=args.target_per_class,
        synthetic_per_class=(
            json.loads(args.synthetic_per_class) if args.synthetic_per_class else None
        ),
        max_synthetic_ratio=args.max_synthetic_ratio,
        max_train_images=args.max_train_images,
    )
    if args.config is not None:
        config = load_gan_config(args.config, **overrides)
    else:
        config = GANConfig(**{k: v for k, v in overrides.items() if v is not None})
    logger.info("config %s: %s", args.config or "(defaults)", config)
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        logger.error("--require-cuda: selected device is %s, not CUDA; refusing to train", device)
        return 2
    logger.info("device %s %s", device, device_facts(device))

    folds = read_folds(audit_dir / CV_DIR_NAME)
    train_rows, validation_rows = fold_members(folds, args.fold)
    final_test = [s for s in read_split(audit_dir / SPLIT_DIR_NAME) if s.split == FINAL_TEST]
    forbidden = forbidden_ids_for_fold(folds, args.fold, final_test)
    logger.info(
        "fold %d: %d training rows; %d forbidden ids (validation %d + final test %d)",
        args.fold,
        len(train_rows),
        len(forbidden),
        len(validation_rows),
        len(final_test),
    )
    dataset = FoldTrainingImages(
        train_rows,
        forbidden_ids=forbidden,
        repo_root=args.repo_root,
        image_size=config.image_size,
        max_images=config.max_train_images,
        seed=config.seed,
        cache_in_memory=not args.no_cache,
    )
    real_counts = dataset.class_counts()
    plan = plan_synthetic_counts(real_counts, config)
    logger.info("real per class %s -> synthetic plan %s", real_counts, plan)

    checkpoint, record = train_gan(
        dataset,
        fold=args.fold,
        config=config,
        device=device,
        out_dir=out_dir,
        train_manifest=audit_dir / CV_DIR_NAME / f"fold_{args.fold:02d}_train.csv",
        forbidden_count=len(forbidden),
        num_workers=args.workers,
    )
    synthetic = generate(
        checkpoint,
        fold=args.fold,
        counts=plan,
        gan_dir=gan_dir,
        repo_root=args.repo_root,
        device=device,
        seed=config.seed,
    )
    validate_synthetic_rows(synthetic, args.fold, args.repo_root)
    write_synthetic_manifest(synthetic, out_dir / "synthetic_manifest.csv")
    augmented = augmented_training_manifest(train_rows, synthetic, args.fold)
    write_augmented_manifest(augmented, out_dir / f"fold_{args.fold:02d}_train_gan.csv")
    smoke_run = args.smoke or config.max_train_images is not None or config.epochs < 5
    update_registry(
        registry,
        registry_row(
            record,
            plan=plan,
            synthetic_manifest=out_dir / "synthetic_manifest.csv",
            with_gan_manifest=out_dir / f"fold_{args.fold:02d}_train_gan.csv",
            smoke_run=smoke_run,
        ),
    )
    summary = {
        "fold": args.fold,
        "real_training_images_in_fold": len(train_rows),
        "real_images_seen_by_gan": record.real_training_images,
        "synthetic_images": len(synthetic),
        "synthetic_per_class": plan,
        "with_gan_manifest_rows": len(augmented),
        "without_gan_manifest": str(audit_dir / CV_DIR_NAME / f"fold_{args.fold:02d}_train.csv"),
        "with_gan_manifest": str(out_dir / f"fold_{args.fold:02d}_train_gan.csv"),
        "run_record": str(out_dir / "gan_run.json"),
        "registry": str(registry),
        "config_file": str(args.config) if args.config else None,
        "smoke_run": smoke_run,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
