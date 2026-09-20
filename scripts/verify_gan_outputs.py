#!/usr/bin/env python
"""Layer 3 — verify generated GAN data against the records and the split (no training).

    python scripts/verify_gan_outputs.py                       # every completed fold, all PNGs
    python scripts/verify_gan_outputs.py --folds 1 --images sample
    python scripts/verify_gan_outputs.py --smoke               # data/gan/smoke + results/v2/smoke

Checks, per fold: run record is for the fold; generator SHA-256 == run record; training
source is fold_XX_train.csv and unchanged (digest); forbidden ids checked == validation +
final test; real images seen == training rows; synthetic rows valid (fold, class, label,
train-only path, file present); ids unique and disjoint from every real id; class directory
== class; no PNG outside fold_XX/train; PNGs on disk == manifest rows; PNGs open as RGB at
the recorded size; WITH-GAN manifest == real training ids + synthetic ids with no validation
or final-test id; registry line digests / counts / device agree. Across folds: no shared id
or path; label index == class. GAN_MANIFEST.csv: rows == union of the per-fold manifests,
classes/labels/sources consistent, image digests match the files. Also proves the split and
fold manifests still verify against their digests.

Exit 0 = VERIFIED, 1 = NOT VERIFIED. Writes results/v2/<gan|smoke>/verification.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.gan_augmentation import (  # noqa: E402
    GAN_DIR_NAME,
    GAN_MANIFEST_NAME,
    build_gan_manifest,
    parse_folds,
    summarize_verification,
    verify_gan_outputs,
    write_gan_manifest,
)
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folds", default=None, help="e.g. 1-10 or 1,2; default: completed")
    parser.add_argument("--smoke", action="store_true", help="verify data/gan/smoke")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--images", default="all", choices=("all", "sample", "none"))
    parser.add_argument(
        "--build-manifest",
        action="store_true",
        help="(re)build GAN_MANIFEST.csv from the per-fold manifests before verifying",
    )
    args = parser.parse_args(argv)

    folds = parse_folds(args.folds) if args.folds else None
    results_dir = (
        Path(args.results_dir) if args.results_dir is not None else Path(args.repo_root) / "results"
    )
    gan_dir = Path(args.data_dir) / GAN_DIR_NAME / ("smoke" if args.smoke else "")
    out_dir = results_dir / "v2" / ("smoke" if args.smoke else "gan")
    manifest_path = out_dir / GAN_MANIFEST_NAME
    if args.build_manifest or not manifest_path.is_file():
        rows = build_gan_manifest(
            gan_dir, args.repo_root, folds=folds, hash_images=args.images != "none"
        )
        write_gan_manifest(rows, manifest_path)
        print(f"wrote {manifest_path} ({len(rows)} rows)")
    report = verify_gan_outputs(
        gan_dir=gan_dir,
        repo_root=args.repo_root,
        audit_dir=Path(args.data_dir) / AUDIT_DIR_NAME,
        registry_path=out_dir / "registry.csv",
        gan_manifest_path=manifest_path,
        folds=folds,
        images=args.images,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verification.json").write_text(json.dumps(report, indent=2))
    print(summarize_verification(report))
    print(json.dumps(report["counts"], indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
