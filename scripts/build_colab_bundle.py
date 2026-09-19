#!/usr/bin/env python
"""Layer 2 — build the clean-dataset bundle for Google Colab (run LOCALLY).

    python scripts/build_colab_bundle.py --out /path/to/aquahealth_bundle [--tar] [--hardlink]

Copies exactly the *included* images of data/audit/clean_manifest.csv (the 5,942 verified
images, development + frozen final test) into `<out>/data/raw/<key>/…` — the same relative
paths the committed manifests use — and writes `bundle_manifest.csv` (image_id, filepath,
sha256, source, class, label, group_id, split, fold) and `bundle.sha256`, which pins the
bundle to the digests of clean_manifest.csv, development.csv, final_test.csv and folds.csv.
Nothing is chosen at random and no raw file is modified. Excluded images (duplicates,
unresolved labels) are not copied. `--tar` additionally writes `<out>.tar` for upload.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colab_data import build_bundle  # noqa: E402
from src.config import ROOT_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="bundle root to create")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--hardlink", action="store_true", help="hard-link instead of copy")
    parser.add_argument("--no-verify-hashes", action="store_true", help="skip re-hashing sources")
    parser.add_argument("--tar", action="store_true", help="also write <out>.tar (uncompressed)")
    args = parser.parse_args(argv)
    summary = build_bundle(
        args.repo_root, args.out, hardlink=args.hardlink, verify_hashes=not args.no_verify_hashes
    )
    if args.tar:
        tar_path = args.out.with_suffix(".tar")
        with tarfile.open(tar_path, "w") as tar:
            tar.add(args.out, arcname=args.out.name)
        summary["tar"] = str(tar_path)
        summary["tar_bytes"] = tar_path.stat().st_size
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
