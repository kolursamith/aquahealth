#!/usr/bin/env python
"""Layer 2 — attach and verify the clean dataset on a Colab runtime.

    export AQUAHEALTH_COLAB_DATASET_ROOT=/content/aquahealth_data/aquahealth_bundle
    python scripts/colab_dataset.py attach            # data/raw/<key> -> bundle (or delivery)
    python scripts/colab_dataset.py verify [--hash all|sample|none] [--sample 200]
    python scripts/colab_dataset.py info

`attach` accepts the clean bundle (scripts/build_colab_bundle.py) or the delivered
Dataset/ drop (then delegates to scripts/link_raw_datasets.py). `verify` proves the
attached images are the verified dataset: manifest/split/fold digests, bundle pins,
every included image present, count, labels, group ids, fold ids and image SHA-256.
Exit 0 = verified, 1 = mismatch, 3 = dataset unavailable (root unset/missing).
Nothing is regenerated or modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.colab_data import (  # noqa: E402
    DATASET_ROOT_ENV,
    DatasetMismatch,
    DatasetUnavailable,
    attach,
    detect_layout,
    resolve_dataset_root,
    summarize,
    verify_dataset,
)
from src.config import ROOT_DIR  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("attach", "verify", "info"))
    parser.add_argument("--root", type=Path, default=None, help=f"overrides ${DATASET_ROOT_ENV}")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--force", action="store_true", help="attach: replace differing links")
    parser.add_argument("--hash", choices=("all", "sample", "none"), default="all")
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--json", action="store_true", help="verify: print the full report")
    args = parser.parse_args(argv)
    try:
        root = resolve_dataset_root(args.root)
        layout = detect_layout(root)
        if args.command == "info":
            print(json.dumps({"root": str(root), "layout": layout, "env": DATASET_ROOT_ENV}))
            return 0
        if args.command == "attach":
            if layout == "bundle":
                for line in attach(root, args.repo_root, force=args.force):
                    print(line)
            else:
                source = root / "Dataset" if (root / "Dataset").is_dir() else root
                cmd = [
                    sys.executable,
                    str(Path(args.repo_root) / "scripts" / "link_raw_datasets.py"),
                    "--source",
                    str(source),
                    "--data-dir",
                    str(Path(args.repo_root) / "data"),
                ] + (["--force"] if args.force else [])
                print("delivery layout ->", " ".join(cmd))
                return subprocess.run(cmd, check=False).returncode
            return 0
        report = verify_dataset(args.repo_root, root, hash_mode=args.hash, sample_size=args.sample)
        print(json.dumps(report, indent=2) if args.json else summarize(report))
        return 0 if report["ok"] else 1
    except DatasetUnavailable as exc:
        print(f"DATASET UNAVAILABLE: {exc}", file=sys.stderr)
        return 3
    except DatasetMismatch as exc:
        print(f"DATASET MISMATCH: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
