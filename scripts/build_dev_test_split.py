#!/usr/bin/env python
"""Phase 8 — the new experiment's development / FROZEN final-test partition.

    python scripts/build_dev_test_split.py [--test-ratio 0.20] [--seed 42]

Reads data/audit/clean_manifest.csv (included rows only) and writes

    data/audit/split_v3/development.csv        everything not in the test set
    data/audit/split_v3/final_test.csv         FROZEN — never train / tune / select / GAN on it
    data/audit/split_v3/split_manifest.sha256  digests of both files
    data/audit/split_v3/split_report.json / .md

Refuses to overwrite an existing final_test.csv (like scripts/build_split_manifest.py
for the baseline). The old baseline split data/split_manifest.csv is not read or
written. No fold, GAN or training is done here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_clean_manifest import _table  # noqa: E402

from src.config import DATA_DIR, ROOT_DIR, SEED  # noqa: E402
from src.dataset_cleaning import CLEAN_MANIFEST_NAME, read_clean_manifest  # noqa: E402
from src.manifest import file_sha256  # noqa: E402
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402
from src.split_v2 import (  # noqa: E402
    DEFAULT_TEST_RATIO,
    SPLIT_DIR_NAME,
    build_dev_test_split,
    split_summary,
    write_split,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("dev_test_split")


def write_report_md(path: Path, s: dict[str, Any]) -> None:
    classes = sorted(s["development"]["per_class"])
    sources = sorted(set(s["development"]["per_source"]) | set(s["final_test"]["per_source"]))
    lines = [
        "# Development / frozen final-test split (Phase 8)",
        "",
        f"- strategy: {s['strategy']}",
        f"- seed: {s['seed']}",
        f"- test ratio: {s['test_ratio']} — {s['ratio_provenance']}",
        f"- clean manifest SHA-256: `{s['clean_manifest_sha256']}`",
        f"- clean images: {s['clean_total']} total, {s['clean_included']} included; excluded "
        f"(not eligible for any partition): {s['excluded_from_clean_manifest']}",
        "",
        "**`final_test.csv` is FROZEN**: not for training, cross-validation, GAN generation, "
        "hyper-parameter tuning or choosing between models. Later phases read the development "
        "partition through `src/split_v2.py::read_development`, which verifies the digests and "
        "never opens the test file.",
        "",
        "## Counts",
        "",
        *_table(
            ["partition", "images", "groups", "share"],
            [
                [
                    "development",
                    s["development"]["images"],
                    s["development"]["groups"],
                    round(1 - s["final_test"]["share"], 4),
                ],
                [
                    "final_test",
                    s["final_test"]["images"],
                    s["final_test"]["groups"],
                    s["final_test"]["share"],
                ],
            ],
        ),
        "",
        "## Per class",
        "",
        *_table(
            ["unified class", "development", "final_test", "test share"],
            [
                [
                    c,
                    s["development"]["per_class"].get(c, 0),
                    s["final_test"]["per_class"].get(c, 0),
                    s["per_class_share_in_test"][c],
                ]
                for c in classes
            ],
        ),
        "",
        "## Per source",
        "",
        *_table(
            ["source", "development", "final_test"],
            [
                [
                    d,
                    s["development"]["per_source"].get(d, 0),
                    s["final_test"]["per_source"].get(d, 0),
                ]
                for d in sources
            ],
        ),
        "",
        "## Group constraints",
        "",
        *[f"- {k.replace('_', ' ')}: {v}" for k, v in s["group_constraints"].items()],
        "",
        "The old baseline split (`data/split_manifest.csv`) and its results are untouched.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = parser.parse_args(argv)
    audit_dir = args.audit_dir or (Path(args.data_dir) / AUDIT_DIR_NAME)
    out_dir = audit_dir / SPLIT_DIR_NAME
    if (out_dir / "final_test.csv").exists() and not args.dry_run:
        logger.error("%s exists: the final test set is frozen and is not rebuilt", out_dir)
        return 2
    clean_path = audit_dir / CLEAN_MANIFEST_NAME
    clean = read_clean_manifest(clean_path)
    split = build_dev_test_split(clean, test_ratio=args.test_ratio, seed=args.seed)
    missing = [s.filepath for s in split if not (Path(args.repo_root) / s.filepath).is_file()]
    if missing:
        logger.error("%d split images are missing on disk, e.g. %s", len(missing), missing[:3])
        return 1
    summary = split_summary(
        split,
        clean,
        test_ratio=args.test_ratio,
        seed=args.seed,
        clean_digest=file_sha256(clean_path),
    )
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    digest = write_split(split, out_dir)
    summary["split_manifest_sha256"] = digest.splitlines()
    (out_dir / "split_report.json").write_text(json.dumps(summary, indent=2))
    write_report_md(out_dir / "split_report.md", summary)
    logger.info(
        "development %d / final_test %d images (%d/%d groups); wrote %s",
        summary["development"]["images"],
        summary["final_test"]["images"],
        summary["development"]["groups"],
        summary["final_test"]["groups"],
        out_dir,
    )
    print((out_dir / "split_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
