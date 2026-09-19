#!/usr/bin/env python
"""Phase 9 — 10-fold cross-validation manifests on the DEVELOPMENT partition only.

    python scripts/build_cv_folds.py [--folds 10] [--seed 42]

Reads data/audit/split_v2/development.csv through src.split_v2.read_development
(digest-verified; final_test.csv is never opened) and writes

    data/audit/cv_v2/folds.csv                       image -> validation fold (1..K)
    data/audit/cv_v2/fold_XX_train.csv               K x
    data/audit/cv_v2/fold_XX_validation.csv          K x
    data/audit/cv_v2/folds.sha256                    digests of every file above
    data/audit/cv_report.csv                         fold, counts, class/source distribution,
                                                     group information, seed
    data/audit/cv_v2/cv_report.md

Folds are group-aware (clean-manifest group_id: exact/near duplicates and, for
MatsyaDx-BD, the same specimen never straddle train/validation) and stratified
by (unified class | majority source), using src.split_v2.assign_groups — the
baseline split rule generalised. scikit-learn is not a project dependency and
its StratifiedGroupKFold would not honour the (class, source) strata, so the
project's own implementation is used. No model is trained; no GAN; the frozen
test set is untouched. Refuses to overwrite existing folds.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_clean_manifest import _table  # noqa: E402

from src.config import DATA_DIR, SEED  # noqa: E402
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402
from src.split_v2 import (  # noqa: E402
    CV_DIR_NAME,
    DEFAULT_FOLDS,
    SPLIT_DIR_NAME,
    assign_folds,
    cv_report_rows,
    read_development,
    write_folds,
)
from src.utils import get_logger  # noqa: E402

logger = get_logger("cv_folds")
CV_REPORT_COLUMNS = (
    "fold",
    "train_count",
    "validation_count",
    "class_distribution",
    "source_distribution",
    "group_information",
    "seed",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    audit_dir = args.audit_dir or (Path(args.data_dir) / AUDIT_DIR_NAME)
    out_dir = audit_dir / CV_DIR_NAME
    if (out_dir / "folds.csv").exists():
        logger.error(
            "%s exists: folds are fixed once written; delete deliberately to rebuild", out_dir
        )
        return 2
    development = read_development(audit_dir / SPLIT_DIR_NAME)
    rows = assign_folds(development, n_folds=args.folds, seed=args.seed)
    digest = write_folds(rows, out_dir, args.folds)
    report = cv_report_rows(rows, args.folds, args.seed)
    with (audit_dir / "cv_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CV_REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(report)
    lines = [
        f"# {args.folds}-fold cross-validation manifests (Phase 9)",
        "",
        f"- source: `data/audit/{SPLIT_DIR_NAME}/development.csv` ({len(development)} images); "
        "`final_test.csv` was not read",
        "- strategy: group-aware stratified K-fold — `src/split_v2.py::assign_folds` over "
        "`assign_groups` (unit = clean-manifest group_id, stratum = unified class | majority "
        f"source), seed {args.seed}; every image is validation exactly once",
        "- sklearn: not installed / not used (its StratifiedGroupKFold would ignore the source "
        "stratum); the project's own baseline rule is generalised instead",
        "- files: `folds.csv`, `fold_XX_train.csv`, `fold_XX_validation.csv`, digests in "
        "`folds.sha256`",
        "",
        *_table(
            ["fold", "train", "validation", "validation classes", "validation sources", "groups"],
            [
                [
                    r["fold"],
                    r["train_count"],
                    r["validation_count"],
                    r["class_distribution"],
                    r["source_distribution"],
                    r["group_information"],
                ]
                for r in report
            ],
        ),
        "",
        "Nothing was trained; no GAN images exist; the frozen test set is untouched.",
    ]
    (out_dir / "cv_report.md").write_text("\n".join(lines) + "\n")
    logger.info("wrote %d folds to %s\n%s", args.folds, out_dir, digest)
    print((out_dir / "cv_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
