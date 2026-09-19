#!/usr/bin/env python
"""Phase 6 — data-leakage audit of the clean manifest, before any split.

    python scripts/audit_leakage.py                 # reads data/audit/clean_manifest.csv

Writes:

    data/audit/leakage_report.csv    one row per check (finding, evidence, affected images/classes,
                                     severity, action, status) — the teacher's seven checks
    data/audit/leakage_report.json   the underlying numbers (src/leakage_audit.leakage_report)
    data/audit/leakage_report.md     human-readable version of both

Checks: duplicate leakage, cross-dataset duplicate leakage, same-specimen leakage
(where specimen ids exist), target leakage, suspicious metadata/features (incl. an
EXIF header scan), class<->resolution shortcut, other source shortcuts. Counting
and header reads only: no model is trained and nothing is modified or deleted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_clean_manifest import _table  # noqa: E402

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.dataset_cleaning import CLEAN_MANIFEST_NAME, read_clean_manifest  # noqa: E402
from src.leakage_audit import (  # noqa: E402
    FINDING_COLUMNS,
    exif_summary,
    findings_rows,
    leakage_report,
)
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("leakage_audit")


def write_leakage_md(path: Path, rep: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    t, d, f = rep["target_leakage"], rep["duplicate_leakage"], rep["suspicious_features"]
    lines = [
        "# Leakage audit (Phase 6, pre-split) — clean manifest",
        "",
        "Counting and EXIF-header reads only; no model was trained. Nothing was modified or "
        "deleted as a result of this audit. Machine-readable: `leakage_report.csv` (one row per "
        "check) and `leakage_report.json` (numbers).",
        "",
        "## Findings (one row per check)",
        "",
        *_table(
            ["check", "finding", "affected images", "affected classes", "action", "status"],
            [
                [
                    x["check"],
                    x["finding"],
                    x["affected_images"],
                    x["affected_classes"],
                    x["action"],
                    x["status"],
                ]
                for x in findings
            ],
        ),
        "",
        f"Severity: {findings[0]['severity']}",
        "",
        "## A. Target leakage — where is the label written outside the pixels?",
        "",
        *[f"- **{k}**: {v}" for k, v in t["channels"].items()],
        "",
        *_table(
            ["dataset", "images (raw)", "filename reveals label", "%", "label in folder path"],
            [
                [
                    k,
                    v["images"],
                    v["filename_reveals_label"],
                    v["filename_reveals_label_pct"],
                    v["label_in_folder_path"],
                ]
                for k, v in t["per_dataset"].items()
            ],
        ),
        "",
        "## B. Duplicate / specimen leakage — what a split must keep together",
        "",
        *_table(
            ["quantity", "value"],
            [[k.replace("_", " "), v] for k, v in d.items() if k != "verdict"],
        ),
        "",
        f"Verdict: {d['verdict']}",
        "",
        "## C. Shortcut features — in-sample upper bounds (majority class per feature value)",
        "",
        *_table(
            [
                "feature",
                "distinct values",
                "majority baseline",
                "feature-only accuracy (upper bound)",
            ],
            [
                [
                    k,
                    v["distinct_values"],
                    v["majority_class_baseline"],
                    v["feature_only_accuracy_upper_bound"],
                ]
                for k, v in f["predictability"].items()
            ],
        ),
        "",
        *_table(
            ["class", "resolutions (top 4)", "sources"],
            [
                [
                    c,
                    ", ".join(f"{k}: {n}" for k, n in f["resolution_by_class_top4"][c].items()),
                    ", ".join(f"{k}: {n}" for k, n in f["source_by_class"][c].items()),
                ]
                for c in f["resolution_by_class_top4"]
            ],
        ),
        "",
        "## D. EXIF header scan (clean images)",
        "",
    ]
    exif = rep.get("exif", {})
    if exif:
        lines += _table(
            ["dataset", "images", "with EXIF", "with orientation tag", "camera make/model (top)"],
            [
                [
                    k,
                    v["images"],
                    v["with_exif"],
                    v["with_orientation"],
                    v["camera_models"] or "none",
                ]
                for k, v in exif.items()
            ],
        )
    else:
        lines.append("EXIF scan skipped.")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--skip-exif", action="store_true", help="do not open image headers")
    args = parser.parse_args(argv)
    audit_dir = args.audit_dir or (Path(args.data_dir) / AUDIT_DIR_NAME)
    clean_path = audit_dir / CLEAN_MANIFEST_NAME
    if not clean_path.is_file():
        logger.error("%s not found; run scripts/build_clean_manifest.py first", clean_path)
        return 2
    rows = read_clean_manifest(clean_path)
    report = leakage_report(rows)
    exif = None
    if not args.skip_exif:
        logger.info("scanning EXIF headers of %d clean images", sum(r.included for r in rows))
        exif = exif_summary(rows, args.repo_root)
        report["exif"] = exif
    findings = findings_rows(rows, report, exif)
    with (audit_dir / "leakage_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FINDING_COLUMNS))
        writer.writeheader()
        writer.writerows(findings)
    (audit_dir / "leakage_report.json").write_text(json.dumps(report, indent=2))
    write_leakage_md(audit_dir / "leakage_report.md", report, findings)
    for x in findings:
        logger.info("%s: %s", x["check"], x["status"])
    print((audit_dir / "leakage_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
