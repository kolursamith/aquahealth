#!/usr/bin/env python
"""Dataset-cleaning stage of the new experiment: clean manifest and de-duplication report.

    python scripts/build_clean_manifest.py          # reads data/audit/master_dataset.csv

Writes (nothing under data/raw/ is touched):

    data/audit/clean_manifest.csv          one row per master image: included / exclusion_reason /
                                           representative_image_id / group_id
    data/audit/dedup_report.json / .md     before/after counts, groups, exclusions, unresolved
    configs/preprocess_v2_clahe.json       the `preprocess` block downstream experiment configs use
                                           (the existing src/preprocessing.py CLAHE setting)

The leakage audit is scripts/audit_leakage.py; the CLAHE validation is
scripts/validate_clahe.py. No split is made here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.dataset_cleaning import (  # noqa: E402
    CLEAN_MANIFEST_NAME,
    build_clean_manifest,
    dedup_summary,
    write_clean_manifest,
)
from src.multi_dataset import (  # noqa: E402
    AUDIT_DIR_NAME,
    MASTER_MANIFEST_NAME,
    read_master_manifest,
)
from src.preprocessing import CLAHEConfig, PreprocessConfig  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("clean_manifest")

# The project's existing CLAHE setting (src/preprocessing.py defaults; the value every
# committed experiment config and the final checkpoint carry): clip 2.0, 8x8 tiles.
PREPROCESS_V2 = PreprocessConfig(clahe=CLAHEConfig())


# --- reports --------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def write_dedup_md(path: Path, s: dict[str, Any]) -> None:
    datasets = sorted(s["raw"]["per_dataset"])
    classes = sorted(set(s["raw"]["mapped_label_per_class"]) | set(s["clean"]["per_class"]))
    lines = [
        "# De-duplication report — clean manifest for the multi-dataset experiment",
        "",
        "Source: `data/audit/master_dataset.csv` (hashes reused from `src/manifest.py`, groups "
        "from `scripts/build_split_manifest.py`). Result: `data/audit/clean_manifest.csv`. "
        "**No raw file was deleted or modified**; exclusion is a manifest flag.",
        "",
        "## Policy applied",
        "",
    ]
    lines += [f"- {k}: {v}" for k, v in s["policy"].items()]
    lines += [
        "",
        "## Before → after",
        "",
        *_table(
            ["dataset", "raw images", "excluded", "clean (included)"],
            [
                [
                    d,
                    s["raw"]["per_dataset"][d],
                    s["raw"]["per_dataset"][d] - s["clean"]["per_dataset"].get(d, 0),
                    s["clean"]["per_dataset"].get(d, 0),
                ]
                for d in datasets
            ]
            + [
                [
                    "**total**",
                    s["raw"]["total"],
                    s["excluded"]["total"],
                    f"**{s['clean']['total']}**",
                ]
            ],
        ),
        "",
        "## Duplicate groups (all active datasets together)",
        "",
        *_table(
            ["quantity", "value"],
            [[k.replace("_", " "), v] for k, v in s["duplicates"].items()]
            + [["corrupt / unreadable", s["raw"]["corrupt"]]],
        ),
        "",
        "## Exclusions by reason",
        "",
        *_table(["reason", "images"], [[k, v] for k, v in s["excluded"]["by_reason"].items()]),
        "",
        *_table(
            ["dataset"]
            + sorted({k for v in s["excluded"]["by_dataset_and_reason"].values() for k in v}),
            [
                [d]
                + [
                    s["excluded"]["by_dataset_and_reason"][d].get(k, 0)
                    for k in sorted(
                        {k for v in s["excluded"]["by_dataset_and_reason"].values() for k in v}
                    )
                ]
                for d in datasets
            ],
        ),
        "",
        "## Final class distribution (clean corpus)",
        "",
        *_table(
            ["unified class", "mapped before dedup", "clean"] + datasets,
            [
                [
                    c,
                    s["raw"]["mapped_label_per_class"].get(c, 0),
                    s["clean"]["per_class"].get(c, 0),
                ]
                + [s["clean"]["per_dataset_and_class"].get(d, {}).get(c, 0) for d in datasets]
                for c in classes
            ],
        ),
        "",
        f"- leakage groups in the clean corpus: {s['clean']['leakage_groups']} "
        f"({s['clean']['leakage_groups_multi_image']} with more than one image; largest "
        f"{s['clean']['largest_leakage_group']}) — a later split must assign whole groups",
        "",
        "## Unresolved / ambiguous",
        "",
        f"- images awaiting a label decision (`LABEL_UNRESOLVED`): "
        f"{s['unresolved']['images_awaiting_label_decision']}",
        *_table(
            ["dataset / original class", "images"],
            [[k, v] for k, v in sorted(s["unresolved"]["per_dataset_and_class"].items())],
        ),
        "",
        "- label-conflict group members excluded "
        f"({len(s['unresolved']['conflict_group_members'])}):",
        "",
        *_table(
            ["group", "dataset", "original path", "unified class"],
            [
                [
                    m["near_dup_group"],
                    m["source_dataset"],
                    f"`{m['original_path']}`",
                    m["unified_class"],
                ]
                for m in s["unresolved"]["conflict_group_members"]
            ],
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


# --- main -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None, help="default: <data-dir>/audit")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--configs-dir", type=Path, default=ROOT_DIR / "configs")
    parser.add_argument(
        "--collapse-near-duplicates",
        action="store_true",
        help="also exclude near-duplicate copies (NOT the established policy; for comparison)",
    )
    args = parser.parse_args(argv)
    audit_dir = args.audit_dir or (Path(args.data_dir) / AUDIT_DIR_NAME)
    master_path = audit_dir / MASTER_MANIFEST_NAME
    if not master_path.is_file():
        logger.error("%s not found; run scripts/build_master_dataset.py first", master_path)
        return 2

    records = read_master_manifest(master_path)
    pending = [r for r in records if r.status == "pending"]
    if pending:
        logger.error(
            "%d master rows were never probed; re-run build_master_dataset.py", len(pending)
        )
        return 2
    rows = build_clean_manifest(records, collapse_near_duplicates=args.collapse_near_duplicates)
    write_clean_manifest(rows, audit_dir / CLEAN_MANIFEST_NAME)
    summary = dedup_summary(records, rows, collapse_near_duplicates=args.collapse_near_duplicates)
    (audit_dir / "dedup_report.json").write_text(json.dumps(summary, indent=2))
    write_dedup_md(audit_dir / "dedup_report.md", summary)
    logger.info(
        "clean manifest: %d raw -> %d included, %d excluded %s",
        summary["raw"]["total"],
        summary["clean"]["total"],
        summary["excluded"]["total"],
        summary["excluded"]["by_reason"],
    )

    args.configs_dir.mkdir(parents=True, exist_ok=True)
    (args.configs_dir / "preprocess_v2_clahe.json").write_text(
        json.dumps(
            {
                "description": "Preprocessing block for the multi-dataset experiment: the existing "
                "CLAHE (LAB-L, clip 2.0, tile 8x8) -> Resize 256 -> CenterCrop 224 -> ImageNet "
                "normalisation, exactly as src/preprocessing.py implements it. Copy into an "
                "experiment config's `preprocess` key.",
                "preprocess": {
                    "image_size": PREPROCESS_V2.image_size,
                    "resize_size": PREPROCESS_V2.resize_size,
                    "resize_mode": PREPROCESS_V2.resize_mode,
                    "clahe": asdict(PREPROCESS_V2.clahe) if PREPROCESS_V2.clahe else None,
                },
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("wrote %s", audit_dir)
    print((audit_dir / "dedup_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
