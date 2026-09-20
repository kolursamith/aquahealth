#!/usr/bin/env python
"""Layer 1 (Colab) — map the uploaded delivery folders to the five dataset keys and
check them against the audited inventory, WITHOUT guessing and without touching a
manifest.

    python scripts/verify_drive_delivery.py --source /content/drive/MyDrive/AquaHealth

The mapping is the committed one (src/multi_dataset.py::DATASET_SOURCES: key ->
delivered_subdir; `current_freshwater` = train_split / test_split / test.csv, loose at
the drop root or under a `Fresh_water_disease` wrapper — the same rules as
scripts/link_raw_datasets.py, which is what then creates data/raw/<key>). For every
key the script reports the resolved folder, the number of image files found there,
the audited inventory count from data/audit/clean_manifest.csv (all rows, included +
excluded) and how many manifest rows (all, and included) resolve to an existing file
under that folder. Exit 1 when a key cannot be resolved or any INCLUDED manifest row
is missing; a surplus / deficit of raw files is reported, not fatal (the SHA-256 proof
is scripts/colab_dataset.py verify --hash all). Writes results/v2/drive_delivery.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR  # noqa: E402
from src.dataset import IMAGE_EXTENSIONS  # noqa: E402
from src.dataset_cleaning import CLEAN_MANIFEST_NAME, read_clean_manifest  # noqa: E402
from src.multi_dataset import DATASET_SOURCES, EXCLUDED_SOURCES  # noqa: E402

CURRENT_KEY = "current_freshwater"
CURRENT_SUBDIRS = ("", "Fresh_water_disease")
CURRENT_PARTS = {
    "train_split": ("train_split",),
    "test_split": ("test_split", "test_split&validation"),
    "test.csv": ("test.csv",),
}


def _is_image(path: Path) -> bool:
    return (
        path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if _is_image(p))


def resolve_key(drop: Path, key: str) -> dict[str, Path]:
    """key -> {relative part under data/raw/<key>: folder in the drop}. Raises FileNotFoundError."""
    if key == CURRENT_KEY:
        for sub in CURRENT_SUBDIRS:
            base = drop / sub
            if (base / "train_split").is_dir():
                parts: dict[str, Path] = {}
                for name, candidates in CURRENT_PARTS.items():
                    hit = next((base / c for c in candidates if (base / c).exists()), None)
                    if hit is None:
                        raise FileNotFoundError(f"{key}: none of {candidates} under {base}")
                    parts[name] = hit
                return parts
        raise FileNotFoundError(
            f"{key}: no train_split/ under {drop} or {drop / 'Fresh_water_disease'}"
        )
    source = next(s for s in DATASET_SOURCES if s.key == key)
    target = drop / source.delivered_subdir
    if not target.is_dir():
        raise FileNotFoundError(
            f"{key}: expected {target} (delivered_subdir {source.delivered_subdir!r})"
        )
    return {"": target}


def check_delivery(drop: Path, repo_root: Path) -> dict[str, Any]:
    clean = read_clean_manifest(repo_root / "data" / "audit" / CLEAN_MANIFEST_NAME)
    inventory_all = Counter(r.source_dataset for r in clean)
    inventory_included = Counter(r.source_dataset for r in clean if r.included)
    report: dict[str, Any] = {"drop": str(drop), "keys": {}, "errors": [], "ok": False}
    top_level = sorted(p.name for p in drop.iterdir() if not p.name.startswith("."))
    report["top_level_entries"] = top_level
    # an EXCLUDED source's folder may still sit in the drop: recorded, never read
    report["excluded_present"] = {
        s.key: str(drop / s.delivered_subdir)
        for s in EXCLUDED_SOURCES
        if (drop / s.delivered_subdir).exists()
    }
    for source in DATASET_SOURCES:
        key = source.key
        entry: dict[str, Any] = {"name": source.name, "delivered_subdir": source.delivered_subdir}
        try:
            parts = resolve_key(drop, key)
        except FileNotFoundError as exc:
            entry["resolved"] = None
            entry["error"] = str(exc)
            report["errors"].append(str(exc))
            report["keys"][key] = entry
            continue
        entry["resolved"] = {k or ".": str(v) for k, v in parts.items()}
        entry["images_found"] = sum(_count_images(v) for v in parts.values() if v.is_dir())
        entry["inventory_all"] = inventory_all.get(key, 0)
        entry["inventory_included"] = inventory_included.get(key, 0)
        rows = [r for r in clean if r.source_dataset == key]

        def locate(filepath: str) -> Path:
            rel = Path(filepath).relative_to(Path("data") / "raw" / key)
            if key == CURRENT_KEY:
                head, *tail = rel.parts
                return parts[head].joinpath(*tail)
            return parts[""] / rel

        missing_all = [r.filepath for r in rows if not locate(r.filepath).is_file()]
        missing_included = [
            r.filepath for r in rows if r.included and not locate(r.filepath).is_file()
        ]
        entry["manifest_rows_resolved"] = f"{len(rows) - len(missing_all)}/{len(rows)}"
        n_inc = inventory_included.get(key, 0)
        entry["included_rows_resolved"] = f"{n_inc - len(missing_included)}/{n_inc}"
        entry["missing_included_examples"] = missing_included[:5]
        entry["surplus_vs_inventory"] = entry["images_found"] - entry["inventory_all"]
        if missing_included:
            report["errors"].append(
                f"{key}: {len(missing_included)} included manifest rows missing"
            )
        report["keys"][key] = entry
    report["ok"] = not report["errors"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, required=True, help="the uploaded delivery folder")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument(
        "--out", type=Path, default=None, help="default results/v2/drive_delivery.json"
    )
    args = parser.parse_args(argv)
    drop = Path(args.source).expanduser()
    if not drop.is_dir():
        parser.error(f"{drop} is not a directory")
    report = check_delivery(drop, args.repo_root)
    out = args.out or (Path(args.repo_root) / "results" / "v2" / "drive_delivery.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"top level: {report['top_level_entries']}")
    for key, e in report["keys"].items():
        if e.get("resolved") is None:
            print(f"[FAIL] {key:18s} {e['error']}")
            continue
        status = "FAIL" if any(err.startswith(key) for err in report["errors"]) else "ok"
        print(
            f"[{status}] {key:18s} found {e['images_found']:5d}  inventory {e['inventory_all']:5d}"
            f" (included {e['inventory_included']:5d})  rows {e['manifest_rows_resolved']:>10s}"
            f"  included {e['included_rows_resolved']:>10s}  -> {list(e['resolved'].values())[0]}"
        )
    for key, path in report["excluded_present"].items():
        print(f"[ignored] {key:14s} EXCLUDED source (dataset configuration v3) present at {path}")
    print("RESULT:", "DELIVERY MAPPED" if report["ok"] else "NOT VERIFIED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
