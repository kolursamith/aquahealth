#!/usr/bin/env python
"""Audit the real dataset and build the frozen 70/15/15 split manifest.

Reads the dataset (never writes into it), then writes:

    results/data_audit.csv          one row per discovered file
    results/data_audit_report.md    human-readable audit + split report
    data/split_manifest.csv         filepath,label,class_name,split  (+ .sha256)

Refuses to overwrite an existing manifest: the split is immutable once made.
Run again with --dry-run to re-audit without touching the manifest.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, UnidentifiedImageError  # noqa: E402

from src.config import ROOT_DIR, SEED  # noqa: E402
from src.dataset import IMAGE_EXTENSIONS  # noqa: E402
from src.manifest import (  # noqa: E402
    CANONICAL_CLASSES,
    MANIFEST_PATH,
    SOURCE_FOLDER_TO_CLASS,
    SPLITS,
    ManifestRow,
    file_sha256,
    group_aware_stratified_split,
    perceptual_dhash,
    validate_manifest,
    write_manifest,
)

DEFAULT_DATASET = ROOT_DIR / "data" / "original" / "aquahealth"
ANNOTATION_EXTENSIONS = {".txt", ".xml", ".json", ".yaml", ".yml"}


@dataclass
class FileRecord:
    filepath: str  # repo-relative
    source_split: str
    source_folder: str
    class_name: str
    label: int
    extension: str
    width: int
    height: int
    mode: str
    sha256: str
    dhash: str
    status: str  # ok | corrupt | unmapped | label_conflict
    exact_dup_group: str = ""
    near_dup_group: str = ""
    split: str = ""


def _test_labels(dataset: Path) -> dict[str, str]:
    csv_path = dataset / "test.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="") as handle:
        return {row["filename"]: row["label"] for row in csv.DictReader(handle)}


def discover(dataset: Path) -> tuple[list[FileRecord], list[str], list[str]]:
    """Walk train_split/<folder>/ and the flat test_split/.

    Returns (records, unexpected directories, unexpected files).
    """
    records: list[FileRecord] = []
    unexpected_dirs: list[str] = []
    unexpected_files: list[str] = []
    test_labels = _test_labels(dataset)

    def record(path: Path, source_split: str, folder: str) -> None:
        class_name = SOURCE_FOLDER_TO_CLASS.get(folder, "")
        status = "ok" if class_name else "unmapped"
        width = height = 0
        mode = ""
        dhash = ""
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height, mode = image.width, image.height, image.mode
                dhash = perceptual_dhash(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError):
            status = "corrupt"
        records.append(
            FileRecord(
                filepath=str(Path("data/original") / path.relative_to(dataset.parent)),
                source_split=source_split,
                source_folder=folder,
                class_name=class_name,
                label=CANONICAL_CLASSES.index(class_name) if class_name else -1,
                extension=path.suffix.lower(),
                width=width,
                height=height,
                mode=mode,
                sha256=file_sha256(path),
                dhash=dhash,
                status=status,
            )
        )

    for entry in sorted(dataset.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_file():
            if entry.name != "test.csv":
                unexpected_files.append(entry.name)
            continue
        if entry.name not in ("train_split", "test_split"):
            unexpected_dirs.append(entry.name)

    train_root = dataset / "train_split"
    for stray in sorted(
        p for p in train_root.iterdir() if p.is_file() and not p.name.startswith(".")
    ):
        unexpected_files.append(str(stray.relative_to(dataset)))
    for folder in sorted(
        p for p in train_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    ):
        for path in sorted(folder.iterdir()):
            if path.name.startswith("."):
                continue
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                record(path, "train_split", folder.name)
            else:
                unexpected_files.append(str(path.relative_to(dataset)))
    for path in sorted((dataset / "test_split").iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            unexpected_dirs.append(str(path.relative_to(dataset)))
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            unexpected_files.append(str(path.relative_to(dataset)))
            continue
        label_name = test_labels.get(path.name) or path.name.split("_")[0]
        record(path, "test_split", label_name)
    return records, unexpected_dirs, unexpected_files


def flag_label_conflicts(records: list[FileRecord]) -> int:
    """Near-duplicate groups whose members carry different classes cannot be trusted:
    the label of every member is set to `label_conflict` and they leave the split."""
    by_group: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r.status == "ok" and r.near_dup_group:
            by_group[r.near_dup_group].add(r.class_name)
    conflicts = {g for g, names in by_group.items() if len(names) > 1}
    for r in records:
        if r.status == "ok" and r.near_dup_group in conflicts:
            r.status = "label_conflict"
    return len(conflicts)


def assign_duplicate_groups(records: list[FileRecord]) -> tuple[int, int]:
    by_sha: dict[str, list[FileRecord]] = defaultdict(list)
    by_dhash: dict[str, list[FileRecord]] = defaultdict(list)
    for r in records:
        if r.status != "corrupt":
            by_sha[r.sha256].append(r)
            if r.dhash:
                by_dhash[r.dhash].append(r)
    exact = sum(1 for v in by_sha.values() if len(v) > 1)
    for index, (digest, group) in enumerate(sorted(by_sha.items())):
        if len(group) > 1:
            for r in group:
                r.exact_dup_group = f"x{index}"
    near = 0
    for index, (digest, group) in enumerate(sorted(by_dhash.items())):
        if len({r.sha256 for r in group}) > 1:
            near += 1
        for r in group:
            r.near_dup_group = f"g{index}"
    return exact, near


def build_split(records: list[FileRecord], seed: int) -> list[ManifestRow]:
    """Exact duplicates collapse to one representative; near-duplicate groups stay together."""
    kept: dict[str, FileRecord] = {}
    for r in sorted(records, key=lambda r: r.filepath):
        if r.status == "ok" and r.sha256 not in kept:
            kept[r.sha256] = r
    items = [(r.filepath, r.label, r.near_dup_group or r.sha256) for r in kept.values()]
    return group_aware_stratified_split(items, seed=seed)


def write_audit_csv(records: list[FileRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def write_report(
    path: Path,
    dataset: Path,
    records: list[FileRecord],
    rows: list[ManifestRow],
    unexpected_dirs: list[str],
    unexpected_files: list[str],
    exact_groups: int,
    near_groups: int,
    conflict_groups: int,
    seed: int,
    manifest_digest: str | None,
) -> None:
    ok = [r for r in records if r.status == "ok"]
    corrupt = [r for r in records if r.status == "corrupt"]
    unmapped = [r for r in records if r.status == "unmapped"]
    conflicts = [r for r in records if r.status == "label_conflict"]
    author_leak = len(
        {
            r.near_dup_group
            for r in records
            if r.near_dup_group
            and len({x.source_split for x in records if x.near_dup_group == r.near_dup_group}) > 1
        }
    )
    exact_images = sum(1 for r in records if r.exact_dup_group)
    by_split_counts = {s: Counter(r.class_name for r in rows if r.split == s) for s in SPLITS}
    lines = [
        "# Dataset audit and frozen split",
        "",
        f"Source (read-only, symlinked): `{dataset}`",
        "",
        "## Totals",
        "",
        f"- files discovered: {len(records)}",
        f"- valid images: {len(ok)}",
        f"- corrupt/unreadable: {len(corrupt)}",
        f"- unmapped class folders: {len(unmapped)}",
        f"- exact-duplicate groups (SHA-256): {exact_groups} ({exact_images} images involved)",
        f"- near-duplicate groups (dHash, distinct bytes): {near_groups}",
        f"- near-duplicate groups spanning the delivered train_split/test_split: {author_leak} "
        "(the delivered test split leaks; superseded by the group-aware split below)",
        f"- near-duplicate groups with conflicting labels: {conflict_groups} "
        f"({len(conflicts)} images excluded from the split as `label_conflict`)",
        f"- unexpected directories: {unexpected_dirs or 'none'}",
        f"- unexpected files: {unexpected_files or 'none'}",
        "- YOLO / bounding-box annotations: none found (only `test.csv` with image labels)",
        "",
        "## Class mapping (source folder → canonical class, index)",
        "",
        "| Source folder | Canonical class | Label |",
        "|---|---|---|",
    ]
    for folder, name in SOURCE_FOLDER_TO_CLASS.items():
        lines.append(f"| {folder} | {name} | {CANONICAL_CLASSES.index(name)} |")
    lines += [
        "",
        "## Per-class counts (valid images, before de-duplication)",
        "",
        "| Class | train_split | test_split | total |",
        "|---|---|---|---|",
    ]
    for name in CANONICAL_CLASSES:
        tr = sum(1 for r in ok if r.class_name == name and r.source_split == "train_split")
        te = sum(1 for r in ok if r.class_name == name and r.source_split == "test_split")
        lines.append(f"| {name} | {tr} | {te} | {tr + te} |")
    lines += ["", "## Image dimensions (valid images)", ""]
    sizes = Counter(f"{r.width}x{r.height}" for r in ok)
    for size, count in sizes.most_common():
        lines.append(f"- {size}: {count}")
    lines += [
        "",
        "### Resolution by class (shortcut-learning risk)",
        "",
        "| Class | " + " | ".join(s for s, _ in sizes.most_common(3)) + " | other |",
        "|---|" + "---|" * 4,
    ]
    top = [s for s, _ in sizes.most_common(3)]
    for name in CANONICAL_CLASSES:
        c = Counter(f"{r.width}x{r.height}" for r in ok if r.class_name == name)
        other = sum(v for k, v in c.items() if k not in top)
        lines.append(f"| {name} | " + " | ".join(str(c.get(s, 0)) for s in top) + f" | {other} |")
    lines += ["", "## Extensions", ""] + [
        f"- {e}: {n}" for e, n in Counter(r.extension for r in records).most_common()
    ]
    lines += [
        "",
        "## Frozen split",
        "",
        f"- seed: {seed}",
        "- ratios: 0.70 / 0.15 / 0.15 per class; exact duplicates collapsed to one image; "
        "near-duplicate groups kept within one split",
        f"- images in manifest: {len(rows)}",
        "- manifest: `data/split_manifest.csv` (sha256 "
        f"{manifest_digest[:16] + '…' if manifest_digest else 'dry-run, not written'})",
        "- **`split == test` is frozen**: evaluate once, at the end, never for model selection",
        "",
        "| Class | train | val | test | total |",
        "|---|---|---|---|---|",
    ]
    for name in CANONICAL_CLASSES:
        per_split = [by_split_counts[s][name] for s in SPLITS]
        lines.append(
            f"| {name} | {per_split[0]} | {per_split[1]} | {per_split[2]} | {sum(per_split)} |"
        )
    tot = [sum(by_split_counts[s].values()) for s in SPLITS]
    lines.append(f"| **total** | {tot[0]} | {tot[1]} | {tot[2]} | {sum(tot)} |")
    lines += [
        "",
        "### Resolution by split",
        "",
        "| Split | " + " | ".join(top) + " | other |",
        "|---|" + "---|" * 4,
    ]
    by_path = {r.filepath: r for r in records}
    for s in SPLITS:
        c = Counter(
            f"{by_path[r.filepath].width}x{by_path[r.filepath].height}"
            for r in rows
            if r.split == s
        )
        other = sum(v for k, v in c.items() if k not in top)
        lines.append(f"| {s} | " + " | ".join(str(c.get(x, 0)) for x in top) + f" | {other} |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument(
        "--dry-run", action="store_true", help="audit only; do not write the manifest"
    )
    args = parser.parse_args(argv)

    if not args.dry_run and args.manifest.exists():
        print(
            f"refusing to overwrite the frozen manifest {args.manifest}; use --dry-run to re-audit"
        )
        return 2

    records, unexpected_dirs, unexpected_files = discover(args.dataset)
    exact_groups, near_groups = assign_duplicate_groups(records)
    conflict_groups = flag_label_conflicts(records)
    rows = build_split(records, args.seed)
    validate_manifest(rows)
    assigned = {r.filepath: r.split for r in rows}
    for r in records:
        r.split = assigned.get(r.filepath, "")

    write_audit_csv(records, args.results_dir / "data_audit.csv")
    digest = None if args.dry_run else write_manifest(rows, args.manifest)
    write_report(
        args.results_dir / "data_audit_report.md",
        args.dataset,
        records,
        rows,
        unexpected_dirs,
        unexpected_files,
        exact_groups,
        near_groups,
        conflict_groups,
        args.seed,
        digest,
    )
    print((args.results_dir / "data_audit_report.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
