#!/usr/bin/env python
"""Audit an image folder before it is used for anything.

Reports, from the data itself: class directories, images per class, file
extensions, image sizes and modes, decode failures, and byte-identical
duplicates. Exits non-zero if any file cannot be decoded or if duplicates
exist, because either would corrupt a train/val/test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.config import ORIGINAL_DIR  # noqa: E402
from src.dataset import ImageDecodeError, discover_classes, index_samples, load_image  # noqa: E402


@dataclass
class AuditReport:
    root: str
    class_names: list[str]
    images_per_class: dict[str, int]
    total_images: int
    extensions: dict[str, int]
    modes: dict[str, int]
    sizes: dict[str, int]
    corrupt: list[str] = field(default_factory=list)
    duplicates: list[list[str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.corrupt and not self.duplicates


def audit(root: Path) -> AuditReport:
    class_names = discover_classes(root)
    samples = index_samples(root, class_names)

    per_class: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    corrupt: list[str] = []
    by_digest: dict[str, list[str]] = defaultdict(list)

    for sample in samples:
        per_class[class_names[sample.label]] += 1
        extensions[sample.path.suffix.lower()] += 1
        by_digest[hashlib.sha256(sample.path.read_bytes()).hexdigest()].append(str(sample.path))
        try:
            with Image.open(sample.path) as image:
                modes[image.mode] += 1
                sizes[f"{image.width}x{image.height}"] += 1
            load_image(sample.path)
        except (ImageDecodeError, OSError):
            corrupt.append(str(sample.path))

    return AuditReport(
        root=str(root),
        class_names=class_names,
        images_per_class={name: per_class.get(name, 0) for name in class_names},
        total_images=len(samples),
        extensions=dict(extensions.most_common()),
        modes=dict(modes.most_common()),
        sizes=dict(sizes.most_common()),
        corrupt=corrupt,
        duplicates=[paths for paths in by_digest.values() if len(paths) > 1],
    )


def format_report(report: AuditReport) -> str:
    lines = [f"Dataset audit — {report.root}", "", "Images per class"]
    lines += [f"  {name:<24} {count}" for name, count in report.images_per_class.items()]
    lines += [f"  {'total':<24} {report.total_images}", "", "Extensions"]
    lines += [f"  {ext:<24} {count}" for ext, count in report.extensions.items()]
    lines += ["", "Image modes"]
    lines += [f"  {mode:<24} {count}" for mode, count in report.modes.items()]
    lines += ["", f"Distinct sizes: {len(report.sizes)}"]
    lines += [f"  {size:<24} {count}" for size, count in list(report.sizes.items())[:10]]
    if len(report.sizes) > 10:
        lines.append(f"  … {len(report.sizes) - 10} more")
    lines += ["", f"Corrupt files: {len(report.corrupt)}"]
    lines += [f"  {path}" for path in report.corrupt]
    lines += ["", f"Duplicate groups: {len(report.duplicates)}"]
    for group in report.duplicates:
        lines.append("  " + "  ==  ".join(group))
    lines.append("")
    lines.append(
        "RESULT: OK" if report.ok else "RESULT: FAIL — fix corrupt/duplicate files before splitting"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=ORIGINAL_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = audit(args.root)
    print(json.dumps(asdict(report), indent=2) if args.json else format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
