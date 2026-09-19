#!/usr/bin/env python
"""Reference the delivered raw datasets under data/raw/<key> without copying them.

    python scripts/link_raw_datasets.py --source /path/to/delivery/Dataset

`--source` (or $AQUAHEALTH_DATASET_SOURCE) is the folder that holds the
unpacked deliveries. Every entry in src.multi_dataset.DATASET_SOURCES becomes
a symlink data/raw/<key> -> <source>/<delivered_subdir>; `current_freshwater`
is a directory of three links (train_split, test_split, test.csv) because the
delivery keeps those parts loose at the drop root, with the flat test folder
named either `test_split` or `test_split&validation`.

Follows the existing convention (data/original/aquahealth is a symlink; images
are never copied into the repository). Idempotent: an existing link that
already points to the right place is left alone; a wrong one is reported and
replaced only with --force. Real directories are never touched.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR  # noqa: E402
from src.multi_dataset import DATASET_SOURCES, DatasetSource, raw_root  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("link_raw_datasets")
SOURCE_ENV = "AQUAHEALTH_DATASET_SOURCE"
CURRENT_PARTS = {
    "train_split": ("train_split",),
    "test_split": ("test_split", "test_split&validation"),
    "test.csv": ("test.csv",),
}
# Folders (relative to the drop) in which the current dataset's loose parts may sit:
# the drop root itself, or a wrapper folder as in the re-packed delivery.
CURRENT_SUBDIRS = ("", "Fresh_water_disease")


def _first_existing(base: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        if (base / name).exists():
            return base / name
    return None


def _current_base(drop: Path, subdirs: tuple[str, ...]) -> Path:
    """The folder that holds train_split/: the first of `subdirs` under `drop` that has it."""
    for sub in subdirs:
        if _first_existing(drop / sub, CURRENT_PARTS["train_split"]) is not None:
            return drop / sub
    raise FileNotFoundError(f"current_freshwater: no train_split/ under {drop} or {subdirs[1:]}")


def link(link_path: Path, target: Path, force: bool) -> str:
    """Create link_path -> target. Returns one of: created | unchanged | replaced | skipped."""
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return "unchanged"
        if not force:
            logger.warning(
                "%s -> %s differs from %s (use --force)", link_path, link_path.resolve(), target
            )
            return "skipped"
        link_path.unlink()
        link_path.symlink_to(target)
        return "replaced"
    if link_path.exists():
        logger.warning("%s exists and is not a symlink; left untouched", link_path)
        return "skipped"
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target)
    return "created"


def link_source(
    source: DatasetSource,
    drop: Path,
    raw_dir: Path,
    force: bool,
    current_subdirs: tuple[str, ...] = CURRENT_SUBDIRS,
) -> list[str]:
    outcomes: list[str] = []
    if source.key == "current_freshwater":
        folder = raw_dir / source.key
        folder.mkdir(parents=True, exist_ok=True)
        base = _current_base(drop, current_subdirs)
        for name, candidates in CURRENT_PARTS.items():
            target = _first_existing(base, candidates)
            if target is None:
                raise FileNotFoundError(f"{source.key}: none of {candidates} found under {base}")
            outcome = link(folder / name, target, force)
            logger.info("%s/%s -> %s (%s)", source.key, name, target, outcome)
            outcomes.append(outcome)
        return outcomes
    target = drop / source.delivered_subdir
    if not target.is_dir():
        raise FileNotFoundError(f"{source.key}: {target} is not a directory")
    outcome = link(raw_dir / source.key, target, force)
    logger.info("%s -> %s (%s)", source.key, target, outcome)
    return [outcome]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=os.environ.get(SOURCE_ENV),
        help=f"delivery folder holding the unpacked datasets (default: ${SOURCE_ENV})",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--only", nargs="*", default=None, help="dataset keys to link (default: all)"
    )
    parser.add_argument("--force", action="store_true", help="replace links that differ")
    parser.add_argument(
        "--current-subdir",
        default=None,
        help="folder under --source holding train_split/test_split/test.csv "
        f"(default: try {CURRENT_SUBDIRS})",
    )
    args = parser.parse_args(argv)
    if args.source is None:
        parser.error(f"--source or ${SOURCE_ENV} is required")
    drop = Path(args.source).expanduser()
    if not drop.is_dir():
        parser.error(f"{drop} is not a directory")
    raw_dir = raw_root(args.data_dir)
    keys = set(args.only) if args.only else {s.key for s in DATASET_SOURCES}
    unknown = keys - {s.key for s in DATASET_SOURCES}
    if unknown:
        parser.error(f"unknown dataset keys {sorted(unknown)}")
    failures = 0
    for source in DATASET_SOURCES:
        if source.key not in keys:
            continue
        try:
            subdirs = (args.current_subdir,) if args.current_subdir is not None else CURRENT_SUBDIRS
            link_source(source, drop, raw_dir, args.force, subdirs)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
