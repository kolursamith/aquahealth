#!/usr/bin/env python
"""Create a stratified, seeded train/val/test split from an image folder.

Every class present in the source is split independently with the same
ratios, so the class distribution of each split matches the source. Files
are copied, never moved; the source is left intact. Refuses to run if a
destination split already contains files, so a split is never silently
mixed with a previous one.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ORIGINAL_DIR, SEED, TEST_DIR, TRAIN_DIR, VAL_DIR  # noqa: E402
from src.dataset import discover_classes, index_samples  # noqa: E402


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split ratios must be non-negative")
        if abs(self.train + self.val + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")


def _partition(items: list[Path], ratios: SplitRatios, rng: random.Random) -> dict[str, list[Path]]:
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(n * ratios.train)
    n_val = round(n * ratios.val)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def create_split(
    source: Path,
    destinations: dict[str, Path],
    ratios: SplitRatios,
    seed: int = SEED,
) -> dict[str, dict[str, int]]:
    """Copy `source/<class>/*` into `destinations[split]/<class>/`; return counts."""
    for name, destination in destinations.items():
        if destination.exists() and any(
            p for p in destination.rglob("*") if p.is_file() and not p.name.startswith(".")
        ):
            raise FileExistsError(f"{name} split already contains files: {destination}")

    class_names = discover_classes(source)
    samples = index_samples(source, class_names)
    rng = random.Random(seed)
    counts: dict[str, dict[str, int]] = {name: {} for name in destinations}

    for label, class_name in enumerate(class_names):
        paths = [s.path for s in samples if s.label == label]
        for split_name, split_paths in _partition(paths, ratios, rng).items():
            target_dir = destinations[split_name] / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for path in split_paths:
                shutil.copy2(path, target_dir / path.name)
            counts[split_name][class_name] = len(split_paths)

    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ORIGINAL_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    ratios = SplitRatios(args.train_ratio, args.val_ratio, args.test_ratio)
    counts = create_split(
        args.source, {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}, ratios, args.seed
    )
    for split_name, per_class in counts.items():
        total = sum(per_class.values())
        print(f"{split_name:<6} {total:>6}  " + "  ".join(f"{k}={v}" for k, v in per_class.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
