#!/usr/bin/env python
"""Creates a stratified train/val/test split from data/original/ into
data/train/, data/val/, data/test/.
"""

import argparse
import random
import shutil

from src.config import CLASS_NAMES, ORIGINAL_DIR, SEED, TEST_DIR, TRAIN_DIR, VAL_DIR


def create_split(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = SEED,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    for class_name in CLASS_NAMES:
        source_dir = ORIGINAL_DIR / class_name
        if not source_dir.is_dir():
            print(f"[WARN] missing class directory: {source_dir}")
            continue

        images = sorted(source_dir.glob("*.*"))
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR: images[n_train : n_train + n_val],
            TEST_DIR: images[n_train + n_val :],
        }

        for split_dir, files in splits.items():
            dest_dir = split_dir / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_path in files:
                shutil.copy2(src_path, dest_dir / src_path.name)

        print(
            f"{class_name}: {len(splits[TRAIN_DIR])} train / "
            f"{len(splits[VAL_DIR])} val / {len(splits[TEST_DIR])} test"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    create_split(args.train_ratio, args.val_ratio, args.test_ratio)
