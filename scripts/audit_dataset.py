#!/usr/bin/env python
"""Reports total images, images/class, corrupt files, and file extensions
found under data/original/.
"""

from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, ORIGINAL_DIR


def audit(root: Path = ORIGINAL_DIR) -> None:
    counts: Counter = Counter()
    extensions: Counter = Counter()
    corrupt: list[Path] = []

    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            print(f"[WARN] missing class directory: {class_dir}")
            continue

        for path in class_dir.glob("*.*"):
            counts[class_name] += 1
            extensions[path.suffix.lower()] += 1
            try:
                with Image.open(path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                corrupt.append(path)

    print("Images per class:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {counts.get(class_name, 0)}")
    print(f"Total: {sum(counts.values())}")

    print("\nFile extensions:")
    for ext, count in extensions.most_common():
        print(f"  {ext}: {count}")

    print(f"\nCorrupt files: {len(corrupt)}")
    for path in corrupt:
        print(f"  {path}")


if __name__ == "__main__":
    audit()
