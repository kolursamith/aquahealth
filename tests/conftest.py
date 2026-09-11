"""Shared fixtures: controlled synthetic image folders.

Images are generated, never real. Each class is rendered in a distinct base
colour, so a test can check that a sample's *content* agrees with its label —
not merely that counts add up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pytest
from PIL import Image

ImageFolderFactory = Callable[..., Path]


def class_colour(class_index: int) -> tuple[int, int, int]:
    """Deterministic, distinct RGB triple per class index."""
    return (30 + 25 * class_index, 200 - 20 * class_index, 60 + 10 * class_index)


def write_image_folder(
    root: Path,
    class_names: Sequence[str],
    per_class: int = 3,
    size: tuple[int, int] = (48, 40),
    formats: Sequence[str] = ("png", "jpg"),
    seed: int = 0,
) -> Path:
    """Create `<root>/<class>/<i>.<fmt>` images filled with the class colour plus noise."""
    rng = np.random.default_rng(seed)
    width, height = size
    for class_index, class_name in enumerate(class_names):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        base = np.array(class_colour(class_index), dtype=np.int16)
        for i in range(per_class):
            noise = rng.integers(-5, 6, size=(height, width, 3), dtype=np.int16)
            pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
            fmt = formats[i % len(formats)]
            Image.fromarray(pixels).save(class_dir / f"{i:03d}.{fmt}", quality=95)
    return root


@pytest.fixture
def make_image_folder(tmp_path: Path) -> ImageFolderFactory:
    def factory(
        class_names: Sequence[str] = ("alpha", "beta", "gamma"),
        per_class: int = 3,
        name: str = "images",
        **kwargs,
    ) -> Path:
        return write_image_folder(tmp_path / name, class_names, per_class=per_class, **kwargs)

    return factory
