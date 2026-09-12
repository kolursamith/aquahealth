"""Images on disk -> labelled samples -> Dataset -> DataLoader (Layer 5).

Layout expected under a split root (train/, val/, test/ each are roots):

    <root>/<class_name>/<image>.{jpg,jpeg,png,bmp,webp}

The class list is discovered from the directory names, sorted, so labels are
determined by the data — never by `src.config.CLASS_NAMES`. Splits other than
the one used for discovery are built with `class_names=` so every split shares
one label mapping.

Images are decoded with Pillow (Layer 2 decision) and handed to `transform`;
without a transform an item is the raw uint8 CHW tensor, so the dataset makes
no preprocessing choice of its own — that is Layer 6's job.

Owner: Student 2
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2.functional import pil_to_tensor

from src.config import BATCH_SIZE, NUM_WORKERS

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

Transform = Callable[[Image.Image], torch.Tensor]


class ImageDecodeError(RuntimeError):
    """A file that was indexed as an image could not be decoded."""


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def discover_classes(root: Path) -> list[str]:
    """Sorted names of the class directories directly under `root`."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not names:
        raise ValueError(f"no class directories found under {root}")
    return names


def is_image_file(path: Path) -> bool:
    return (
        path.is_file() and not path.name.startswith(".") and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def index_samples(root: Path, class_names: list[str]) -> list[Sample]:
    """Every image file under `root/<class>/`, in a deterministic (sorted) order.

    Directories under `root` that are not in `class_names` are an error: they
    would otherwise be silently dropped, hiding a label-mapping mismatch
    between splits.
    """
    root = Path(root)
    known = set(class_names)
    unknown = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in known
    )
    if unknown:
        raise ValueError(f"unexpected class directories under {root}: {unknown}")

    samples: list[Sample] = []
    for label, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if is_image_file(path):
                samples.append(Sample(path=path, label=label))
    return samples


def load_image(path: Path) -> Image.Image:
    """Decode an image file to an RGB PIL image; fail loudly naming the file."""
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError(f"could not decode image {path}: {exc}") from exc


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        *,
        transform: Transform | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.class_names = (
            list(class_names) if class_names is not None else discover_classes(self.root)
        )
        self.class_to_idx = {name: index for index, name in enumerate(self.class_names)}
        self.samples = index_samples(self.root, self.class_names)
        if not self.samples:
            raise ValueError(f"no image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = load_image(sample.path)
        tensor = self.transform(image) if self.transform is not None else pil_to_tensor(image)
        return tensor, sample.label

    @property
    def targets(self) -> list[int]:
        return [sample.label for sample in self.samples]

    def class_counts(self) -> dict[str, int]:
        counts = Counter(self.targets)
        return {name: counts.get(index, 0) for index, name in enumerate(self.class_names)}


def build_dataloader(
    dataset: Dataset[tuple[torch.Tensor, int]],
    *,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
    seed: int | None = None,
    drop_last: bool = False,
    persistent_workers: bool = False,
) -> DataLoader[tuple[torch.Tensor, int]]:
    """DataLoader whose shuffle order is reproducible when `seed` is given.

    `persistent_workers` keeps worker processes alive between epochs so the
    (slow, on macOS `spawn`) start-up cost is paid once per loader, not once
    per epoch. It is only meaningful with `num_workers > 0`.
    """
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        persistent_workers=persistent_workers and num_workers > 0,
    )
