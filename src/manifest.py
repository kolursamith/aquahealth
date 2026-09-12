"""Frozen split manifest: the single source of truth for which image belongs to
which split, with a fixed label index per class.

    data/split_manifest.csv      filepath,label,class_name,split   (repo-relative paths)
    data/split_manifest.sha256   digest of the CSV; the manifest is immutable once written

Labels are NOT derived from directory order here (unlike ImageFolderDataset):
they follow `CANONICAL_CLASSES`, the project's agreed 0..7 mapping, so the
index is stable no matter how the source folders are named or sorted.
"""

from __future__ import annotations

import csv
import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import pil_to_tensor

from src.config import ROOT_DIR, SEED
from src.dataset import Sample, load_image

MANIFEST_PATH = ROOT_DIR / "data" / "split_manifest.csv"
MANIFEST_DIGEST_PATH = ROOT_DIR / "data" / "split_manifest.sha256"
MANIFEST_COLUMNS = ("filepath", "label", "class_name", "split")
SPLITS = ("train", "val", "test")
FROZEN_SPLIT = "test"

# Agreed project mapping (index = label). Source folders are mapped onto these
# names explicitly (see scripts/build_split_manifest.py); folders are never renamed.
CANONICAL_CLASSES: tuple[str, ...] = (
    "Bacterial Red Disease",
    "Aeromoniasis",
    "Bacterial Gill Disease",
    "EUS Disease",
    "Saprolegniasis",
    "Parasitic Disease",
    "White Tail Disease",
    "Healthy Fish",
)

# Exact source-folder names as delivered -> canonical class. Reported, not silent.
SOURCE_FOLDER_TO_CLASS: dict[str, str] = {
    "Bacterial Red disease": "Bacterial Red Disease",
    "Bacterial diseases - Aeromoniasis": "Aeromoniasis",
    "Bacterial gill disease": "Bacterial Gill Disease",
    "EUS": "EUS Disease",
    "Fungal diseases Saprolegniasis": "Saprolegniasis",
    "Parasitic diseases": "Parasitic Disease",
    "Viral diseases White tail disease": "White Tail Disease",
    "Healthy Fish": "Healthy Fish",
}

Transform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class ManifestRow:
    filepath: str
    label: int
    class_name: str
    split: str

    def resolve(self, root: Path = ROOT_DIR) -> Path:
        return root / self.filepath


# --- hashing -----------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def perceptual_dhash(image: Image.Image, size: int = 8) -> str:
    """Difference hash: robust to re-encoding/resizing, so it groups near-duplicates."""
    grey = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(grey, dtype=np.int16)
    bits = 0
    for left, right in zip(pixels[:, :-1].flatten(), pixels[:, 1:].flatten()):
        bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:0{size * size // 4}x}"


# --- manifest I/O and validation -------------------------------------------------------


def write_manifest(rows: Iterable[ManifestRow], path: Path = MANIFEST_PATH) -> str:
    """Write the CSV and its digest file; returns the digest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow([row.filepath, row.label, row.class_name, row.split])
    digest = file_sha256(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def read_manifest(path: Path = MANIFEST_PATH) -> list[ManifestRow]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"manifest columns {reader.fieldnames} != {list(MANIFEST_COLUMNS)}")
        return [
            ManifestRow(r["filepath"], int(r["label"]), r["class_name"], r["split"]) for r in reader
        ]


def verify_manifest_digest(path: Path = MANIFEST_PATH) -> str:
    """Raise if the manifest no longer matches its recorded digest (immutability guard)."""
    path = Path(path)
    recorded = path.with_suffix(".sha256").read_text().split()[0]
    actual = file_sha256(path)
    if recorded != actual:
        raise ValueError(
            f"{path.name} has been modified: digest {actual[:12]} != recorded {recorded[:12]}"
        )
    return actual


def validate_manifest(
    rows: list[ManifestRow], class_names: tuple[str, ...] = CANONICAL_CLASSES
) -> dict[str, Counter]:
    """Structural checks; returns per-split class counts. Raises on any violation."""
    if not rows:
        raise ValueError("manifest is empty")
    seen: dict[str, str] = {}
    for row in rows:
        if row.split not in SPLITS:
            raise ValueError(f"unknown split {row.split!r} for {row.filepath}")
        if not 0 <= row.label < len(class_names):
            raise ValueError(f"label {row.label} out of range for {row.filepath}")
        if class_names[row.label] != row.class_name:
            raise ValueError(
                f"{row.filepath}: label {row.label} is {class_names[row.label]!r}, "
                f"row says {row.class_name!r}"
            )
        if row.filepath in seen:
            raise ValueError(f"{row.filepath} appears in both {seen[row.filepath]} and {row.split}")
        seen[row.filepath] = row.split
    counts = {split: Counter(r.class_name for r in rows if r.split == split) for split in SPLITS}
    for split in SPLITS:
        missing = [c for c in class_names if counts[split][c] == 0]
        if missing:
            raise ValueError(f"split {split!r} has no images for {missing}")
    return counts


# --- split construction --------------------------------------------------------------------


def largest_remainder_quota(n: int, ratios: tuple[float, ...]) -> list[int]:
    """Integer counts summing to n that best match `ratios` (Hamilton apportionment)."""
    raw = [n * r for r in ratios]
    base = [int(x) for x in raw]
    for index in sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)[
        : n - sum(base)
    ]:
        base[index] += 1
    return base


def group_aware_stratified_split(
    items: list[tuple[str, int, str]],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = SEED,
) -> list[ManifestRow]:
    """Assign items (filepath, label, group_id) to train/val/test, per class, keeping
    every group (near-duplicate cluster) inside one split.

    Groups are shuffled with `seed` and handed, largest first, to whichever
    split is furthest below its quota; the result is deterministic.
    """
    if abs(sum(ratios) - 1.0) > 1e-9 or min(ratios) < 0:
        raise ValueError("ratios must be non-negative and sum to 1")
    rows: list[ManifestRow] = []
    by_class: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for filepath, label, group_id in items:
        by_class[label][group_id].append(filepath)

    rng = random.Random(seed)
    for label in sorted(by_class):
        groups = [sorted(paths) for _, paths in sorted(by_class[label].items())]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)
        total = sum(len(g) for g in groups)
        quota = largest_remainder_quota(total, ratios)
        filled = [0, 0, 0]
        for group in groups:
            deficit = [(quota[i] - filled[i]) / max(quota[i], 1) for i in range(3)]
            target = max(range(3), key=lambda i: (deficit[i], -i))
            filled[target] += len(group)
            rows.extend(
                ManifestRow(p, label, CANONICAL_CLASSES[label], SPLITS[target]) for p in group
            )
    rows.sort(key=lambda r: (SPLITS.index(r.split), r.label, r.filepath))
    return rows


# --- dataset ----------------------------------------------------------------------------------


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    """Images of one split of the manifest, labelled by the canonical index."""

    def __init__(
        self,
        rows: list[ManifestRow],
        split: str,
        *,
        transform: Transform | None = None,
        root: Path = ROOT_DIR,
        class_names: tuple[str, ...] = CANONICAL_CLASSES,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        self.split = split
        self.root = Path(root)
        self.transform = transform
        self.class_names = list(class_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.rows = [r for r in rows if r.split == split]
        if not self.rows:
            raise ValueError(f"manifest has no rows for split {split!r}")
        self.samples = [Sample(path=self.root / r.filepath, label=r.label) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

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
        return {name: counts.get(i, 0) for i, name in enumerate(self.class_names)}

    def missing_files(self) -> list[Path]:
        return [s.path for s in self.samples if not s.path.is_file()]


def build_split_dataset(
    split: str,
    *,
    transform: Transform | None = None,
    manifest_path: Path = MANIFEST_PATH,
    root: Path = ROOT_DIR,
    class_names: list[str] | None = None,
) -> ManifestDataset:
    """One split of the frozen manifest, digest-verified, ready for a DataLoader.

    `class_names` (e.g. from a checkpoint) must equal the canonical list; the
    manifest's label indices are only meaningful under that mapping.
    """
    verify_manifest_digest(manifest_path)
    if class_names is not None and list(class_names) != list(CANONICAL_CLASSES):
        raise ValueError(
            f"class names {class_names} do not match the manifest's canonical classes "
            f"{list(CANONICAL_CLASSES)}"
        )
    return ManifestDataset(read_manifest(manifest_path), split, transform=transform, root=root)
