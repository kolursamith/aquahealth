"""Multi-dataset inventory for the Phase-2 experiment (Phases 1, 2, 4 and 5 of the
new data pipeline): discover every image of every raw dataset, keep its
provenance, probe it, hash it and write a master manifest.

    data/raw/<key>/            one entry per raw dataset (symlink or directory,
                               created by scripts/link_raw_datasets.py; never
                               written into)
    data/audit/master_dataset.csv
                               one row per discovered file, MASTER_COLUMNS

Nothing here decides a split, removes a file or renames a label. Hashing reuses
`src.manifest.file_sha256` and `src.manifest.perceptual_dhash` so the numbers
are comparable with `results/data_audit.csv` from the first dataset audit.

Each raw dataset is described by a `DatasetSource` whose `layout` names the
walker that turns the delivered folder structure into `ImageRecord`s. The
layouts are the ones actually observed in the delivered copies; a dataset with
a different structure is an error, not a guess.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from PIL import Image, UnidentifiedImageError

from src.dataset import IMAGE_EXTENSIONS
from src.manifest import file_sha256, perceptual_dhash

RAW_ROOT_NAME = "raw"
AUDIT_DIR_NAME = "audit"
MASTER_MANIFEST_NAME = "master_dataset.csv"

# Files the delivered datasets are known to contain besides images; recorded,
# never silently ignored. Anything else is reported as unexpected.
DOCUMENTATION_EXTENSIONS = frozenset({".txt", ".csv", ".md", ".pdf"})
ARCHIVE_EXTENSIONS = frozenset({".zip", ".7z", ".rar", ".tar", ".gz"})
IGNORED_NAMES = frozenset({".DS_Store", "Thumbs.db"})


@dataclass(frozen=True)
class DatasetSource:
    """One raw dataset as delivered."""

    key: str  # directory name under data/raw/ and the `source_dataset` value
    name: str  # human-readable name as printed on the delivered folder/README
    layout: str  # walker name, see LAYOUTS
    delivered_subdir: str  # folder name inside the delivery drop, for link_raw_datasets
    documentation: tuple[str, ...] = ()  # documentation files found inside the dataset
    notes: str = ""


# The keys follow the structure agreed for the new experiment. `delivered_subdir`
# is the folder name as it exists in the download drop (relative to the drop
# root); `current_freshwater` is assembled from three entries by the link script.
DATASET_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        key="current_freshwater",
        name="Current AquaHealth dataset (train_split / test_split / test.csv)",
        layout="train_split_flat_test",
        delivered_subdir="",
        documentation=("test.csv",),
        notes="The dataset of the existing baseline; delivered as DATASET.zip. "
        "test_split is flat and labelled by test.csv (filename,label).",
    ),
    DatasetSource(
        key="kaptai",
        name="Fresh Water Fish Dataset",
        layout="class_folders",
        delivered_subdir="Fresh Water Fish Dataset",
        notes="Delivered as 'Fresh water fish disease dataset.zip'; no README or "
        "metadata inside the dataset. <class>/<image>.",
    ),
    DatasetSource(
        key="roboflow",
        name="Fish Disease - v1 2023-10-04 (Roboflow export)",
        layout="split_class_folders",
        delivered_subdir="Fish Disease.v1i.folder",
        documentation=("README.dataset.txt", "README.roboflow.txt"),
        notes="Roboflow 'folder' export: <train|valid|test>/<class>/<image>; README says "
        "454 images, auto-orient + resize 640x640 (stretch), no augmentation.",
    ),
    DatasetSource(
        key="mendeley",
        name="MatsyaDx-BD",
        layout="matsyadx",
        delivered_subdir="MatsyaDx-BD An image dataset of freshwater fish di/MatsyaDx-BD",
        documentation=("metadata.csv",),
        notes="<health_condition>/<fish_category>/<specimen>/<image> plus metadata.csv "
        "(image_id, health_condition, fish_category, specimen_id, ...). The four "
        "per-class .7z archives next to the folders are the delivered originals.",
    ),
    DatasetSource(
        key="paper_dataset",
        name="SalmonScan",
        layout="class_folders",
        delivered_subdir=(
            "SalmonScan A Novel Image Dataset for Fish Disease Detection in Salmon "
            "Aquaculture System/SalmonScan"
        ),
        notes="<FreshFish|InfectedFish>/<image>.png; no README or metadata inside the "
        "dataset (SalmonScan.zip next to the folder is the delivered original).",
    ),
)

SOURCE_BY_KEY: dict[str, DatasetSource] = {s.key: s for s in DATASET_SOURCES}


@dataclass
class ImageRecord:
    """One discovered file with its provenance. Label fields are filled later by
    src/label_harmonization.py; hash/duplicate fields by the probing and
    duplicate-analysis steps (the attribute names match
    scripts/build_split_manifest.FileRecord so its grouping helpers apply)."""

    image_id: str
    source_dataset: str
    filepath: str  # repo-relative, through data/raw/<key>/...
    original_path: str  # relative to the dataset root, exactly as delivered
    original_filename: str
    original_class: str
    original_split: str  # train/valid/test folder of the delivery, or ""
    species: str  # only when the dataset states it
    specimen_id: str  # only when the dataset states it
    extension: str
    unified_class: str = ""
    mapping_status: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    file_size: int = 0
    sha256: str = ""
    dhash: str = ""
    status: str = "pending"  # ok | corrupt
    exact_dup_group: str = ""
    near_dup_group: str = ""
    dedup_action: str = ""

    @property
    def class_name(self) -> str:
        """Alias used by the reused duplicate helpers (they compare `class_name`)."""
        return self.unified_class


MASTER_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ImageRecord))


@dataclass(frozen=True)
class NonImageFile:
    source_dataset: str
    original_path: str
    kind: str  # documentation | archive | ignored | unexpected
    size: int


def make_image_id(source_key: str, filepath: str) -> str:
    """Deterministic id from the repo-relative path (stable across re-runs)."""
    digest = hashlib.sha256(filepath.encode("utf-8")).hexdigest()[:12]
    return f"{source_key}-{digest}"


def classify_non_image(path: Path, documentation: tuple[str, ...]) -> str:
    if path.name in IGNORED_NAMES:
        return "ignored"
    if path.name in documentation or path.suffix.lower() in DOCUMENTATION_EXTENSIONS:
        return "documentation"
    if path.suffix.lower() in ARCHIVE_EXTENSIONS:
        return "archive"
    return "unexpected"


# --- layout walkers -------------------------------------------------------------
#
# Each walker yields (relative_path, original_split, original_class, species,
# specimen_id) for every image file, plus the NonImageFile entries it passes.
# They read the directory tree only; hashing happens in `probe`.

WalkItem = tuple[Path, str, str, str, str]


def _iter_files(directory: Path) -> Iterator[Path]:
    """Every regular file below `directory`, sorted; symlinked directories are
    followed because data/raw/ entries are links into the delivery drop."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=True):
        dirnames.sort()
        found.extend(Path(dirpath) / name for name in filenames)
    for path in sorted(found):
        if path.is_file():
            yield path


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and path.name not in IGNORED_NAMES


def walk_class_folders(root: Path) -> Iterator[WalkItem | Path]:
    """<root>/<class>/<image>."""
    for path in _iter_files(root):
        rel = path.relative_to(root)
        if _is_image(path) and len(rel.parts) == 2:
            yield path, "", rel.parts[0], "", ""
        else:
            yield path


def walk_split_class_folders(root: Path) -> Iterator[WalkItem | Path]:
    """<root>/<train|valid|test>/<class>/<image>."""
    for path in _iter_files(root):
        rel = path.relative_to(root)
        if _is_image(path) and len(rel.parts) == 3:
            yield path, rel.parts[0], rel.parts[1], "", ""
        else:
            yield path


def walk_train_split_flat_test(root: Path) -> Iterator[WalkItem | Path]:
    """<root>/train_split/<class>/<image> and flat <root>/test_split/<image>
    labelled by <root>/test.csv (filename,label) — the current dataset's layout,
    read exactly as scripts/build_split_manifest.py reads it."""
    test_labels: dict[str, str] = {}
    csv_path = root / "test.csv"
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            test_labels = {row["filename"]: row["label"] for row in csv.DictReader(handle)}
    for path in _iter_files(root):
        rel = path.relative_to(root)
        if not _is_image(path):
            yield path
        elif rel.parts[0] == "train_split" and len(rel.parts) == 3:
            yield path, "train_split", rel.parts[1], "", ""
        elif rel.parts[0] == "test_split" and len(rel.parts) == 2:
            label = test_labels.get(path.name) or path.name.split("_")[0]
            yield path, "test_split", label, "", ""
        else:
            yield path


def walk_matsyadx(root: Path) -> Iterator[WalkItem | Path]:
    """<root>/<health_condition>/<fish_category>/<specimen>/<image>, cross-checked
    against <root>/metadata.csv when present (the CSV is authoritative for the
    label fields; a disagreement with the folder path is an error)."""
    meta: dict[str, dict[str, str]] = {}
    csv_path = root / "metadata.csv"
    if csv_path.is_file():
        with csv_path.open(newline="") as handle:
            meta = {row["image_path"]: row for row in csv.DictReader(handle)}
    for path in _iter_files(root):
        rel = path.relative_to(root)
        if _is_image(path) and len(rel.parts) == 4:
            condition, species, specimen = rel.parts[:3]
            row = meta.get(rel.as_posix())
            if row is not None and (
                row["health_condition"] != condition
                or row["fish_category"] != species
                or row["specimen_id"] != specimen
            ):
                raise ValueError(f"metadata.csv disagrees with the folder path for {rel}")
            yield path, "", condition, species, specimen
        else:
            yield path


LAYOUTS: dict[str, Callable[[Path], Iterator[WalkItem | Path]]] = {
    "class_folders": walk_class_folders,
    "split_class_folders": walk_split_class_folders,
    "train_split_flat_test": walk_train_split_flat_test,
    "matsyadx": walk_matsyadx,
}


# --- discovery -----------------------------------------------------------------


def raw_root(data_dir: Path) -> Path:
    return Path(data_dir) / RAW_ROOT_NAME


def resolve_source_root(source: DatasetSource, raw_dir: Path) -> Path:
    """`data/raw/<key>`, which must exist (symlink or directory)."""
    path = Path(raw_dir) / source.key
    if not path.is_dir():
        raise FileNotFoundError(
            f"raw dataset {source.key!r} not found at {path}; "
            "run scripts/link_raw_datasets.py first"
        )
    return path


def discover_source(
    source: DatasetSource, raw_dir: Path, repo_root: Path
) -> tuple[list[ImageRecord], list[NonImageFile]]:
    """Every file under data/raw/<key>: images become ImageRecords (unprobed),
    everything else is classified and returned so nothing is silently skipped."""
    root = resolve_source_root(source, raw_dir)
    walker = LAYOUTS[source.layout]
    records: list[ImageRecord] = []
    others: list[NonImageFile] = []
    for item in walker(root):
        if isinstance(item, Path):
            rel = item.relative_to(root).as_posix()
            kind = classify_non_image(item, source.documentation)
            others.append(NonImageFile(source.key, rel, kind, item.stat().st_size))
            continue
        path, split, original_class, species, specimen = item
        rel = path.relative_to(root).as_posix()
        filepath = (Path("data") / RAW_ROOT_NAME / source.key / rel).as_posix()
        records.append(
            ImageRecord(
                image_id=make_image_id(source.key, filepath),
                source_dataset=source.key,
                filepath=filepath,
                original_path=rel,
                original_filename=path.name,
                original_class=original_class,
                original_split=split,
                species=species,
                specimen_id=specimen,
                extension=path.suffix.lower(),
                file_size=path.stat().st_size,
            )
        )
    records.sort(key=lambda r: r.filepath)
    return records, others


def discover_all(
    raw_dir: Path,
    repo_root: Path,
    sources: Iterable[DatasetSource] = DATASET_SOURCES,
) -> tuple[list[ImageRecord], list[NonImageFile]]:
    records: list[ImageRecord] = []
    others: list[NonImageFile] = []
    for source in sources:
        found, non_images = discover_source(source, raw_dir, repo_root)
        records.extend(found)
        others.extend(non_images)
    ids = Counter(r.image_id for r in records)
    clashes = [i for i, n in ids.items() if n > 1]
    if clashes:
        raise ValueError(f"image_id collision: {clashes[:5]}")
    return records, others


# --- probing (dimensions + hashes) -------------------------------------------------


def probe(record: ImageRecord, repo_root: Path) -> ImageRecord:
    """Fill dimensions, mode, SHA-256 and dHash in place; a file that cannot be
    decoded is kept with status 'corrupt' (its SHA-256 is still recorded)."""
    path = Path(repo_root) / record.filepath
    record.sha256 = file_sha256(path)
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record.width, record.height, record.mode = image.width, image.height, image.mode
            record.dhash = perceptual_dhash(image.convert("RGB"))
        record.status = "ok"
    except (UnidentifiedImageError, OSError, ValueError):
        record.width = record.height = 0
        record.mode = ""
        record.dhash = ""
        record.status = "corrupt"
    return record


def reuse_probe(record: ImageRecord, previous: dict[str, ImageRecord]) -> bool:
    """Copy probe results from an earlier manifest row for the same path and
    byte size (restartability); returns whether anything was reused."""
    old = previous.get(record.filepath)
    if old is None or old.file_size != record.file_size or old.status == "pending":
        return False
    record.width, record.height, record.mode = old.width, old.height, old.mode
    record.sha256, record.dhash, record.status = old.sha256, old.dhash, old.status
    return True


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit distance between two equal-length hex dHashes."""
    if len(hash_a) != len(hash_b):
        raise ValueError("hashes must have equal length")
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


# --- manifest I/O ---------------------------------------------------------------------


def write_master_manifest(records: Iterable[ImageRecord], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MASTER_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_master_manifest(path: Path) -> list[ImageRecord]:
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MASTER_COLUMNS:
            raise ValueError(f"{path.name} columns {reader.fieldnames} != {list(MASTER_COLUMNS)}")
        rows = []
        for row in reader:
            typed: dict[str, Any] = dict(row)
            for key in ("width", "height", "file_size"):
                typed[key] = int(row[key] or 0)
            rows.append(ImageRecord(**typed))
        return rows


# --- inventory ---------------------------------------------------------------------


INVENTORY_COLUMNS = (
    "dataset",
    "dataset_name",
    "local_path",
    "split_or_structure",
    "original_class",
    "image_count",
    "extensions",
    "species",
    "specimens",
    "dimensions",
    "corrupt",
    "metadata_available",
    "notes",
)


def inventory_rows(
    records: list[ImageRecord],
    raw_dir: Path,
    sources: Iterable[DatasetSource] = DATASET_SOURCES,
) -> list[dict[str, object]]:
    """One row per (dataset, delivered split, original class), counted from `records`."""
    rows: list[dict[str, object]] = []
    for source in sources:
        mine = [r for r in records if r.source_dataset == source.key]
        local = Path(raw_dir) / source.key
        target = local.resolve() if local.exists() else local
        groups: dict[tuple[str, str], list[ImageRecord]] = {}
        for r in mine:
            groups.setdefault((r.original_split, r.original_class), []).append(r)
        for (split, original_class), items in sorted(groups.items()):
            dims = Counter(f"{r.width}x{r.height}" for r in items if r.status == "ok")
            species = sorted({r.species for r in items if r.species})
            specimens = sorted({r.specimen_id for r in items if r.specimen_id})
            rows.append(
                {
                    "dataset": source.key,
                    "dataset_name": source.name,
                    "local_path": str(target),
                    "split_or_structure": split or source.layout,
                    "original_class": original_class,
                    "image_count": len(items),
                    "extensions": " ".join(
                        f"{e}:{n}" for e, n in sorted(Counter(r.extension for r in items).items())
                    ),
                    "species": "; ".join(species),
                    "specimens": len(specimens) if specimens else "",
                    "dimensions": " ".join(f"{d}:{n}" for d, n in dims.most_common(4)),
                    "corrupt": sum(1 for r in items if r.status == "corrupt"),
                    "metadata_available": "; ".join(source.documentation) or "none",
                    "notes": source.notes,
                }
            )
    return rows


def write_inventory_csv(rows: list[dict[str, object]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INVENTORY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
