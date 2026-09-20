"""Making the verified clean dataset available to a Google Colab runtime (Layer 2).

The clean manifest (`data/audit/clean_manifest.csv`) is the only description of
the dataset; the raw delivery stays on the local machine and is never copied
into git. Colab receives either

  * a **clean bundle** — exactly the manifest's *included* images, stored at their
    manifest `filepath` (`data/raw/<key>/…`) under one root, plus
    `bundle_manifest.csv` and `bundle.sha256` that pin the bundle to the digests
    of the manifest, split and fold files it was built from; or
  * the **full delivery** drop, attached with `scripts/link_raw_datasets.py`.

The root is configured with the environment variable
`AQUAHEALTH_COLAB_DATASET_ROOT` (or passed explicitly); nothing here knows a
machine-specific path. When the root is unset or absent the caller gets
`DatasetUnavailable` with a message that says what to do.

Attaching a root only creates `data/raw/<key>` symlinks in the repository, so
every committed manifest (`filepath` relative to the repository root) works
unchanged and the runner's leakage guards see the same rows as locally.
`verify_dataset` proves, inside the runtime, that what is on disk is the
verified dataset: same manifest/split/fold digests, every included image
present, counts, labels, group ids and fold ids equal to the manifests, and
image SHA-256 equal to the manifest (all images or a seeded sample).
"""

from __future__ import annotations

import csv
import os
import random
import shutil
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from src.config import SEED
from src.dataset_cleaning import CLEAN_MANIFEST_NAME, read_clean_manifest
from src.manifest import CANONICAL_CLASSES, file_sha256
from src.split_v2 import (
    CV_DIR_NAME,
    DEVELOPMENT,
    FINAL_TEST,
    SPLIT_DIR_NAME,
    read_folds,
    read_split,
    verify_split_digest,
)

DATASET_ROOT_ENV = "AQUAHEALTH_COLAB_DATASET_ROOT"
BUNDLE_MANIFEST_NAME = "bundle_manifest.csv"
BUNDLE_DIGEST_NAME = "bundle.sha256"
RAW_DIR_PARTS = ("data", "raw")
# Names that identify the full delivery drop (see scripts/link_raw_datasets.py).
DELIVERY_MARKERS = ("Fresh_water_disease", "train_split", "Fresh Water Fish Dataset", "MatsyaDx-BD")


class DatasetUnavailable(RuntimeError):
    """The configured Colab dataset root is missing or unusable."""


class DatasetMismatch(RuntimeError):
    """What is on disk is not the verified dataset described by the manifests."""


# --- root resolution -------------------------------------------------------------------------


def resolve_dataset_root(explicit: Path | str | None = None) -> Path:
    """Explicit argument, else $AQUAHEALTH_COLAB_DATASET_ROOT. Raises DatasetUnavailable
    with an actionable message when unset or not a directory."""
    value = str(explicit) if explicit is not None else os.environ.get(DATASET_ROOT_ENV, "")
    if not value.strip():
        raise DatasetUnavailable(
            f"{DATASET_ROOT_ENV} is not set. Point it at the clean bundle "
            f"(built by scripts/build_colab_bundle.py) or at the delivered Dataset/ folder, "
            f"e.g. os.environ['{DATASET_ROOT_ENV}'] = '/content/aquahealth_data/bundle'."
        )
    root = Path(value).expanduser()
    if not root.is_dir():
        raise DatasetUnavailable(
            f"{DATASET_ROOT_ENV}={value!r} is not a directory on this runtime. "
            "Copy/extract the bundle (or mount the drive that holds it) first."
        )
    return root.resolve()


def detect_layout(root: Path) -> str:
    """'bundle' (has bundle_manifest.csv and data/raw/), 'delivery' (the raw drop), or raise."""
    root = Path(root)
    if (root / BUNDLE_MANIFEST_NAME).is_file() and root.joinpath(*RAW_DIR_PARTS).is_dir():
        return "bundle"
    candidates = [root, root / "Dataset"]
    for base in candidates:
        if base.is_dir() and any((base / m).exists() for m in DELIVERY_MARKERS):
            return "delivery"
    raise DatasetUnavailable(
        f"{root} is neither a clean bundle ({BUNDLE_MANIFEST_NAME} + data/raw/) nor the "
        f"delivered Dataset/ drop (expected one of {DELIVERY_MARKERS})."
    )


# --- bundle -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class BundleRow:
    image_id: str
    filepath: str  # relative to the repository root AND to the bundle root
    sha256: str
    source_dataset: str
    unified_class: str
    label: int
    group_id: str
    split: str  # development | final_test
    fold: str  # 1..K for development rows, "" for final_test rows


BUNDLE_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(BundleRow))


def _audit_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "audit"


def _pinned_files(repo_root: Path) -> dict[str, Path]:
    audit = _audit_dir(repo_root)
    return {
        f"data/audit/{CLEAN_MANIFEST_NAME}": audit / CLEAN_MANIFEST_NAME,
        f"data/audit/{SPLIT_DIR_NAME}/{DEVELOPMENT}.csv": audit
        / SPLIT_DIR_NAME
        / f"{DEVELOPMENT}.csv",
        f"data/audit/{SPLIT_DIR_NAME}/{FINAL_TEST}.csv": audit
        / SPLIT_DIR_NAME
        / f"{FINAL_TEST}.csv",
        f"data/audit/{CV_DIR_NAME}/folds.csv": audit / CV_DIR_NAME / "folds.csv",
    }


def bundle_rows(repo_root: Path) -> list[BundleRow]:
    """The included clean rows joined with their split and fold — the bundle's content list."""
    repo_root = Path(repo_root)
    audit = _audit_dir(repo_root)
    clean = read_clean_manifest(audit / CLEAN_MANIFEST_NAME)
    verify_split_digest(audit / SPLIT_DIR_NAME)
    split = {s.image_id: s for s in read_split(audit / SPLIT_DIR_NAME)}
    fold_of = {f.image_id: f.fold for f in read_folds(audit / CV_DIR_NAME)}
    included = [r for r in clean if r.included]
    if {r.image_id for r in included} != set(split):
        raise DatasetMismatch("split_v2 rows are not exactly the included clean rows")
    rows = []
    for r in included:
        s = split[r.image_id]
        if (s.group_id, s.unified_class, s.filepath) != (r.group_id, r.unified_class, r.filepath):
            raise DatasetMismatch(f"{r.image_id}: split row disagrees with the clean manifest")
        if s.split == DEVELOPMENT and r.image_id not in fold_of:
            raise DatasetMismatch(f"{r.image_id}: development image without a fold")
        if s.split == FINAL_TEST and r.image_id in fold_of:
            raise DatasetMismatch(f"{r.image_id}: final-test image inside the folds")
        rows.append(
            BundleRow(
                image_id=r.image_id,
                filepath=r.filepath,
                sha256=r.sha256,
                source_dataset=r.source_dataset,
                unified_class=r.unified_class,
                label=CANONICAL_CLASSES.index(r.unified_class),
                group_id=r.group_id,
                split=s.split,
                fold=str(fold_of[r.image_id]) if r.image_id in fold_of else "",
            )
        )
    rows.sort(key=lambda b: b.filepath)
    return rows


def build_bundle(
    repo_root: Path, out_dir: Path, *, hardlink: bool = False, verify_hashes: bool = True
) -> dict[str, Any]:
    """Copy exactly the manifest's included images into `out_dir/<filepath>`; write
    bundle_manifest.csv and bundle.sha256. Raw files are read, never modified."""
    repo_root, out_dir = Path(repo_root), Path(out_dir)
    rows = bundle_rows(repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = reused = 0
    for b in rows:
        src = repo_root / b.filepath
        if not src.is_file():
            raise DatasetUnavailable(f"source image missing locally: {src}")
        if verify_hashes and file_sha256(src) != b.sha256:
            raise DatasetMismatch(f"{b.filepath}: local file differs from the manifest SHA-256")
        dst = out_dir / b.filepath
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            reused += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if hardlink:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        copied += 1
    with (out_dir / BUNDLE_MANIFEST_NAME).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BUNDLE_COLUMNS))
        writer.writeheader()
        writer.writerows(asdict(b) for b in rows)
    digest_lines = [
        f"{file_sha256(path)}  {name}" for name, path in _pinned_files(repo_root).items()
    ]
    digest_lines.append(f"{file_sha256(out_dir / BUNDLE_MANIFEST_NAME)}  {BUNDLE_MANIFEST_NAME}")
    (out_dir / BUNDLE_DIGEST_NAME).write_text("\n".join(digest_lines) + "\n")
    return {
        "bundle_root": str(out_dir),
        "images": len(rows),
        "copied": copied,
        "reused": reused,
        "bytes": sum((out_dir / b.filepath).stat().st_size for b in rows),
        "per_split": dict(Counter(b.split for b in rows)),
        "per_source": dict(Counter(b.source_dataset for b in rows)),
        "digests": digest_lines,
    }


def read_bundle_manifest(root: Path) -> list[BundleRow]:
    path = Path(root) / BUNDLE_MANIFEST_NAME
    if not path.is_file():
        raise DatasetUnavailable(f"{path} not found — is {DATASET_ROOT_ENV} the bundle root?")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BUNDLE_COLUMNS:
            raise DatasetMismatch(f"{path} columns {reader.fieldnames} != {list(BUNDLE_COLUMNS)}")
        rows = []
        for raw in reader:
            typed: dict[str, Any] = dict(raw)
            typed["label"] = int(raw["label"])
            rows.append(BundleRow(**typed))
    return rows


def read_bundle_digests(root: Path) -> dict[str, str]:
    path = Path(root) / BUNDLE_DIGEST_NAME
    if not path.is_file():
        raise DatasetUnavailable(f"{path} not found — the bundle is incomplete")
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


# --- attach -----------------------------------------------------------------------------------


def attach(root: Path, repo_root: Path, *, force: bool = False) -> list[str]:
    """Make `repo_root/data/raw/<key>` point at `root/data/raw/<key>` for every key in the
    bundle (bundle layout only; the delivery layout uses scripts/link_raw_datasets.py).
    Returns one line per link. Existing links are replaced only with force=True."""
    root, repo_root = Path(root), Path(repo_root)
    if detect_layout(root) != "bundle":
        raise DatasetUnavailable(
            f"{root} is the delivery layout; attach it with scripts/link_raw_datasets.py --source"
        )
    src_raw = root.joinpath(*RAW_DIR_PARTS)
    dst_raw = repo_root.joinpath(*RAW_DIR_PARTS)
    dst_raw.mkdir(parents=True, exist_ok=True)
    lines = []
    for key_dir in sorted(p for p in src_raw.iterdir() if p.is_dir()):
        link = dst_raw / key_dir.name
        target = key_dir.resolve()
        if link.is_symlink():
            if link.resolve() == target:
                lines.append(f"{link} -> {target} (unchanged)")
                continue
            if not force:
                raise DatasetUnavailable(
                    f"{link} already points at {link.resolve()}; pass force=True to replace"
                )
            link.unlink()
        elif link.exists():
            # scripts/link_raw_datasets.py builds current_freshwater as a real directory that
            # holds only symlinks (train_split, test_split, test.csv). With force=True such a
            # link-only directory is replaced; a directory holding real files never is.
            entries = list(link.iterdir()) if link.is_dir() else []
            if force and link.is_dir() and entries and all(e.is_symlink() for e in entries):
                for e in entries:
                    e.unlink()
                link.rmdir()
            else:
                raise DatasetUnavailable(
                    f"{link} exists and is not a symlink; refusing to replace it"
                )
        link.symlink_to(target, target_is_directory=True)
        lines.append(f"{link} -> {target}")
    return lines


# --- verification -----------------------------------------------------------------------------


def _check(report: dict[str, Any], name: str, ok: bool, detail: Any = "") -> None:
    report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        report["errors"].append(f"{name}: {detail}")


def verify_dataset(
    repo_root: Path,
    root: Path | None = None,
    *,
    hash_mode: str = "all",
    sample_size: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:
    """Prove that the images reachable through `repo_root/data/raw` are the verified clean
    dataset. `hash_mode`: 'all' (every included image), 'sample' (seeded sample), 'none'.
    Never regenerates or modifies anything; returns a report with `ok` and `errors`."""
    if hash_mode not in ("all", "sample", "none"):
        raise ValueError("hash_mode must be all|sample|none")
    repo_root = Path(repo_root)
    report: dict[str, Any] = {"ok": False, "checks": [], "errors": [], "layout": None}
    audit = _audit_dir(repo_root)

    # 1. manifests exist and their digests still verify
    manifest = audit / CLEAN_MANIFEST_NAME
    _check(report, "clean manifest exists", manifest.is_file(), str(manifest))
    if not manifest.is_file():
        return report
    try:
        verify_split_digest(audit / SPLIT_DIR_NAME)
        _check(report, "split digests verify", True, "development.csv, final_test.csv")
    except (OSError, ValueError) as exc:
        _check(report, "split digests verify", False, str(exc))
    try:
        folds = read_folds(audit / CV_DIR_NAME)
        _check(report, "fold digests verify", True, f"{len(folds)} rows")
    except (OSError, ValueError) as exc:
        _check(report, "fold digests verify", False, str(exc))
        folds = []
    if report["errors"]:
        return report
    rows = bundle_rows(repo_root)  # also cross-checks manifest <-> split <-> folds
    _check(report, "manifest/split/fold rows agree", True, f"{len(rows)} included images")
    report["manifest_sha256"] = file_sha256(manifest)

    # 2. bundle identity (bundle layout only)
    if root is not None:
        root = Path(root)
        layout = detect_layout(root)
        report["layout"] = layout
        if layout == "bundle":
            recorded = read_bundle_digests(root)
            for name, path in _pinned_files(repo_root).items():
                actual = file_sha256(path)
                _check(
                    report,
                    f"bundle pinned to {name}",
                    recorded.get(name) == actual,
                    f"recorded {str(recorded.get(name))[:12]} actual {actual[:12]}",
                )
            bm_actual = file_sha256(root / BUNDLE_MANIFEST_NAME)
            _check(
                report,
                "bundle_manifest.csv digest",
                recorded.get(BUNDLE_MANIFEST_NAME) == bm_actual,
                bm_actual[:12],
            )
            bundle = read_bundle_manifest(root)
            expected = {b.image_id: b for b in rows}
            got = {b.image_id: b for b in bundle}
            _check(
                report,
                "bundle rows == included manifest rows",
                set(got) == set(expected),
                f"bundle {len(got)} manifest {len(expected)}",
            )
            diff: Counter[str] = Counter()
            for image_id, b in got.items():
                e = expected.get(image_id)
                if e is None:
                    continue
                for key in (
                    "filepath",
                    "sha256",
                    "unified_class",
                    "label",
                    "group_id",
                    "split",
                    "fold",
                    "source_dataset",
                ):
                    if getattr(b, key) != getattr(e, key):
                        diff[key] += 1
            _check(report, "bundle fields match manifests", not diff, dict(diff) or "0 mismatches")

    # 3. files: existence, count, checksums
    missing = [b.filepath for b in rows if not (repo_root / b.filepath).is_file()]
    _check(
        report, "every included image exists", not missing, f"missing {len(missing)}: {missing[:3]}"
    )
    present = len(rows) - len(missing)
    _check(report, "image count matches manifest", present == len(rows), f"{present}/{len(rows)}")
    if hash_mode != "none" and not missing:
        to_hash = rows
        if hash_mode == "sample":
            rng = random.Random(seed)
            to_hash = rng.sample(rows, min(sample_size, len(rows)))
        bad = [b.filepath for b in to_hash if file_sha256(repo_root / b.filepath) != b.sha256]
        _check(
            report,
            f"image SHA-256 match manifest ({hash_mode}, {len(to_hash)} files)",
            not bad,
            f"mismatches {len(bad)}: {bad[:3]}",
        )

    # 4. labels, groups, folds
    _check(
        report,
        "labels canonical",
        all(CANONICAL_CLASSES[b.label] == b.unified_class for b in rows),
        f"{len(CANONICAL_CLASSES)} classes",
    )
    fold_of = {f.image_id: f.fold for f in folds}
    group_fold: dict[str, set[int]] = {}
    for f in folds:
        group_fold.setdefault(f.group_id, set()).add(f.fold)
    _check(
        report,
        "fold ids: development rows all folded, test rows none",
        all((b.fold != "") == (b.split == DEVELOPMENT) for b in rows)
        and all(b.fold == str(fold_of.get(b.image_id, "")) for b in rows if b.fold),
        dict(Counter(b.fold or "test" for b in rows)),
    )
    _check(
        report,
        "no group in two folds",
        all(len(s) == 1 for s in group_fold.values()),
        f"{len(group_fold)} groups",
    )
    dev_groups = {b.group_id for b in rows if b.split == DEVELOPMENT}
    test_groups = {b.group_id for b in rows if b.split == FINAL_TEST}
    _check(
        report,
        "no group in both development and final_test",
        not (dev_groups & test_groups),
        len(dev_groups & test_groups),
    )
    report["counts"] = {
        "images": len(rows),
        "per_split": dict(Counter(b.split for b in rows)),
        "per_class": dict(sorted(Counter(b.unified_class for b in rows).items())),
        "per_source": dict(sorted(Counter(b.source_dataset for b in rows).items())),
    }
    report["ok"] = not report["errors"]
    return report


def summarize(report: dict[str, Any]) -> str:
    lines = [
        f"[{'ok' if c['ok'] else 'FAIL'}] {c['check']} — {c['detail']}" for c in report["checks"]
    ]
    lines.append("RESULT: " + ("VERIFIED" if report["ok"] else "NOT VERIFIED"))
    return "\n".join(lines)


def iter_keys(rows: Iterable[BundleRow]) -> list[str]:
    """Dataset keys (data/raw/<key>) present in the rows, for attach/link reporting."""
    return sorted({Path(b.filepath).parts[2] for b in rows if len(Path(b.filepath).parts) > 2})
