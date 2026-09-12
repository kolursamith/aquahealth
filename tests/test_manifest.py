"""Frozen split manifest: schema, digest guard, group-aware split, manifest dataset,
and the audit/split script on a synthetic copy of the real dataset's layout."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.manifest import (
    CANONICAL_CLASSES,
    SOURCE_FOLDER_TO_CLASS,
    SPLITS,
    ManifestDataset,
    ManifestRow,
    group_aware_stratified_split,
    largest_remainder_quota,
    perceptual_dhash,
    read_manifest,
    validate_manifest,
    verify_manifest_digest,
    write_manifest,
)
from src.preprocessing import PreprocessConfig, build_eval_transform
from tests.conftest import class_colour

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_split_manifest.py"


# --- mapping ---


def test_canonical_mapping_is_the_agreed_eight_classes():
    assert len(CANONICAL_CLASSES) == 8
    assert (
        CANONICAL_CLASSES[0] == "Bacterial Red Disease" and CANONICAL_CLASSES[7] == "Healthy Fish"
    )
    assert set(SOURCE_FOLDER_TO_CLASS.values()) == set(CANONICAL_CLASSES)
    assert len(SOURCE_FOLDER_TO_CLASS) == 8


# --- quota and split ---


@pytest.mark.parametrize(
    "n,expected", [(100, [70, 15, 15]), (2400, [1680, 360, 360]), (7, [5, 1, 1]), (3, [2, 1, 0])]
)
def test_largest_remainder_quota(n, expected):
    assert largest_remainder_quota(n, (0.7, 0.15, 0.15)) == expected


def _items(per_class: dict[int, int], group_size: int = 1):
    items = []
    for label, count in per_class.items():
        for i in range(count):
            items.append((f"c{label}/img{i:04d}.jpg", label, f"c{label}-g{i // group_size}"))
    return items


def test_split_is_stratified_disjoint_and_seeded():
    items = _items({0: 100, 1: 40, 2: 13}, group_size=1)
    rows = group_aware_stratified_split(items, seed=42)
    counts = validate_manifest(rows, class_names=CANONICAL_CLASSES[:3])
    assert sum(sum(c.values()) for c in counts.values()) == 153
    assert [counts[s]["Bacterial Red Disease"] for s in SPLITS] == [70, 15, 15]
    assert [counts[s]["Aeromoniasis"] for s in SPLITS] == [28, 6, 6]
    assert [counts[s]["Bacterial Gill Disease"] for s in SPLITS] == [9, 2, 2]
    again = group_aware_stratified_split(items, seed=42)
    other = group_aware_stratified_split(items, seed=7)
    assert rows == again
    assert rows != other
    assert {r.split for r in other} == set(SPLITS)


def test_split_keeps_every_group_inside_one_split():
    items = _items({0: 60, 1: 30}, group_size=3)
    rows = group_aware_stratified_split(items, seed=1)
    split_of = {r.filepath: r.split for r in rows}
    for _, _, group in items:
        members = [p for p, _, g in items if g == group]
        assert len({split_of[p] for p in members}) == 1, group


def test_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        group_aware_stratified_split(_items({0: 10}), ratios=(0.5, 0.5, 0.5))


# --- validation and digest ---


def test_validate_manifest_catches_duplicates_bad_labels_and_missing_classes():
    good = ManifestRow("a.jpg", 0, "Bacterial Red Disease", "train")
    with pytest.raises(ValueError, match="appears in both"):
        validate_manifest([good, ManifestRow("a.jpg", 0, "Bacterial Red Disease", "test")])
    with pytest.raises(ValueError, match="label 0 is"):
        validate_manifest([ManifestRow("a.jpg", 0, "Healthy Fish", "train")])
    with pytest.raises(ValueError, match="out of range"):
        validate_manifest([ManifestRow("a.jpg", 9, "Healthy Fish", "train")])
    with pytest.raises(ValueError, match="unknown split"):
        validate_manifest([ManifestRow("a.jpg", 0, "Bacterial Red Disease", "holdout")])
    with pytest.raises(ValueError, match="has no images"):
        validate_manifest([good])


def test_write_read_and_digest_guard(tmp_path):
    rows = group_aware_stratified_split(_items({i: 10 for i in range(8)}), seed=3)
    path = tmp_path / "split_manifest.csv"
    digest = write_manifest(rows, path)
    assert read_manifest(path) == rows
    assert verify_manifest_digest(path) == digest
    with path.open() as handle:
        assert handle.readline().strip() == "filepath,label,class_name,split"
    path.write_text(path.read_text().replace("train", "val", 1))
    with pytest.raises(ValueError, match="has been modified"):
        verify_manifest_digest(path)


def test_read_manifest_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("path,label\nx,0\n")
    with pytest.raises(ValueError, match="manifest columns"):
        read_manifest(path)


# --- perceptual hash ---


def test_dhash_groups_reencoded_copies_but_not_different_images(tmp_path):
    """Photo-like content (smooth gradients) survives JPEG re-encoding and resizing with an
    identical dHash; unrelated content does not. Pure noise is the known worst case and is
    deliberately not the fixture: real photos are smooth at 9x8."""
    yy, xx = np.mgrid[0:120, 0:160]
    pixels = np.stack([xx * 255 // 160, yy * 255 // 120, (xx + yy) * 255 // 280], axis=-1)
    original = Image.fromarray(pixels.astype(np.uint8))
    buffer_jpeg = tmp_path / "copy.jpg"
    original.save(buffer_jpeg, quality=80)
    resized = original.resize((80, 60))
    assert perceptual_dhash(original) == perceptual_dhash(Image.open(buffer_jpeg))
    assert perceptual_dhash(original) == perceptual_dhash(resized)
    other = Image.fromarray(pixels[::-1, ::-1].astype(np.uint8))
    assert perceptual_dhash(original) != perceptual_dhash(other)


# --- synthetic replica of the delivered layout ---


def _write_delivered_layout(root: Path, per_class: int = 12) -> Path:
    """train_split/<source folder>/*.jpg + flat test_split/ + test.csv, like the real data."""
    rng = np.random.default_rng(0)
    (root / "train_split").mkdir(parents=True)
    (root / "test_split").mkdir()
    test_rows = []
    for index, folder in enumerate(SOURCE_FOLDER_TO_CLASS):
        base = np.array(class_colour(index), np.int16)
        d = root / "train_split" / folder
        d.mkdir()
        for i in range(per_class):
            noise = rng.integers(-5, 6, (40, 48, 3), np.int16)
            img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
            img.save(d / f"{i:03d}.jpg", quality=95)
        for i in range(3):
            noise = rng.integers(-5, 6, (40, 48, 3), np.int16)
            name = f"{folder}_{folder}_{i}.jpg"
            Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).save(
                root / "test_split" / name, quality=95
            )
            test_rows.append((name, folder))
    shutil.copy2(root / "train_split" / "EUS" / "000.jpg", root / "train_split" / "EUS" / "dup.jpg")
    with (root / "test.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "label"])
        writer.writerows(test_rows)
    return root


@pytest.fixture
def replica(tmp_path, monkeypatch) -> Path:
    """A replica under a fake repo root so manifest paths are `data/original/...`."""
    fake_root = tmp_path / "repo"
    dataset = _write_delivered_layout(fake_root / "data" / "original" / "aquahealth")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    return dataset


def _run_script(dataset: Path, *extra: str) -> subprocess.CompletedProcess:
    repo_root = dataset.parents[2]
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(dataset),
            "--manifest",
            str(repo_root / "data" / "split_manifest.csv"),
            "--results-dir",
            str(repo_root / "results"),
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_script_audits_and_freezes_a_split(replica):
    repo_root = replica.parents[2]
    result = _run_script(replica)
    assert result.returncode == 0, result.stderr
    manifest = repo_root / "data" / "split_manifest.csv"
    assert manifest.exists() and manifest.with_suffix(".sha256").exists()
    rows = read_manifest(manifest)
    counts = validate_manifest(rows)
    # 8 classes x (12 train + 3 test) + 1 exact duplicate = 121 files -> 120 unique images
    assert len(rows) == 120
    assert all(r.filepath.startswith("data/original/aquahealth/") for r in rows)
    assert counts["test"]["Healthy Fish"] == 2 and counts["train"]["Healthy Fish"] == 11
    report = (repo_root / "results" / "data_audit_report.md").read_text()
    assert "exact-duplicate groups (SHA-256): 1 (2 images involved)" in report
    assert "corrupt/unreadable: 0" in report
    assert "| EUS | EUS Disease | 3 |" in report
    with (repo_root / "results" / "data_audit.csv").open() as handle:
        audit = list(csv.DictReader(handle))
    assert len(audit) == 121
    assert Counter(r["source_split"] for r in audit) == {"train_split": 97, "test_split": 24}
    assert sum(1 for r in audit if r["exact_dup_group"]) == 2
    assert all(r["status"] == "ok" for r in audit)

    second = _run_script(replica)
    assert second.returncode == 2 and "refusing to overwrite" in second.stdout
    assert verify_manifest_digest(manifest)


def test_script_flags_corrupt_and_unexpected_files(replica):
    (replica / "train_split" / "EUS" / "broken.jpg").write_bytes(b"nope")
    (replica / "train_split" / "notes.txt").write_text("x")
    (replica / "extra_dir").mkdir()
    result = _run_script(replica, "--dry-run")
    assert result.returncode == 0, result.stderr
    report = (replica.parents[2] / "results" / "data_audit_report.md").read_text()
    assert "corrupt/unreadable: 1" in report
    assert "unexpected directories: ['extra_dir']" in report
    assert "notes.txt" in report
    assert not (replica.parents[2] / "data" / "split_manifest.csv").exists()


# --- manifest dataset ---


def test_manifest_dataset_labels_by_canonical_index_not_folder_order(replica):
    repo_root = replica.parents[2]
    assert _run_script(replica).returncode == 0
    rows = read_manifest(repo_root / "data" / "split_manifest.csv")
    small = PreprocessConfig(image_size=32, resize_size=36)
    train = ManifestDataset(rows, "train", transform=build_eval_transform(small), root=repo_root)
    assert train.class_names == list(CANONICAL_CLASSES)
    assert train.missing_files() == []
    assert set(train.class_counts()) == set(CANONICAL_CLASSES)
    tensor, label = train[0]
    assert tensor.shape == (3, 32, 32) and tensor.dtype == torch.float32
    healthy = [s for s in train.samples if "Healthy Fish" in str(s.path)]
    assert healthy and all(s.label == 7 for s in healthy), "Healthy Fish must be label 7"
    eus = [s for s in train.samples if "/EUS/" in str(s.path)]
    assert eus and all(s.label == 3 for s in eus)
    test = ManifestDataset(rows, "test", root=repo_root)
    assert not {s.path for s in test.samples} & {s.path for s in train.samples}
    with pytest.raises(ValueError, match="unknown split"):
        ManifestDataset(rows, "holdout", root=repo_root)
