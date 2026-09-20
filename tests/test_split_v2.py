"""Phase 8 — development / frozen final-test partition: assignment rule, invariants,
I/O + digests, the script, and the real committed split."""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.dataset_cleaning import (
    CLEAN_MANIFEST_NAME,
    build_clean_manifest,
    read_clean_manifest,
    write_clean_manifest,
)
from src.manifest import MANIFEST_PATH, verify_manifest_digest
from src.split_v2 import (
    DEVELOPMENT,
    FINAL_TEST,
    SPLIT_COLUMNS,
    SPLIT_DIR_NAME,
    assign_groups,
    build_dev_test_split,
    read_development,
    read_split,
    strata,
    validate_split,
    verify_split_digest,
    write_split,
)
from tests.test_dataset_cleaning import build_master_records, rec

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "audit" / SPLIT_DIR_NAME
AUDIT = ROOT / "data" / "audit"


def _many_records(n_per_class: int = 40) -> list:
    """Synthetic master records: 3 classes x 2 sources, groups of 1-3 images."""
    records = []
    classes = ["EUS Disease", "Healthy Fish", "Aeromoniasis"]
    originals = ["EUS", "Healthy Fish", "Bacterial diseases - Aeromoniasis"]
    i = 0
    for cls, orig in zip(classes, originals):
        for k in range(n_per_class):
            source = "current_freshwater" if k % 3 else "roboflow"
            group = f"g{cls[:3]}{k // 2}" if k % 5 else ""  # pairs share a near-dup group
            records.append(
                rec(
                    f"{source[:3]}-{i:04d}",
                    source,
                    f"{'train_split' if source == 'current_freshwater' else 'train'}"
                    f"/{orig}/{i}.jpg",
                    orig,
                    cls,
                    sha=f"s{i}",
                    dhash=f"h{k // 2 if k % 5 else i}{cls[:2]}",
                    near=group,
                )
            )
            i += 1
    return records


def test_assign_groups_keeps_groups_together_and_respects_ratios():
    items = [(f"i{k}", "A", f"g{k // 4}") for k in range(100)]  # 25 groups of 4
    parts = assign_groups(items, ratios=(0.8, 0.2), seed=1)
    by_group = defaultdict(set)
    for item, part in parts.items():
        by_group[item.split("i")[1]].add(part)
    for k in range(100):
        assert parts[f"i{k}"] == parts[f"i{k - k % 4}"]  # same group -> same part
    counts = Counter(parts.values())
    assert counts == {0: 80, 1: 20}
    assert assign_groups(items, ratios=(0.8, 0.2), seed=1) == parts  # reproducible
    assert assign_groups(items, ratios=(0.8, 0.2), seed=2) != parts
    with pytest.raises(ValueError):
        assign_groups(items, ratios=(0.5, 0.4), seed=1)
    with pytest.raises(ValueError, match="two strata"):
        assign_groups([("a", "A", "g"), ("b", "B", "g")], ratios=(0.5, 0.5))


def test_strata_use_majority_source_of_the_group():
    rows = build_clean_manifest(_many_records())
    s = strata(r for r in rows if r.included)
    assert all("|" in v for v in s.values())
    by_group = defaultdict(set)
    for r in rows:
        if r.included:
            by_group[r.group_id].add(s[r.image_id])
    assert all(len(v) == 1 for v in by_group.values())  # one stratum per group


def test_build_dev_test_split_invariants_and_exclusions():
    master = _many_records()
    clean = build_clean_manifest(master)
    split = build_dev_test_split(clean, test_ratio=0.25, seed=42)
    assert {s.image_id for s in split} == {r.image_id for r in clean if r.included}
    test = [s for s in split if s.split == FINAL_TEST]
    dev = [s for s in split if s.split == DEVELOPMENT]
    assert 0.2 <= len(test) / len(split) <= 0.3
    assert {s.unified_class for s in test} == {s.unified_class for s in dev}
    assert not ({s.group_id for s in test} & {s.group_id for s in dev})
    assert build_dev_test_split(clean, test_ratio=0.25, seed=42) == split  # reproducible
    # an excluded row never enters
    clean[0].included, clean[0].exclusion_reason = False, "LABEL_UNRESOLVED"
    split2 = build_dev_test_split(clean, test_ratio=0.25, seed=42)
    assert clean[0].image_id not in {s.image_id for s in split2}
    with pytest.raises(ValueError):
        build_dev_test_split(clean, test_ratio=1.5)


def test_validate_split_catches_violations():
    clean = build_clean_manifest(_many_records())
    split = build_dev_test_split(clean, test_ratio=0.25, seed=42)
    broken = [s for s in split]
    broken[0].split = FINAL_TEST if broken[0].split == DEVELOPMENT else DEVELOPMENT
    if any(s.group_id == broken[0].group_id and s is not broken[0] for s in broken):
        with pytest.raises(ValueError, match="straddle"):
            validate_split(broken)
    split = build_dev_test_split(clean, test_ratio=0.25, seed=42)
    split[0].label = (split[0].label + 1) % 8
    with pytest.raises(ValueError, match="label"):
        validate_split(split)


def test_write_read_and_digest_guard(tmp_path):
    clean = build_clean_manifest(_many_records())
    split = build_dev_test_split(clean, test_ratio=0.25, seed=42)
    write_split(split, tmp_path)
    with (tmp_path / "final_test.csv").open() as handle:
        assert tuple(next(csv.reader(handle))) == SPLIT_COLUMNS
    assert read_split(tmp_path) == split
    dev = read_development(tmp_path)
    assert {s.image_id for s in dev} == {s.image_id for s in split if s.split == DEVELOPMENT}
    verify_split_digest(tmp_path)
    (tmp_path / "final_test.csv").write_text((tmp_path / "final_test.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="modified"):
        verify_split_digest(tmp_path)
    with pytest.raises(ValueError, match="modified"):
        read_development(tmp_path)


def test_script_writes_reports_and_refuses_to_overwrite(tmp_path):
    master = build_master_records() + _many_records()
    audit = tmp_path / "data" / "audit"
    audit.mkdir(parents=True)
    rows = build_clean_manifest(master)
    for r in rows:
        if r.included:
            p = tmp_path / r.filepath
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")  # existence check only
    write_clean_manifest(rows, audit / CLEAN_MANIFEST_NAME)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_dev_test_split.py"),
        "--data-dir",
        str(tmp_path / "data"),
        "--repo-root",
        str(tmp_path),
        "--test-ratio",
        "0.25",
    ]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    out = audit / SPLIT_DIR_NAME
    for name in (
        "development.csv",
        "final_test.csv",
        "split_manifest.sha256",
        "split_report.json",
        "split_report.md",
    ):
        assert (out / name).is_file(), name
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 2 and "frozen" in second.stderr


# --- the real, committed split ---


@pytest.mark.skipif(not (SPLIT_DIR / "final_test.csv").is_file(), reason="split not built")
def test_real_split_is_valid_disjoint_and_frozen():
    verify_split_digest(SPLIT_DIR)
    clean = read_clean_manifest(AUDIT / CLEAN_MANIFEST_NAME)
    split = read_split(SPLIT_DIR)
    validate_split(split, clean)
    dev = {s.image_id for s in split if s.split == DEVELOPMENT}
    test = {s.image_id for s in split if s.split == FINAL_TEST}
    assert len(dev) == 3044 and len(test) == 761 and not (dev & test)  # dataset configuration v3
    # no known duplicate pair crosses the partitions (exact SHA-256 or identical dHash)
    part = {s.image_id: s.split for s in split}
    with (AUDIT / "duplicate_report.csv").open() as handle:
        crossing = [
            p
            for p in csv.DictReader(handle)
            if p["image_a"] in part
            and p["image_b"] in part
            and part[p["image_a"]] != part[p["image_b"]]
        ]
    assert crossing == []
    # specimen ids never straddle
    by_specimen = defaultdict(set)
    for s in split:
        if s.specimen_id:
            by_specimen[(s.source_dataset, s.specimen_id)].add(s.split)
    assert all(len(v) == 1 for v in by_specimen.values())
    # every selected image exists on disk and the labels follow the canonical mapping
    assert all((ROOT / s.filepath).is_file() for s in split)
    # reproducible from the clean manifest with the recorded seed / ratio
    assert build_dev_test_split(clean, test_ratio=0.20, seed=42) == split
    # the old baseline split is untouched
    assert verify_manifest_digest(MANIFEST_PATH).startswith("b7d1fccbb21e73a8")


# --- Phase 9: K-fold on the development partition ---


def test_assign_folds_every_image_validation_once_groups_intact():
    from src.split_v2 import assign_folds, fold_members, validate_folds

    clean = build_clean_manifest(_many_records(60))
    split = build_dev_test_split(clean, test_ratio=0.2, seed=42)
    development = [s for s in split if s.split == DEVELOPMENT]
    rows = assign_folds(development, n_folds=5, seed=42)
    assert {r.image_id for r in rows} == {s.image_id for s in development}
    assert Counter(r.fold for r in rows).keys() == {1, 2, 3, 4, 5}
    seen = Counter()
    for fold in range(1, 6):
        train, validation = fold_members(rows, fold)
        assert len(train) + len(validation) == len(rows)
        assert not ({r.image_id for r in train} & {r.image_id for r in validation})
        assert not ({r.group_id for r in train} & {r.group_id for r in validation})
        seen.update(r.image_id for r in validation)
    assert all(n == 1 for n in seen.values()) and len(seen) == len(rows)
    assert assign_folds(development, n_folds=5, seed=42) == rows  # reproducible
    with pytest.raises(ValueError, match="development rows only"):
        assign_folds([s for s in split if s.split == FINAL_TEST][:3], n_folds=2)
    bad = list(rows)
    bad[0].fold = 99
    with pytest.raises(ValueError):
        validate_folds(bad, 5)


def test_fold_files_round_trip_with_digests(tmp_path):
    from src.split_v2 import assign_folds, read_folds, write_folds

    clean = build_clean_manifest(_many_records(30))
    development = [s for s in build_dev_test_split(clean, test_ratio=0.2) if s.split == DEVELOPMENT]
    rows = assign_folds(development, n_folds=3, seed=42)
    write_folds(rows, tmp_path, 3)
    assert read_folds(tmp_path) == rows
    for fold in (1, 2, 3):
        assert (tmp_path / f"fold_{fold:02d}_train.csv").is_file()
        assert (tmp_path / f"fold_{fold:02d}_validation.csv").is_file()
    (tmp_path / "fold_02_train.csv").write_text("tampered")
    with pytest.raises(ValueError, match="modified"):
        read_folds(tmp_path)


CV_DIR = ROOT / "data" / "audit" / "cv_v3"


@pytest.mark.skipif(not (CV_DIR / "folds.csv").is_file(), reason="folds not built")
def test_real_folds_cover_development_exactly_and_never_touch_test():
    from src.split_v2 import assign_folds, fold_members, read_folds

    rows = read_folds(CV_DIR)
    split = read_split(SPLIT_DIR)
    dev = {s.image_id for s in split if s.split == DEVELOPMENT}
    test = {s.image_id for s in split if s.split == FINAL_TEST}
    assert {r.image_id for r in rows} == dev
    assert not ({r.image_id for r in rows} & test)
    assert sorted(Counter(r.fold for r in rows)) == list(range(1, 11))
    part = {r.image_id: r.fold for r in rows}
    with (AUDIT / "duplicate_report.csv").open() as handle:
        crossing = [
            p
            for p in csv.DictReader(handle)
            if p["image_a"] in part
            and p["image_b"] in part
            and part[p["image_a"]] != part[p["image_b"]]
        ]
    assert crossing == []  # no known duplicate pair crosses train/validation in any fold
    for fold in range(1, 11):
        train, validation = fold_members(rows, fold)
        assert len(train) + len(validation) == 3044
        with (CV_DIR / f"fold_{fold:02d}_validation.csv").open() as handle:
            assert sum(1 for _ in csv.DictReader(handle)) == len(validation)
        assert {r.unified_class for r in validation} == {r.unified_class for r in train}
    development = [s for s in split if s.split == DEVELOPMENT]
    assert assign_folds(development, n_folds=10, seed=42) == rows  # reproducible
