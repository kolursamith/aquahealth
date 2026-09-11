"""Layer 5 — dataset tooling (`scripts/audit_dataset.py`, `scripts/create_split.py`)."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from audit_dataset import audit, format_report  # noqa: E402
from create_split import SplitRatios, _partition, create_split  # noqa: E402

from src.dataset import ImageFolderDataset  # noqa: E402

CLASSES = ("alpha", "beta", "gamma")


# --- audit ---


def test_audit_reports_counts_from_the_data(make_image_folder):
    root = make_image_folder(CLASSES, per_class=3, size=(48, 40))
    report = audit(root)
    assert report.class_names == list(CLASSES)
    assert report.images_per_class == {"alpha": 3, "beta": 3, "gamma": 3}
    assert report.total_images == 9
    assert report.extensions == {".png": 6, ".jpg": 3}
    assert report.modes == {"RGB": 9}
    assert report.sizes == {"48x40": 9}
    assert report.ok
    assert "RESULT: OK" in format_report(report)


def test_audit_flags_corrupt_files(make_image_folder):
    root = make_image_folder(CLASSES, per_class=2)
    (root / "beta" / "zz_bad.jpg").write_bytes(b"not an image")
    report = audit(root)
    assert report.total_images == 7
    assert [Path(p).name for p in report.corrupt] == ["zz_bad.jpg"]
    assert not report.ok
    assert "RESULT: FAIL" in format_report(report)


def test_audit_flags_byte_identical_duplicates_across_classes(make_image_folder):
    root = make_image_folder(CLASSES, per_class=2)
    shutil.copy2(root / "alpha" / "000.png", root / "gamma" / "copy_of_alpha.png")
    report = audit(root)
    assert len(report.duplicates) == 1
    assert {Path(p).name for p in report.duplicates[0]} == {"000.png", "copy_of_alpha.png"}
    assert not report.ok


def test_audit_cli_exit_code_and_json(make_image_folder):
    root = make_image_folder(CLASSES, per_class=1)
    script = ROOT / "scripts" / "audit_dataset.py"
    ok = subprocess.run(
        [sys.executable, str(script), str(root), "--json"], capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["total_images"] == 3

    (root / "alpha" / "bad.png").write_bytes(b"x")
    bad = subprocess.run([sys.executable, str(script), str(root)], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "Corrupt files: 1" in bad.stdout


# --- split ---


@pytest.mark.parametrize("ratios", [(0.5, 0.5, 0.1), (-0.1, 0.6, 0.5), (0.7, 0.2, 0.2)])
def test_split_ratios_are_validated(ratios):
    with pytest.raises(ValueError):
        SplitRatios(*ratios)


def test_partition_is_exhaustive_and_disjoint():
    items = [Path(f"{i}.png") for i in range(23)]
    parts = _partition(items, SplitRatios(0.7, 0.15, 0.15), random.Random(0))
    assert sum(len(v) for v in parts.values()) == 23
    assert len(set().union(*map(set, parts.values()))) == 23
    assert [len(parts[k]) for k in ("train", "val", "test")] == [16, 3, 4]


def _destinations(tmp_path: Path) -> dict[str, Path]:
    return {name: tmp_path / "splits" / name for name in ("train", "val", "test")}


def test_create_split_is_stratified_disjoint_and_leaves_source_intact(make_image_folder, tmp_path):
    source = make_image_folder(CLASSES, per_class=10, name="original")
    before = sorted(p.name for p in source.rglob("*") if p.is_file())
    counts = create_split(source, _destinations(tmp_path), SplitRatios(0.7, 0.2, 0.1), seed=0)

    assert counts == {
        "train": {"alpha": 7, "beta": 7, "gamma": 7},
        "val": {"alpha": 2, "beta": 2, "gamma": 2},
        "test": {"alpha": 1, "beta": 1, "gamma": 1},
    }
    assert sorted(p.name for p in source.rglob("*") if p.is_file()) == before

    files = {
        name: {(p.parent.name, p.name) for p in dest.rglob("*") if p.is_file()}
        for name, dest in _destinations(tmp_path).items()
    }
    assert not (files["train"] & files["val"]) and not (files["train"] & files["test"])
    assert not (files["val"] & files["test"])
    assert files["train"] | files["val"] | files["test"] == {
        (p.parent.name, p.name) for p in source.rglob("*") if p.is_file()
    }


def test_create_split_is_reproducible_under_seed(make_image_folder, tmp_path):
    source = make_image_folder(CLASSES, per_class=6, name="original")
    first = _destinations(tmp_path / "a")
    second = _destinations(tmp_path / "b")
    create_split(source, first, SplitRatios(0.5, 0.25, 0.25), seed=3)
    create_split(source, second, SplitRatios(0.5, 0.25, 0.25), seed=3)
    for name in first:
        assert sorted(p.name for p in first[name].rglob("*.*")) == sorted(
            p.name for p in second[name].rglob("*.*")
        )


def test_create_split_refuses_to_overwrite_existing_split(make_image_folder, tmp_path):
    source = make_image_folder(CLASSES, per_class=2, name="original")
    destinations = _destinations(tmp_path)
    create_split(source, destinations, SplitRatios(0.5, 0.5, 0.0))
    with pytest.raises(FileExistsError, match="train split already contains files"):
        create_split(source, destinations, SplitRatios(0.5, 0.5, 0.0))


def test_split_outputs_load_with_a_shared_label_mapping(make_image_folder, tmp_path):
    source = make_image_folder(CLASSES, per_class=4, name="original")
    destinations = _destinations(tmp_path)
    create_split(source, destinations, SplitRatios(0.5, 0.25, 0.25))
    train = ImageFolderDataset(destinations["train"])
    val = ImageFolderDataset(destinations["val"], class_names=train.class_names)
    assert train.class_to_idx == val.class_to_idx == {"alpha": 0, "beta": 1, "gamma": 2}
    assert len(train) == 6 and len(val) == 3
