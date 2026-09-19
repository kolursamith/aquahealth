"""Phase 6 (leakage audit findings, EXIF scan, script) and Phase 7 (existing
CLAHE pipeline on the new manifests, corrupt-file reporting) — plus loading the
real committed master/clean manifests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import ImageDecodeError, load_image
from src.dataset_cleaning import (
    CLEAN_MANIFEST_NAME,
    build_clean_manifest,
    read_clean_manifest,
    validate_clean_manifest,
    write_clean_manifest,
)
from src.leakage_audit import (
    FINDING_COLUMNS,
    NOT_ESTABLISHED,
    exif_summary,
    findings_rows,
    leakage_report,
)
from src.manifest import MANIFEST_PATH, verify_manifest_digest
from src.multi_dataset import MASTER_MANIFEST_NAME, read_master_manifest
from src.preprocessing import CLAHE, CLAHEConfig, PreprocessConfig, build_eval_transform
from tests.test_dataset_cleaning import build_master_records

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
AUDIT = ROOT / "data" / "audit"
sys.path.insert(0, str(SCRIPTS))

from validate_clahe import corrupt_file_check, representative_samples  # noqa: E402


def _write_image(path: Path, colour, size=(40, 30), orientation: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pixels = np.clip(
        np.array(colour, dtype=np.int16) + rng.integers(-40, 41, size=(size[1], size[0], 3)),
        0,
        255,
    ).astype(np.uint8)
    image = Image.fromarray(pixels)
    if orientation is None:
        image.save(path)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        exif[0x010F], exif[0x0110] = "TestMake", "TestModel"
        image.save(path, exif=exif.tobytes())


# --- Phase 6 ---


def test_findings_cover_the_seven_checks():
    rows = build_clean_manifest(build_master_records())
    report = leakage_report(rows)
    findings = findings_rows(rows, report, exif=None)
    assert [f["check"][0] for f in findings] == ["1", "2", "3", "4", "5", "6", "7"]
    assert all(tuple(f.keys()) == FINDING_COLUMNS for f in findings)
    assert all(f["severity"].startswith("n/a") for f in findings)  # no project convention
    specimen = findings[2]
    assert NOT_ESTABLISHED in specimen["finding"]  # datasets without specimen ids
    assert "mendeley" in specimen["finding"]
    assert "EXIF not scanned" in findings[4]["finding"]
    assert all(f["affected_images"] >= 0 for f in findings)


def test_exif_summary_reports_orientation_and_camera(tmp_path):
    rows = build_clean_manifest(build_master_records())
    repo = tmp_path
    for r in rows:
        if r.included:
            _write_image(
                repo / r.filepath,
                (90, 140, 90),
                orientation=6 if r.source_dataset == "mendeley" else None,
            )
    summary = exif_summary(rows, repo)
    assert summary["mendeley"]["with_exif"] == 2
    assert summary["mendeley"]["rotated_by_tag"] == 2
    assert summary["mendeley"]["rotated_by_class"] == {"Healthy Fish": 2}
    assert summary["mendeley"]["camera_models"] == {"TestMake TestModel": 2}
    assert summary["current_freshwater"]["with_exif"] == 0
    findings = findings_rows(rows, leakage_report(rows), summary)
    assert "stored rotated (tag != 1) 2" in findings[4]["finding"]


def test_audit_leakage_script_writes_csv_json_md(tmp_path):
    audit = tmp_path / "data" / "audit"
    audit.mkdir(parents=True)
    write_clean_manifest(build_clean_manifest(build_master_records()), audit / CLEAN_MANIFEST_NAME)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "audit_leakage.py"),
            "--data-dir",
            str(tmp_path / "data"),
            "--repo-root",
            str(tmp_path),
            "--skip-exif",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    with (audit / "leakage_report.csv").open() as handle:
        findings = list(csv.DictReader(handle))
    assert len(findings) == 7 and tuple(findings[0].keys()) == FINDING_COLUMNS
    report = json.loads((audit / "leakage_report.json").read_text())
    assert set(report) >= {"target_leakage", "duplicate_leakage", "suspicious_features"}
    assert "## Findings" in (audit / "leakage_report.md").read_text()


# --- Phase 7 ---


def test_existing_clahe_is_lab_l_channel_and_runs_on_new_manifest_images(tmp_path):
    rows = build_clean_manifest(build_master_records())
    picks = representative_samples(rows)
    assert {label for label, _ in picks} >= {
        "class_EUS_Disease",
        "class_Healthy_Fish",
        "source_current_freshwater",
        "source_mendeley",
    }
    transform = build_eval_transform(PreprocessConfig(clahe=CLAHEConfig()))
    assert [type(t).__name__ for t in transform.transforms][0] == "CLAHE"
    for _, row in picks:
        path = tmp_path / row.filepath
        _write_image(path, (200, 60, 60), size=(100, 70))
        image = load_image(path)
        enhanced = CLAHE(CLAHEConfig())(image)
        # LAB: chroma untouched, only lightness changed
        import cv2

        before = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
        after = cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_RGB2LAB)
        assert np.abs(before[:, :, 1:].astype(int) - after[:, :, 1:].astype(int)).mean() < 3
        tensor = transform(image)
        assert tensor.shape == (3, 224, 224) and tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_corrupt_images_are_reported_not_skipped(tmp_path):
    transform = build_eval_transform(PreprocessConfig(clahe=CLAHEConfig()))
    assert corrupt_file_check(transform)["status"] == "reported"
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"garbage")
    with pytest.raises(ImageDecodeError, match="bad.jpg"):
        transform(load_image(bad))


def test_validate_clahe_script_end_to_end(tmp_path):
    rows = build_clean_manifest(build_master_records())
    for r in rows:
        if r.included:
            _write_image(tmp_path / r.filepath, (60, 160, 90), size=(64, 48))
    audit = tmp_path / "data" / "audit"
    audit.mkdir(parents=True)
    write_clean_manifest(rows, audit / CLEAN_MANIFEST_NAME)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_clahe.py"),
            "--data-dir",
            str(tmp_path / "data"),
            "--repo-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    sheets = sorted(p.name for p in (audit / "clahe_samples").glob("*.png"))
    assert "class_EUS_Disease.png" in sheets and "source_mendeley.png" in sheets
    with (audit / "clahe_validation.csv").open() as handle:
        samples = list(csv.DictReader(handle))
    assert samples and all(
        s["output_shape"] == "3x224x224" and s["status"] == "ok" for s in samples
    )
    cfg = json.loads((audit / "preprocessing_config.json").read_text())
    assert cfg["clahe"] == {
        "colour_space": "LAB",
        "channel": "L (lightness) only; A and B untouched",
        "clip_limit": 2.0,
        "tile_grid_size": [8, 8],
        "position": "first stage, on the full-resolution decoded RGB image, before resize/crop",
        "library": "OpenCV (opencv-python-headless, requirements/base.txt)",
    }
    assert cfg["corrupt_file_handling"]["status"] == "reported"
    assert cfg["eval_pipeline_stages"] == [
        "CLAHE",
        "Resize",
        "CenterCrop",
        "PILToTensor",
        "ToDtype",
        "Normalize",
    ]


# --- the real committed manifests load and are consistent ---


@pytest.mark.skipif(not (AUDIT / MASTER_MANIFEST_NAME).is_file(), reason="master manifest absent")
def test_real_master_and_clean_manifests_load():
    master = read_master_manifest(AUDIT / MASTER_MANIFEST_NAME)
    assert len(master) == 7435
    assert {r.status for r in master} == {"ok"}
    assert {r.source_dataset for r in master} == {
        "current_freshwater",
        "kaptai",
        "roboflow",
        "mendeley",
        "paper_dataset",
    }
    assert all(len(r.sha256) == 64 and len(r.dhash) == 16 for r in master)
    clean = read_clean_manifest(AUDIT / CLEAN_MANIFEST_NAME)
    validate_clean_manifest(clean)
    assert len(clean) == 7435 and sum(r.included for r in clean) == 5942
    # the old baseline manifest is untouched
    assert verify_manifest_digest(MANIFEST_PATH).startswith("b7d1fccbb21e73a8")
