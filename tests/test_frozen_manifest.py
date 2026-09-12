"""Guards on the committed frozen split (data/split_manifest.csv), plus the AMP and
manifest-driven paths of the training engine."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.manifest import (
    CANONICAL_CLASSES,
    MANIFEST_PATH,
    SPLITS,
    build_split_dataset,
    read_manifest,
    validate_manifest,
    verify_manifest_digest,
)

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TOTALS = {"train": 2384, "val": 512, "test": 509}

pytestmark = pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="no frozen manifest in repo")


def test_frozen_manifest_matches_its_digest():
    assert verify_manifest_digest(MANIFEST_PATH).startswith("b7d1fccbb21e73a8")


def test_frozen_manifest_structure_and_counts():
    rows = read_manifest(MANIFEST_PATH)
    counts = validate_manifest(rows)
    assert {s: sum(c.values()) for s, c in counts.items()} == EXPECTED_TOTALS
    assert len(rows) == sum(EXPECTED_TOTALS.values())
    assert all(r.filepath.startswith("data/original/aquahealth/") for r in rows)
    for name in CANONICAL_CLASSES:
        per_split = [counts[s][name] for s in SPLITS]
        total = sum(per_split)
        assert abs(per_split[0] / total - 0.70) < 0.01, name
        assert abs(per_split[2] / total - 0.15) < 0.01, name
    assert len({r.filepath for r in rows}) == len(rows)


def test_frozen_manifest_labels_follow_canonical_mapping():
    rows = read_manifest(MANIFEST_PATH)
    for row in rows:
        assert CANONICAL_CLASSES[row.label] == row.class_name
    folder_labels = {
        ("Healthy Fish", 7),
        ("EUS", 3),
        ("Bacterial Red disease", 0),
        ("Viral diseases White tail disease", 6),
    }
    for folder, label in folder_labels:
        matching = [r for r in rows if f"/train_split/{folder}/" in r.filepath]
        assert matching and all(r.label == label for r in matching), folder


def test_build_split_dataset_rejects_foreign_class_names():
    with pytest.raises(ValueError, match="canonical"):
        build_split_dataset("val", class_names=["a", "b"])


@pytest.mark.skipif(
    not (ROOT / "data" / "original" / "aquahealth").exists(), reason="real dataset not present"
)
def test_frozen_manifest_files_exist_locally():
    for split in SPLITS:
        assert build_split_dataset(split).missing_files() == []


# --- engine paths exercised on the synthetic fixture (no real data needed) ---


def test_amp_training_step_produces_finite_updates(make_image_folder):
    from src.dataset import ImageFolderDataset, build_dataloader
    from src.model import build_classifier
    from src.preprocessing import PreprocessConfig, build_eval_transform
    from src.train import TrainConfig, build_optimizer, build_scaler, train_one_epoch

    root = make_image_folder(("a", "b", "c"), per_class=4, size=(80, 60))
    dataset = ImageFolderDataset(root, transform=build_eval_transform(PreprocessConfig(64, 72)))
    loader = build_dataloader(dataset, batch_size=6, num_workers=0)
    model = build_classifier(3, pretrained=False, freeze_backbone=True)
    config = TrainConfig(learning_rate=1e-3, amp=True)
    device = torch.device("cpu")
    scaler = build_scaler(device, config.amp)
    assert not scaler.is_enabled(), "GradScaler is only enabled on CUDA"
    before = model.classifier[1].weight.clone()
    stats = train_one_epoch(
        model, loader, build_optimizer(model, config), device, epoch=1, amp=True, scaler=scaler
    )
    assert torch.isfinite(torch.tensor(stats.loss))
    assert not torch.equal(before, model.classifier[1].weight)
    assert torch.isfinite(model.classifier[1].weight).all()


def test_train_cli_accepts_manifest_and_rejects_both_sources(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--train-root",
            "x",
            "--manifest",
            "--checkpoint-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0 and "not allowed with argument" in result.stderr
