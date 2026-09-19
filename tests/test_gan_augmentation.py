"""Phase 10 — training-fold-only cDCGAN pipeline: leakage guards, class-aware
amounts, training/generation smoke on a tiny synthetic fold (CPU), provenance
records, ablation manifests and the script."""

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

from src.dataset_cleaning import CLEAN_MANIFEST_NAME, build_clean_manifest, write_clean_manifest
from src.gan_augmentation import (
    ARCHITECTURE,
    ConditionalDiscriminator,
    ConditionalGenerator,
    FoldTrainingImages,
    GANConfig,
    assert_train_only_path,
    augmented_training_manifest,
    forbidden_ids_for_fold,
    generate,
    load_generator,
    plan_synthetic_counts,
    train_gan,
    validate_synthetic_rows,
    write_augmented_manifest,
    write_synthetic_manifest,
)
from src.manifest import CANONICAL_CLASSES
from src.split_v2 import (
    DEVELOPMENT,
    FINAL_TEST,
    assign_folds,
    build_dev_test_split,
    fold_members,
    write_folds,
    write_split,
)
from tests.test_split_v2 import _many_records

ROOT = Path(__file__).resolve().parent.parent
CPU = torch.device("cpu")
TINY = GANConfig(image_size=16, latent_dim=8, base_channels=8, epochs=1, batch_size=8, seed=1)


def _write_images(rows, repo: Path, size=(24, 20)) -> None:
    rng = np.random.default_rng(0)
    for r in rows:
        p = repo / r.filepath
        p.parent.mkdir(parents=True, exist_ok=True)
        pixels = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(p)


@pytest.fixture
def fold_setup(tmp_path):
    """Clean manifest -> dev/test split -> 3 folds, with real image files on disk."""
    clean = build_clean_manifest(_many_records(30))
    split = build_dev_test_split(clean, test_ratio=0.2, seed=42)
    development = [s for s in split if s.split == DEVELOPMENT]
    final_test = [s for s in split if s.split == FINAL_TEST]
    folds = assign_folds(development, n_folds=3, seed=42)
    _write_images(split, tmp_path)
    audit = tmp_path / "data" / "audit"
    write_clean_manifest(clean, audit / CLEAN_MANIFEST_NAME)
    write_split(split, audit / "split_v2")
    write_folds(folds, audit / "cv_v2", 3)
    return tmp_path, folds, final_test


# --- networks ---


def test_networks_produce_expected_shapes():
    g = ConditionalGenerator(8, TINY)
    d = ConditionalDiscriminator(8, TINY)
    z = torch.randn(4, TINY.latent_dim)
    labels = torch.tensor([0, 3, 7, 7])
    images = g(z, labels)
    assert images.shape == (4, 3, 16, 16) and images.min() >= -1 and images.max() <= 1
    assert d(images, labels).shape == (4,)
    bigger = GANConfig(image_size=64, latent_dim=8, base_channels=8, epochs=1)
    assert ConditionalGenerator(8, bigger)(z, labels).shape == (4, 3, 64, 64)
    with pytest.raises(ValueError):
        GANConfig(image_size=48)


# --- isolation guards ---


def test_validation_and_test_ids_cannot_enter_gan_training(fold_setup):
    repo, folds, final_test = fold_setup
    train, validation = fold_members(folds, 1)
    forbidden = forbidden_ids_for_fold(folds, 1, final_test)
    assert {r.image_id for r in validation} <= forbidden
    assert {s.image_id for s in final_test} <= forbidden
    FoldTrainingImages(train, forbidden_ids=forbidden, repo_root=repo, image_size=16)  # ok
    with pytest.raises(ValueError, match="validation/test images offered"):
        FoldTrainingImages(
            train + validation[:1], forbidden_ids=forbidden, repo_root=repo, image_size=16
        )
    leaked_test = list(train) + [
        type(train[0])(**{**train[0].__dict__, "image_id": final_test[0].image_id})
    ]
    with pytest.raises(ValueError, match="validation/test images offered"):
        FoldTrainingImages(leaked_test, forbidden_ids=forbidden, repo_root=repo, image_size=16)
    with pytest.raises(ValueError, match="final_test rows expected"):
        forbidden_ids_for_fold(
            folds,
            1,
            validation
            and [type(final_test[0])(**{**final_test[0].__dict__, "split": DEVELOPMENT})],
        )


def test_synthetic_paths_must_be_train_directories(tmp_path):
    assert_train_only_path(tmp_path / "data" / "gan" / "fold_01" / "train" / "EUS_Disease")
    for bad in ("validation", "test", "final_test"):
        with pytest.raises(ValueError, match="train directory"):
            assert_train_only_path(tmp_path / "data" / "gan" / "fold_01" / bad / "x")
    with pytest.raises(ValueError, match="train directory"):
        assert_train_only_path(tmp_path / "data" / "gan" / "fold_01" / "images")


# --- amounts from the actual distribution ---


def test_plan_synthetic_counts_uses_actual_distribution():
    real = {"Healthy Fish": 800, "Aeromoniasis": 300, "EUS Disease": 200}
    plan = plan_synthetic_counts(real, GANConfig())
    assert plan == {"Healthy Fish": 0, "Aeromoniasis": 300, "EUS Disease": 200}  # capped 1x real
    plan = plan_synthetic_counts(real, GANConfig(target_per_class=500, max_synthetic_ratio=0.5))
    assert plan == {"Healthy Fish": 0, "Aeromoniasis": 150, "EUS Disease": 100}
    plan = plan_synthetic_counts(real, GANConfig(synthetic_per_class={"EUS Disease": 50}))
    assert plan == {"Healthy Fish": 0, "Aeromoniasis": 0, "EUS Disease": 50}
    with pytest.raises(ValueError):
        GANConfig(synthetic_per_class={"Not A Class": 5})


# --- train + generate on a tiny fold (CPU) ---


def test_train_generate_and_record_provenance(fold_setup):
    repo, folds, final_test = fold_setup
    train, _ = fold_members(folds, 2)
    forbidden = forbidden_ids_for_fold(folds, 2, final_test)
    dataset = FoldTrainingImages(train, forbidden_ids=forbidden, repo_root=repo, image_size=16)
    out = repo / "data" / "gan" / "fold_02"
    manifest = repo / "data" / "audit" / "cv_v2" / "fold_02_train.csv"
    checkpoint, record = train_gan(
        dataset,
        fold=2,
        config=TINY,
        device=CPU,
        out_dir=out,
        train_manifest=manifest,
        forbidden_count=len(forbidden),
    )
    assert checkpoint.is_file() and (out / "gan_run.json").is_file()
    rec = json.loads((out / "gan_run.json").read_text())
    for key in (
        "architecture",
        "fold",
        "config",
        "seed",
        "device",
        "real_training_images",
        "real_per_class",
        "epochs_run",
        "checkpoint",
        "checkpoint_sha256",
        "train_manifest",
        "train_manifest_sha256",
    ):
        assert key in rec
    assert rec["architecture"] == ARCHITECTURE and rec["fold"] == 2 and rec["epochs_run"] == 1
    assert rec["real_training_images"] == len(train)
    assert all(np.isfinite(v) for v in rec["final_losses"].values())

    counts = {"EUS Disease": 3, "Healthy Fish": 2}
    rows = generate(
        checkpoint,
        fold=2,
        counts=counts,
        gan_dir=repo / "data" / "gan",
        repo_root=repo,
        device=CPU,
        seed=7,
    )
    assert len(rows) == 5
    validate_synthetic_rows(rows, 2, repo)
    assert all(r["filepath"].startswith("data/gan/fold_02/train/") for r in rows)
    assert {r["unified_class"] for r in rows} == set(counts)
    assert all(CANONICAL_CLASSES[r["label"]] == r["unified_class"] for r in rows)
    with Image.open(repo / rows[0]["filepath"]) as image:
        assert image.size == (16, 16) and image.mode == "RGB"
    # deterministic in the seed
    again = generate(
        checkpoint,
        fold=2,
        counts=counts,
        gan_dir=repo / "data" / "gan",
        repo_root=repo,
        device=CPU,
        seed=7,
    )
    assert [(r["image_id"], (repo / r["filepath"]).read_bytes()) for r in rows] == [
        (r["image_id"], (repo / r["filepath"]).read_bytes()) for r in again
    ]
    # the checkpoint refuses to generate for another fold
    with pytest.raises(ValueError, match="trained on fold 2"):
        generate(
            checkpoint,
            fold=1,
            counts=counts,
            gan_dir=repo / "data" / "gan",
            repo_root=repo,
            device=CPU,
        )
    generator, payload = load_generator(checkpoint, CPU)
    assert payload["class_names"] == list(CANONICAL_CLASSES)

    write_synthetic_manifest(rows, out / "synthetic_manifest.csv")
    augmented = augmented_training_manifest(train, rows, 2)
    assert len(augmented) == len(train) + 5
    assert sum(r["synthetic"] for r in augmented) == 5
    assert all(r["fold"] == -1 for r in augmented if r["synthetic"])
    write_augmented_manifest(augmented, out / "fold_02_train_gan.csv")
    with (out / "fold_02_train_gan.csv").open() as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == len(augmented)
    # a synthetic row from another fold, or a validation row, is refused
    with pytest.raises(ValueError, match="another fold"):
        augmented_training_manifest(train, [{**rows[0], "fold": 3}], 2)
    _, validation = fold_members(folds, 2)
    with pytest.raises(ValueError, match="validation fold"):
        augmented_training_manifest(train + validation[:1], rows, 2)


def test_validate_synthetic_rows_rejects_bad_rows(tmp_path):
    good = tmp_path / "data" / "gan" / "fold_01" / "train" / "EUS_Disease" / "s.png"
    good.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(good)
    row = {
        "image_id": "x",
        "fold": 1,
        "unified_class": "EUS Disease",
        "label": 3,
        "filepath": str(good.relative_to(tmp_path)),
    }
    validate_synthetic_rows([row], 1, tmp_path)
    with pytest.raises(ValueError, match="belongs to fold"):
        validate_synthetic_rows([{**row, "fold": 2}], 1, tmp_path)
    with pytest.raises(ValueError, match="invalid class"):
        validate_synthetic_rows([{**row, "unified_class": "Nope"}], 1, tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        validate_synthetic_rows([{**row, "label": 0}], 1, tmp_path)
    with pytest.raises(ValueError, match="train directory"):
        validate_synthetic_rows(
            [{**row, "filepath": "data/gan/fold_01/validation/s.png"}], 1, tmp_path
        )


# --- script ---


def test_run_gan_fold_script_end_to_end(fold_setup):
    repo, folds, _ = fold_setup
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_gan_fold.py"),
        "--fold",
        "3",
        "--data-dir",
        str(repo / "data"),
        "--repo-root",
        str(repo),
        "--epochs",
        "1",
        "--image-size",
        "16",
        "--batch-size",
        "8",
        "--device",
        "cpu",
        "--synthetic-per-class",
        '{"EUS Disease": 2}',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = repo / "data" / "gan" / "fold_03"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["synthetic_images"] == 2 and summary["smoke_run"] is True
    assert (out / "generator.pt").is_file() and (out / "synthetic_manifest.csv").is_file()
    assert (out / "fold_03_train_gan.csv").is_file()
    # nothing written outside the fold's train directory
    pngs = list((repo / "data" / "gan").rglob("*.png"))
    assert pngs and all("train" in p.parts and "fold_03" in p.parts for p in pngs)
    # the validation/test files were not modified
    from src.split_v2 import read_folds, verify_split_digest

    read_folds(repo / "data" / "audit" / "cv_v2")
    verify_split_digest(repo / "data" / "audit" / "split_v2")
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 2
