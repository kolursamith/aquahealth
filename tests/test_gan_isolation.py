"""Layer 2 — the seven isolation proofs for training-fold-only GAN augmentation.

Each test is one requirement, by name:

    1. test IDs never enter GAN
    2. validation IDs never enter GAN
    3. generated data belongs only to the training fold
    4. fold isolation holds across folds
    5. generated images have valid dimensions
    6. generated files are traceable (manifest, run record, registry, config)
    7. raw data remains unchanged

All on the tiny synthetic fixture (CPU, 16x16, 1 epoch) so the suite stays fast;
the real-data path is exercised by scripts/run_gan_fold.py --smoke.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from src.gan_augmentation import (
    ARCHITECTURE,
    REGISTRY_COLUMNS,
    FoldTrainingImages,
    GANConfig,
    augmented_training_manifest,
    forbidden_ids_for_fold,
    generate,
    load_gan_config,
    plan_synthetic_counts,
    read_registry,
    registry_row,
    train_gan,
    update_registry,
    validate_synthetic_rows,
    write_augmented_manifest,
    write_synthetic_manifest,
)
from src.manifest import CANONICAL_CLASSES
from src.split_v2 import FINAL_TEST, fold_members, read_folds, read_split, verify_split_digest
from tests.test_gan_augmentation import TINY, fold_setup  # noqa: F401  (fixture re-export)

ROOT = Path(__file__).resolve().parent.parent
CPU = __import__("torch").device("cpu")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_fold(repo: Path, folds, final_test, fold: int, counts: dict[str, int]):
    """Train + generate for one fold on the fixture; returns (train_rows, synthetic, record)."""
    train, _ = fold_members(folds, fold)
    forbidden = forbidden_ids_for_fold(folds, fold, final_test)
    dataset = FoldTrainingImages(train, forbidden_ids=forbidden, repo_root=repo, image_size=16)
    out = repo / "data" / "gan" / f"fold_{fold:02d}"
    checkpoint, record = train_gan(
        dataset,
        fold=fold,
        config=TINY,
        device=CPU,
        out_dir=out,
        train_manifest=repo / "data" / "audit" / "cv_v2" / f"fold_{fold:02d}_train.csv",
        forbidden_count=len(forbidden),
    )
    synthetic = generate(
        checkpoint,
        fold=fold,
        counts=counts,
        gan_dir=repo / "data" / "gan",
        repo_root=repo,
        device=CPU,
        seed=TINY.seed,
    )
    validate_synthetic_rows(synthetic, fold, repo)
    write_synthetic_manifest(synthetic, out / "synthetic_manifest.csv")
    write_augmented_manifest(
        augmented_training_manifest(train, synthetic, fold), out / f"fold_{fold:02d}_train_gan.csv"
    )
    return train, synthetic, record


# 1 ------------------------------------------------------------------------------------------


def test_1_test_ids_never_enter_gan(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    test_ids = {s.image_id for s in final_test}
    assert test_ids, "fixture must have a final test partition"
    for fold in (1, 2, 3):
        train, _ = fold_members(folds, fold)
        forbidden = forbidden_ids_for_fold(folds, fold, final_test)
        # every test id is forbidden, and none of them is in what the GAN would read
        assert test_ids <= forbidden
        assert not ({r.image_id for r in train} & test_ids)
        dataset = FoldTrainingImages(train, forbidden_ids=forbidden, repo_root=repo, image_size=16)
        assert not ({r.image_id for r in dataset.rows} & test_ids)
        # and smuggling a test id in (same path, test id) aborts before any image is decoded
        smuggled = train + [
            type(train[0])(**{**train[0].__dict__, "image_id": next(iter(test_ids))})
        ]
        with pytest.raises(ValueError, match="validation/test images offered"):
            FoldTrainingImages(smuggled, forbidden_ids=forbidden, repo_root=repo, image_size=16)


# 2 ------------------------------------------------------------------------------------------


def test_2_validation_ids_never_enter_gan(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    for fold in (1, 2, 3):
        train, validation = fold_members(folds, fold)
        val_ids = {r.image_id for r in validation}
        forbidden = forbidden_ids_for_fold(folds, fold, final_test)
        assert val_ids <= forbidden
        assert not ({r.image_id for r in train} & val_ids)
        dataset = FoldTrainingImages(train, forbidden_ids=forbidden, repo_root=repo, image_size=16)
        assert not ({r.image_id for r in dataset.rows} & val_ids)
        with pytest.raises(ValueError, match="validation/test images offered"):
            FoldTrainingImages(
                train + validation[:1], forbidden_ids=forbidden, repo_root=repo, image_size=16
            )
    # the GAN's real input is exactly the union of the OTHER folds' validation sets
    for fold in (1, 2, 3):
        train, _ = fold_members(folds, fold)
        assert {r.image_id for r in train} == {r.image_id for r in folds if r.fold != fold}


# 3 ------------------------------------------------------------------------------------------


def test_3_generated_data_belongs_only_to_training_fold(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    train, synthetic, _ = _run_fold(
        repo, folds, final_test, 2, {"EUS Disease": 3, "Healthy Fish": 2}
    )
    assert len(synthetic) == 5
    for row in synthetic:
        path = repo / row["filepath"]
        assert row["fold"] == 2 and "fold_02" in path.parts and "train" in path.parts
        assert not ({"validation", "val", "test", "final_test"} & {p.lower() for p in path.parts})
    # synthetic ids are new: never a real id, never in validation, never in test
    real_ids = {r.image_id for r in folds} | {s.image_id for s in final_test}
    assert not ({r["image_id"] for r in synthetic} & real_ids)
    # the WITH-GAN manifest = real training rows + synthetic rows, nothing from validation
    with (repo / "data" / "gan" / "fold_02" / "fold_02_train_gan.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    _, validation = fold_members(folds, 2)
    assert {r["image_id"] for r in rows if r["synthetic"] == "False"} == {r.image_id for r in train}
    assert not ({r["image_id"] for r in rows} & {r.image_id for r in validation})
    assert all(int(r["fold"]) == -1 for r in rows if r["synthetic"] == "True")
    # the validation / test manifests are byte-identical to before (digest-guarded readers pass)
    read_folds(repo / "data" / "audit" / "cv_v2")
    verify_split_digest(repo / "data" / "audit" / "split_v2")


# 4 ------------------------------------------------------------------------------------------


def test_4_fold_isolation_holds_across_folds(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    produced = {}
    for fold in (1, 2, 3):
        _, synthetic, record = _run_fold(repo, folds, final_test, fold, {"EUS Disease": 2})
        produced[fold] = synthetic
        # each generator saw exactly its own training fold and checked all forbidden ids
        assert record.fold == fold
        assert record.real_training_images == len(fold_members(folds, fold)[0])
        assert record.forbidden_ids_checked == len(forbidden_ids_for_fold(folds, fold, final_test))
    for fold, synthetic in produced.items():
        paths = {repo / r["filepath"] for r in synthetic}
        ids = {r["image_id"] for r in synthetic}
        for other, other_synthetic in produced.items():
            if other == fold:
                continue
            assert not (paths & {repo / r["filepath"] for r in other_synthetic})
            assert not (ids & {r["image_id"] for r in other_synthetic})
            # fold k's synthetic images never appear in fold j's WITH-GAN manifest
            with (
                repo / "data" / "gan" / f"fold_{other:02d}" / f"fold_{other:02d}_train_gan.csv"
            ).open(newline="") as handle:
                assert not (ids & {r["image_id"] for r in csv.DictReader(handle)})
        # a checkpoint from fold k refuses to generate "for" fold j
        with pytest.raises(ValueError, match="trained on fold"):
            generate(
                repo / "data" / "gan" / f"fold_{fold:02d}" / "generator.pt",
                fold=(fold % 3) + 1,
                counts={"EUS Disease": 1},
                gan_dir=repo / "data" / "gan",
                repo_root=repo,
                device=CPU,
            )
    # every synthetic file on disk sits under exactly one fold's train directory
    for png in (repo / "data" / "gan").rglob("*.png"):
        fold_parts = [p for p in png.parts if p.startswith("fold_")]
        assert len(fold_parts) == 1 and "train" in png.parts


# 5 ------------------------------------------------------------------------------------------


def test_5_generated_images_have_valid_dimensions(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    _, synthetic, _ = _run_fold(repo, folds, final_test, 1, {c: 1 for c in CANONICAL_CLASSES})
    assert len(synthetic) == len(CANONICAL_CLASSES)
    for row in synthetic:
        with Image.open(repo / row["filepath"]) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (TINY.image_size, TINY.image_size)
            extrema = image.getextrema()
            assert all(0 <= lo <= hi <= 255 for lo, hi in extrema)
    # the resolution recorded in the run record is the one on disk
    record = json.loads((repo / "data" / "gan" / "fold_01" / "gan_run.json").read_text())
    assert record["config"]["image_size"] == TINY.image_size


# 6 ------------------------------------------------------------------------------------------


def test_6_generated_files_are_traceable(fold_setup, tmp_path):  # noqa: F811
    repo, folds, final_test = fold_setup
    train, synthetic, record = _run_fold(repo, folds, final_test, 3, {"Healthy Fish": 4})
    out = repo / "data" / "gan" / "fold_03"
    checkpoint = out / "generator.pt"
    # every synthetic row names its fold, class, label, seed, generator and the generator's digest
    for row in synthetic:
        assert row["fold"] == 3 and row["seed"] == TINY.seed
        assert row["architecture"] == ARCHITECTURE
        assert Path(row["generator_checkpoint"]) == checkpoint
        assert row["generator_sha256"] == _sha(checkpoint)
        assert CANONICAL_CLASSES[row["label"]] == row["unified_class"]
        assert row["image_id"].startswith("gan-f03-")
    # the run record names the source training fold manifest and its digest
    assert Path(record.train_manifest) == repo / "data" / "audit" / "cv_v2" / "fold_03_train.csv"
    assert record.train_manifest_sha256 == _sha(Path(record.train_manifest))
    assert record.checkpoint_sha256 == _sha(checkpoint)
    assert record.optimizer.startswith("Adam(")
    assert record.real_per_class == {
        CANONICAL_CLASSES[k]: v
        for k, v in __import__("collections").Counter(r.label for r in train).items()
    }
    # the registry line carries every required field and the digests match the files
    registry = tmp_path / "registry.csv"
    row = registry_row(
        record,
        plan={"Healthy Fish": 4},
        synthetic_manifest=out / "synthetic_manifest.csv",
        with_gan_manifest=out / "fold_03_train_gan.csv",
        smoke_run=False,
    )
    update_registry(registry, row)
    (line,) = read_registry(registry)
    for key in (
        "architecture",
        "latent_dim",
        "resolution",
        "epochs",
        "batch_size",
        "optimizer",
        "learning_rate",
        "seed",
        "fold",
        "generated_count",
        "source_training_fold",
    ):
        assert key in REGISTRY_COLUMNS and line[key] != ""
    assert line["architecture"] == ARCHITECTURE
    assert int(line["fold"]) == 3 and int(line["generated_count"]) == 4
    assert line["resolution"] == f"{TINY.image_size}x{TINY.image_size}"
    assert int(line["latent_dim"]) == TINY.latent_dim and int(line["epochs"]) == TINY.epochs
    assert line["source_training_fold_sha256"] == _sha(Path(line["source_training_fold"]))
    assert line["generator_sha256"] == _sha(checkpoint)
    assert line["synthetic_manifest_sha256"] == _sha(out / "synthetic_manifest.csv")
    # re-registering the same fold replaces, not duplicates; other folds are kept and sorted
    update_registry(registry, {**row, "fold": 1})
    update_registry(registry, row)
    assert [int(r["fold"]) for r in read_registry(registry)] == [1, 3]
    # the committed config files load into the same dataclass the run used
    default = load_gan_config(ROOT / "configs" / "gan_v2" / "default.json")
    smoke = load_gan_config(ROOT / "configs" / "gan_v2" / "smoke.json")
    assert isinstance(default, GANConfig) and default.max_train_images is None
    assert smoke.max_train_images == 256 and smoke.epochs == 1
    assert load_gan_config(ROOT / "configs" / "gan_v2" / "default.json", epochs=2).epochs == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"architecture": "StyleGAN2", "gan": {}}))
    with pytest.raises(ValueError, match="architecture"):
        load_gan_config(bad)
    bad.write_text(json.dumps({"gan": {"nope": 1}}))
    with pytest.raises(ValueError, match="unknown GANConfig fields"):
        load_gan_config(bad)


# 7 ------------------------------------------------------------------------------------------


def test_7_raw_data_remains_unchanged(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    split = read_split(repo / "data" / "audit" / "split_v2")
    raw = sorted({repo / s.filepath for s in split})
    before = {p: (_sha(p), p.stat().st_size) for p in raw}
    manifests = sorted((repo / "data" / "audit").rglob("*.csv")) + sorted(
        (repo / "data" / "audit").rglob("*.sha256")
    )
    manifests_before = {p: _sha(p) for p in manifests}
    _run_fold(repo, folds, final_test, 1, {"EUS Disease": 3})
    # nothing of the raw corpus (dev + test) or the split/fold manifests changed
    assert {p: (_sha(p), p.stat().st_size) for p in raw} == before
    assert {p: _sha(p) for p in manifests} == manifests_before
    # no file was added next to the raw images; everything new is under data/gan
    raw_dirs = {p.parent for p in raw}
    for directory in raw_dirs:
        assert {q for q in directory.iterdir() if q.is_file()} <= set(raw)
    new_files = {p for p in repo.rglob("*") if p.is_file()} - set(raw) - set(manifests)
    assert new_files and all("gan" in p.parts for p in new_files)
    assert not any(s.split == FINAL_TEST and "gan" in Path(s.filepath).parts for s in split)


# class-aware policy (justification recorded in configs/gan_v2/default.json) ------------------


def test_policy_never_generates_the_same_count_for_every_class():
    real = {"Healthy Fish": 1000, "Aeromoniasis": 340, "EUS Disease": 560, "Saprolegniasis": 310}
    plan = plan_synthetic_counts(real, GANConfig())
    assert plan["Healthy Fish"] == 0  # the largest class is never topped up
    assert plan["Aeromoniasis"] == 340  # capped at 1.0 x real, not (1000 - 340)
    assert plan["EUS Disease"] == 440 and plan["Saprolegniasis"] == 310
    assert len(set(plan.values())) > 1
