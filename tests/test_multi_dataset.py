"""Phase 1-5 multi-dataset pipeline: discovery with provenance, label mapping,
master manifest I/O, hashing (reused from src.manifest) and duplicate reporting
— on a synthetic delivery drop that mirrors the delivered layouts."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.label_harmonization import (
    LABEL_MAPPING,
    LABEL_MAPPING_COLUMNS,
    LabelMapping,
    lookup,
    read_label_mapping_csv,
    validate_mapping_table,
    write_label_mapping_csv,
)
from src.manifest import CANONICAL_CLASSES, SOURCE_FOLDER_TO_CLASS, file_sha256, perceptual_dhash
from src.multi_dataset import (
    DATASET_SOURCES,
    MASTER_COLUMNS,
    ImageRecord,
    discover_all,
    discover_source,
    hamming_distance,
    inventory_rows,
    make_image_id,
    probe,
    read_master_manifest,
    reuse_probe,
    write_master_manifest,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from build_master_dataset import (  # noqa: E402
    apply_label_mapping,
    duplicate_pairs,
    flag_conflicts_and_actions,
)
from build_split_manifest import assign_duplicate_groups  # noqa: E402


def _image(path: Path, colour: tuple[int, int, int], size=(32, 24), seed=0) -> None:
    rng = np.random.default_rng(seed)
    base = np.array(colour, dtype=np.int16)
    noise = rng.integers(-8, 9, size=(size[1], size[0], 3), dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8)).save(path, quality=95)


@pytest.fixture
def delivery(tmp_path: Path) -> Path:
    """A tiny delivery drop with every delivered layout, plus planted duplicates:
    - current/train_split/EUS/a.jpg == roboflow/train/.../dup.jpg (byte copy, cross-dataset)
    - kaptai/Redspot/near.png is a PNG re-encode of the same picture (near duplicate)
    - mendeley has metadata.csv agreeing with its folder path
    - one corrupt file in paper_dataset
    """
    drop = tmp_path / "drop"
    cur = drop
    _image(cur / "train_split" / "EUS" / "a.jpg", (200, 40, 40), seed=1)
    _image(cur / "train_split" / "Healthy Fish" / "h.jpg", (40, 200, 40), seed=2)
    _image(cur / "test_split" / "EUS_EUS_9.jpg", (200, 40, 40), seed=3)
    (cur / "test.csv").write_text("filename,label\nEUS_EUS_9.jpg,EUS\n")

    rob = drop / "Fish Disease.v1i.folder"
    (rob / "train" / "Healthy Fish").mkdir(parents=True)
    shutil.copy(cur / "train_split" / "EUS" / "a.jpg", rob / "train" / "Healthy Fish" / "dup.jpg")
    _image(rob / "valid" / "Bacterial Red disease" / "r.jpg", (150, 30, 30), seed=4)
    (rob / "README.roboflow.txt").write_text("Fish Disease - v1\n")

    kap = drop / "Fresh Water Fish Dataset"
    with Image.open(cur / "train_split" / "EUS" / "a.jpg") as im:
        (kap / "Redspot").mkdir(parents=True)
        im.save(kap / "Redspot" / "near.png")
    _image(kap / "Argulus" / "p.jpg", (90, 90, 200), seed=5)

    men = drop / "MatsyaDx-BD An image dataset of freshwater fish di" / "MatsyaDx-BD"
    rel = "Healthy Fish/Rui_(Labeo rohita)/Fish_1/IMG1.jpg"
    _image(men / rel, (40, 200, 40), seed=6)
    (men / "metadata.csv").write_text(
        "image_id,image_path,health_condition,fish_category,specimen_id\n"
        f"IMG1.jpg,{rel},Healthy Fish,Rui_(Labeo rohita),Fish_1\n"
    )
    (men / "Healthy Fish.7z").write_bytes(b"7z")

    sal = (
        drop
        / "SalmonScan A Novel Image Dataset for Fish Disease Detection in Salmon Aquaculture System"
        / "SalmonScan"
    )
    _image(sal / "FreshFish" / "fresh_0.png", (120, 120, 120), seed=7)
    (sal / "InfectedFish").mkdir(parents=True)
    (sal / "InfectedFish" / "broken.png").write_bytes(b"not an image")
    return drop


def _link(drop: Path, repo: Path) -> Path:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "link_raw_datasets.py"),
            "--source",
            str(drop),
            "--data-dir",
            str(repo / "data"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return repo / "data" / "raw"


# --- registry and mapping table ---


def test_registry_covers_the_five_agreed_keys():
    assert [s.key for s in DATASET_SOURCES] == [
        "current_freshwater",
        "kaptai",
        "roboflow",
        "mendeley",
        "paper_dataset",
    ]


def test_mapping_table_is_consistent_with_the_project_classes():
    validate_mapping_table()
    for m in LABEL_MAPPING:
        if m.mapping_status in ("EXACT_MATCH", "SUPPORTED_MAPPING"):
            assert m.unified_class in CANONICAL_CLASSES
        else:
            assert m.unified_class == ""
        assert m.reason and m.evidence
    # the current dataset keeps the project's own mapping, unchanged
    for folder, name in SOURCE_FOLDER_TO_CLASS.items():
        m = lookup("current_freshwater", folder)
        assert m is not None and m.unified_class == name and m.mapping_status == "EXACT_MATCH"
    # similar wording is not treated as equivalence
    assert lookup("kaptai", "THE BACTERIAL GILL ROT").mapping_status == "UNRESOLVED"
    assert lookup("kaptai", "Redspot").mapping_status == "UNRESOLVED"
    assert lookup("paper_dataset", "InfectedFish").mapping_status == "UNRESOLVED"
    assert lookup("kaptai", "Broken antennae and rostrum").mapping_status == "EXCLUDED"
    assert lookup("nowhere", "EUS") is None


def test_label_mapping_rejects_inconsistent_rows():
    with pytest.raises(ValueError):
        LabelMapping("x", "y", "Not A Class", "EXACT_MATCH", "r", "e")
    with pytest.raises(ValueError):
        LabelMapping("x", "y", "EUS Disease", "UNRESOLVED", "r", "e")
    with pytest.raises(ValueError):
        LabelMapping("x", "y", "", "MAYBE", "r", "e")


def test_label_mapping_csv_round_trip(tmp_path):
    path = tmp_path / "label_mapping.csv"
    write_label_mapping_csv(path)
    with path.open() as handle:
        assert tuple(next(csv.reader(handle))) == LABEL_MAPPING_COLUMNS
    assert read_label_mapping_csv(path) == list(LABEL_MAPPING)


# --- linking and discovery ---


def test_link_script_is_idempotent_and_handles_the_renamed_test_folder(delivery, tmp_path):
    repo = tmp_path / "repo"
    (delivery / "test_split").rename(delivery / "test_split&validation")
    raw = _link(delivery, repo)
    assert (raw / "current_freshwater" / "test_split").resolve() == (
        delivery / "test_split&validation"
    ).resolve()
    assert (raw / "kaptai").is_symlink() and (raw / "mendeley").is_symlink()
    before = {p: p.resolve() for p in raw.rglob("*") if p.is_symlink()}
    _link(delivery, repo)  # second run: nothing changes
    assert {p: p.resolve() for p in raw.rglob("*") if p.is_symlink()} == before


def test_discovery_preserves_provenance_and_reports_non_images(delivery, tmp_path):
    repo = tmp_path / "repo"
    raw = _link(delivery, repo)
    records, others = discover_all(raw, repo)
    assert len(records) == 10
    by_id = {r.image_id: r for r in records}
    assert len(by_id) == 10
    cur = [r for r in records if r.source_dataset == "current_freshwater"]
    assert {(r.original_split, r.original_class) for r in cur} == {
        ("train_split", "EUS"),
        ("train_split", "Healthy Fish"),
        ("test_split", "EUS"),  # label from test.csv, not from a folder
    }
    rob = next(r for r in records if r.original_filename == "dup.jpg")
    assert rob.source_dataset == "roboflow" and rob.original_split == "train"
    assert rob.original_path == "train/Healthy Fish/dup.jpg"
    assert rob.filepath == "data/raw/roboflow/train/Healthy Fish/dup.jpg"
    men = next(r for r in records if r.source_dataset == "mendeley")
    assert (men.species, men.specimen_id) == ("Rui_(Labeo rohita)", "Fish_1")
    assert all(r.unified_class == "" and r.status == "pending" for r in records)
    kinds = {(o.source_dataset, o.original_path): o.kind for o in others}
    assert kinds[("current_freshwater", "test.csv")] == "documentation"
    assert kinds[("roboflow", "README.roboflow.txt")] == "documentation"
    assert kinds[("mendeley", "metadata.csv")] == "documentation"
    assert kinds[("mendeley", "Healthy Fish.7z")] == "archive"


def test_matsyadx_metadata_disagreement_is_an_error(delivery, tmp_path):
    repo = tmp_path / "repo"
    raw = _link(delivery, repo)
    meta = raw / "mendeley" / "metadata.csv"
    meta.write_text(meta.read_text().replace("Fish_1\n", "Fish_9\n"))
    with pytest.raises(ValueError, match="metadata.csv disagrees"):
        discover_source(DATASET_SOURCES[3], raw, repo)


def test_image_id_is_deterministic_and_path_specific():
    assert make_image_id("kaptai", "a/b.jpg") == make_image_id("kaptai", "a/b.jpg")
    assert make_image_id("kaptai", "a/b.jpg") != make_image_id("kaptai", "a/c.jpg")
    assert make_image_id("kaptai", "a/b.jpg").startswith("kaptai-")


# --- probing and hashing (reused implementations) ---


def test_probe_reuses_manifest_hashes_and_flags_corrupt_files(delivery, tmp_path):
    repo = tmp_path / "repo"
    raw = _link(delivery, repo)
    records, _ = discover_all(raw, repo)
    for r in records:
        probe(r, repo)
    ok = [r for r in records if r.status == "ok"]
    bad = [r for r in records if r.status == "corrupt"]
    assert len(bad) == 1 and bad[0].original_filename == "broken.png"
    assert bad[0].sha256 and bad[0].dhash == "" and bad[0].width == 0
    sample = ok[0]
    assert sample.sha256 == file_sha256(repo / sample.filepath)
    with Image.open(repo / sample.filepath) as image:
        assert sample.dhash == perceptual_dhash(image.convert("RGB"))
    assert (sample.width, sample.height, sample.mode) == (32, 24, "RGB")


def test_master_manifest_round_trip_and_reuse(delivery, tmp_path):
    repo = tmp_path / "repo"
    raw = _link(delivery, repo)
    records, _ = discover_all(raw, repo)
    for r in records:
        probe(r, repo)
    path = repo / "data" / "audit" / "master_dataset.csv"
    write_master_manifest(records, path)
    with path.open() as handle:
        assert tuple(next(csv.reader(handle))) == MASTER_COLUMNS
    again = read_master_manifest(path)
    assert again == records
    fresh, _ = discover_all(raw, repo)
    previous = {r.filepath: r for r in again}
    assert all(reuse_probe(r, previous) for r in fresh)
    assert fresh == records
    fresh[0].file_size += 1  # a changed file is re-probed, not trusted
    assert not reuse_probe(fresh[0], previous)


def test_hamming_distance():
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "000000000000000f") == 4
    with pytest.raises(ValueError):
        hamming_distance("00", "0000")


# --- duplicate analysis on the reused grouping helper ---


def _analysed(delivery: Path, repo: Path) -> list[ImageRecord]:
    raw = _link(delivery, repo)
    records, _ = discover_all(raw, repo)
    apply_label_mapping(records)
    for r in records:
        probe(r, repo)
    assign_duplicate_groups(records)  # type: ignore[arg-type]
    flag_conflicts_and_actions(records)
    return records


def test_cross_dataset_exact_and_near_duplicates_are_reported(delivery, tmp_path):
    records = _analysed(delivery, tmp_path / "repo")
    pairs = duplicate_pairs(records)
    exact = [p for p in pairs if p["duplicate_status"] == "EXACT_DUPLICATE"]
    near = [p for p in pairs if p["duplicate_status"] == "NEAR_DUPLICATE"]
    assert len(exact) == 1
    (e,) = exact
    assert (e["source_dataset_a"], e["source_dataset_b"]) == ("current_freshwater", "roboflow")
    assert e["cross_dataset"] is True
    assert e["original_path_a"] == "train_split/EUS/a.jpg"
    assert e["original_path_b"] == "train/Healthy Fish/dup.jpg"
    assert e["hash_type"] == "sha256" and e["similarity_or_distance"] == 0
    # EUS vs Healthy Fish: byte-identical files with conflicting unified labels
    assert e["label_relation"] == "LABEL_CONFLICT" and e["recommended_action"] == "EXCLUDE_BOTH"
    # the PNG re-encode shares the dHash but not the bytes; its label is unresolved
    assert {p["hash_type"] for p in near} == {"dhash"}
    assert any(
        p["source_dataset_b"] == "kaptai"
        and p["label_relation"] == "UNRESOLVED_LABEL"
        and p["recommended_action"] == "REVIEW"
        for p in near
    )
    for p in pairs:
        assert p["image_a"] != p["image_b"]
        assert p["original_path_a"] and p["original_path_b"]


def test_dedup_actions_follow_the_established_policy_without_touching_files(delivery, tmp_path):
    repo = tmp_path / "repo"
    before = sorted(p.stat().st_size for p in delivery.rglob("*") if p.is_file())
    records = _analysed(delivery, repo)
    actions = {r.original_filename: r.dedup_action for r in records}
    assert actions["a.jpg"] == "EXCLUDE_LABEL_CONFLICT"
    assert actions["dup.jpg"] == "EXCLUDE_LABEL_CONFLICT"
    assert actions["near.png"] == "REVIEW_LABEL_CONFLICT"  # unresolved member of that group
    assert actions["broken.png"] == "CORRUPT"
    assert actions["h.jpg"] in ("KEEP", "KEEP_GROUPED")
    assert sorted(p.stat().st_size for p in delivery.rglob("*") if p.is_file()) == before


def test_exact_duplicates_without_conflict_collapse_to_the_first_path(delivery, tmp_path):
    repo = tmp_path / "repo"
    # a byte copy of the current Healthy image inside roboflow's Healthy folder: same label
    shutil.copy(
        delivery / "train_split" / "Healthy Fish" / "h.jpg",
        delivery / "Fish Disease.v1i.folder" / "train" / "Healthy Fish" / "h_copy.jpg",
    )
    records = _analysed(delivery, repo)
    actions = {r.original_filename: r.dedup_action for r in records}
    assert actions["h.jpg"] == "KEEP"  # current_freshwater sorts first -> representative
    assert actions["h_copy.jpg"] == "DROP_EXACT_DUPLICATE"
    pair = next(p for p in duplicate_pairs(records) if p["original_path_b"].endswith("h_copy.jpg"))
    assert pair["recommended_action"] == "DROP_B_KEEP_A" and pair["label_relation"] == "SAME_LABEL"


def test_unmapped_class_is_reported_not_guessed(delivery, tmp_path):
    (delivery / "Fresh Water Fish Dataset" / "Mystery Disease").mkdir()
    _image(delivery / "Fresh Water Fish Dataset" / "Mystery Disease" / "m.jpg", (1, 2, 3))
    raw = _link(delivery, tmp_path / "repo")
    records, _ = discover_all(raw, tmp_path / "repo")
    with pytest.raises(ValueError, match="kaptai/Mystery Disease"):
        apply_label_mapping(records)


def test_inventory_counts_from_records(delivery, tmp_path):
    repo = tmp_path / "repo"
    records = _analysed(delivery, repo)
    rows = inventory_rows(records, repo / "data" / "raw")
    cur = {
        (r["split_or_structure"], r["original_class"]): r
        for r in rows
        if r["dataset"] == "current_freshwater"
    }
    assert cur[("train_split", "EUS")]["image_count"] == 1
    assert cur[("test_split", "EUS")]["image_count"] == 1
    men = next(r for r in rows if r["dataset"] == "mendeley")
    assert men["species"] == "Rui_(Labeo rohita)" and men["specimens"] == 1
    sal = next(
        r for r in rows if r["dataset"] == "paper_dataset" and r["original_class"] == "InfectedFish"
    )
    assert sal["corrupt"] == 1 and sal["image_count"] == 1


# --- end-to-end script on the synthetic drop ---


def test_build_master_dataset_script_end_to_end(delivery, tmp_path):
    repo = tmp_path / "repo"
    drop_files = sorted(p.relative_to(delivery) for p in delivery.rglob("*"))
    _link(delivery, repo)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_master_dataset.py"),
            "--data-dir",
            str(repo / "data"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    audit = repo / "data" / "audit"
    for name in (
        "dataset_inventory.csv",
        "dataset_inventory.md",
        "non_image_files.csv",
        "label_mapping.csv",
        "master_dataset.csv",
        "duplicate_report.csv",
        "duplicate_report.md",
    ):
        assert (audit / name).is_file(), name
    manifest = read_master_manifest(audit / "master_dataset.csv")
    assert len(manifest) == 10
    assert {r.mapping_status for r in manifest} >= {"EXACT_MATCH", "UNRESOLVED"}
    with (audit / "duplicate_report.csv").open() as handle:
        pairs = list(csv.DictReader(handle))
    assert any(
        p["duplicate_status"] == "EXACT_DUPLICATE" and p["cross_dataset"] == "True" for p in pairs
    )
    # restartable: a second run reuses the hashes and produces the same manifest
    second = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_master_dataset.py"),
            "--data-dir",
            str(repo / "data"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0 and "reusing probe results" in second.stderr
    assert read_master_manifest(audit / "master_dataset.csv") == manifest
    # nothing was written into the delivery drop
    assert sorted(p.relative_to(delivery) for p in delivery.rglob("*")) == drop_files
