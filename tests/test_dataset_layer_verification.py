"""Dataset-cleaning layer verification (teacher pipeline: load -> preprocess -> leakage ->
CLAHE): five sources discovered, deterministic hashes and duplicate groups, clean manifest
invariants, ambiguous labels never accepted, raw files untouched by CLAHE, manifest
regeneration reproducible. Real-data checks run against the committed manifests and the
linked raw datasets when present."""

from __future__ import annotations

import csv
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from PIL import Image

from src.dataset import load_image
from src.dataset_cleaning import (
    CLEAN_MANIFEST_NAME,
    read_clean_manifest,
    validate_clean_manifest,
)
from src.label_harmonization import (
    LABEL_MAPPING,
    REVIEW_STATUS,
    read_label_mapping_csv,
    write_label_mapping_csv,
)
from src.manifest import file_sha256, perceptual_dhash
from src.multi_dataset import (
    DATASET_SOURCES,
    MASTER_MANIFEST_NAME,
    discover_all,
    probe,
    read_master_manifest,
)
from src.preprocessing import CLAHE, CLAHEConfig, PreprocessConfig, build_eval_transform
from tests.test_multi_dataset import _link, delivery  # noqa: F401  (fixture re-export)

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "data" / "audit"
RAW = ROOT / "data" / "raw"
sys.path.insert(0, str(ROOT / "scripts"))
from build_split_manifest import assign_duplicate_groups  # noqa: E402

real_data = pytest.mark.skipif(
    not (AUDIT / MASTER_MANIFEST_NAME).is_file() or not (RAW / "mendeley").exists(),
    reason="real manifests or raw links absent",
)


# --- synthetic: determinism and reproducibility ---


def test_hashes_and_duplicate_groups_are_deterministic(delivery, tmp_path):  # noqa: F811
    repo = tmp_path / "repo"
    raw = _link(delivery, repo)
    first, _ = discover_all(raw, repo)
    second, _ = discover_all(raw, repo)
    for r in first + second:
        probe(r, repo)
    assert [(r.image_id, r.sha256, r.dhash) for r in first] == [
        (r.image_id, r.sha256, r.dhash) for r in second
    ]
    assign_duplicate_groups(first)  # type: ignore[arg-type]
    assign_duplicate_groups(second)  # type: ignore[arg-type]
    assert [(r.exact_dup_group, r.near_dup_group) for r in first] == [
        (r.exact_dup_group, r.near_dup_group) for r in second
    ]
    shuffled = list(first)
    random.Random(3).shuffle(shuffled)
    assign_duplicate_groups(shuffled)  # type: ignore[arg-type]
    assert {r.image_id: (r.exact_dup_group, r.near_dup_group) for r in shuffled} == {
        r.image_id: (r.exact_dup_group, r.near_dup_group) for r in first
    }


def test_master_manifest_regenerates_identically(delivery, tmp_path):  # noqa: F811
    repo = tmp_path / "repo"
    _link(delivery, repo)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_master_dataset.py"),
        "--data-dir",
        str(repo / "data"),
        "--repo-root",
        str(repo),
        "--no-reuse",
    ]
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    first = (repo / "data" / "audit" / MASTER_MANIFEST_NAME).read_bytes()
    assert subprocess.run(cmd, capture_output=True, text=True).returncode == 0
    assert (repo / "data" / "audit" / MASTER_MANIFEST_NAME).read_bytes() == first


def test_ambiguous_labels_are_never_accepted_and_review_status_is_written(tmp_path):
    path = tmp_path / "label_mapping.csv"
    write_label_mapping_csv(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert {r["review_status"] for r in rows} == {
        "accepted",
        "ambiguous (requires review)",
        "excluded",
    }
    for r in rows:
        assert r["review_status"] == REVIEW_STATUS[r["mapping_status"]]
        if r["review_status"] != "accepted":
            assert r["unified_class"] == ""  # never silently mapped
    assert read_label_mapping_csv(path) == list(LABEL_MAPPING)
    kaptai = {
        r["original_class"]: r["review_status"] for r in rows if r["source_dataset"] == "kaptai"
    }
    assert kaptai["THE BACTERIAL GILL ROT"].startswith("ambiguous")
    assert kaptai["Redspot"].startswith("ambiguous")
    assert kaptai["Broken antennae and rostrum"] == "excluded"


def test_clahe_does_not_modify_raw_files(tmp_path):
    path = tmp_path / "raw.jpg"
    Image.new("RGB", (64, 48), (120, 80, 60)).save(path, quality=90)
    before = file_sha256(path)
    image = load_image(path)
    enhanced = CLAHE(CLAHEConfig())(image)
    tensor = build_eval_transform(PreprocessConfig(clahe=CLAHEConfig()))(image)
    assert enhanced.size == image.size and tensor.shape == (3, 224, 224)
    assert file_sha256(path) == before  # untouched on disk


# --- real data: the committed manifests against the linked corpus ---


@real_data
def test_all_five_sources_discovered_and_counts_match_manifest():
    master = read_master_manifest(AUDIT / MASTER_MANIFEST_NAME)
    counts = Counter(r.source_dataset for r in master)
    assert set(counts) == {s.key for s in DATASET_SOURCES}
    assert counts == {
        "current_freshwater": 3503,
        "kaptai": 133,
        "roboflow": 454,
        "mendeley": 2137,
        "paper_dataset": 1208,
    }
    assert all((ROOT / r.filepath).is_file() for r in master)  # every path resolves


@real_data
def test_raw_files_unchanged_sample_rehash():
    """Re-hash a seeded sample of raw files (all sources) and compare to the manifest."""
    master = read_master_manifest(AUDIT / MASTER_MANIFEST_NAME)
    rng = random.Random(42)
    sample = []
    for key in {r.source_dataset for r in master}:
        mine = [r for r in master if r.source_dataset == key]
        sample += rng.sample(mine, min(12, len(mine)))
    for r in sample:
        path = ROOT / r.filepath
        assert file_sha256(path) == r.sha256, r.filepath
        with Image.open(path) as image:
            assert perceptual_dhash(image.convert("RGB")) == r.dhash, r.filepath


@real_data
def test_clean_manifest_excludes_duplicate_copies_and_keeps_groups_indivisible():
    clean = read_clean_manifest(AUDIT / CLEAN_MANIFEST_NAME)
    validate_clean_manifest(clean)
    included = [r for r in clean if r.included]
    assert len(clean) == 7435 and len(included) == 5942
    by_sha = defaultdict(list)
    for r in clean:
        by_sha[r.sha256].append(r)
    for rows in by_sha.values():  # every exact group: at most one included copy
        assert sum(r.included for r in rows) <= 1
    excluded_dups = [r for r in clean if r.exclusion_reason == "EXACT_DUPLICATE"]
    assert len(excluded_dups) == 162
    ids = {r.image_id: r for r in clean}
    for r in excluded_dups:
        rep = ids[r.representative_image_id]
        assert rep.included and rep.sha256 == r.sha256 and rep.group_id == r.group_id
    # a group carries one label and is the indivisible unit for any split
    by_group = defaultdict(set)
    for r in included:
        by_group[r.group_id].add(r.unified_class)
    assert all(len(v) == 1 for v in by_group.values())
    assert Counter(r.exclusion_reason for r in clean if not r.included) == {
        "LABEL_UNRESOLVED": 1277,
        "EXACT_DUPLICATE": 162,
        "LABEL_CONFLICT_GROUP": 47,
        "LABEL_EXCLUDED": 7,
    }
    assert all(r.leakage_flags != "" or r.group_size == 1 for r in included)


@real_data
def test_duplicate_pairs_never_straddle_group_boundaries():
    clean = {r.image_id: r for r in read_clean_manifest(AUDIT / CLEAN_MANIFEST_NAME)}
    with (AUDIT / "duplicate_report.csv").open() as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 1405
    assert all(clean[p["image_a"]].group_id == clean[p["image_b"]].group_id for p in pairs)
