"""Clean manifest (exclusion policy as annotations), de-duplication summary,
leakage audit and the clean-manifest script — on hand-built master records."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.dataset_cleaning import (
    CLEAN_COLUMNS,
    EXCLUSION_REASONS,
    build_clean_manifest,
    dedup_summary,
    leakage_groups,
    read_clean_manifest,
    validate_clean_manifest,
    write_clean_manifest,
)
from src.leakage_audit import (
    duplicate_leakage,
    filename_label_tokens,
    leakage_report,
    shortcut_predictability,
    target_leakage,
)
from src.multi_dataset import ImageRecord, write_master_manifest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_clean_manifest.py"


def rec(
    image_id: str,
    dataset: str,
    path: str,
    original_class: str,
    unified: str,
    status: str = "EXACT_MATCH",
    *,
    sha: str,
    dhash: str,
    near: str = "",
    exact: str = "",
    specimen: str = "",
    split: str = "",
    size: tuple[int, int] = (64, 64),
    ok: bool = True,
) -> ImageRecord:
    return ImageRecord(
        image_id=image_id,
        source_dataset=dataset,
        filepath=f"data/raw/{dataset}/{path}",
        original_path=path,
        original_filename=path.rsplit("/", 1)[-1],
        original_class=original_class,
        original_split=split,
        species="Rui" if specimen else "",
        specimen_id=specimen,
        extension=".jpg",
        unified_class=unified,
        mapping_status=status,
        width=size[0],
        height=size[1],
        mode="RGB",
        file_size=100,
        sha256=sha,
        dhash=dhash,
        status="ok" if ok else "corrupt",
        exact_dup_group=exact,
        near_dup_group=near,
    )


def build_master_records() -> list[ImageRecord]:
    """Hand-built master rows covering every exclusion path:
    - a1/r1: byte-identical, same label, two datasets  -> r1 EXACT_DUPLICATE of a1
    - a2/r2: byte-identical but EUS vs Healthy          -> both LABEL_CONFLICT_GROUP
    - a3/k1: same dHash, k1 label UNRESOLVED            -> k1 LABEL_UNRESOLVED, a3 kept (grouped)
    - m1/m2: same specimen, different hashes            -> one leakage group
    - k2: EXCLUDED class; c1: corrupt
    """
    return [
        rec(
            "cur-a1",
            "current_freshwater",
            "train_split/EUS/a1.jpg",
            "EUS",
            "EUS Disease",
            sha="s1",
            dhash="h1",
            near="g1",
            exact="x1",
            split="train_split",
        ),
        rec(
            "rob-r1",
            "roboflow",
            "train/EUS/r1.jpg",
            "EUS",
            "EUS Disease",
            sha="s1",
            dhash="h1",
            near="g1",
            exact="x1",
            split="train",
        ),
        rec(
            "cur-a2",
            "current_freshwater",
            "train_split/EUS/a2.jpg",
            "EUS",
            "EUS Disease",
            sha="s2",
            dhash="h2",
            near="g2",
            exact="x2",
        ),
        rec(
            "rob-r2",
            "roboflow",
            "test/Healthy Fish/r2.jpg",
            "Healthy Fish",
            "Healthy Fish",
            sha="s2",
            dhash="h2",
            near="g2",
            exact="x2",
            split="test",
        ),
        rec(
            "cur-a3",
            "current_freshwater",
            "train_split/EUS/EUS_aug_3.jpg",
            "EUS",
            "EUS Disease",
            sha="s3",
            dhash="h3",
            near="g3",
        ),
        rec(
            "kap-k1",
            "kaptai",
            "Redspot/EUS (3).jpg",
            "Redspot",
            "",
            "UNRESOLVED",
            sha="s4",
            dhash="h3",
            near="g3",
        ),
        rec(
            "men-m1",
            "mendeley",
            "Healthy Fish/Rui/Fish_1/IMG1.jpg",
            "Healthy Fish",
            "Healthy Fish",
            sha="s5",
            dhash="h5",
            specimen="Fish_1",
            size=(4000, 3000),
        ),
        rec(
            "men-m2",
            "mendeley",
            "Healthy Fish/Rui/Fish_1/IMG2.jpg",
            "Healthy Fish",
            "Healthy Fish",
            sha="s6",
            dhash="h6",
            specimen="Fish_1",
            size=(4000, 3000),
        ),
        rec(
            "kap-k2",
            "kaptai",
            "Broken antennae and rostrum/b.jpg",
            "Broken antennae and rostrum",
            "",
            "EXCLUDED",
            sha="s7",
            dhash="h7",
        ),
        rec(
            "cur-c1",
            "current_freshwater",
            "train_split/EUS/c1.jpg",
            "EUS",
            "EUS Disease",
            sha="s8",
            dhash="",
            ok=False,
        ),
        rec(
            "cur-a4",
            "current_freshwater",
            "train_split/Healthy Fish/a4.jpg",
            "Healthy Fish",
            "Healthy Fish",
            sha="s9",
            dhash="h9",
        ),
    ]


@pytest.fixture
def master() -> list[ImageRecord]:
    return build_master_records()


def test_clean_manifest_applies_the_policy_as_flags(master):
    rows = {r.image_id: r for r in build_clean_manifest(master)}
    assert rows["cur-a1"].included and rows["cur-a1"].exclusion_reason == ""
    assert rows["rob-r1"].exclusion_reason == "EXACT_DUPLICATE"
    assert rows["rob-r1"].representative_image_id == "cur-a1"  # first path wins
    assert rows["cur-a2"].exclusion_reason == "LABEL_CONFLICT_GROUP"
    assert rows["rob-r2"].exclusion_reason == "LABEL_CONFLICT_GROUP"
    assert rows["cur-a2"].representative_image_id == ""  # no label chosen
    assert rows["cur-a3"].included  # near-duplicate of an unresolved image: kept, grouped
    assert rows["kap-k1"].exclusion_reason == "LABEL_UNRESOLVED"
    assert rows["kap-k2"].exclusion_reason == "LABEL_EXCLUDED"
    assert rows["cur-c1"].exclusion_reason == "CORRUPT"
    assert rows["men-m1"].included and rows["men-m2"].included
    assert all(r.exclusion_reason in EXCLUSION_REASONS + ("",) for r in rows.values())
    included = [r for r in rows.values() if r.included]
    assert {r.image_id for r in included} == {"cur-a1", "cur-a3", "men-m1", "men-m2", "cur-a4"}


def test_leakage_groups_join_near_duplicates_and_specimens(master):
    groups = leakage_groups(master)
    assert groups["cur-a1"] == groups["rob-r1"]  # exact copies
    assert groups["cur-a3"] == groups["kap-k1"]  # same dHash
    assert groups["men-m1"] == groups["men-m2"]  # same specimen, different pictures
    assert groups["men-m1"] != groups["cur-a4"]
    assert groups["cur-a4"] == "cur-a4"  # singleton keeps its own id
    rows = {r.image_id: r for r in build_clean_manifest(master)}
    assert rows["men-m1"].group_size == 2 and rows["cur-a4"].group_size == 1


def test_collapse_near_duplicates_is_opt_in(master):
    master = master + [
        rec(
            "rob-r5",
            "roboflow",
            "train/EUS/r5.jpg",
            "EUS",
            "EUS Disease",
            sha="s10",
            dhash="h3",
            near="g3",
        )
    ]
    default = {r.image_id: r for r in build_clean_manifest(master)}
    assert default["rob-r5"].included
    collapsed = {r.image_id: r for r in build_clean_manifest(master, collapse_near_duplicates=True)}
    assert collapsed["rob-r5"].exclusion_reason == "NEAR_DUPLICATE"
    assert collapsed["rob-r5"].representative_image_id == "cur-a3"


def test_representative_is_the_first_eligible_copy(master):
    # the current-dataset copy is unresolved, so the roboflow copy must be kept instead
    master = [r for r in master if r.image_id not in ("cur-a1", "rob-r1")] + [
        rec(
            "cur-u1",
            "current_freshwater",
            "train_split/EUS/u1.jpg",
            "EUS",
            "",
            "UNRESOLVED",
            sha="s1",
            dhash="h1",
            near="g1",
            exact="x1",
        ),
        rec(
            "rob-r1",
            "roboflow",
            "train/EUS/r1.jpg",
            "EUS",
            "EUS Disease",
            sha="s1",
            dhash="h1",
            near="g1",
            exact="x1",
        ),
    ]
    rows = {r.image_id: r for r in build_clean_manifest(master)}
    assert rows["rob-r1"].included and rows["cur-u1"].exclusion_reason == "LABEL_UNRESOLVED"


def test_validate_rejects_broken_invariants(master):
    rows = build_clean_manifest(master)
    rows[0].included = False
    with pytest.raises(ValueError, match="without a reason"):
        validate_clean_manifest(rows)
    rows = build_clean_manifest(master)
    dup = next(r for r in rows if r.exclusion_reason == "EXACT_DUPLICATE")
    dup.included, dup.exclusion_reason = True, ""
    with pytest.raises(ValueError, match="share SHA-256"):
        validate_clean_manifest(rows)


def test_clean_manifest_csv_round_trip(master, tmp_path):
    rows = build_clean_manifest(master)
    path = tmp_path / "clean_manifest.csv"
    write_clean_manifest(rows, path)
    with path.open() as handle:
        assert tuple(next(csv.reader(handle))) == CLEAN_COLUMNS
    assert read_clean_manifest(path) == rows


def test_dedup_summary_counts_before_and_after(master):
    rows = build_clean_manifest(master)
    s = dedup_summary(master, rows, collapse_near_duplicates=False)
    assert s["raw"]["total"] == 11 and s["raw"]["corrupt"] == 1
    assert s["clean"]["total"] == 5
    assert s["excluded"]["total"] == 6
    assert s["excluded"]["by_reason"] == {
        "EXACT_DUPLICATE": 1,
        "LABEL_CONFLICT_GROUP": 2,
        "LABEL_UNRESOLVED": 1,
        "LABEL_EXCLUDED": 1,
        "CORRUPT": 1,
    }
    assert s["duplicates"]["exact_groups"] == 2
    assert s["duplicates"]["exact_groups_cross_dataset"] == 2
    assert s["duplicates"]["near_groups_distinct_bytes"] == 1  # g3 (s3 vs s4)
    assert s["duplicates"]["conflicting_label_groups"] == 1
    assert s["clean"]["per_class"] == {"EUS Disease": 2, "Healthy Fish": 3}
    assert s["clean"]["per_dataset"] == {"current_freshwater": 3, "mendeley": 2}
    assert s["unresolved"]["per_dataset_and_class"] == {
        "kaptai/Redspot": 1,
        "kaptai/Broken antennae and rostrum": 1,
    }
    assert {m["image_id"] for m in s["unresolved"]["conflict_group_members"]} == {
        "cur-a2",
        "rob-r2",
    }
    json.dumps(s)  # machine-readable


# --- leakage audit ---


def test_filename_label_tokens_and_target_leakage(master):
    rows = {r.image_id: r for r in build_clean_manifest(master)}
    assert "eus" in filename_label_tokens(rows["cur-a3"])
    assert "<original class name>" in filename_label_tokens(rows["cur-a3"])
    assert filename_label_tokens(rows["men-m1"]) == []
    report = target_leakage(list(rows.values()))
    assert report["per_dataset"]["mendeley"]["filename_reveals_label"] == 0
    assert report["per_dataset"]["current_freshwater"]["filename_reveals_label"] == 1


def test_duplicate_leakage_measures_groups(master):
    d = duplicate_leakage(build_clean_manifest(master))
    assert d["included_images"] == 5
    assert d["multi_image_groups"] == 1  # men-m1 + men-m2 (a3's partner k1 is excluded)
    assert d["specimen_groups"] == 1
    assert d["included_group_with_mixed_labels"] == 0
    assert d["excluded_exact_copies"] == 1


def test_shortcut_predictability_bounds(master):
    rows = [r for r in build_clean_manifest(master) if r.included]
    res = shortcut_predictability(rows, lambda r: f"{r.width}x{r.height}")
    # 4000x3000 -> Healthy (2/2), 64x64 -> 2 EUS + 1 Healthy -> majority EUS: (2+2)/5
    assert res["feature_only_accuracy_upper_bound"] == 0.8
    assert res["majority_class_baseline"] == 0.6
    constant = shortcut_predictability(rows, lambda r: "same")
    assert constant["feature_only_accuracy_upper_bound"] == constant["majority_class_baseline"]
    full = leakage_report(build_clean_manifest(master))
    assert set(full) == {"target_leakage", "duplicate_leakage", "suspicious_features"}
    json.dumps(full)


# --- script ---


def test_clean_manifest_script_writes_every_artifact(master, tmp_path):
    audit = tmp_path / "data" / "audit"
    audit.mkdir(parents=True)
    write_master_manifest(master, audit / "master_dataset.csv")
    configs = tmp_path / "configs"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir",
            str(tmp_path / "data"),
            "--repo-root",
            str(tmp_path),
            "--configs-dir",
            str(configs),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for name in ("clean_manifest.csv", "dedup_report.json", "dedup_report.md"):
        assert (audit / name).is_file(), name
    rows = read_clean_manifest(audit / "clean_manifest.csv")
    assert sum(r.included for r in rows) == 5
    pre = json.loads((configs / "preprocess_v2_clahe.json").read_text())["preprocess"]
    assert pre == {
        "image_size": 224,
        "resize_size": 256,
        "resize_mode": "crop",
        "clahe": {"clip_limit": 2.0, "tile_grid_size": 8},
    }
    text = (audit / "dedup_report.md").read_text()
    assert "| **total** | 11 | 6 | **5** |" in text
