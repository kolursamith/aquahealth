"""Layer 3 — GAN_MANIFEST.csv, the output verifier, the all-folds driver and the
CUDA-only rule, on the tiny synthetic fixture (CPU, 16x16, 1 epoch).

Requirement -> test:
    class mapping                 test_class_mapping_label_index_equals_class
    manifest consistency          test_gan_manifest_lists_every_synthetic_image_with_provenance
    verifier proves isolation     test_verifier_passes_on_clean_outputs_and_catches_tampering
    fold isolation / resume       test_all_folds_driver_generates_skips_complete_and_verifies
    raw-data protection           test_verifier_detects_changed_training_source
    CUDA-only real training       test_require_cuda_refuses_cpu_and_mps
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.gan_augmentation import (
    GAN_MANIFEST_COLUMNS,
    N_FOLDS,
    build_gan_manifest,
    completed_folds,
    parse_folds,
    read_gan_manifest,
    read_registry,
    registry_row,
    update_registry,
    validate_synthetic_rows,
    verify_gan_outputs,
    write_gan_manifest,
)
from src.manifest import CANONICAL_CLASSES
from tests.test_gan_augmentation import TINY, fold_setup  # noqa: F401  (fixture re-export)
from tests.test_gan_isolation import _run_fold

ROOT = Path(__file__).resolve().parent.parent
ALL_FOLDS = ROOT / "scripts" / "run_gan_all_folds.py"
ONE_FOLD = ROOT / "scripts" / "run_gan_fold.py"
VERIFY = ROOT / "scripts" / "verify_gan_outputs.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_fold(repo, folds, final_test, fold, counts):
    """_run_fold + summary.json + registry line, i.e. what scripts/run_gan_fold.py leaves."""
    train, synthetic, record = _run_fold(repo, folds, final_test, fold, counts)
    out = repo / "data" / "gan" / f"fold_{fold:02d}"
    (out / "summary.json").write_text(
        json.dumps({"fold": fold, "synthetic_images": len(synthetic), "config_file": None})
    )
    update_registry(
        repo / "results" / "v2" / "gan" / "registry.csv",
        registry_row(
            record,
            plan=counts,
            synthetic_manifest=out / "synthetic_manifest.csv",
            with_gan_manifest=out / f"fold_{fold:02d}_train_gan.csv",
            smoke_run=False,
        ),
    )
    return train, synthetic, record


def _verify(repo, **kw):
    return verify_gan_outputs(
        gan_dir=repo / "data" / "gan",
        repo_root=repo,
        audit_dir=repo / "data" / "audit",
        registry_path=repo / "results" / "v2" / "gan" / "registry.csv",
        gan_manifest_path=repo / "results" / "v2" / "gan" / "GAN_MANIFEST.csv",
        **kw,
    )


def _failed(report) -> list[str]:
    return [c["check"] for c in report["checks"] if not c["ok"]]


# --- class mapping ------------------------------------------------------------------------------


def test_class_mapping_label_index_equals_class(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    _, synthetic, _ = _complete_fold(repo, folds, final_test, 1, {c: 1 for c in CANONICAL_CLASSES})
    # every synthetic row: label index -> CANONICAL_CLASSES -> class -> directory name
    for row in synthetic:
        assert CANONICAL_CLASSES[row["label"]] == row["unified_class"]
        assert Path(row["filepath"]).parent.name == row["unified_class"].replace(" ", "_")
    assert {r["unified_class"] for r in synthetic} == set(CANONICAL_CLASSES)
    # a label that names another class is rejected before anything is written
    bad = [{**synthetic[0], "label": (synthetic[0]["label"] + 1) % len(CANONICAL_CLASSES)}]
    with pytest.raises(ValueError, match="label/class mismatch"):
        validate_synthetic_rows(bad, 1, repo)
    # the consolidated manifest carries the same mapping
    rows = build_gan_manifest(repo / "data" / "gan", repo)
    assert all(CANONICAL_CLASSES[r["label"]] == r["class"] for r in rows)


# --- GAN_MANIFEST.csv ---------------------------------------------------------------------------


def test_gan_manifest_lists_every_synthetic_image_with_provenance(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    produced = {}
    for fold, counts in ((1, {"EUS Disease": 2}), (2, {"Healthy Fish": 3, "Aeromoniasis": 1})):
        _, synthetic, record = _complete_fold(repo, folds, final_test, fold, counts)
        produced[fold] = (synthetic, record)
    assert completed_folds(repo / "data" / "gan") == [1, 2]
    rows = build_gan_manifest(repo / "data" / "gan", repo)
    path = repo / "results" / "v2" / "gan" / "GAN_MANIFEST.csv"
    write_gan_manifest(rows, path)
    read = read_gan_manifest(path)
    assert len(read) == len(rows) == 6
    for required in (
        "synthetic_id",
        "fold",
        "class",
        "generation_seed",
        "generation_config",
        "training_source",
        "path",
    ):
        assert required in GAN_MANIFEST_COLUMNS
    by_id = {r["synthetic_id"]: r for r in read}
    for fold, (synthetic, record) in produced.items():
        for s in synthetic:
            m = by_id[s["image_id"]]
            assert int(m["fold"]) == fold and m["class"] == s["unified_class"]
            assert int(m["generation_seed"]) == TINY.seed
            assert json.loads(m["generation_config"]) == record.config
            assert m["training_source"] == f"data/audit/cv_v3/fold_{fold:02d}_train.csv"
            assert m["training_source_sha256"] == _sha(repo / m["training_source"])
            assert m["path"] == s["filepath"] and (repo / m["path"]).is_file()
            assert m["sha256"] == _sha(repo / m["path"])
            assert m["generator_sha256"] == record.checkpoint_sha256
            assert m["generator_checkpoint"] == f"data/gan/fold_{fold:02d}/generator.pt"
            assert m["device"] == "cpu" and m["gpu_name"] == ""
            assert int(m["width"]) == int(m["height"]) == TINY.image_size
            assert m["format"] == "PNG" and m["source_dataset"] == "gan"
            assert m["generated_at"][:4] == "2026" and s["generated_at"] == m["generated_at"]
    # ids and paths are unique across folds
    assert len({r["synthetic_id"] for r in read}) == 6 and len({r["path"] for r in read}) == 6
    # a fold whose synthetic rows name another generator is refused
    bad = repo / "data" / "gan" / "fold_02" / "gan_run.json"
    record = json.loads(bad.read_text())
    record["checkpoint_sha256"] = "0" * 64
    bad.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="generator digest"):
        build_gan_manifest(repo / "data" / "gan", repo)


# --- verifier -----------------------------------------------------------------------------------


def test_verifier_passes_on_clean_outputs_and_catches_tampering(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    for fold in (1, 2, 3):
        _complete_fold(repo, folds, final_test, fold, {"EUS Disease": 2, "Saprolegniasis": 1})
    write_gan_manifest(
        build_gan_manifest(repo / "data" / "gan", repo),
        repo / "results" / "v2" / "gan" / "GAN_MANIFEST.csv",
    )
    report = _verify(repo)
    assert report["ok"], _failed(report)
    assert report["counts"] == {"folds": 3, "synthetic_images": 9, "per_fold": {1: 3, 2: 3, 3: 3}}
    assert all(f["device"] == "cpu" and f["synthetic"] == 3 for f in report["folds"])
    # sample / none modes also pass and say so
    assert _verify(repo, images="sample")["ok"] and _verify(repo, images="none")["ok"]

    out = repo / "data" / "gan" / "fold_02"
    # (a) a validation image smuggled into the WITH-GAN manifest
    with_gan = out / "fold_02_train_gan.csv"
    original = with_gan.read_bytes()
    _, validation = __import__("src.split_v2", fromlist=["fold_members"]).fold_members(folds, 2)
    v = validation[0]
    with_gan.write_bytes(
        original
        + (
            f"{v.image_id},{v.source_dataset},{v.filepath},{v.original_path},{v.original_class},"
            f"{v.unified_class},{v.label},{v.group_id},{v.specimen_id},{v.stratum},2,False\r\n"
        ).encode()
    )
    failed = _failed(_verify(repo, images="none"))
    assert "fold 02: no validation id in WITH-GAN manifest" in failed
    assert "fold 02: registry digests match generator / manifests / source" in failed
    with_gan.write_bytes(original)
    assert _verify(repo, images="none")["ok"]
    # (b) a synthetic PNG replaced on disk -> digest / validity checks fail
    png = repo / read_gan_manifest(repo / "results" / "v2" / "gan" / "GAN_MANIFEST.csv")[0]["path"]
    png_bytes = png.read_bytes()
    png.write_bytes(b"not a png")
    failed = _failed(_verify(repo))
    assert "GAN_MANIFEST.csv image digests match files" in failed
    assert any(c.startswith("fold 01: PNGs open as RGB") for c in failed)
    png.write_bytes(png_bytes)
    # (c) a PNG outside train/ (e.g. under a validation directory) is caught
    stray = out / "validation" / "x.png"
    stray.parent.mkdir()
    stray.write_bytes(png_bytes)
    assert "fold 02: no PNG outside fold_02/train" in _failed(_verify(repo, images="none"))
    stray.unlink()
    stray.parent.rmdir()
    # (d) an extra PNG next to the synthetic ones that no manifest lists
    extra = out / "train" / "EUS_Disease" / "synthetic_99999.png"
    extra.write_bytes(png_bytes)
    assert "fold 02: PNGs on disk == synthetic manifest rows" in _failed(
        _verify(repo, images="none")
    )
    extra.unlink()
    # (e) a synthetic id shared between two folds
    manifest = repo / "results" / "v2" / "gan" / "GAN_MANIFEST.csv"
    sm3 = out.parent / "fold_03" / "synthetic_manifest.csv"
    sm3_bytes = sm3.read_bytes()
    first_id_fold2 = read_gan_manifest(manifest)[3]["synthetic_id"]
    lines = sm3_bytes.decode().split("\r\n")
    cells = lines[1].split(",")
    cells[0] = first_id_fold2
    sm3.write_bytes("\r\n".join([lines[0], ",".join(cells)] + lines[2:]).encode())
    failed = _failed(_verify(repo, images="none"))
    assert "no synthetic id shared between folds" in failed
    sm3.write_bytes(sm3_bytes)
    assert _verify(repo, images="none")["ok"]


def test_verifier_detects_changed_training_source(fold_setup):  # noqa: F811
    repo, folds, final_test = fold_setup
    _complete_fold(repo, folds, final_test, 1, {"EUS Disease": 1})
    write_gan_manifest(
        build_gan_manifest(repo / "data" / "gan", repo),
        repo / "results" / "v2" / "gan" / "GAN_MANIFEST.csv",
    )
    assert _verify(repo)["ok"]
    # the run record says it trained on a manifest whose digest differs from the file now on disk
    record_path = repo / "data" / "gan" / "fold_01" / "gan_run.json"
    record = json.loads(record_path.read_text())
    record["train_manifest_sha256"] = "f" * 64
    record_path.write_text(json.dumps(record))
    failed = _failed(_verify(repo, images="none"))
    assert "fold 01: training source is fold_01_train.csv and unchanged" in failed
    # and a fold manifest edited after the fact breaks the digest-guarded readers
    fold_file = repo / "data" / "audit" / "cv_v3" / "fold_01_train.csv"
    text = fold_file.read_text()
    fold_file.write_text(text + "\n")
    assert "split and fold manifests verify" in _failed(_verify(repo, images="none"))
    fold_file.write_text(text)


# --- all-folds driver ---------------------------------------------------------------------------


def test_parse_folds():
    assert parse_folds("1-3") == [1, 2, 3]
    assert parse_folds("2,1,5") == [1, 2, 5]
    assert parse_folds(f"1-{N_FOLDS}") == list(range(1, N_FOLDS + 1))
    for bad in ("0", "11", "", "1-11"):
        with pytest.raises(ValueError):
            parse_folds(bad)


def test_all_folds_driver_generates_skips_complete_and_verifies(fold_setup, tmp_path):  # noqa: F811
    repo, folds, final_test = fold_setup
    config = tmp_path / "tiny.json"
    config.write_text(
        json.dumps(
            {
                "architecture": "cDCGAN (class-conditional DCGAN)",
                "gan": {
                    **{k: v for k, v in TINY.__dict__.items()},
                    "synthetic_per_class": {"EUS Disease": 2},
                },
            }
        )
    )
    # fold 2 is already complete (from a previous session): it must be kept, not regenerated
    _, synthetic2, _ = _complete_fold(repo, folds, final_test, 2, {"Healthy Fish": 1})
    generator2 = _sha(repo / "data" / "gan" / "fold_02" / "generator.pt")

    def run(*extra):
        return subprocess.run(
            [
                sys.executable,
                str(ALL_FOLDS),
                "--folds",
                "1-3",
                "--config",
                str(config),
                "--data-dir",
                str(repo / "data"),
                "--results-dir",
                str(repo / "results"),
                "--repo-root",
                str(repo),
                "--device",
                "cpu",
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )

    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fold 2 already complete" in result.stderr
    assert _sha(repo / "data" / "gan" / "fold_02" / "generator.pt") == generator2
    assert completed_folds(repo / "data" / "gan") == [1, 2, 3]
    gan_results = repo / "results" / "v2" / "gan"
    registry = read_registry(gan_results / "registry.csv")
    assert [int(r["fold"]) for r in registry] == [1, 2, 3]
    assert all(r["device"] == "cpu" and r["gpu_name"] == "" for r in registry)
    manifest = read_gan_manifest(gan_results / "GAN_MANIFEST.csv")
    assert len(manifest) == 2 + len(synthetic2) + 2
    assert {int(m["fold"]) for m in manifest} == {1, 2, 3}
    assert {m["config_file"] for m in manifest} == {str(config), "(defaults)"}
    verification = json.loads((gan_results / "verification.json").read_text())
    assert verification["ok"], [c for c in verification["checks"] if not c["ok"]]
    summary = json.loads((gan_results / "generation_summary.json").read_text())
    assert summary["folds_completed"] == [1, 2, 3] and summary["synthetic_images_total"] == len(
        manifest
    )
    assert [f["status"] for f in summary["per_fold"]] == [
        "generated",
        "skipped (complete)",
        "generated",
    ]
    assert summary["device"] == "cpu" and summary["verification_ok"] is True
    # a second run regenerates nothing and verifies the same set
    again = run()
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stderr.count("already complete") == 3
    assert read_gan_manifest(gan_results / "GAN_MANIFEST.csv") == manifest
    # the standalone verifier agrees, and refuses to pass a fold whose registry line is gone
    check = subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--data-dir",
            str(repo / "data"),
            "--results-dir",
            str(repo / "results"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert check.returncode == 0 and "RESULT: VERIFIED" in check.stdout, check.stdout + check.stderr
    (gan_results / "registry.csv").write_text(
        (gan_results / "registry.csv").read_text().splitlines()[0] + "\n"
    )
    check = subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "--data-dir",
            str(repo / "data"),
            "--results-dir",
            str(repo / "results"),
            "--repo-root",
            str(repo),
            "--images",
            "none",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert check.returncode == 1 and "registry line present" in check.stdout


# --- CUDA-only rule -----------------------------------------------------------------------------


@pytest.mark.skipif(torch.cuda.is_available(), reason="this machine has CUDA")
def test_require_cuda_refuses_cpu_and_mps(fold_setup, tmp_path):  # noqa: F811
    repo, _, _ = fold_setup
    common = [
        "--data-dir",
        str(repo / "data"),
        "--results-dir",
        str(repo / "results"),
        "--repo-root",
        str(repo),
        "--require-cuda",
    ]
    for script, extra in ((ONE_FOLD, ["--fold", "1"]), (ALL_FOLDS, ["--folds", "1"])):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                *extra,
                "--config",
                str(ROOT / "configs/gan_v2/smoke.json"),
                *common,
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        assert result.returncode == 2, result.stdout + result.stderr
        assert "not CUDA" in result.stderr
    # nothing was trained or written
    assert not (repo / "data" / "gan").exists() or not list((repo / "data" / "gan").rglob("*.pt"))
