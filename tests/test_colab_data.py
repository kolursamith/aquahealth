"""Layer 2 — Colab data pipeline: bundle build, attach, verification, root resolution,
GPU smoke test and pre-flight, all on a synthetic mini-repository (never the raw dataset)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src import colab_data
from src.colab_data import (
    BUNDLE_COLUMNS,
    BUNDLE_DIGEST_NAME,
    BUNDLE_MANIFEST_NAME,
    DATASET_ROOT_ENV,
    DatasetUnavailable,
    attach,
    build_bundle,
    bundle_rows,
    detect_layout,
    read_bundle_manifest,
    resolve_dataset_root,
    verify_dataset,
)
from src.dataset_cleaning import CleanRow, write_clean_manifest
from src.manifest import CANONICAL_CLASSES
from src.split_v2 import (
    CV_DIR_NAME,
    DEVELOPMENT,
    SPLIT_DIR_NAME,
    assign_folds,
    build_dev_test_split,
    write_folds,
    write_split,
)

ROOT = Path(__file__).resolve().parent.parent
PER_CLASS = 5
N_FOLDS = 2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dhash(i: int) -> str:
    return f"{i:016x}"


def make_repo(root: Path) -> Path:
    """A mini repository: data/raw/<key>/<class>/img.png + clean manifest + split + folds."""
    rows: list[CleanRow] = []
    rng = np.random.default_rng(0)
    n = 0
    for c_idx, cls in enumerate(CANONICAL_CLASSES):
        for k in range(PER_CLASS):
            key = "src_b" if k == PER_CLASS - 1 else "src_a"
            rel = Path("data/raw") / key / cls.replace(" ", "_") / f"{cls[:3]}_{k}.png"
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            arr = rng.integers(0, 255, size=(20, 24, 3), dtype=np.uint8)
            arr[..., 0] = 20 * c_idx
            Image.fromarray(arr).save(path)
            image_id = f"{key}-{n:06d}"
            # two images of the first class share a specimen -> one group
            specimen = "S1" if (c_idx == 0 and k < 2) else ""
            rows.append(
                CleanRow(
                    image_id=image_id,
                    source_dataset=key,
                    filepath=str(rel),
                    original_path=str(rel.relative_to("data/raw")),
                    original_filename=rel.name,
                    original_class=cls,
                    original_split="",
                    unified_class=cls,
                    mapping_status="EXACT_MATCH",
                    species="",
                    specimen_id=specimen,
                    width=24,
                    height=20,
                    extension=".png",
                    sha256=_sha(path),
                    dhash=_dhash(n),
                    exact_dup_group="",
                    near_dup_group="",
                    group_id="",
                    group_size=1,
                    included=True,
                    exclusion_reason="",
                    representative_image_id="",
                )
            )
            n += 1
    # one excluded duplicate row (must never enter the bundle)
    dup = rows[0]
    rows.append(
        CleanRow(
            **{
                **dup.__dict__,
                "image_id": "src_a-dup",
                "filepath": dup.filepath,
                "included": False,
                "exclusion_reason": "EXACT_DUPLICATE",
                "representative_image_id": dup.image_id,
            }
        )
    )
    # groups: specimen pair shares a group, everyone else is a singleton
    group_of = {}
    for r in rows:
        group_of[r.image_id] = "grp-S1" if r.specimen_id == "S1" else r.image_id
    for r in rows:
        r.group_id = group_of[r.image_id]
        r.group_size = sum(1 for x in rows if x.group_id == r.group_id and x.included)
    audit = root / "data" / "audit"
    audit.mkdir(parents=True)
    write_clean_manifest(rows, audit / "clean_manifest.csv")
    split = build_dev_test_split(rows, test_ratio=0.2, seed=42)
    write_split(split, audit / SPLIT_DIR_NAME)
    folds = assign_folds([s for s in split if s.split == DEVELOPMENT], n_folds=N_FOLDS, seed=42)
    write_folds(folds, audit / CV_DIR_NAME, N_FOLDS)
    return root


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo")


@pytest.fixture()
def colab_repo(tmp_path: Path, repo: Path) -> Path:
    """A second checkout with the manifests but no data/raw (what Colab clones)."""
    other = tmp_path / "colab_repo"
    (other / "data").mkdir(parents=True)
    import shutil

    shutil.copytree(repo / "data" / "audit", other / "data" / "audit")
    return other


# --- root resolution ---------------------------------------------------------------------


def test_root_unset_fails_clearly(monkeypatch):
    monkeypatch.delenv(DATASET_ROOT_ENV, raising=False)
    with pytest.raises(DatasetUnavailable, match=DATASET_ROOT_ENV):
        resolve_dataset_root()


def test_root_missing_directory_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv(DATASET_ROOT_ENV, str(tmp_path / "nowhere"))
    with pytest.raises(DatasetUnavailable, match="not a directory"):
        resolve_dataset_root()


def test_root_from_env_and_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv(DATASET_ROOT_ENV, str(tmp_path))
    assert resolve_dataset_root() == tmp_path.resolve()
    assert resolve_dataset_root(tmp_path / ".") == tmp_path.resolve()


def test_no_machine_path_in_code():
    text = (ROOT / "src" / "colab_data.py").read_text()
    for script in ("colab_dataset.py", "build_colab_bundle.py", "gpu_smoke.py"):
        text += (ROOT / "scripts" / script).read_text()
    assert "/Users/" not in text and "AI_TECHTAHON" not in text


def test_detect_layout(tmp_path):
    with pytest.raises(DatasetUnavailable):
        detect_layout(tmp_path)
    (tmp_path / "Fresh_water_disease").mkdir()
    assert detect_layout(tmp_path) == "delivery"
    nested = tmp_path / "drop"
    (nested / "Dataset" / "MatsyaDx-BD").mkdir(parents=True)
    assert detect_layout(nested) == "delivery"


# --- bundle -------------------------------------------------------------------------------


def test_bundle_rows_are_exactly_the_included_manifest_rows(repo):
    rows = bundle_rows(repo)
    assert len(rows) == PER_CLASS * len(CANONICAL_CLASSES)
    assert all(r.image_id != "src_a-dup" for r in rows)
    assert {r.split for r in rows} == {"development", "final_test"}
    assert all((r.fold != "") == (r.split == "development") for r in rows)
    assert all(CANONICAL_CLASSES[r.label] == r.unified_class for r in rows)
    grp = [r for r in rows if r.group_id == "grp-S1"]
    assert len(grp) == 2 and len({r.split for r in grp}) == 1 and len({r.fold for r in grp}) == 1


def test_build_bundle_copies_only_included_images_with_relative_paths(repo, tmp_path):
    out = tmp_path / "bundle"
    summary = build_bundle(repo, out)
    assert summary["images"] == PER_CLASS * len(CANONICAL_CLASSES)
    assert detect_layout(out) == "bundle"
    manifest = read_bundle_manifest(out)
    assert tuple(manifest[0].__dataclass_fields__) == BUNDLE_COLUMNS
    for b in manifest:
        assert (out / b.filepath).is_file()
        assert b.filepath.startswith("data/raw/")
        assert _sha(out / b.filepath) == b.sha256
    digests = (out / BUNDLE_DIGEST_NAME).read_text()
    assert "data/audit/clean_manifest.csv" in digests and "folds.csv" in digests
    assert _sha(repo / "data/audit/clean_manifest.csv") in digests
    # a second build reuses the files and is idempotent
    again = build_bundle(repo, out)
    assert again["reused"] == summary["images"] and again["copied"] == 0


def test_build_bundle_refuses_when_local_file_differs_from_manifest(repo, tmp_path):
    rows = bundle_rows(repo)
    (repo / rows[0].filepath).write_bytes(b"not the image")
    with pytest.raises(colab_data.DatasetMismatch, match="differs from the manifest"):
        build_bundle(repo, tmp_path / "bundle")


# --- attach + verify (the Colab side) ---------------------------------------------------------


def test_attach_and_full_verify(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    lines = attach(out, colab_repo)
    assert len(lines) == 2  # src_a, src_b
    for key in ("src_a", "src_b"):
        link = colab_repo / "data" / "raw" / key
        assert link.is_symlink() and link.resolve() == (out / "data" / "raw" / key).resolve()
    report = verify_dataset(colab_repo, out, hash_mode="all")
    assert report["ok"], report["errors"]
    names = {c["check"] for c in report["checks"]}
    assert {
        "every included image exists",
        "image count matches manifest",
        "labels canonical",
    } <= names
    assert report["counts"]["images"] == PER_CLASS * len(CANONICAL_CLASSES)
    assert report["layout"] == "bundle"
    # attaching again is a no-op; pointing elsewhere without force is refused
    assert all("unchanged" in line for line in attach(out, colab_repo))
    other = tmp_path / "bundle2"
    build_bundle(repo, other)
    with pytest.raises(DatasetUnavailable, match="force"):
        attach(other, colab_repo)
    attach(other, colab_repo, force=True)


def test_verify_detects_modified_image(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    attach(out, colab_repo)
    victim = read_bundle_manifest(out)[3]
    Image.fromarray(np.zeros((20, 24, 3), dtype=np.uint8)).save(out / victim.filepath)
    report = verify_dataset(colab_repo, out, hash_mode="all")
    assert not report["ok"]
    assert any("SHA-256" in e and victim.filepath in e for e in report["errors"])


def test_verify_detects_missing_image_and_count(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    attach(out, colab_repo)
    victim = read_bundle_manifest(out)[0]
    (out / victim.filepath).unlink()
    report = verify_dataset(colab_repo, out, hash_mode="none")
    assert not report["ok"]
    assert any(e.startswith("every included image exists") for e in report["errors"])
    assert any(e.startswith("image count matches manifest") for e in report["errors"])


def test_verify_detects_tampered_fold_manifest(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    attach(out, colab_repo)
    folds = colab_repo / "data" / "audit" / CV_DIR_NAME / "folds.csv"
    text = folds.read_text().splitlines()
    text[1] = (
        text[1].rsplit(",", 1)[0] + ",2"
        if text[1].endswith(",1")
        else text[1].rsplit(",", 1)[0] + ",1"
    )
    folds.write_text("\n".join(text) + "\n")
    report = verify_dataset(colab_repo, out, hash_mode="none")
    assert not report["ok"]
    assert any(e.startswith("fold digests verify") for e in report["errors"])


def test_verify_detects_bundle_built_from_other_manifest(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    attach(out, colab_repo)
    # the runtime's manifest differs from the one the bundle was pinned to
    manifest = colab_repo / "data" / "audit" / "clean_manifest.csv"
    manifest.write_text(manifest.read_text() + "\n")
    report = verify_dataset(colab_repo, out, hash_mode="none")
    assert not report["ok"]
    assert any("bundle pinned to data/audit/clean_manifest.csv" in e for e in report["errors"])


def test_verify_sample_mode_hashes_a_seeded_subset(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    attach(out, colab_repo)
    report = verify_dataset(colab_repo, out, hash_mode="sample", sample_size=7, seed=1)
    assert report["ok"]
    assert any("(sample, 7 files)" in c["check"] for c in report["checks"])


def test_excluded_rows_never_in_bundle(repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    assert all(b.image_id != "src_a-dup" for b in read_bundle_manifest(out))


# --- CLIs -----------------------------------------------------------------------------------


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = {**os.environ, **(env or {})}
    merged.pop(DATASET_ROOT_ENV, None) if env is None else None
    return subprocess.run(
        [sys.executable, *args], capture_output=True, text=True, cwd=ROOT, env=merged
    )


def test_colab_dataset_cli_unavailable_exit_code(colab_repo):
    env = {k: v for k, v in os.environ.items() if k != DATASET_ROOT_ENV}
    proc = subprocess.run(
        [sys.executable, "scripts/colab_dataset.py", "verify", "--repo-root", str(colab_repo)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 3 and DATASET_ROOT_ENV in proc.stderr


def test_colab_dataset_cli_attach_and_verify(repo, colab_repo, tmp_path):
    out = tmp_path / "bundle"
    build_bundle(repo, out)
    env = {DATASET_ROOT_ENV: str(out)}
    proc = _run("scripts/colab_dataset.py", "attach", "--repo-root", str(colab_repo), env=env)
    assert proc.returncode == 0, proc.stderr
    proc = _run(
        "scripts/colab_dataset.py", "verify", "--repo-root", str(colab_repo), "--json", env=env
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] and report["counts"]["images"] == PER_CLASS * len(CANONICAL_CLASSES)


def test_build_colab_bundle_cli(repo, tmp_path):
    out = tmp_path / "bundle"
    proc = _run(
        "scripts/build_colab_bundle.py", "--out", str(out), "--repo-root", str(repo), "--tar"
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["images"] == PER_CLASS * len(CANONICAL_CLASSES)
    assert Path(summary["tar"]).is_file()
    assert (out / BUNDLE_MANIFEST_NAME).is_file()


# --- GPU smoke + preflight --------------------------------------------------------------------


def test_gpu_smoke_steps_on_cpu():
    sys.path.insert(0, str(ROOT / "scripts"))
    import gpu_smoke  # noqa: E402
    import torch

    report = gpu_smoke.run(torch.device("cpu"))
    assert report["ok"]
    assert report["steps"]["forward_shape"] == [8, 8]
    assert report["steps"]["backward_ok"] and report["steps"]["optimizer_changed_weights"]


def test_gpu_smoke_cli_require_cuda_exit_code():
    import torch

    proc = _run("scripts/gpu_smoke.py", "--require-cuda", env={})
    assert proc.returncode == (0 if torch.cuda.is_available() else 2)


def test_preflight_env_only_reports_actual_dependencies():
    proc = _run("scripts/colab_preflight.py", "--env-only", env={})
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    env = report["environment"]
    assert env["ok"] and set(env["required"]) == {
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "matplotlib",
    }
    assert set(env["optional_not_required"]) == {"scikit-learn", "timm", "transformers"}
    assert "dataset" not in report  # --env-only never opens fold manifests
    assert report["repository"]["split_digest_ok"] and report["repository"]["folds_digest_ok"]


def test_preflight_requires_fold_and_arm_without_env_only():
    proc = _run("scripts/colab_preflight.py", env={})
    assert proc.returncode == 2 and "--fold and --data-arm" in proc.stderr


# --- notebook --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "notebook", ["AquaHealthAI_Phase12_Colab.ipynb", "AquaHealthAI_Layer3_GAN_Colab.ipynb"]
)
def test_notebook_uses_develop_and_configurable_root(notebook):
    text = (ROOT / "notebooks" / notebook).read_text()
    assert "BRANCH = 'develop'" in text
    assert DATASET_ROOT_ENV in text and "colab_dataset.py verify" in text
    assert "gpu_smoke.py --require-cuda" in text and "colab_preflight.py --env-only" in text
    assert "release/final-completion" not in text and "/Users/" not in text


def test_gan_notebook_trains_on_cuda_only_through_the_scripts():
    text = (ROOT / "notebooks" / "AquaHealthAI_Layer3_GAN_Colab.ipynb").read_text()
    assert (
        "run_gan_fold.py --fold 1 --config configs/gan_v2/smoke.json --smoke --require-cuda" in text
    )
    assert "run_gan_all_folds.py --config configs/gan_v2/default.json --require-cuda" in text
    assert "verify_gan_outputs.py --smoke" in text and "verify_gan_outputs.py --images all" in text
    assert "--device cpu" not in text and "--device mps" not in text
