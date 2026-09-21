"""Phase 12 — generic CV runner: model factory, manifest path logic, hard leakage guards,
configuration/provenance recording, checkpoints, resume, status, the 100-experiment
matrix and CUDA enforcement. Local runs use tiny synthetic images on CPU only —
structural checks, not training."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.cv_runner import (
    DATA_ARMS,
    HISTORY_COLUMNS,
    CVConfig,
    LeakageError,
    RunArgs,
    TrainRow,
    check_isolation,
    experiment_id,
    is_completed,
    load_cv_config,
    load_preprocess_config,
    manifest_paths,
    read_manifest_rows,
    read_status,
    run_experiment,
    stage_for_epoch,
)
from src.dataset_cleaning import CLEAN_MANIFEST_NAME, build_clean_manifest, write_clean_manifest
from src.experiment_matrix import (
    MATRIX_COLUMNS,
    build_matrix,
    read_matrix,
    refresh_statuses,
    validate_matrix,
    write_matrix,
)
from src.gan_augmentation import (
    augmented_training_manifest,
    write_augmented_manifest,
    write_synthetic_manifest,
)
from src.manifest import CANONICAL_CLASSES
from src.model_factory import (
    MODEL_KEYS,
    backbone_parameters,
    build_model,
    head_parameters,
    set_backbone_trainable,
    set_train_mode,
)
from src.preprocessing import CLAHEConfig, PreprocessConfig
from src.split_v2 import (
    DEVELOPMENT,
    assign_folds,
    build_dev_test_split,
    fold_members,
    write_folds,
    write_split,
)
from tests.test_split_v2 import _many_records

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "cv_v2" / "default.json"
SMOKE = ROOT / "configs" / "cv_v2" / "smoke.json"


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, size=(36, 40, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def repo(tmp_path):
    """A miniature repository: clean manifest, dev/test split, 3 folds, a WITH-GAN
    manifest for fold 1 (fake synthetic PNGs under data/gan/fold_01/train), configs copied."""
    clean = build_clean_manifest(_many_records(30))
    split = build_dev_test_split(clean, test_ratio=0.2, seed=42)
    development = [s for s in split if s.split == DEVELOPMENT]
    folds = assign_folds(development, n_folds=3, seed=42)
    for i, s in enumerate(split):
        _write_image(tmp_path / s.filepath, i)
    audit = tmp_path / "data" / "audit"
    write_clean_manifest(clean, audit / CLEAN_MANIFEST_NAME)
    write_split(split, audit / "split_v3")
    write_folds(folds, audit / "cv_v3", 3)
    train, _ = fold_members(folds, 1)
    synthetic = []
    for k, cls in enumerate(["EUS Disease", "Healthy Fish"]):
        rel = f"data/gan/fold_01/train/{cls.replace(' ', '_')}/synthetic_{k:05d}.png"
        _write_image(tmp_path / rel, 100 + k)
        synthetic.append(
            {
                "image_id": f"gan-f01-{k}",
                "fold": 1,
                "unified_class": cls,
                "label": CANONICAL_CLASSES.index(cls),
                "filepath": rel,
            }
        )
    write_augmented_manifest(
        augmented_training_manifest(train, synthetic, 1),
        tmp_path / "data" / "gan" / "fold_01" / "fold_01_train_gan.csv",
    )
    # the provenance the real GAN pipeline leaves next to the images: a run record for THIS
    # fold and a synthetic manifest whose rows name that run's generator digest
    gan_dir = tmp_path / "data" / "gan" / "fold_01"
    generator_sha = "ab" * 32
    (gan_dir / "gan_run.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "checkpoint_sha256": generator_sha,
                "train_manifest_sha256": "cd" * 32,
                "device": "cuda",
                "gpu_name": "Tesla T4",
            }
        )
    )
    write_synthetic_manifest(
        [
            {
                **s,
                "index": k,
                "seed": 42,
                "generator_checkpoint": str(gan_dir / "generator.pt"),
                "generator_sha256": generator_sha,
                "architecture": "cDCGAN (class-conditional DCGAN)",
                "image_size": 16,
                "generated_at": "2026-09-20T00:00:00",
                "synthetic": True,
            }
            for k, s in enumerate(synthetic)
        ],
        gan_dir / "synthetic_manifest.csv",
    )
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "preprocess_v2_clahe.json", tmp_path / "configs")
    (tmp_path / "configs" / "cv_v2").mkdir()
    shutil.copy(CONFIG, tmp_path / "configs" / "cv_v2")
    shutil.copy(SMOKE, tmp_path / "configs" / "cv_v2")
    return tmp_path


def _args(repo: Path, **overrides) -> RunArgs:
    base = dict(
        model_key="cnn_bilstm",
        fold=1,
        data_arm="without_gan",
        config_path=repo / "configs" / "cv_v2" / "smoke.json",
        data_dir=repo / "data",
        repo_root=repo,
        out_root=repo / "results" / "v2" / "experiments",
        device="cpu",
        max_train_samples=12,
        max_validation_samples=6,
    )
    base.update(overrides)
    return RunArgs(**base)


# --- 1-3: factory and common interface ---


def test_efficientnet_baseline_and_all_hybrids_construct_through_the_factory():
    assert MODEL_KEYS[0] == "efficientnet_b0" and len(MODEL_KEYS) == 6
    x = torch.randn(2, 3, 224, 224)
    for key in MODEL_KEYS:
        model = build_model(key, 8, pretrained=False).eval()
        with torch.no_grad():
            assert model(x).shape == (2, 8)
        assert head_parameters(model) and backbone_parameters(model)
        set_backbone_trainable(model, False)
        assert all(not p.requires_grad for p in backbone_parameters(model))
        assert all(p.requires_grad for p in head_parameters(model))
        set_train_mode(model, backbone_frozen=True)
        assert model.classifier.training
        set_backbone_trainable(model, True)
    with pytest.raises(ValueError):
        build_model("nope", 8)


# --- 4-6: manifest path logic ---


def test_manifest_paths_for_both_arms_and_validation(tmp_path):
    train, val = manifest_paths(tmp_path, 1, "without_gan")
    assert train == tmp_path / "audit" / "cv_v3" / "fold_01_train.csv"
    assert val == tmp_path / "audit" / "cv_v3" / "fold_01_validation.csv"
    train, val = manifest_paths(tmp_path, 1, "with_gan")
    assert train == tmp_path / "gan" / "fold_01" / "fold_01_train_gan.csv"
    assert val == tmp_path / "audit" / "cv_v3" / "fold_01_validation.csv"
    with pytest.raises(ValueError):
        manifest_paths(tmp_path, 1, "with_test")


# --- 7-11: hard leakage guards ---


def _rows(repo: Path, fold: int, arm: str):
    train_path, val_path = manifest_paths(repo / "data", fold, arm)
    return train_path, val_path, read_manifest_rows(train_path), read_manifest_rows(val_path)


def _test_ids(repo: Path) -> set[str]:
    with (repo / "data" / "audit" / "split_v3" / "final_test.csv").open() as handle:
        return {r["image_id"] for r in csv.DictReader(handle)}


def test_guards_accept_the_correct_manifests(repo):
    for arm in DATA_ARMS:
        train_path, val_path, train, val = _rows(repo, 1, arm)
        info = check_isolation(
            fold=1,
            data_arm=arm,
            train_path=train_path,
            validation_path=val_path,
            train_rows=train,
            validation_rows=val,
            test_ids=_test_ids(repo),
        )
        assert info["train_synthetic"] == (2 if arm == "with_gan" else 0)
        assert info["groups_straddling"] == 0


def test_guards_reject_every_violation(repo):
    train_path, val_path, train, val = _rows(repo, 1, "without_gan")
    test_ids = _test_ids(repo)
    common = dict(train_path=train_path, validation_path=val_path, test_ids=test_ids)
    final_test = repo / "data" / "audit" / "split_v3" / "final_test.csv"
    # final test as training / validation manifest
    with pytest.raises(LeakageError, match="not the without_gan manifest"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_path=final_test,
            validation_path=val_path,
            train_rows=train,
            validation_rows=val,
            test_ids=test_ids,
        )
    with pytest.raises(LeakageError, match="final_test"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_path=train_path,
            validation_path=final_test,
            train_rows=train,
            validation_rows=val,
            test_ids=test_ids,
        )
    # wrong fold manifests
    with pytest.raises(LeakageError, match="not the without_gan manifest of fold 2"):
        check_isolation(
            fold=2, data_arm="without_gan", train_rows=train, validation_rows=val, **common
        )
    # training/validation overlap
    with pytest.raises(LeakageError, match="both training and validation"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_rows=train + val[:1],
            validation_rows=val,
            **common,
        )
    # final-test ids in training / validation
    leaked = TrainRow(next(iter(test_ids)), "x", "data/raw/x.jpg", "EUS Disease", 3, "g", 2, False)
    with pytest.raises(LeakageError, match="final-test image ids are in the training"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_rows=train + [leaked],
            validation_rows=val,
            **common,
        )
    with pytest.raises(LeakageError, match="final-test image ids are in the validation"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_rows=train,
            validation_rows=val
            + [TrainRow(leaked.image_id, "x", "data/raw/x.jpg", "EUS Disease", 3, "g", 1, False)],
            **common,
        )
    # synthetic rows in the WITHOUT-GAN arm, synthetic from another fold, synthetic in validation
    syn_other = TrainRow(
        "gan-f02-9",
        "gan",
        "data/gan/fold_02/train/EUS_Disease/s.png",
        "EUS Disease",
        3,
        "gan-f02-9",
        -1,
        True,
    )
    with pytest.raises(LeakageError, match="contains synthetic rows"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_rows=train + [syn_other],
            validation_rows=val,
            **common,
        )
    gan_train_path, _, gan_train, _ = _rows(repo, 1, "with_gan")
    with pytest.raises(LeakageError, match="is not under data/gan/fold_01/train"):
        check_isolation(
            fold=1,
            data_arm="with_gan",
            train_path=gan_train_path,
            validation_path=val_path,
            train_rows=gan_train + [syn_other],
            validation_rows=val,
            test_ids=test_ids,
        )
    syn_val = TrainRow(
        "gan-f01-val-only",
        "gan",
        "data/gan/fold_01/train/EUS_Disease/synthetic_00000.png",
        "EUS Disease",
        3,
        "g",
        1,
        True,
    )
    with pytest.raises(LeakageError, match="synthetic / under data/gan"):
        check_isolation(
            fold=1,
            data_arm="with_gan",
            train_path=gan_train_path,
            validation_path=val_path,
            train_rows=gan_train,
            validation_rows=val + [syn_val],
            test_ids=test_ids,
        )
    # real training row that is actually the validation fold
    wrong_fold = TrainRow(
        "cur-zzz", "current_freshwater", "data/raw/x.jpg", "EUS Disease", 3, "gz", 1, False
    )
    with pytest.raises(LeakageError, match="validation fold is 1"):
        check_isolation(
            fold=1,
            data_arm="without_gan",
            train_rows=train + [wrong_fold],
            validation_rows=val,
            **common,
        )


# --- 12, 20-22: configuration, checkpoints, provenance ---


def test_config_loading_and_stages():
    config = load_cv_config(CONFIG)
    assert config.strategy == "head_then_full" and config.selection_metric == "f1_macro"
    assert (
        stage_for_epoch(1, config) == "head"
        and stage_for_epoch(config.head_epochs + 1, config) == "full"
    )
    pre = load_preprocess_config(ROOT / "configs" / "preprocess_v2_clahe.json")
    assert pre == PreprocessConfig(clahe=CLAHEConfig())  # the existing CLAHE setting, unchanged
    with pytest.raises(ValueError):
        CVConfig(strategy="head_then_full", head_epochs=5, epochs=5)
    with pytest.raises(ValueError):
        CVConfig(scheduler="plateau")


def test_run_writes_every_artifact_records_provenance_and_refuses_overwrite(repo):
    summary = run_experiment(_args(repo))
    out = repo / "results" / "v2" / "experiments" / "cnn_bilstm_fold01_without_gan"
    for name in (
        "config.json",
        "history.csv",
        "metrics.json",
        "confusion_matrix.csv",
        "best.pt",
        "latest.pt",
        "run_summary.json",
        "status.json",
        "logs/train.log",
    ):
        assert (out / name).is_file(), name
    assert is_completed(out) and read_status(out) == "COMPLETED"
    record = json.loads((out / "config.json").read_text())
    for key in (
        "experiment_id",
        "model",
        "model_key",
        "fold",
        "data_arm",
        "seed",
        "num_classes",
        "image_size",
        "preprocessing",
        "augmentation",
        "training",
        "device",
        "environment",
        "git_commit",
        "manifests",
        "timestamp",
        "config_hash",
        "isolation",
    ):
        assert key in record, key
    assert record["preprocessing"]["config_path"] == "configs/preprocess_v2_clahe.json"
    assert len(record["preprocessing"]["config_sha256"]) == 64
    assert record["preprocessing"]["resolved"]["clahe"] == {"clip_limit": 2.0, "tile_grid_size": 8}
    for k in ("train_sha256", "validation_sha256", "final_test_sha256"):
        assert len(record["manifests"][k]) == 64
    assert record["manifests"]["validation_path"].endswith("fold_01_validation.csv")
    assert record["training"]["early_stopping"] == {"metric": "f1_macro", "patience": None}
    env = record["environment"]
    assert {"python", "torch", "torchvision", "cuda", "mps", "device"} <= set(env)
    assert record["smoke"] is True and summary["note"].startswith("INFRASTRUCTURE SMOKE TEST")
    with (out / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0].keys()) == HISTORY_COLUMNS and len(rows) == 2
    assert [r["stage"] for r in rows] == ["head", "full"]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["class_names"] == list(CANONICAL_CLASSES)
    assert len(metrics["per_class"]) == 8 and len(metrics["confusion"]) == 8
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "model_key",
        "fold",
        "data_arm",
        "seed",
        "best_val_f1_macro",
        "config",
        "history",
        "rng",
        "class_names",
    ):
        assert key in ck, key
    assert ck["model_key"] == "cnn_bilstm" and ck["fold"] == 1 and ck["data_arm"] == "without_gan"
    # a completed experiment is never retrained
    with pytest.raises(FileExistsError, match="COMPLETED"):
        run_experiment(_args(repo))
    with pytest.raises(FileExistsError, match="COMPLETED"):
        run_experiment(_args(repo, resume=True))


def test_with_gan_arm_trains_on_real_plus_synthetic_and_validates_on_fold(repo):
    summary = run_experiment(_args(repo, data_arm="with_gan", max_train_samples=None))
    assert summary["train_synthetic"] == 2
    record = json.loads(
        (repo / "results/v2/experiments/cnn_bilstm_fold01_with_gan/config.json").read_text()
    )
    assert record["manifests"]["train_path"].endswith("data/gan/fold_01/fold_01_train_gan.csv")
    assert record["manifests"]["validation_path"].endswith("cv_v3/fold_01_validation.csv")
    assert record["isolation"]["train_synthetic"] == 2
    # dataset configuration version and the reproducibility metadata travel with the checkpoint
    assert record["dataset_config_version"] == "v3"
    assert (
        record["manifests"]["split_dir"] == "split_v3" and record["manifests"]["cv_dir"] == "cv_v3"
    )
    ckpt = torch.load(
        repo / "results/v2/experiments/cnn_bilstm_fold01_with_gan/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["dataset_config_version"] == "v3" and ckpt["seed"] == record["seed"]
    assert ckpt["optimizer"] == "adamw" and ckpt["scheduler"] in ("cosine", "none", "step")
    assert ckpt["manifest_sha256"]["train"] == record["manifests"]["train_sha256"]
    assert ckpt["preprocessing_sha256"] == record["preprocessing"]["config_sha256"]
    assert ckpt["environment"]["torch"] == torch.__version__ and "python" in ckpt["environment"]
    # WITH-GAN provenance travels with the record and the checkpoint
    assert record["gan"]["synthetic_rows"] == 2 and record["gan"]["generator_sha256"] == "ab" * 32
    assert ckpt["manifest_sha256"]["gan_synthetic"] == record["gan"]["synthetic_manifest_sha256"]
    assert (
        ckpt["gan_generator_sha256"] == "ab" * 32 and ckpt["config_hash"] == record["config_hash"]
    )


def test_with_gan_arm_refuses_a_generator_of_another_fold(repo):
    run_record = repo / "data" / "gan" / "fold_01" / "gan_run.json"
    payload = json.loads(run_record.read_text())
    payload["fold"] = 2
    run_record.write_text(json.dumps(payload))
    with pytest.raises(LeakageError, match="belongs to fold 2"):
        run_experiment(_args(repo, data_arm="with_gan", max_train_samples=None))


def test_efficientnet_baseline_runs_through_the_same_runner(repo):
    summary = run_experiment(_args(repo, model_key="efficientnet_b0"))
    assert summary["model_key"] == "efficientnet_b0" and summary["epochs_run"] == 2


def test_resume_continues_and_does_not_restart(repo):
    """Interrupt after epoch 1 (by running a 1-epoch config), then resume to 2 epochs."""
    first = run_experiment(_args(repo, epochs_override=1, experiment_id="resume_case"))
    out = repo / "results" / "v2" / "experiments" / "resume_case"
    assert first["epochs_run"] == 1
    # simulate an interruption: status back to RUNNING, summary removed
    (out / "run_summary.json").unlink()
    (out / "status.json").write_text(json.dumps({"status": "INTERRUPTED"}))
    assert not is_completed(out)
    with pytest.raises(FileExistsError, match="--resume"):
        run_experiment(_args(repo, epochs_override=1, experiment_id="resume_case"))
    # resuming with a different configuration is refused
    with pytest.raises(ValueError, match="configuration hash"):
        run_experiment(_args(repo, epochs_override=2, experiment_id="resume_case", resume=True))
    second = run_experiment(
        _args(repo, epochs_override=1, experiment_id="resume_case", resume=True)
    )
    assert second["epochs_run"] == 1 and is_completed(out)  # nothing to add, completes cleanly


def test_final_test_cannot_be_used_and_cuda_is_enforced(repo):
    with pytest.raises(ValueError, match="data_arm"):
        run_experiment(_args(repo, data_arm="final_test"))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA required"):
            run_experiment(_args(repo, require_cuda=True, device=None))
        # CPU is never silently selected when CUDA is required, even with an explicit cpu device
        with pytest.raises(RuntimeError, match="CUDA required"):
            run_experiment(_args(repo, require_cuda=True, device="cpu"))


def test_cli_script_runs_smoke_config(repo):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_cv_experiment.py"),
            "--model",
            "yolo_transformer",
            "--fold",
            "2",
            "--data-arm",
            "without_gan",
            "--config",
            "configs/cv_v2/smoke.json",
            "--data-dir",
            str(repo / "data"),
            "--repo-root",
            str(repo),
            "--out-root",
            str(repo / "results" / "v2" / "smoke"),
            "--device",
            "cpu",
            "--max-train-samples",
            "8",
            "--max-validation-samples",
            "4",
            "--smoke",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (
        repo / "results" / "v2" / "smoke" / "yolo_transformer_fold02_without_gan" / "best.pt"
    ).is_file()


# --- 13-19: matrix ---


def test_matrix_has_exactly_100_pending_unique_experiments(tmp_path):
    rows = build_matrix(tmp_path / "data", tmp_path)
    validate_matrix(rows)
    assert len(rows) == 100 and all(r["status"] == "PENDING" for r in rows)
    assert len({r["experiment_id"] for r in rows}) == 100
    assert len({r["model"] for r in rows}) == 5 and len({r["fold"] for r in rows}) == 10
    assert {r["data_arm"] for r in rows} == set(DATA_ARMS)
    assert all(
        not ("final_test" in r["train_manifest"] or "final_test" in r["validation_manifest"])
        for r in rows
    )
    assert all(r["validation_manifest"].endswith(f"fold_{r['fold']}_validation.csv") for r in rows)
    path = tmp_path / "results" / "v2" / "experiment_matrix.csv"
    write_matrix(rows, path)
    with path.open() as handle:
        assert tuple(next(csv.reader(handle))) == MATRIX_COLUMNS
    assert read_matrix(path) == rows
    with pytest.raises(ValueError, match="100 rows"):
        validate_matrix(rows[:99])
    dup = rows[:99] + [dict(rows[0])]
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(dup)


def test_matrix_refresh_reads_statuses_without_faking_completion(tmp_path):
    rows = build_matrix(tmp_path / "data", tmp_path)
    exp = tmp_path / "results" / "v2" / "experiments" / rows[0]["experiment_id"]
    exp.mkdir(parents=True)
    (exp / "status.json").write_text(json.dumps({"status": "COMPLETED"}))  # files missing -> FAILED
    refreshed = refresh_statuses(rows, tmp_path / "results" / "v2" / "experiments")
    assert refreshed[0]["status"] == "FAILED" and refreshed[1]["status"] == "PENDING"


def test_real_committed_matrix_has_100_experiments_with_record_backed_statuses():
    """The committed matrix carries the statuses of the Colab runs whose records are
    committed: a COMPLETED row must have its run_summary.json + metrics.json in the repo."""
    path = ROOT / "results" / "v2" / "experiment_matrix.csv"
    rows = read_matrix(path)
    assert len(rows) == 100 and len({r["experiment_id"] for r in rows}) == 100
    assert {r["status"] for r in rows} <= {"PENDING", "COMPLETED", "FAILED", "INTERRUPTED"}
    assert experiment_id("cnn_vit_lstm", 1, "without_gan") == rows[0]["experiment_id"]
    for r in rows:
        if r["status"] == "COMPLETED":
            exp = ROOT / "results" / "v2" / "experiments" / r["experiment_id"]
            assert (exp / "run_summary.json").is_file() and (exp / "metrics.json").is_file(), r


# --- 23-25: Colab preflight, CUDA detection, Colab configuration ---


def test_preflight_reports_cuda_honestly_and_never_selects_cpu_silently(repo):
    sys.path.insert(0, str(ROOT / "scripts"))
    from colab_preflight import cuda_report, dataset_report, repository_report, version_report

    cuda = cuda_report(require=True)
    assert cuda["available"] == torch.cuda.is_available()
    assert cuda["ok"] == torch.cuda.is_available()  # required + absent -> not ok, never "cpu"
    assert cuda_report(require=False)["ok"] is True
    assert version_report(ROOT)["ok"] is True
    (repo / "results" / "v2").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results" / "v2" / "experiment_matrix.csv", repo / "results" / "v2")
    rep = repository_report(repo, repo / "data")
    assert rep["split_digest_ok"] and rep["folds_digest_ok"] and rep["ok"]
    assert len(rep["final_test_sha256"]) == 64
    ds = dataset_report(repo, repo / "data", 1, "without_gan")
    assert ds["ok"] and ds["missing_images"] == 0
    ds_gan = dataset_report(repo, repo / "data", 2, "with_gan")  # no GAN manifest for fold 2
    assert ds_gan["ok"] is False and "run_gan_fold" in ds_gan["error"]


def test_preflight_cli_exit_codes(repo):
    (repo / "results" / "v2").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results" / "v2" / "experiment_matrix.csv", repo / "results" / "v2")
    (repo / "requirements").mkdir()
    shutil.copy(ROOT / "requirements" / "base.txt", repo / "requirements")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "colab_preflight.py"),
        "--fold",
        "1",
        "--data-arm",
        "without_gan",
        "--data-dir",
        str(repo / "data"),
        "--repo-root",
        str(repo),
    ]
    ok = subprocess.run(cmd, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert json.loads(ok.stdout)["ready"] is True
    if not torch.cuda.is_available():
        required = subprocess.run(cmd + ["--require-cuda"], capture_output=True, text=True)
        assert required.returncode == 2


def test_colab_configurations_load_and_smoke_config_is_flagged():
    smoke = load_cv_config(SMOKE)
    assert smoke.epochs == 2 and smoke.head_epochs == 1 and smoke.early_stopping_patience is None
    default = load_cv_config(CONFIG)
    assert default.epochs == 30 and default.preprocess_config == "configs/preprocess_v2_clahe.json"
    text = json.loads(SMOKE.read_text())["description"]
    assert "NOT A PERFORMANCE RESULT" in text
    nb = json.loads((ROOT / "notebooks" / "AquaHealthAI_Phase12_Colab.ipynb").read_text())
    sources = "".join("".join(c["source"]) for c in nb["cells"])
    assert "--require-cuda" in sources and "colab_preflight.py" in sources
    # the frozen test file appears only as a digest pin and an ids-absence check, never as
    # an evaluation split (the v3 flow's section 4 hashes it and asserts no id leaks into a fold)
    uses = re.findall(r"final_test[^\n]*", sources)
    assert uses and all("final_test.csv" in u for u in uses), uses
    assert not re.search(r"evaluate|predict|test_loader|--split\s+final_test", sources)
    assert all(c.get("outputs", []) == [] for c in nb["cells"] if c["cell_type"] == "code")
