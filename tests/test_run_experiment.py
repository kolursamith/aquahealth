"""Experiment runner (`scripts/run_experiment.py`) on a synthetic manifest.

The manifest's `test` rows point at files that do not exist, so the runner
would crash if it ever read the frozen test split during an experiment.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.manifest import CANONICAL_CLASSES, ManifestRow, write_manifest
from src.train import load_checkpoint
from tests.conftest import write_image_folder

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_experiment.py"


@pytest.fixture
def experiment_env(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    folders = [f"c{i}" for i in range(8)]
    root = write_image_folder(
        repo / "data" / "original" / "syn", folders, per_class=6, size=(80, 60)
    )
    rows: list[ManifestRow] = []
    for label, folder in enumerate(folders):
        files = sorted((root / folder).iterdir())
        for i, path in enumerate(files):
            split = "train" if i < 4 else "val"
            rows.append(
                ManifestRow(str(path.relative_to(repo)), label, CANONICAL_CLASSES[label], split)
            )
        rows.append(
            ManifestRow(
                f"data/original/syn/{folder}/MISSING_test.jpg",
                label,
                CANONICAL_CLASSES[label],
                "test",
            )
        )
    manifest = repo / "data" / "split_manifest.csv"
    write_manifest(rows, manifest)
    config = {
        "description": "runner test",
        "preprocess": {"image_size": 64, "resize_size": 72, "resize_mode": "crop", "clahe": True},
        "augment": {
            "crop_scale": [0.9, 1.0],
            "rotation_degrees": 5.0,
            "brightness": 0.1,
            "contrast": 0.1,
        },
        "stages": [
            {"name": "head", "epochs": 2, "trainable_blocks": 0, "learning_rate": 0.001},
            {
                "name": "partial",
                "epochs": 1,
                "trainable_blocks": 2,
                "learning_rate": 0.0003,
                "backbone_learning_rate": 0.00003,
            },
        ],
        "optimizer": {"optimizer": "adamw", "lr_step_size": 1, "lr_gamma": 0.5},
        "batch_size": 8,
    }
    config_path = repo / "exp.json"
    config_path.write_text(json.dumps(config))
    return {
        "repo": repo,
        "manifest": manifest,
        "config": config_path,
        "out": repo / "results" / "experiments",
    }


def _run(env: dict[str, Path], *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--id",
            "EXP-T",
            "--config",
            str(env["config"]),
            "--manifest",
            str(env["manifest"]),
            "--out-root",
            str(env["out"]),
            "--device",
            "cpu",
            "--workers",
            "0",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=env["repo"],
    )


def test_runner_produces_every_artifact_without_touching_the_test_split(experiment_env):
    env = experiment_env
    result = _run(env)
    assert result.returncode == 0, result.stderr[-3000:]
    out = env["out"] / "EXP-T"
    for name in (
        "config.json",
        "metrics.csv",
        "training_curves.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "best_model.pth",
        "inference_benchmark.json",
        "experiment_report.md",
        "checkpoints/head/best.pt",
        "checkpoints/partial/last.pt",
    ):
        assert (out / name).exists(), name

    config = json.loads((out / "config.json").read_text())
    assert config["train_images"] == 32 and config["val_images"] == 16
    assert config["preprocess"]["clahe"] == {"clip_limit": 2.0, "tile_grid_size": 8}
    assert config["stages"][1]["trainable_blocks"] == 2
    assert config["optimizer"]["lr_step_size"] == 1

    with (out / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["stage"], r["epoch"]) for r in rows] == [
        ("head", "1"),
        ("head", "2"),
        ("partial", "1"),
    ]
    assert [r["global_epoch"] for r in rows] == ["1", "2", "3"]
    assert float(rows[1]["lr_head"]) == pytest.approx(0.0005), "StepLR(1, 0.5) halves the head lr"
    assert rows[2]["lr_backbone"] == "3e-05"
    assert all(float(r["val_f1_macro"]) >= 0 for r in rows)
    assert int(rows[0]["trainable_parameters"]) < int(rows[2]["trainable_parameters"])

    best = load_checkpoint(out / "best_model.pth")
    assert best.class_names == list(CANONICAL_CLASSES)
    bench = json.loads((out / "inference_benchmark.json").read_text())
    assert bench["total_parameters"] == 4_017_796 and bench["ms_per_image_batch1"] > 0
    assert bench["best_stage"] in ("head", "partial")
    report = (out / "experiment_report.md").read_text()
    assert "frozen test split was not read" in report and "macro F1" in report

    again = _run(env)
    assert again.returncode == 2 and "already holds a completed run" in again.stdout


def test_runner_resumes_from_a_stage_checkpoint(experiment_env):
    env = experiment_env
    assert _run(env).returncode == 0
    out = env["out"] / "EXP-T"
    resumed = _run(env, "--resume", str(out / "checkpoints" / "head" / "last.pt"))
    assert resumed.returncode == 0, resumed.stderr[-2000:]
    with (out / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["stage"], r["epoch"]) for r in rows] == [
        ("head", "1"),
        ("head", "2"),
        ("partial", "1"),
    ]
