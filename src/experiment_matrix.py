"""The 100-experiment matrix (Phase 12): 5 hybrid models x 10 folds x 2 data arms.

    results/v2/experiment_matrix.csv   experiment_id, model, fold, data_arm,
                                       train_manifest, validation_manifest, status

Generated once with every status PENDING; `refresh_statuses` reads each experiment
directory's status.json (src/cv_runner.py) and updates the column. The frozen
final_test.csv never appears in the matrix.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from src.cv_runner import (
    DATA_ARMS,
    EXPERIMENTS_DIR,
    RESULTS_V2,
    experiment_id,
    is_completed,
    manifest_paths,
    read_status,
)
from src.hybrid_models import HYBRID_MODELS

MATRIX_NAME = "experiment_matrix.csv"
MATRIX_COLUMNS = (
    "experiment_id",
    "model",
    "fold",
    "data_arm",
    "train_manifest",
    "validation_manifest",
    "status",
)
MATRIX_MODELS: tuple[str, ...] = tuple(HYBRID_MODELS)  # the five hybrids, in registry order
FOLDS = tuple(range(1, 11))


def build_matrix(data_dir: Path, repo_root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in MATRIX_MODELS:
        for fold in FOLDS:
            for arm in DATA_ARMS:
                train, validation = manifest_paths(data_dir, fold, arm)
                rows.append(
                    {
                        "experiment_id": experiment_id(model, fold, arm),
                        "model": model,
                        "fold": f"{fold:02d}",
                        "data_arm": arm,
                        "train_manifest": _relative(train, repo_root),
                        "validation_manifest": _relative(validation, repo_root),
                        "status": "PENDING",
                    }
                )
    validate_matrix(rows)
    return rows


def _relative(path: Path, root: Path) -> str:
    path, root = Path(path), Path(root)
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()


def validate_matrix(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 100:
        raise ValueError(f"matrix must have 100 rows, has {len(rows)}")
    combos = Counter((r["model"], r["fold"], r["data_arm"]) for r in rows)
    if len(combos) != 100 or max(combos.values()) != 1:
        raise ValueError("matrix has duplicate or missing model/fold/data_arm combinations")
    ids = {r["experiment_id"] for r in rows}
    if len(ids) != 100:
        raise ValueError("experiment ids are not unique")
    models = {r["model"] for r in rows}
    folds = {r["fold"] for r in rows}
    arms = {r["data_arm"] for r in rows}
    if (
        models != set(MATRIX_MODELS)
        or folds != {f"{f:02d}" for f in FOLDS}
        or arms != set(DATA_ARMS)
    ):
        raise ValueError("matrix does not cover exactly 5 models x 10 folds x 2 data arms")
    for f in folds:
        if sum(r["fold"] == f for r in rows) != 10:
            raise ValueError(f"fold {f} must have 10 experiments")
    for m in models:
        if sum(r["model"] == m for r in rows) != 20:
            raise ValueError(f"model {m} must have 20 experiments")
    for a in arms:
        if sum(r["data_arm"] == a for r in rows) != 50:
            raise ValueError(f"data arm {a} must have 50 experiments")
    for r in rows:
        train, val = Path(r["train_manifest"]).name, Path(r["validation_manifest"]).name
        expected = (
            f"fold_{r['fold']}_train.csv"
            if r["data_arm"] == "without_gan"
            else f"fold_{r['fold']}_train_gan.csv"
        )
        if train != expected:
            raise ValueError(f"{r['experiment_id']}: train manifest {train} != {expected}")
        if val != f"fold_{r['fold']}_validation.csv":
            raise ValueError(f"{r['experiment_id']}: validation manifest {val} is wrong")
        if "final_test" in r["train_manifest"] or "final_test" in r["validation_manifest"]:
            raise ValueError(f"{r['experiment_id']}: final_test.csv must not appear")
        if r["status"] not in ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED"):
            raise ValueError(f"{r['experiment_id']}: bad status {r['status']!r}")


def write_matrix(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MATRIX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_matrix(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_matrix(rows)
    return rows


def refresh_statuses(rows: list[dict[str, Any]], experiments_root: Path) -> list[dict[str, Any]]:
    """Status column <- each experiment directory's status.json; COMPLETED only when the
    required output files all exist."""
    for r in rows:
        out_dir = Path(experiments_root) / r["experiment_id"]
        status = read_status(out_dir)
        if status == "COMPLETED" and not is_completed(out_dir):
            status = "FAILED"
        r["status"] = status
    validate_matrix(rows)
    return rows


def matrix_path(repo_root: Path) -> Path:
    return Path(repo_root) / RESULTS_V2 / MATRIX_NAME


def experiments_root(repo_root: Path) -> Path:
    return Path(repo_root) / EXPERIMENTS_DIR
