"""Phase 12 batch driver + fit analysis + summaries, on the runner's miniature repository
(CPU, the 2-epoch smoke config). Nothing here touches the real data or the final test."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from src.fit_analysis import analyse, write_fit_analysis
from tests.test_cv_runner import repo  # noqa: F401  (fixture re-export)

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def _batch(repo_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_cv_batch.py"),
            "--models",
            "cnn_bilstm",
            "--folds",
            "1",
            "--data-arm",
            "without_gan",
            "--config",
            "configs/cv_v2/smoke.json",
            "--repo-root",
            str(repo_dir),
            "--data-dir",
            str(repo_dir / "data"),
            "--device",
            "cpu",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_batch_trains_gates_analyses_summarises_and_never_reruns(repo):  # noqa: F811
    # the preflight wants the pins file and the experiment matrix of the repository
    (repo / "requirements").mkdir()
    shutil.copy(ROOT / "requirements" / "base.txt", repo / "requirements" / "base.txt")
    built = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "build_experiment_matrix.py"),
            "--data-dir",
            str(repo / "data"),
            "--repo-root",
            str(repo),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert built.returncode == 0, built.stderr
    first = _batch(repo)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    exp = repo / "results" / "v2" / "experiments" / "cnn_bilstm_fold01_without_gan"
    for name in (
        "config.json",
        "history.csv",
        "metrics.json",
        "confusion_matrix.csv",
        "best.pt",
        "latest.pt",
        "run_summary.json",
        "status.json",
        "fit_analysis.json",
        "fit_analysis.md",
    ):
        assert (exp / name).is_file(), name
    assert json.loads((exp / "status.json").read_text())["status"] == "COMPLETED"
    analysis = json.loads((exp / "fit_analysis.json").read_text())
    assert analysis["verdict"] in (
        "underfitting",
        "harmful_overfitting",
        "mild_generalisation_gap",
        "reasonable_fit",
    )
    assert analysis["correction_applied"] is False and analysis["epochs_run"] == 2
    md = (exp / "fit_analysis.md").read_text()
    assert "no hyper-parameter was changed" in md and "Correction applied: none" in md
    # the gate output was written before training and the matrix refreshed after it
    logs = list((repo / "results" / "v2" / "batches").glob("batch_without_gan_*.console.log"))
    assert logs and '"ready": true' in logs[0].read_text()
    matrix = (repo / "results" / "v2" / "experiment_matrix.csv").read_text()
    assert "cnn_bilstm_fold01_without_gan,cnn_bilstm,01,without_gan" in matrix
    assert matrix.count("COMPLETED") == 1
    # per-model summary with the aggregate statistics
    summary = repo / "results" / "v2" / "summaries" / "cnn_bilstm_without_gan.md"
    assert summary.is_file() and "| 01 |" in summary.read_text()
    agg = json.loads(
        (repo / "results" / "v2" / "summaries" / "cnn_bilstm_without_gan.json").read_text()
    )
    assert agg["completed"] == 1 and agg["f1_macro"]["n"] == 1
    # a second batch never reruns a COMPLETED experiment (checkpoint bytes untouched)
    before = (exp / "best.pt").stat().st_mtime_ns
    second = _batch(repo)
    assert second.returncode == 0, second.stdout[-2000:]
    assert (exp / "best.pt").stat().st_mtime_ns == before
    log_lines = [
        json.loads(line)
        for path in (repo / "results" / "v2" / "batches").glob("batch_without_gan_*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert {entry["outcome"] for entry in log_lines} == {"completed", "skipped_completed"}


def test_fit_analysis_verdicts_from_curves(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    cols = (
        "epoch,stage,train_loss,train_accuracy,train_precision_macro,train_recall_macro,"
        "train_f1_macro,val_loss,val_accuracy,val_precision_macro,val_recall_macro,val_f1_macro,"
        "val_f1_weighted,lr_head,lr_backbone,seconds,improved"
    )

    def write(rows):
        lines = [cols]
        for e, (tl, tf, vl, vf) in enumerate(rows, start=1):
            lines.append(
                f"{e},full,{tl},{tf},{tf},{tf},{tf},{vl},{vf},{vf},{vf},{vf},{vf},1e-4,1e-5,10,True"
            )
        (exp / "history.csv").write_text("\n".join(lines) + "\n")
        best = max(range(len(rows)), key=lambda i: rows[i][3]) + 1
        (exp / "metrics.json").write_text(
            json.dumps(
                {
                    "epoch": best,
                    "summary": {},
                    "per_class": [
                        {"name": "A", "precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 10},
                        {"name": "B", "precision": 0.5, "recall": 0.4, "f1": 0.44, "support": 10},
                    ],
                }
            )
        )
        (exp / "run_summary.json").write_text(
            json.dumps(
                {
                    "experiment_id": "x_fold01_without_gan",
                    "model_key": "x",
                    "fold": 1,
                    "data_arm": "without_gan",
                    "epochs_configured": len(rows),
                    "stopped_early": False,
                    "best_epoch": best,
                }
            )
        )

    # validation loss climbs 40% after an early best epoch -> harmful overfitting
    write(
        [(1.0, 0.5, 1.0, 0.5), (0.4, 0.8, 0.6, 0.75), (0.2, 0.9, 0.5, 0.80)]
        + [(0.05, 0.99, 0.5 + 0.03 * k, 0.78 - 0.005 * k) for k in range(1, 9)]
    )
    a = analyse(exp)
    assert a["verdict"] == "harmful_overfitting" and a["best_epoch"] == 3
    # tiny gap, flat loss -> reasonable fit
    write([(0.5, 0.80, 0.5, 0.79), (0.3, 0.88, 0.35, 0.87), (0.25, 0.90, 0.30, 0.89)])
    assert analyse(exp)["verdict"] == "reasonable_fit"
    # never learnt -> underfitting
    write([(2.0, 0.2, 2.0, 0.2), (1.9, 0.25, 1.95, 0.22), (1.9, 0.26, 1.94, 0.23)])
    assert analyse(exp)["verdict"] == "underfitting"
    # large gap but validation loss still at its minimum -> mild gap, files written
    write([(1.0, 0.6, 1.0, 0.55), (0.2, 0.95, 0.6, 0.80), (0.05, 0.99, 0.55, 0.85)])
    a = write_fit_analysis(exp)
    assert a["verdict"] == "mild_generalisation_gap" and (exp / "fit_analysis.md").is_file()
    assert a["weakest_class"]["name"] == "B"
