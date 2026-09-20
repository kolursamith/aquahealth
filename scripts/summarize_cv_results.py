#!/usr/bin/env python
"""Phase 12 — tabulate finished CV experiments from their saved records (no training,
no test split, no ranking).

    python scripts/summarize_cv_results.py --model cnn_vit_lstm --data-arm with_gan
    python scripts/summarize_cv_results.py --data-arm with_gan --all-models

Per (model, data arm): one row per fold — best validation Macro-F1, accuracy,
precision, recall, weighted F1, best epoch, epochs run, training seconds, fit
verdict, status — plus mean / sd / min / max over the COMPLETED folds. Written to
results/v2/summaries/<model>_<arm>.csv / .md. With --all-models, also
results/v2/summaries/<arm>_matrix_report.md: every model's per-fold table, per-class
mean F1 over folds, pooled confusion matrix, training time, successful / failed
counts and the reproducibility facts shared by all runs. Validation-fold numbers only;
final_test.csv is never opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR  # noqa: E402
from src.cv_runner import DATA_ARMS, EXPERIMENTS_DIR, experiment_id, read_status  # noqa: E402
from src.manifest import CANONICAL_CLASSES  # noqa: E402
from src.model_factory import HYBRID_MODELS  # noqa: E402

N_FOLDS = 10
RECORD_FILES = (
    "config.json",
    "history.csv",
    "metrics.json",
    "confusion_matrix.csv",
    "run_summary.json",
    "status.json",
)


def records_complete(out: Path) -> bool:
    """COMPLETED status and every small record present. Checkpoints (best.pt / latest.pt)
    live on Drive and are not required to tabulate; the runner's own is_completed() still
    demands them where the run happened."""
    return read_status(out) == "COMPLETED" and all((out / n).is_file() for n in RECORD_FILES)


COLUMNS = (
    "experiment_id",
    "model",
    "fold",
    "data_arm",
    "status",
    "best_epoch",
    "epochs_run",
    "val_f1_macro",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_weighted",
    "val_loss",
    "train_f1_macro_at_best",
    "gap_f1",
    "fit_verdict",
    "unstable",
    "train_seconds",
    "train_images",
    "train_synthetic",
    "validation_images",
    "git_commit",
    "config_hash",
)


def fold_row(root: Path, model: str, fold: int, arm: str) -> dict[str, Any]:
    exp = experiment_id(model, fold, arm)
    out = root / exp
    row: dict[str, Any] = {k: "" for k in COLUMNS}
    row.update({"experiment_id": exp, "model": model, "fold": fold, "data_arm": arm})
    row["status"] = read_status(out) if out.is_dir() else "PENDING"
    if row["status"] == "COMPLETED" and not records_complete(out):
        row["status"] = "FAILED"  # status says COMPLETED but records are missing
    if row["status"] != "COMPLETED":
        if (out / "status.json").is_file():
            row["fit_verdict"] = json.loads((out / "status.json").read_text()).get("error", "")
        return row
    summary = json.loads((out / "run_summary.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    config = json.loads((out / "config.json").read_text())
    with (out / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    best = max(history, key=lambda r: float(r["val_f1_macro"]))
    fit = {}
    if (out / "fit_analysis.json").is_file():
        fit = json.loads((out / "fit_analysis.json").read_text())
    s = metrics["summary"]
    row.update(
        {
            "best_epoch": summary["best_epoch"],
            "epochs_run": summary["epochs_run"],
            "val_f1_macro": round(s["f1_macro"], 6),
            "val_accuracy": round(s["accuracy"], 6),
            "val_precision_macro": round(s["precision_macro"], 6),
            "val_recall_macro": round(s["recall_macro"], 6),
            "val_f1_weighted": round(s["f1_weighted"], 6),
            "val_loss": round(s["loss"], 6),
            "train_f1_macro_at_best": round(float(best["train_f1_macro"]), 6),
            "gap_f1": round(float(best["train_f1_macro"]) - s["f1_macro"], 6),
            "fit_verdict": fit.get("verdict", ""),
            "unstable": fit.get("unstable", ""),
            "train_seconds": round(sum(float(r["seconds"]) for r in history), 1),
            "train_images": summary["train_images"],
            "train_synthetic": summary["train_synthetic"],
            "validation_images": summary["validation_images"],
            "git_commit": (config.get("git_commit") or "")[:12],
            "config_hash": config.get("config_hash", "")[:12],
        }
    )
    return row


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(root: Path, model: str, arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [fold_row(root, model, f, arm) for f in range(1, N_FOLDS + 1)]
    done = [r for r in rows if r["status"] == "COMPLETED"]
    agg = {
        "model": model,
        "data_arm": arm,
        "completed": len(done),
        "failed": sum(r["status"] == "FAILED" for r in rows),
        "pending_or_running": sum(
            r["status"] in ("PENDING", "RUNNING", "INTERRUPTED") for r in rows
        ),
        "f1_macro": stats([r["val_f1_macro"] for r in done]),
        "accuracy": stats([r["val_accuracy"] for r in done]),
        "precision_macro": stats([r["val_precision_macro"] for r in done]),
        "recall_macro": stats([r["val_recall_macro"] for r in done]),
        "train_seconds_total": sum(r["train_seconds"] for r in done),
        "verdicts": dict(Counter(r["fit_verdict"] for r in done)),
    }
    return rows, agg


def _fmt(x: float | None, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def per_class_and_confusion(
    root: Path, model: str, arm: str
) -> tuple[dict[str, dict], list[list[int]]]:
    """Mean per-class P/R/F1 over COMPLETED folds and the confusion matrices summed over
    folds (each validation image appears in exactly one fold, so the pooled matrix covers
    the development set once)."""
    acc: dict[str, dict[str, list[float]]] = {
        c: {"precision": [], "recall": [], "f1": [], "support": []} for c in CANONICAL_CLASSES
    }
    pooled = [[0] * len(CANONICAL_CLASSES) for _ in CANONICAL_CLASSES]
    for fold in range(1, N_FOLDS + 1):
        out = root / experiment_id(model, fold, arm)
        if not records_complete(out):
            continue
        m = json.loads((out / "metrics.json").read_text())
        for c in m["per_class"]:
            for k in ("precision", "recall", "f1", "support"):
                acc[c["name"]][k].append(c[k])
        for i, row in enumerate(m["confusion"]):
            for j, v in enumerate(row):
                pooled[i][j] += int(v)
    means = {
        c: {k: (statistics.mean(v) if v else None) for k, v in d.items()}
        | {"folds": len(d["f1"]), "support_total": sum(d["support"])}
        for c, d in acc.items()
    }
    return means, pooled


def write_model_summary(out_dir: Path, rows: list[dict[str, Any]], agg: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{agg['model']}_{agg['data_arm']}"
    with (out_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    f1, acc = agg["f1_macro"], agg["accuracy"]
    lines = [
        f"# {agg['model']} — {agg['data_arm'].replace('_', '-').upper()} — 10-fold CV "
        "(dataset configuration v3, validation folds only)",
        "",
        "| Fold | Val Macro-F1 | Val Accuracy | Precision | Recall | Weighted F1 | Best Epoch | "
        "Epochs | Train s | Fit | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["status"] == "COMPLETED":
            lines.append(
                f"| {r['fold']:02d} | {r['val_f1_macro']:.4f} | {r['val_accuracy']:.4f} | "
                f"{r['val_precision_macro']:.4f} | {r['val_recall_macro']:.4f} | "
                f"{r['val_f1_weighted']:.4f} | {r['best_epoch']} | {r['epochs_run']} | "
                f"{r['train_seconds']:.0f} | {r['fit_verdict']}"
                f"{' (unstable)' if r['unstable'] is True else ''} | COMPLETED |"
            )
        else:
            lines.append(
                f"| {r['fold']:02d} | — | — | — | — | — | — | — | — | {r['fit_verdict']} | "
                f"{r['status']} |"
            )
    lines += [
        "",
        f"Completed {agg['completed']}/{N_FOLDS} · failed {agg['failed']} · "
        f"pending/running {agg['pending_or_running']}",
        "",
        "| statistic | Macro-F1 | Accuracy |",
        "|---|---|---|",
        f"| mean | {_fmt(f1['mean'])} | {_fmt(acc['mean'])} |",
        f"| standard deviation (population, over folds) | {_fmt(f1['sd'])} | {_fmt(acc['sd'])} |",
        f"| minimum | {_fmt(f1['min'])} | {_fmt(acc['min'])} |",
        f"| maximum | {_fmt(f1['max'])} | {_fmt(acc['max'])} |",
        "",
        f"Total training time {agg['train_seconds_total'] / 3600:.2f} h; fit verdicts "
        f"{agg['verdicts']}. No architecture ranking is made here; the frozen final test was "
        "not used.",
        "",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(lines))
    (out_dir / f"{stem}.json").write_text(json.dumps(agg, indent=2))
    return out_dir / f"{stem}.md"


def write_matrix_report(root: Path, out_dir: Path, arm: str, models: tuple[str, ...]) -> Path:
    lines = [
        f"# Phase 12 — {arm.replace('_', '-').upper()} matrix report "
        f"({len(models)} architectures × {N_FOLDS} folds, dataset configuration v3)",
        "",
        "Validation-fold results only. **No ranking is implied by row order**; the frozen final "
        "test was never opened. GAN data (WITH-GAN arm) = official Layer 3 artefacts only.",
        "",
        "## Per-architecture aggregate",
        "",
        "| architecture | completed | failed | mean Macro-F1 | sd | min | max | mean accuracy "
        "| sd | total train h |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    total_done = total_failed = 0
    commits: Counter[str] = Counter()
    hashes: Counter[str] = Counter()
    for model in models:
        rows, agg = summarize(root, model, arm)
        write_model_summary(out_dir, rows, agg)
        f1, acc = agg["f1_macro"], agg["accuracy"]
        total_done += agg["completed"]
        total_failed += agg["failed"]
        commits.update(r["git_commit"] for r in rows if r["status"] == "COMPLETED")
        hashes.update(r["config_hash"] for r in rows if r["status"] == "COMPLETED")
        lines.append(
            f"| {model} | {agg['completed']} | {agg['failed']} | {_fmt(f1['mean'])} | "
            f"{_fmt(f1['sd'])} | {_fmt(f1['min'])} | {_fmt(f1['max'])} | {_fmt(acc['mean'])} | "
            f"{_fmt(acc['sd'])} | "
            f"{agg['train_seconds_total'] / 3600:.2f} |"
        )
    lines += [
        "",
        f"Successful experiments: {total_done} · failed: {total_failed} · "
        f"expected: {len(models) * N_FOLDS}",
        "",
    ]
    for model in models:
        rows, _ = summarize(root, model, arm)
        lines += [
            f"## {model} — per fold",
            "",
            "| Fold | Macro-F1 | Accuracy | Best epoch | Status |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            done = r["status"] == "COMPLETED"
            f1_cell = _fmt(r["val_f1_macro"]) if done else "—"
            acc_cell = _fmt(r["val_accuracy"]) if done else "—"
            epoch_cell = r["best_epoch"] if done else "—"
            lines.append(
                f"| {r['fold']:02d} | {f1_cell} | {acc_cell} | {epoch_cell} | {r['status']} |"
            )
        means, pooled = per_class_and_confusion(root, model, arm)
        lines += [
            "",
            f"### {model} — per-class mean over completed folds",
            "",
            "| class | precision | recall | F1 | folds | validation images |",
            "|---|---|---|---|---|---|",
        ]
        for c, d in means.items():
            lines.append(
                f"| {c} | {_fmt(d['precision'], 3)} | {_fmt(d['recall'], 3)} | "
                f"{_fmt(d['f1'], 3)} | {d['folds']} | {d['support_total']} |"
            )
        lines += [
            "",
            f"### {model} — pooled confusion matrix (rows = true, sum over completed folds)",
            "",
            "| true \\ predicted | " + " | ".join(CANONICAL_CLASSES) + " |",
            "|---|" + "---|" * len(CANONICAL_CLASSES),
        ]
        for name, row in zip(CANONICAL_CLASSES, pooled):
            lines.append(f"| {name} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    lines += [
        "## Reproducibility",
        "",
        f"git commits of completed runs: {dict(commits)}; configuration hashes: {dict(hashes)}; "
        "seed 42; preprocessing `configs/preprocess_v2_clahe.json`; manifests `data/audit/cv_v3/` "
        "(WITH-GAN train lists `data/gan/fold_XX/fold_XX_train_gan.csv` from the official "
        "Layer 3 run); per-run environment (Python, torch, torchvision, CUDA, GPU) in each "
        "`config.json` / checkpoint.",
        "",
        "Confirmations: `final_test.csv` was read for ids only (isolation guard) and never "
        "evaluated; MatsyaDx-BD/Mendeley is not part of dataset configuration v3; synthetic "
        "images come only from `results/v2/gan/` (Colab CUDA, verified 229/229) — the runner "
        "refuses any other source.",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}_matrix_report.md"
    path.write_text("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--data-arm", choices=DATA_ARMS, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    args = parser.parse_args(argv)
    root = Path(args.repo_root) / EXPERIMENTS_DIR
    out_dir = Path(args.repo_root) / "results" / "v2" / "summaries"
    if args.all_models:
        path = write_matrix_report(root, out_dir, args.data_arm, HYBRID_MODELS)
    elif args.model:
        rows, agg = summarize(root, args.model, args.data_arm)
        path = write_model_summary(out_dir, rows, agg)
    else:
        parser.error("--model or --all-models is required")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
