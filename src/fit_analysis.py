"""Phase 12 — fit analysis of ONE finished experiment from its saved records.

Reads history.csv (per-epoch train/validation loss, accuracy, precision, recall,
Macro-F1) and metrics.json (best epoch, per-class P/R/F1, confusion) of an
experiment directory and writes fit_analysis.json + fit_analysis.md: the
train/validation Macro-F1 gap, the loss curves, convergence and stability, and
an evidence-based classification into

    underfitting | harmful_overfitting | mild_generalisation_gap | reasonable_fit

plus an `unstable` flag. The thresholds below are stated in the report; they
are an observation aid, not a decision rule — no hyper-parameter is changed
here (Phase 13 handles corrections under separate `_corrected` ids). The
frozen final test is never read.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

# thresholds (documented in every report)
UNDERFIT_VAL_F1 = 0.60  # best validation Macro-F1 below this = model has not learnt the task
UNDERFIT_TRAIN_F1 = 0.80  # training Macro-F1 below this at the end = not fitting the training set
GAP_MILD = 0.05  # train - validation Macro-F1 above this = a generalisation gap worth noting
LOSS_RISE_HARMFUL = 1.15  # final validation loss > this x its minimum = validation deterioration
EARLY_BEST_MARGIN = 5  # ... together with a best epoch at least this many epochs before the end
UNSTABLE_SD = 0.03  # standard deviation of the last-10 validation Macro-F1 above this = unstable


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def analyse(exp_dir: Path) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    with (exp_dir / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    if not history:
        raise ValueError(f"{exp_dir}: empty history")
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    summary = json.loads((exp_dir / "run_summary.json").read_text())
    best = max(history, key=lambda r: _f(r, "val_f1_macro"))
    last = history[-1]
    tail = history[-10:]
    val_losses = [_f(r, "val_loss") for r in history]
    min_loss_row = min(history, key=lambda r: _f(r, "val_loss"))
    epochs = len(history)
    best_epoch = int(best["epoch"])
    gap = _f(best, "train_f1_macro") - _f(best, "val_f1_macro")
    tail_f1 = [_f(r, "val_f1_macro") for r in tail]
    tail_sd = statistics.pstdev(tail_f1) if len(tail_f1) > 1 else 0.0
    loss_ratio = _f(last, "val_loss") / min(val_losses) if min(val_losses) > 0 else 1.0
    underfitting = (
        _f(best, "val_f1_macro") < UNDERFIT_VAL_F1 or _f(last, "train_f1_macro") < UNDERFIT_TRAIN_F1
    )
    harmful = loss_ratio > LOSS_RISE_HARMFUL and best_epoch <= epochs - EARLY_BEST_MARGIN
    unstable = tail_sd > UNSTABLE_SD
    if underfitting:
        verdict = "underfitting"
    elif harmful:
        verdict = "harmful_overfitting"
    elif gap > GAP_MILD:
        verdict = "mild_generalisation_gap"
    else:
        verdict = "reasonable_fit"
    per_class = metrics["per_class"]
    weakest = min(per_class, key=lambda c: c["f1"])
    return {
        "experiment_id": summary["experiment_id"],
        "model": summary["model_key"],
        "fold": summary["fold"],
        "data_arm": summary["data_arm"],
        "epochs_run": epochs,
        "epochs_configured": summary.get("epochs_configured"),
        "stopped_early": summary.get("stopped_early"),
        "best_epoch": best_epoch,
        "best": {
            "val_f1_macro": _f(best, "val_f1_macro"),
            "val_accuracy": _f(best, "val_accuracy"),
            "val_precision_macro": _f(best, "val_precision_macro"),
            "val_recall_macro": _f(best, "val_recall_macro"),
            "val_f1_weighted": _f(best, "val_f1_weighted"),
            "val_loss": _f(best, "val_loss"),
            "train_f1_macro": _f(best, "train_f1_macro"),
            "train_loss": _f(best, "train_loss"),
        },
        "final": {
            "val_f1_macro": _f(last, "val_f1_macro"),
            "val_loss": _f(last, "val_loss"),
            "train_f1_macro": _f(last, "train_f1_macro"),
            "train_loss": _f(last, "train_loss"),
        },
        "gap_train_minus_val_f1_at_best": gap,
        "val_loss_min": _f(min_loss_row, "val_loss"),
        "val_loss_min_epoch": int(min_loss_row["epoch"]),
        "val_loss_final_over_min": loss_ratio,
        "last10_val_f1_mean": statistics.mean(tail_f1),
        "last10_val_f1_sd": tail_sd,
        "last10_val_f1_min": min(tail_f1),
        "last10_val_f1_max": max(tail_f1),
        "train_seconds": sum(_f(r, "seconds") for r in history),
        "seconds_per_epoch": statistics.mean(_f(r, "seconds") for r in history),
        "weakest_class": {
            "name": weakest["name"],
            "f1": weakest["f1"],
            "recall": weakest["recall"],
        },
        "per_class": per_class,
        "verdict": verdict,
        "unstable": unstable,
        "thresholds": {
            "underfit_val_f1": UNDERFIT_VAL_F1,
            "underfit_train_f1": UNDERFIT_TRAIN_F1,
            "gap_mild": GAP_MILD,
            "loss_rise_harmful": LOSS_RISE_HARMFUL,
            "early_best_margin": EARLY_BEST_MARGIN,
            "unstable_sd": UNSTABLE_SD,
        },
        "correction_applied": False,
    }


def _explanation(a: dict[str, Any]) -> str:
    t = a["thresholds"]
    if a["verdict"] == "underfitting":
        return (
            "The model has not fitted the task (validation Macro-F1 below "
            f"{t['underfit_val_f1']} or training Macro-F1 below {t['underfit_train_f1']})."
        )
    if a["verdict"] == "harmful_overfitting":
        return (
            f"The validation loss ends more than {t['loss_rise_harmful']:.2f}x above its minimum "
            f"while the best epoch lies at least {t['early_best_margin']} epochs before the end: "
            "validation deteriorated after the best epoch."
        )
    if a["verdict"] == "mild_generalisation_gap":
        return (
            f"Training Macro-F1 exceeds validation Macro-F1 by more than {t['gap_mild']} but the "
            f"validation loss did not deteriorate (final loss {a['val_loss_final_over_min']:.2f}x "
            "its minimum): a capacity/data-size gap, not harmful overfitting."
        )
    return (
        f"Training and validation Macro-F1 are within {t['gap_mild']} and the validation loss "
        "did not deteriorate."
    )


def render_markdown(a: dict[str, Any]) -> str:
    b, f, t = a["best"], a["final"], a["thresholds"]
    arm = a["data_arm"].replace("_", "-").upper()
    row = "| {} | {} |".format
    lines = [
        f"# Fit analysis — {a['experiment_id']}",
        "",
        f"Model `{a['model']}` · fold {a['fold']:02d} · {arm} · dataset configuration v3. "
        "Observation record generated from `history.csv` / `metrics.json`; "
        "**no hyper-parameter was changed** (Phase 13 handles corrections under separate "
        "`_corrected` ids).",
        "",
        "| quantity | value |",
        "|---|---|",
        row(
            "epochs run / configured",
            f"{a['epochs_run']} / {a['epochs_configured']} (early stop: {a['stopped_early']})",
        ),
        row("best epoch (validation Macro-F1)", a["best_epoch"]),
        row(
            "validation at best: Macro-F1 / accuracy / precision / recall / weighted F1",
            f"{b['val_f1_macro']:.4f} / {b['val_accuracy']:.4f} / {b['val_precision_macro']:.4f}"
            f" / {b['val_recall_macro']:.4f} / {b['val_f1_weighted']:.4f}",
        ),
        row(
            "train vs validation Macro-F1 at best",
            f"{b['train_f1_macro']:.3f} vs {b['val_f1_macro']:.3f} "
            f"(gap **{a['gap_train_minus_val_f1_at_best']:+.3f}**)",
        ),
        row("train vs validation loss at best", f"{b['train_loss']:.3f} vs {b['val_loss']:.3f}"),
        row(
            "validation loss minimum",
            f"{a['val_loss_min']:.3f} at epoch {a['val_loss_min_epoch']}; final {f['val_loss']:.3f}"
            f" (= {a['val_loss_final_over_min']:.2f} x minimum)",
        ),
        row(
            "last-10-epoch validation Macro-F1",
            f"mean {a['last10_val_f1_mean']:.3f}, sd {a['last10_val_f1_sd']:.3f} "
            f"(min {a['last10_val_f1_min']:.3f}, max {a['last10_val_f1_max']:.3f})",
        ),
        row(
            "runtime",
            f"{a['train_seconds']:.0f} s ({a['train_seconds'] / 60:.1f} min), "
            f"{a['seconds_per_epoch']:.1f} s/epoch",
        ),
        row(
            "weakest class",
            f"{a['weakest_class']['name']} (F1 {a['weakest_class']['f1']:.3f}, "
            f"recall {a['weakest_class']['recall']:.3f})",
        ),
        "",
        f"## Verdict: `{a['verdict']}`" + (" — **unstable**" if a["unstable"] else ""),
        "",
        _explanation(a),
        "",
        f"Stability: last-10-epoch validation Macro-F1 sd {a['last10_val_f1_sd']:.3f} "
        + ("exceeds" if a["unstable"] else "is within")
        + f" the {t['unstable_sd']} threshold.",
        "",
        "## Per-class (best epoch, real validation fold only)",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    for c in a["per_class"]:
        lines.append(
            f"| {c['name']} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | "
            f"{c['support']} |"
        )
    lines += ["", "Confusion matrix: `confusion_matrix.csv`. **Correction applied: none.**", ""]
    return "\n".join(lines)


def write_fit_analysis(exp_dir: Path) -> dict[str, Any]:
    exp_dir = Path(exp_dir)
    a = analyse(exp_dir)
    (exp_dir / "fit_analysis.json").write_text(json.dumps(a, indent=2))
    (exp_dir / "fit_analysis.md").write_text(render_markdown(a))
    return a
