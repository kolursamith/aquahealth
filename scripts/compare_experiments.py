#!/usr/bin/env python
"""Tabulate finished experiments from their saved records (no training, no test split).

Reads, per experiment directory under --root: config.json, metrics.csv,
inference_benchmark.json and (if present) eval_val/metrics.json, and writes one
CSV row per experiment plus the delta against --baseline. Only validation
numbers appear; the test split is never touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "experiment_id",
    "description",
    "best_stage",
    "best_epoch",
    "best_global_epoch",
    "val_f1_macro",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_weighted",
    "val_loss",
    "delta_f1_vs_baseline",
    "final_epoch_train_accuracy",
    "final_epoch_val_accuracy",
    "min_per_class_f1",
    "min_per_class_name",
    "ece",
    "epochs",
    "train_seconds",
    "ms_per_image_batch1",
    "total_parameters",
    "trainable_parameters",
    "clahe",
    "optimizer",
    "batch_size",
    "seed",
    "test_set_read",
]


def load_row(exp_dir: Path) -> dict:
    config = json.loads((exp_dir / "config.json").read_text())
    with (exp_dir / "metrics.csv").open() as handle:
        epochs = list(csv.DictReader(handle))
    best = max(epochs, key=lambda r: float(r["val_f1_macro"]))
    last = epochs[-1]
    bench_path = exp_dir / "inference_benchmark.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    eval_path = exp_dir / "eval_val" / "metrics.json"
    row = {
        "experiment_id": config.get("id", exp_dir.name),
        "description": config.get("description", ""),
        "best_stage": best["stage"],
        "best_epoch": int(best["epoch"]),
        "best_global_epoch": int(best["global_epoch"]),
        "val_f1_macro": round(float(best["val_f1_macro"]), 4),
        "val_accuracy": round(float(best["val_accuracy"]), 4),
        "val_precision_macro": round(float(best["val_precision_macro"]), 4),
        "val_recall_macro": round(float(best["val_recall_macro"]), 4),
        "val_f1_weighted": "",
        "val_loss": round(float(best["val_loss"]), 4),
        "delta_f1_vs_baseline": "",
        "final_epoch_train_accuracy": round(float(last["train_accuracy"]), 4),
        "final_epoch_val_accuracy": round(float(last["val_accuracy"]), 4),
        "min_per_class_f1": "",
        "min_per_class_name": "",
        "ece": "",
        "epochs": len(epochs),
        "train_seconds": bench.get("train_seconds", ""),
        "ms_per_image_batch1": bench.get("ms_per_image_batch1", ""),
        "total_parameters": bench.get("total_parameters", ""),
        "trainable_parameters": bench.get("trainable_parameters", ""),
        "clahe": "on" if config.get("preprocess", {}).get("clahe") else "off",
        "optimizer": config.get("optimizer", {}).get("optimizer", ""),
        "batch_size": config.get("batch_size", ""),
        "seed": config.get("seed", ""),
        "test_set_read": "NO",
    }
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        worst = min(ev["metrics"]["per_class"], key=lambda c: c["f1"])
        row["val_f1_weighted"] = round(ev["metrics"]["f1_weighted"], 4)
        row["min_per_class_f1"] = round(worst["f1"], 4)
        row["min_per_class_name"] = worst["name"]
        row["ece"] = round(ev["confidence"]["expected_calibration_error"], 4)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--ids", nargs="+", required=True, help="experiment ids, in table order")
    parser.add_argument("--baseline", default=None, help="id whose val F1 the deltas refer to")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [load_row(args.root / exp_id) for exp_id in args.ids]
    if args.baseline:
        base = next(r for r in rows if r["experiment_id"] == args.baseline)["val_f1_macro"]
        for row in rows:
            row["delta_f1_vs_baseline"] = round(row["val_f1_macro"] - base, 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['experiment_id']:<10} f1 {row['val_f1_macro']:.4f} "
            f"({row['delta_f1_vs_baseline']:+.4f})  acc {row['val_accuracy']:.4f}  "
            f"best {row['best_stage']}/{row['best_epoch']}  worst {row['min_per_class_name']} "
            f"{row['min_per_class_f1']}"
            if row["delta_f1_vs_baseline"] != ""
            else f"{row['experiment_id']:<10} f1 {row['val_f1_macro']:.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
