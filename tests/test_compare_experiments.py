"""compare_experiments tabulates saved experiment records without touching any split."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "compare_experiments.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare_experiments", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exp001_record_tabulates_with_zero_delta(tmp_path):
    out = tmp_path / "cmp.csv"
    module = _load()
    assert module.main(["--ids", "EXP-001", "--baseline", "EXP-001", "--out", str(out)]) == 0
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert [r["experiment_id"] for r in rows] == ["EXP-001"]
    row = rows[0]
    assert row["best_stage"] == "partial" and row["best_epoch"] == "9"
    assert float(row["val_f1_macro"]) == 0.9004
    assert float(row["delta_f1_vs_baseline"]) == 0.0
    assert row["min_per_class_name"] == "Bacterial Red Disease"
    assert row["test_set_read"] == "NO"
    assert list(row) == module.COLUMNS
