"""scripts/verify_drive_delivery.py: committed key mapping, counts, resolution — no guessing."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from src.dataset_cleaning import CLEAN_MANIFEST_NAME, build_clean_manifest, write_clean_manifest
from src.multi_dataset import DATASET_SOURCES
from tests.test_split_v2 import _many_records

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "verify_drive_delivery.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_drive_delivery", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _drop_from_manifest(repo: Path, drop: Path, clean) -> None:
    """Materialise every manifest row inside a delivery-shaped drop (wrapper folder for
    current_freshwater, delivered_subdir for the others)."""
    subdir = {s.key: s.delivered_subdir for s in DATASET_SOURCES}
    for r in clean:
        rel = Path(r.filepath).relative_to(Path("data") / "raw" / r.source_dataset)
        base = drop / (
            "Fresh_water_disease"
            if r.source_dataset == "current_freshwater"
            else subdir[r.source_dataset]
        )
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x89PNG fake " + r.image_id.encode())
    (drop / "Fresh_water_disease" / "test.csv").write_text("filename,label\n")
    (drop / "Fresh_water_disease" / "test_split").mkdir(exist_ok=True)
    (drop / "Fresh_water_disease" / "train_split").mkdir(exist_ok=True)
    for sub in subdir.values():  # every delivered folder exists, even without fixture rows
        if sub:
            (drop / sub).mkdir(parents=True, exist_ok=True)


def test_maps_keys_counts_and_resolution(tmp_path):
    repo = tmp_path / "repo"
    clean = build_clean_manifest(_many_records(40))
    audit = repo / "data" / "audit"
    write_clean_manifest(clean, audit / CLEAN_MANIFEST_NAME)
    drop = tmp_path / "MyDrive" / "AquaHealth"
    _drop_from_manifest(repo, drop, clean)
    module = _load()
    report = module.check_delivery(drop, repo)
    assert report["ok"], report["errors"]
    keys_in_fixture = {r.source_dataset for r in clean}
    for key, e in report["keys"].items():
        if key in keys_in_fixture:
            n = sum(1 for r in clean if r.source_dataset == key)
            assert e["images_found"] == e["inventory_all"] == n
            assert e["manifest_rows_resolved"] == f"{n}/{n}"
            assert e["surplus_vs_inventory"] == 0
    # the CLI writes the report and exits 0
    out = tmp_path / "drive_delivery.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source",
            str(drop),
            "--repo-root",
            str(repo),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0 and "RESULT: DELIVERY MAPPED" in result.stdout, result.stderr
    assert json.loads(out.read_text())["ok"]
    # a renamed delivery folder is reported as unresolved, never guessed
    src = next(s for s in DATASET_SOURCES if s.key == "kaptai")
    (drop / src.delivered_subdir).rename(drop / "Fresh Water Fish Dataset (1)")
    report = module.check_delivery(drop, repo)
    assert not report["ok"] and report["keys"]["kaptai"]["resolved"] is None
    assert any("kaptai" in err for err in report["errors"])
    (drop / "Fresh Water Fish Dataset (1)").rename(drop / src.delivered_subdir)
    # a missing INCLUDED image is fatal; a surplus raw file is only reported
    included = next(r for r in clean if r.included and r.source_dataset == "roboflow")
    path = module.resolve_key(drop, "roboflow")[""] / Path(included.filepath).relative_to(
        Path("data") / "raw" / "roboflow"
    )
    path.unlink()
    report = module.check_delivery(drop, repo)
    assert not report["ok"] and "roboflow: 1 included manifest rows missing" in report["errors"]
    path.write_bytes(b"x")
    (module.resolve_key(drop, "roboflow")[""] / "extra.jpg").write_bytes(b"x")
    report = module.check_delivery(drop, repo)
    assert report["ok"] and report["keys"]["roboflow"]["surplus_vs_inventory"] == 1
