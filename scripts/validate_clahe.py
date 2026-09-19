#!/usr/bin/env python
"""Phase 7 — validate the EXISTING preprocessing / CLAHE implementation on the new
clean dataset. Nothing new is defined here: the pipeline is
src/preprocessing.py::build_eval_transform (CLAHE -> Resize -> CenterCrop ->
tensor -> ImageNet normalisation) with the project's CLAHEConfig defaults, and
images are decoded by src/dataset.py::load_image, exactly as training and the
app do.

    python scripts/validate_clahe.py                # reads data/audit/clean_manifest.csv

Writes:

    data/audit/clahe_samples/class_<unified class>.png   original | CLAHE, one image per class
    data/audit/clahe_samples/source_<dataset>.png        original | CLAHE, one image per source
    data/audit/clahe_validation.csv                      per sample: L-channel contrast before/
                                                         after, tensor shape/dtype/mean/std
    data/audit/preprocessing_config.json / .md           the configuration as the code defines it,
                                                         plus this validation's results

The comparison sheets are documentation, not training data. Raw files are
never written; the transform runs in memory only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from build_clean_manifest import PREPROCESS_V2, _table  # noqa: E402
from PIL import Image  # noqa: E402

from src.augmentation import AugmentConfig, build_train_transform  # noqa: E402
from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.dataset import ImageDecodeError, load_image  # noqa: E402
from src.dataset_cleaning import CLEAN_MANIFEST_NAME, CleanRow, read_clean_manifest  # noqa: E402
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402
from src.preprocessing import CLAHE, CLAHEConfig, build_eval_transform, denormalize  # noqa: E402
from src.utils import get_logger  # noqa: E402
from src.validation import transform_stage_names  # noqa: E402

logger = get_logger("clahe_validation")
VALIDATION_COLUMNS = (
    "sample_for",
    "image_id",
    "source_dataset",
    "unified_class",
    "original_path",
    "input_width",
    "input_height",
    "l_std_before",
    "l_std_after",
    "l_std_ratio",
    "output_shape",
    "output_dtype",
    "output_mean",
    "output_std",
    "denorm_min",
    "denorm_max",
    "seconds",
    "sheet",
    "status",
)


def l_channel_std(image: Image.Image) -> float:
    lab = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].std())


def representative_samples(rows: list[CleanRow]) -> list[tuple[str, CleanRow]]:
    """First included image (by image_id) per unified class and per source."""
    included = sorted((r for r in rows if r.included), key=lambda r: r.image_id)
    picks: list[tuple[str, CleanRow]] = []
    seen_class: set[str] = set()
    seen_source: set[str] = set()
    for r in included:
        if r.unified_class not in seen_class:
            seen_class.add(r.unified_class)
            picks.append((f"class_{r.unified_class.replace(' ', '_')}", r))
    for r in included:
        if r.source_dataset not in seen_source:
            seen_source.add(r.source_dataset)
            picks.append((f"source_{r.source_dataset}", r))
    return picks


def validate_sample(
    label: str, row: CleanRow, repo_root: Path, sample_dir: Path, transform: Any, clahe: CLAHE
) -> dict[str, Any]:
    image = load_image(repo_root / row.filepath)  # the project's decoder (Pillow, RGB)
    started = time.perf_counter()
    tensor = transform(image)  # the project's eval pipeline, CLAHE included
    seconds = time.perf_counter() - started
    enhanced = clahe(image)  # CLAHE stage alone, for the comparison sheet
    sheet = Image.new("RGB", (image.width * 2, image.height))
    sheet.paste(image, (0, 0))
    sheet.paste(enhanced, (image.width, 0))
    sheet.thumbnail((1400, 700))
    path = sample_dir / f"{label}.png"
    sheet.save(path)
    before, after = l_channel_std(image), l_channel_std(enhanced)
    dn = denormalize(tensor)
    return {
        "sample_for": label,
        "image_id": row.image_id,
        "source_dataset": row.source_dataset,
        "unified_class": row.unified_class,
        "original_path": row.original_path,
        "input_width": image.width,
        "input_height": image.height,
        "l_std_before": round(before, 3),
        "l_std_after": round(after, 3),
        "l_std_ratio": round(after / before, 3) if before else "",
        "output_shape": "x".join(map(str, tensor.shape)),
        "output_dtype": str(tensor.dtype),
        "output_mean": round(float(tensor.mean()), 4),
        "output_std": round(float(tensor.std()), 4),
        "denorm_min": round(float(dn.min()), 4),
        "denorm_max": round(float(dn.max()), 4),
        "seconds": round(seconds, 4),
        "sheet": str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path),
        "status": "ok",
    }


def corrupt_file_check(transform: Any) -> dict[str, Any]:
    """The pipeline must *report* an undecodable file, not skip it: load_image
    raises ImageDecodeError naming the file. Exercised on a throw-away file."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "corrupt.jpg"
        bad.write_bytes(b"\xff\xd8\xff not really a jpeg")
        try:
            transform(load_image(bad))
        except ImageDecodeError as exc:
            return {"status": "reported", "exception": type(exc).__name__, "message": str(exc)}
        return {"status": "NOT REPORTED", "exception": "", "message": ""}


def preprocessing_config(
    validation: list[dict[str, Any]], corrupt: dict[str, Any]
) -> dict[str, Any]:
    cfg = PREPROCESS_V2
    assert cfg.clahe is not None
    return {
        "implementation": {
            "decode": "src/dataset.py::load_image — Pillow Image.open(...).convert('RGB'); failure "
            "raises ImageDecodeError naming the file",
            "pipeline": "src/preprocessing.py::build_eval_transform (validation/test/inference) "
            "and "
            "src/augmentation.py::build_train_transform (training) — reused unchanged",
            "clahe_class": "src/preprocessing.py::CLAHE (lines 79-118): cv2.createCLAHE on the L "
            "channel of cv2.COLOR_RGB2LAB, back with cv2.COLOR_LAB2RGB",
        },
        "clahe": {
            "colour_space": "LAB",
            "channel": "L (lightness) only; A and B untouched",
            "clip_limit": cfg.clahe.clip_limit,
            "tile_grid_size": [cfg.clahe.tile_grid_size, cfg.clahe.tile_grid_size],
            "position": "first stage, on the full-resolution decoded RGB image, before resize/crop",
            "library": "OpenCV (opencv-python-headless, requirements/base.txt)",
        },
        "resize": f"Resize(shorter side -> {cfg.resize_size}, bicubic, antialias) then "
        f"CenterCrop({cfg.image_size}) [eval]; RandomResizedCrop({cfg.image_size}) [train]",
        "normalisation": {"mean": list(cfg.mean), "std": list(cfg.std), "note": "ImageNet"},
        "tensor": "PILToTensor -> ToDtype(float32, scale=True) -> Normalize; output (3, 224, 224)",
        "preprocess": asdict(cfg),
        "eval_pipeline_stages": transform_stage_names(build_eval_transform(cfg)),
        "train_pipeline_stages": transform_stage_names(build_train_transform(cfg, AugmentConfig())),
        "note_on_order": "the team method note sketches Resize -> CLAHE; the existing project "
        "applies CLAHE -> Resize -> CenterCrop. Kept as implemented; changing it is a decision "
        "for approval.",
        "evidence_of_effect_so_far": "EXP-002 (CLAHE off) val Macro-F1 0.9182 vs EXP-001 "
        "(CLAHE on) "
        "0.9004 under AdamW on the current dataset; whether CLAHE helps is to be measured.",
        "validation_samples": validation,
        "corrupt_file_handling": corrupt,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
    }


def write_md(path: Path, cfg: dict[str, Any]) -> None:
    c = cfg["clahe"]
    lines = [
        "# Preprocessing / CLAHE validation (Phase 7) — existing implementation on the new dataset",
        "",
        "## What is reused",
        "",
        *[f"- **{k}**: {v}" for k, v in cfg["implementation"].items()],
        "",
        "## Configuration (read from the code, not invented)",
        "",
        *_table(
            ["item", "value"],
            [
                ["colour space / channel", f"{c['colour_space']} / {c['channel']}"],
                ["clip limit", c["clip_limit"]],
                ["tile grid size", c["tile_grid_size"]],
                ["position", c["position"]],
                ["resize", cfg["resize"]],
                [
                    "normalisation",
                    f"mean {cfg['normalisation']['mean']}, std {cfg['normalisation']['std']}",
                ],
                ["tensor", cfg["tensor"]],
                ["eval stages", " → ".join(cfg["eval_pipeline_stages"])],
                ["train stages", " → ".join(cfg["train_pipeline_stages"])],
            ],
        ),
        "",
        f"Order note: {cfg['note_on_order']}",
        "",
        f"Evidence so far: {cfg['evidence_of_effect_so_far']}",
        "",
        "## Validation samples (in memory; one clean image per class and per source)",
        "",
        "`l_std` is the standard deviation of the LAB lightness channel before and after CLAHE "
        "(ratio > 1 = contrast increased). `output` is the tensor the model would receive.",
        "",
        *_table(
            [
                "sample",
                "dataset",
                "class",
                "input",
                "L std before → after (×)",
                "output",
                "mean / std",
                "sheet",
            ],
            [
                [
                    v["sample_for"],
                    v["source_dataset"],
                    v["unified_class"],
                    f"{v['input_width']}x{v['input_height']}",
                    f"{v['l_std_before']} → {v['l_std_after']} (×{v['l_std_ratio']})",
                    f"{v['output_shape']} {v['output_dtype']}",
                    f"{v['output_mean']} / {v['output_std']}",
                    f"`{v['sheet']}`",
                ]
                for v in cfg["validation_samples"]
            ],
        ),
        "",
        "## Corrupted-image handling",
        "",
        f"- status: **{cfg['corrupt_file_handling']['status']}** — "
        f"{cfg['corrupt_file_handling']['exception']}: {cfg['corrupt_file_handling']['message']}",
        "- the master-manifest probe records undecodable files with status `corrupt` "
        "(0 found in the five datasets); the training loader raises instead of skipping",
        "",
        "**CLAHE has not been applied to the dataset on disk.** It runs inside the transform of "
        "any "
        "experiment whose config sets `preprocess.clahe` (`configs/preprocess_v2_clahe.json`).",
    ]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    args = parser.parse_args(argv)
    audit_dir = args.audit_dir or (Path(args.data_dir) / AUDIT_DIR_NAME)
    rows = read_clean_manifest(audit_dir / CLEAN_MANIFEST_NAME)
    sample_dir = audit_dir / "clahe_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for stale in sample_dir.glob("*.png"):
        stale.unlink()
    transform = build_eval_transform(PREPROCESS_V2)
    clahe = CLAHE(PREPROCESS_V2.clahe or CLAHEConfig())
    validation = []
    for label, row in representative_samples(rows):
        result = validate_sample(label, row, args.repo_root, sample_dir, transform, clahe)
        logger.info(
            "%s: %s -> %s (L std ×%s)",
            label,
            row.original_path,
            result["output_shape"],
            result["l_std_ratio"],
        )
        validation.append(result)
    corrupt = corrupt_file_check(transform)
    logger.info("corrupt file: %s", corrupt["status"])
    with (audit_dir / "clahe_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(VALIDATION_COLUMNS))
        writer.writeheader()
        writer.writerows(validation)
    cfg = preprocessing_config(validation, corrupt)
    (audit_dir / "preprocessing_config.json").write_text(json.dumps(cfg, indent=2))
    write_md(audit_dir / "preprocessing_config.md", cfg)
    print((audit_dir / "preprocessing_config.md").read_text())
    return 0 if corrupt["status"] == "reported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
