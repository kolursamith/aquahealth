#!/usr/bin/env python
"""Layer 3 — the required GAN generation: one cDCGAN per CV fold, trained on that
fold's training rows only, for every fold, then the consolidated GAN_MANIFEST.csv
and the verification report. Runs on Colab CUDA.

    python scripts/run_gan_all_folds.py --config configs/gan_v2/default.json --require-cuda
    python scripts/run_gan_all_folds.py --folds 1-3 --config configs/gan_v2/default.json \
        --require-cuda

Resumable: a fold whose data/gan/fold_XX/ already holds generator.pt, gan_run.json,
synthetic_manifest.csv and summary.json is skipped (its registry line is kept); a
half-written fold directory is refused by scripts/run_gan_fold.py, so remove it
deliberately before retrying. Each fold is scripts/run_gan_fold.py with the same
arguments; nothing is trained here.

Writes

    data/gan/fold_XX/…                                   per fold (see run_gan_fold.py)
    results/v2/gan/registry.csv                          one line per fold (tracked)
    results/v2/gan/GAN_MANIFEST.csv                      one line per synthetic image (tracked)
    results/v2/gan/verification.json                     verify_gan_outputs report (tracked)
    results/v2/gan/generation_summary.json               per-fold time / counts, GPU, CUDA (tracked)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.device import resolve_device  # noqa: E402
from src.gan_augmentation import (  # noqa: E402
    GAN_DIR_NAME,
    GAN_MANIFEST_NAME,
    N_FOLDS,
    build_gan_manifest,
    device_facts,
    parse_folds,
    summarize_verification,
    verify_gan_outputs,
    write_gan_manifest,
)
from src.multi_dataset import AUDIT_DIR_NAME  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("gan_all_folds")


def _load_run_gan_fold():
    """scripts/ is not a package; load the single-fold runner from its file."""
    import importlib.util

    path = Path(__file__).resolve().parent / "run_gan_fold.py"
    spec = importlib.util.spec_from_file_location("run_gan_fold", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


COMPLETE_MARKERS = ("generator.pt", "gan_run.json", "synthetic_manifest.csv", "summary.json")


def fold_complete(out_dir: Path) -> bool:
    return all((out_dir / name).is_file() for name in COMPLETE_MARKERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--folds", default=f"1-{N_FOLDS}", help="e.g. 1-10 or 1,2,5")
    parser.add_argument("--config", type=Path, required=True, help="configs/gan_v2/<name>.json")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--images",
        default="all",
        choices=("all", "sample", "none"),
        help="how many PNGs the final verification opens and hashes",
    )
    args = parser.parse_args(argv)

    folds = parse_folds(args.folds)
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        logger.error("--require-cuda: selected device is %s, not CUDA", device)
        return 2
    facts = device_facts(device)
    results_dir = (
        Path(args.results_dir) if args.results_dir is not None else Path(args.repo_root) / "results"
    )
    gan_dir = Path(args.data_dir) / GAN_DIR_NAME
    gan_results = results_dir / "v2" / "gan"
    logger.info("folds %s on %s %s; config %s", folds, device, facts, args.config)

    run_gan_fold = _load_run_gan_fold()
    per_fold: list[dict] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for fold in folds:
        out_dir = gan_dir / f"fold_{fold:02d}"
        if fold_complete(out_dir):
            summary = json.loads((out_dir / "summary.json").read_text())
            logger.info(
                "fold %d already complete (%d synthetic) — skipped",
                fold,
                summary["synthetic_images"],
            )
            per_fold.append({"fold": fold, "status": "skipped (complete)", **summary})
            continue
        fold_argv = [
            "--fold",
            str(fold),
            "--config",
            str(args.config),
            "--data-dir",
            str(args.data_dir),
            "--results-dir",
            str(results_dir),
            "--repo-root",
            str(args.repo_root),
            "--workers",
            str(args.workers),
        ]
        if args.device:
            fold_argv += ["--device", args.device]
        if args.require_cuda:
            fold_argv.append("--require-cuda")
        t0 = time.perf_counter()
        rc = run_gan_fold.main(fold_argv)
        wall = round(time.perf_counter() - t0, 1)
        if rc != 0:
            logger.error("fold %d failed with exit code %d after %.0f s; stopping", fold, rc, wall)
            return rc
        summary = json.loads((out_dir / "summary.json").read_text())
        record = json.loads((out_dir / "gan_run.json").read_text())
        per_fold.append(
            {
                "fold": fold,
                "status": "generated",
                "wall_seconds_incl_decode_and_generation": wall,
                "train_seconds": record["seconds"],
                **summary,
            }
        )
        logger.info(
            "fold %d done: %d synthetic in %.0f s wall", fold, summary["synthetic_images"], wall
        )

    # consolidated manifest + verification over every completed fold
    rows = build_gan_manifest(gan_dir, args.repo_root, hash_images=args.images != "none")
    manifest_path = gan_results / GAN_MANIFEST_NAME
    write_gan_manifest(rows, manifest_path)
    report = verify_gan_outputs(
        gan_dir=gan_dir,
        repo_root=args.repo_root,
        audit_dir=Path(args.data_dir) / AUDIT_DIR_NAME,
        registry_path=gan_results / "registry.csv",
        gan_manifest_path=manifest_path,
        images=args.images,
    )
    (gan_results / "verification.json").write_text(json.dumps(report, indent=2))
    print(summarize_verification(report))
    overall = {
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(device),
        **facts,
        "torch_version": torch.__version__,
        "config_file": str(args.config),
        "folds_requested": folds,
        "folds_completed": [f["fold"] for f in report["folds"]],
        "synthetic_images_total": report["counts"].get("synthetic_images", 0),
        "gan_manifest": str(manifest_path),
        "gan_manifest_rows": len(rows),
        "verification_ok": report["ok"],
        "per_fold": per_fold,
    }
    (gan_results / "generation_summary.json").write_text(json.dumps(overall, indent=2))
    print(json.dumps({k: v for k, v in overall.items() if k != "per_fold"}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
