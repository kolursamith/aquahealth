#!/usr/bin/env python
"""Phase 12 — pre-flight checks for a Google Colab (CUDA) experiment session.

    python scripts/colab_preflight.py --fold 1 --data-arm without_gan [--require-cuda]
    python scripts/colab_preflight.py --env-only [--require-cuda]      # Layer 2: runtime only

Verifies, and prints a JSON report of:
  0. environment: Python, platform, torch/torchvision/numpy/pillow/opencv/matplotlib versions
     (the project's actual dependencies, requirements/*.txt); optional libraries that the
     code does NOT import (scikit-learn, timm, transformers) are reported but never required
  1. CUDA: available, GPU name, CUDA version, memory (aborts with --require-cuda if absent)
  2. PyTorch / torchvision versions match requirements/base.txt pins
  3. repository integrity: git commit, frozen split digests, fold digests, matrix present
  4. dataset availability: every image of the fold's training + validation manifests exists
  5. GAN manifest for the fold when --data-arm with_gan
  6. final-test SHA-256 (recorded, never opened as images)
Exit code 0 = ready, 1 = a check failed, 2 = CUDA required but absent.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torchvision  # noqa: E402

from src.config import DATA_DIR, ROOT_DIR  # noqa: E402
from src.cv_runner import (  # noqa: E402
    DATA_ARMS,
    check_isolation,
    final_test_ids,
    manifest_paths,
    read_manifest_rows,
)
from src.manifest import file_sha256  # noqa: E402
from src.split_v2 import CV_DIR_NAME, SPLIT_DIR_NAME, read_folds, verify_split_digest  # noqa: E402

# Libraries the code base actually imports (requirements/base.txt + experiments.txt) and the
# optional ones it does not: reported so nobody adds them by accident, never required.
REQUIRED_MODULES: dict[str, str] = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "matplotlib": "matplotlib",
}
OPTIONAL_MODULES: dict[str, str] = {
    "sklearn": "scikit-learn",
    "timm": "timm",
    "transformers": "transformers",
}


def _version(module: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 - absence is the information
        return None
    return str(getattr(mod, "__version__", "present"))


def environment_report() -> dict[str, Any]:
    required = {dist: _version(mod) for mod, dist in REQUIRED_MODULES.items()}
    optional = {dist: _version(mod) for mod, dist in OPTIONAL_MODULES.items()}
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "required": required,
        "optional_not_required": optional,
        "ok": all(v is not None for v in required.values()),
    }


def cuda_report(require: bool) -> dict[str, Any]:
    available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "available": available,
        "version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "memory_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if available
            else None
        ),
        "torch_built_with_cuda": torch.version.cuda is not None,
        "required": require,
        "ok": available or not require,
    }
    return report


def pinned_versions(requirements: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not requirements.is_file():
        return pins
    for line in requirements.read_text().splitlines():
        m = re.match(r"^(torch|torchvision)==([\w.+]+)", line.strip())
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def version_report(repo_root: Path) -> dict[str, Any]:
    pins = pinned_versions(repo_root / "requirements" / "base.txt")
    installed = {
        "torch": torch.__version__.split("+")[0],
        "torchvision": torchvision.__version__.split("+")[0],
    }
    return {
        "pinned": pins,
        "installed": {"torch": torch.__version__, "torchvision": torchvision.__version__},
        "ok": bool(pins) and all(installed[k] == v.split("+")[0] for k, v in pins.items()),
    }


def repository_report(repo_root: Path, data_dir: Path) -> dict[str, Any]:
    split_dir = data_dir / "audit" / SPLIT_DIR_NAME
    cv_dir = data_dir / "audit" / CV_DIR_NAME
    report: dict[str, Any] = {
        "git_commit": None,
        "split_digest_ok": False,
        "folds_digest_ok": False,
    }
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True
        )
        report["git_commit"] = out.stdout.strip() or None
    except OSError:
        pass
    try:
        verify_split_digest(split_dir)
        report["split_digest_ok"] = True
        report["final_test_sha256"] = file_sha256(split_dir / "final_test.csv")
        report["development_sha256"] = file_sha256(split_dir / "development.csv")
    except (OSError, ValueError) as exc:
        report["split_error"] = str(exc)
    try:
        folds = read_folds(cv_dir)
        report["folds_digest_ok"] = True
        report["development_images"] = len(folds)
        report["folds"] = len({r.fold for r in folds})
    except (OSError, ValueError) as exc:
        report["folds_error"] = str(exc)
    matrix = repo_root / "results" / "v2" / "experiment_matrix.csv"
    report["matrix_present"] = matrix.is_file()
    report["ok"] = (
        report["split_digest_ok"] and report["folds_digest_ok"] and report["matrix_present"]
    )
    return report


def dataset_report(repo_root: Path, data_dir: Path, fold: int, data_arm: str) -> dict[str, Any]:
    train_path, val_path = manifest_paths(data_dir, fold, data_arm)
    report: dict[str, Any] = {
        "train_manifest": str(train_path),
        "validation_manifest": str(val_path),
    }
    if not train_path.is_file():
        report["ok"] = False
        report["error"] = f"{train_path} missing" + (
            " — run scripts/run_gan_fold.py for this fold first" if data_arm == "with_gan" else ""
        )
        return report
    train = read_manifest_rows(train_path)
    val = read_manifest_rows(val_path)
    isolation = check_isolation(
        fold=fold,
        data_arm=data_arm,
        train_path=train_path,
        validation_path=val_path,
        train_rows=train,
        validation_rows=val,
        test_ids=final_test_ids(data_dir / "audit" / SPLIT_DIR_NAME),
    )
    missing = [r.filepath for r in train + val if not (repo_root / r.filepath).is_file()]
    report.update(
        {
            "train_sha256": file_sha256(train_path),
            "validation_sha256": file_sha256(val_path),
            "isolation": isolation,
            "missing_images": len(missing),
            "missing_examples": missing[:5],
            "ok": not missing,
        }
    )
    return report


def model_report(model_key: str, require_cuda: bool) -> dict[str, Any]:
    """Build the model (no pretrained download) and prove it maps (N, 3, 224, 224) to
    (N, num_classes) finite logits on the device that training will use."""
    from src.manifest import CANONICAL_CLASSES
    from src.model_factory import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "model": model_key,
        "device": str(device),
        "num_classes": len(CANONICAL_CLASSES),
        "input_shape": [2, 3, 224, 224],
    }
    try:
        model = build_model(model_key, len(CANONICAL_CLASSES), pretrained=False).to(device).eval()
        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224, device=device))
        report["output_shape"] = list(logits.shape)
        report["parameters"] = sum(p.numel() for p in model.parameters())
        report["ok"] = (
            tuple(logits.shape) == (2, len(CANONICAL_CLASSES))
            and bool(torch.isfinite(logits).all())
            and (device.type == "cuda" or not require_cuda)
        )
    except Exception as exc:  # any failure is a preflight failure, reported not raised
        report["error"] = repr(exc)
        report["ok"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--data-arm", choices=DATA_ARMS, default=None)
    parser.add_argument(
        "--model",
        default=None,
        help="also build this model and check (N,3,224,224) -> (N,num_classes) logits",
    )
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="environment + CUDA + versions + repository digests only (no fold manifests)",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    if not args.env_only and (args.fold is None or args.data_arm is None):
        parser.error("--fold and --data-arm are required unless --env-only is given")
    report: dict[str, Any] = {
        "environment": environment_report(),
        "cuda": cuda_report(args.require_cuda),
        "versions": version_report(args.repo_root),
        "repository": repository_report(args.repo_root, args.data_dir),
    }
    if not args.env_only:
        report["dataset"] = dataset_report(args.repo_root, args.data_dir, args.fold, args.data_arm)
    if args.model:
        report["model"] = model_report(args.model, args.require_cuda)
    report["ready"] = all(section["ok"] for section in report.values() if isinstance(section, dict))
    print(json.dumps(report, indent=2))
    if not report["cuda"]["ok"]:
        return 2
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
