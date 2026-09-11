#!/usr/bin/env python
"""Checks that the local environment has the packages this project needs,
and reports GPU/CUDA availability.
"""

import importlib
import sys

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "cv2",
    "timm",
    "albumentations",
    "streamlit",
]


def verify() -> None:
    print(f"Python: {sys.version.split()[0]}")

    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"  [OK] {package} ({version})")
        except ImportError:
            missing.append(package)
            print(f"  [MISSING] {package}")

    try:
        import torch

        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nCUDA available: unknown (torch not installed)")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements-dev.txt")
        sys.exit(1)


if __name__ == "__main__":
    verify()
