"""Compute-device selection and PyTorch runtime facts (Layer 1).

This is the single place the project decides which device to run on. Every
later layer (model, training, inference) asks `resolve_device` rather than
touching `torch.cuda` / `torch.backends.mps` directly, so the policy lives in
one file and is tested in one file.

Policy: an explicit request must be honoured or fail loudly — silently
falling back from a requested GPU to the CPU would make a 50x slowdown look
like a working run. Only `preferred=None` performs automatic selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

DEVICE_PRIORITY: tuple[str, ...] = ("cuda", "mps", "cpu")


@dataclass(frozen=True)
class TorchReport:
    version: str
    cuda_built: bool
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    mps_built: bool
    mps_available: bool
    num_threads: int
    selected_device: str


def available_backends() -> dict[str, bool]:
    """Availability of every backend in `DEVICE_PRIORITY`, determined at call time."""
    return {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }


def resolve_device(preferred: str | None = None) -> torch.device:
    """Return the device to compute on.

    `None` selects the best available backend in `DEVICE_PRIORITY` order.
    An explicit name (e.g. "cpu", "mps", "cuda", "cuda:1") is returned only
    if that backend is available; otherwise `RuntimeError` is raised.
    """
    backends = available_backends()

    if preferred is None:
        for name in DEVICE_PRIORITY:
            if backends[name]:
                return torch.device(name)
        raise RuntimeError("no compute backend available")  # pragma: no cover - cpu is always True

    device = torch.device(preferred)
    if device.type not in backends:
        raise ValueError(
            f"unsupported device type {device.type!r}; expected one of {DEVICE_PRIORITY}"
        )
    if not backends[device.type]:
        available = [name for name, ok in backends.items() if ok]
        raise RuntimeError(
            f"requested device {preferred!r} is not available on this machine "
            f"(available: {', '.join(available)})"
        )
    return device


def torch_report(preferred: str | None = None) -> TorchReport:
    cuda_available = torch.cuda.is_available()
    return TorchReport(
        version=torch.__version__,
        cuda_built=torch.version.cuda is not None,
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
        cuda_device_name=torch.cuda.get_device_name(0) if cuda_available else None,
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        num_threads=torch.get_num_threads(),
        selected_device=str(resolve_device(preferred)),
    )
