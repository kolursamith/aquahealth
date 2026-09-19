#!/usr/bin/env python
"""Layer 2 — tiny GPU smoke test: CUDA visible, tensor on device, forward, backward,
optimizer step. Not training; a few hundred parameters, one synthetic batch.

    python scripts/gpu_smoke.py [--device auto|cuda|cpu] [--require-cuda]

Exit 0 = pass, 1 = a step failed, 2 = --require-cuda and no CUDA device.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from typing import Any

import torch
from torch import nn


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def run(device: torch.device, batch: int = 8, size: int = 32, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    report: dict[str, Any] = {"device": str(device), "steps": {}}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["gpu"] = props.name
        report["vram_gb"] = round(props.total_memory / 1024**3, 2)
        report["cuda"] = torch.version.cuda
        report["cudnn"] = torch.backends.cudnn.version()
        torch.cuda.reset_peak_memory_stats(device)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 8),
    ).to(device)
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(0, 8, (batch,), device=device)
    report["steps"]["tensor_on_device"] = x.device.type == device.type
    t0 = time.perf_counter()
    logits = model(x)
    report["steps"]["forward_shape"] = list(logits.shape)
    report["steps"]["forward_ok"] = (
        logits.shape == (batch, 8) and torch.isfinite(logits).all().item()
    )
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    report["steps"]["backward_ok"] = all(
        g is not None and torch.isfinite(g).all().item() for g in grads
    )
    before = [p.detach().clone() for p in model.parameters()]
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    report["steps"]["optimizer_changed_weights"] = any(
        not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters())
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        report["peak_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
    report["runtime_s"] = round(time.perf_counter() - t0, 4)
    report["loss"] = round(float(loss.detach()), 6)
    report["ok"] = all(v is True for k, v in report["steps"].items() if k != "forward_shape")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if args.require_cuda and not torch.cuda.is_available():
        print(json.dumps({"environment": env, "ok": False, "error": "CUDA required but absent"}))
        return 2
    device = pick_device(args.device)
    report = run(device)
    report["environment"] = env
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
