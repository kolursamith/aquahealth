#!/usr/bin/env python
"""Verify that the local environment matches what the project declares.

Checks the interpreter against `.python-version` and every distribution
declared in the requirements files. Exits non-zero when the environment does
not satisfy what is declared, so the same command works as a local check and
as a CI gate.

GPU/CUDA/MPS availability is reported only when PyTorch is installed (Layer 1
onwards), because establishing it requires importing torch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import EnvironmentReport, build_report  # noqa: E402

if TYPE_CHECKING:
    from src.device import TorchReport

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUIREMENTS = [ROOT / "requirements" / "base.txt", ROOT / "requirements" / "dev.txt"]
PYTHON_VERSION_FILE = ROOT / ".python-version"


def torch_section(report: EnvironmentReport) -> TorchReport | None:
    """Torch runtime facts, or None when torch is not installed.

    Imported lazily so this script still runs (and reports the absence) in an
    environment where torch is not installed.
    """
    if not report.accelerator.torch_installed:
        return None
    from src.device import torch_report

    return torch_report()


def format_report(report: EnvironmentReport, torch: TorchReport | None = None) -> str:
    py = report.python
    plat = report.platform
    acc = report.accelerator

    memory = plat.total_memory_bytes
    memory_text = f"{memory / 1024**3:.1f} GiB" if memory else "unknown"
    pin_text = ".".join(str(p) for p in report.python_pin) if report.python_pin else "none"

    lines = [
        "AquaHealth AI — environment verification",
        "",
        "Python",
        f"  version          {py.version} ({py.implementation})",
        f"  declared pin     {pin_text}",
        f"  matches pin      {_tri_state(report.python_matches_pin)}",
        f"  executable       {py.executable}",
        f"  virtualenv       {'yes' if py.in_virtualenv else 'NO — not running inside a venv'}",
        "",
        "Platform",
        f"  system           {plat.system} {plat.release}",
        f"  machine          {plat.machine}",
        f"  cpu              {plat.cpu_brand}",
        f"  logical cpus     {plat.logical_cpus}",
        f"  memory           {memory_text}",
        "",
        "Accelerator",
        f"  machine          {acc.machine}",
        f"  nvidia-smi       {'present' if acc.nvidia_smi_present else 'not present'}",
        f"  torch installed  {'yes' if acc.torch_installed else 'no'}",
    ]

    if torch is None:
        lines.append("  cuda / mps       not determined (requires torch)")
    else:
        cuda_text = "no"
        if torch.cuda_available:
            cuda_text = f"yes, {torch.cuda_device_count} device(s) ({torch.cuda_device_name})"
        lines += [
            "",
            "PyTorch",
            f"  version          {torch.version}",
            f"  cuda built       {'yes' if torch.cuda_built else 'no'}",
            f"  cuda available   {cuda_text}",
            f"  mps built        {'yes' if torch.mps_built else 'no'}",
            f"  mps available    {'yes' if torch.mps_available else 'no'}",
            f"  cpu threads      {torch.num_threads}",
            f"  selected device  {torch.selected_device}",
        ]

    lines += ["", "Declared distributions"]

    if not report.distributions:
        lines.append("  (none declared yet)")
    for dist in report.distributions:
        mark = "OK     " if dist.installed else "MISSING"
        lines.append(f"  [{mark}] {dist.name} {dist.version or ''}".rstrip())

    lines.append("")
    if report.ok:
        lines.append("RESULT: environment satisfies everything the project declares.")
    else:
        if report.python_matches_pin is False:
            lines.append(f"RESULT: interpreter is {py.version}, project declares {pin_text}.")
        for dist in report.missing_distributions:
            lines.append(f"RESULT: missing distribution: {dist.name}")
        lines.append("Install with: pip install -r requirements/dev.txt")

    return "\n".join(lines)


def _tri_state(value: bool | None) -> str:
    if value is None:
        return "no pin declared"
    return "yes" if value else "NO"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        action="append",
        help="Requirements file to check (repeatable). Defaults to requirements/{base,dev}.txt",
    )
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON")
    args = parser.parse_args(argv)

    report = build_report(
        requirement_files=args.requirements or DEFAULT_REQUIREMENTS,
        python_version_file=PYTHON_VERSION_FILE if PYTHON_VERSION_FILE.exists() else None,
    )

    torch = torch_section(report)

    if args.json:
        payload = dataclasses.asdict(report)
        payload["torch"] = dataclasses.asdict(torch) if torch else None
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_report(report, torch))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
