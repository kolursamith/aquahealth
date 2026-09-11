#!/usr/bin/env python
"""Verify that the local environment matches what the project declares.

Checks the interpreter against `.python-version` and every distribution
declared in the requirements files. Exits non-zero when the environment does
not satisfy what is declared, so the same command works as a local check and
as a CI gate.

GPU/CUDA/MPS availability is not reported here: establishing it requires
PyTorch, which Layer 0 does not install. Layer 1 verifies it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment import EnvironmentReport, build_report  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUIREMENTS = [ROOT / "requirements" / "base.txt", ROOT / "requirements" / "dev.txt"]
PYTHON_VERSION_FILE = ROOT / ".python-version"


def format_report(report: EnvironmentReport) -> str:
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
        "  cuda / mps       not determined in Layer 0 (requires torch — Layer 1)",
        "",
        "Declared distributions",
    ]

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

    if args.json:
        print(json.dumps(dataclasses.asdict(report), indent=2, default=str))
    else:
        print(format_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
