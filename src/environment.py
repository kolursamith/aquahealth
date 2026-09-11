"""Inspection of the local development environment (Layer 0).

This module deliberately imports nothing outside the standard library: it has
to run correctly in an environment where no runtime dependency has been
installed yet, which is exactly the situation it is meant to diagnose.

It reports facts and never infers them. Anything that cannot be established
without a dependency that is not installed is reported as unknown rather than
guessed — notably GPU/CUDA/MPS availability, which requires PyTorch and is
therefore determined in Layer 1, not here.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Callable, Iterable

_NAME_TERMINATORS = "[=<>!~;@ \t"
_INCLUDE_PREFIXES = ("-r ", "--requirement ")

CommandRunner = Callable[[list[str]], str]
VersionLookup = Callable[[str], str]


@dataclass(frozen=True)
class Requirement:
    """A single declared distribution, with the raw line it came from."""

    name: str
    raw: str
    source: Path


@dataclass(frozen=True)
class DistributionStatus:
    name: str
    installed: bool
    version: str | None


@dataclass(frozen=True)
class PythonReport:
    version: str
    version_info: tuple[int, int, int]
    implementation: str
    executable: str
    in_virtualenv: bool
    prefix: str
    base_prefix: str


@dataclass(frozen=True)
class PlatformReport:
    system: str
    release: str
    machine: str
    cpu_brand: str
    logical_cpus: int | None
    total_memory_bytes: int | None


@dataclass(frozen=True)
class AcceleratorReport:
    """Accelerator facts observable without importing a deep-learning library.

    `cuda_available` and `mps_available` are intentionally absent: determining
    them requires PyTorch, which Layer 0 does not install. Layer 1 owns that
    verification.
    """

    machine: str
    nvidia_smi_present: bool
    torch_installed: bool


@dataclass(frozen=True)
class EnvironmentReport:
    python: PythonReport
    platform: PlatformReport
    accelerator: AcceleratorReport
    python_pin: tuple[int, int] | None
    python_matches_pin: bool | None
    distributions: list[DistributionStatus] = field(default_factory=list)

    @property
    def missing_distributions(self) -> list[DistributionStatus]:
        return [d for d in self.distributions if not d.installed]

    @property
    def ok(self) -> bool:
        return not self.missing_distributions and self.python_matches_pin is not False


def requirement_name(line: str) -> str:
    """Extract the distribution name from a requirement line.

    Handles the forms this project uses: bare names, version specifiers,
    extras, and environment markers.
    """
    stripped = line.strip()
    for index, char in enumerate(stripped):
        if char in _NAME_TERMINATORS:
            return stripped[:index]
    return stripped


def parse_requirements(path: Path, _seen: frozenset[Path] = frozenset()) -> list[Requirement]:
    """Parse a pip requirements file, following `-r` includes.

    Unsupported constructs (editable installs, direct URLs, options) raise
    rather than being skipped, so the environment check can never silently
    under-report what the project declares.
    """
    resolved = path.resolve()
    if resolved in _seen:
        raise ValueError(f"circular requirement include detected at {resolved}")

    requirements: list[Requirement] = []
    for raw_line in resolved.read_text().splitlines():
        line = raw_line.split(" #", 1)[0].split("\t#", 1)[0].strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith(_INCLUDE_PREFIXES):
            include = line.split(maxsplit=1)[1].strip()
            requirements.extend(parse_requirements(resolved.parent / include, _seen | {resolved}))
            continue

        if line.startswith("-"):
            raise ValueError(f"unsupported requirement option in {resolved}: {line!r}")

        name = requirement_name(line)
        if not name:
            raise ValueError(f"could not parse requirement in {resolved}: {line!r}")
        requirements.append(Requirement(name=name, raw=line, source=resolved))

    return requirements


def distribution_status(
    names: Iterable[str],
    lookup: VersionLookup = distribution_version,
) -> list[DistributionStatus]:
    """Report whether each named distribution is installed, without importing it."""
    statuses: list[DistributionStatus] = []
    for name in names:
        try:
            statuses.append(DistributionStatus(name=name, installed=True, version=lookup(name)))
        except PackageNotFoundError:
            statuses.append(DistributionStatus(name=name, installed=False, version=None))
    return statuses


def read_python_pin(path: Path) -> tuple[int, int]:
    """Read the project's declared Python version from a `.python-version` file."""
    text = path.read_text().strip()
    parts = text.split(".")
    if len(parts) < 2:
        raise ValueError(f"expected at least MAJOR.MINOR in {path}, got {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"could not parse Python pin in {path}: {text!r}") from exc


def matches_pin(version_info: tuple[int, ...], pin: tuple[int, int]) -> bool:
    return tuple(version_info[:2]) == pin


def _run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def cpu_brand(runner: CommandRunner = _run, system: str | None = None) -> str:
    """Best-effort CPU model name, falling back to what `platform` reports."""
    system = system or platform.system()
    try:
        if system == "Darwin":
            return runner(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def total_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def python_report() -> PythonReport:
    info = sys.version_info
    return PythonReport(
        version=platform.python_version(),
        version_info=(info.major, info.minor, info.micro),
        implementation=platform.python_implementation(),
        executable=sys.executable,
        in_virtualenv=sys.prefix != sys.base_prefix,
        prefix=sys.prefix,
        base_prefix=sys.base_prefix,
    )


def platform_report(runner: CommandRunner = _run) -> PlatformReport:
    return PlatformReport(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        cpu_brand=cpu_brand(runner),
        logical_cpus=os.cpu_count(),
        total_memory_bytes=total_memory_bytes(),
    )


def accelerator_report(lookup: VersionLookup = distribution_version) -> AcceleratorReport:
    return AcceleratorReport(
        machine=platform.machine(),
        nvidia_smi_present=shutil.which("nvidia-smi") is not None,
        torch_installed=bool(distribution_status(["torch"], lookup)[0].installed),
    )


def build_report(
    requirement_files: Iterable[Path],
    python_version_file: Path | None = None,
    lookup: VersionLookup = distribution_version,
    runner: CommandRunner = _run,
) -> EnvironmentReport:
    declared: list[str] = []
    for path in requirement_files:
        declared.extend(r.name for r in parse_requirements(path))

    pin = read_python_pin(python_version_file) if python_version_file else None
    python = python_report()

    return EnvironmentReport(
        python=python,
        platform=platform_report(runner),
        accelerator=accelerator_report(lookup),
        python_pin=pin,
        python_matches_pin=matches_pin(python.version_info, pin) if pin else None,
        distributions=distribution_status(dict.fromkeys(declared), lookup),
    )
