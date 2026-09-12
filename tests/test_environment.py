import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from src.environment import (
    build_report,
    cpu_brand,
    distribution_status,
    matches_pin,
    parse_requirements,
    python_report,
    read_python_pin,
    requirement_name,
)

ROOT = Path(__file__).resolve().parent.parent
VERIFY_SCRIPT = ROOT / "scripts" / "verify_environment.py"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("pytest", "pytest"),
        ("pytest==9.1.1", "pytest"),
        ("torch>=2.2", "torch"),
        ("opencv-python-headless==5.0.0.93", "opencv-python-headless"),
        ("uvicorn[standard]==0.30.0", "uvicorn"),
        ("numpy!=2.0.0", "numpy"),
        ("tomli; python_version < '3.11'", "tomli"),
        ("black ==26.5.1", "black"),
    ],
)
def test_requirement_name_parsing(line, expected):
    assert requirement_name(line) == expected


def test_parse_requirements_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "base.txt"
    path.write_text("# a comment\n" "\n" "torch==2.14.0  # inline comment\n" "   \n" "numpy>=2.0\n")
    assert [r.name for r in parse_requirements(path)] == ["torch", "numpy"]


def test_parse_requirements_keeps_raw_line_without_inline_comment(tmp_path):
    path = tmp_path / "base.txt"
    path.write_text("torch==2.14.0  # needed by Layer 1\n")
    (requirement,) = parse_requirements(path)
    assert requirement.raw == "torch==2.14.0"
    assert requirement.source == path.resolve()


def test_parse_requirements_follows_includes(tmp_path):
    (tmp_path / "base.txt").write_text("torch==2.14.0\n")
    dev = tmp_path / "dev.txt"
    dev.write_text("-r base.txt\npytest==9.1.1\n")
    assert [r.name for r in parse_requirements(dev)] == ["torch", "pytest"]


def test_parse_requirements_follows_long_form_include(tmp_path):
    (tmp_path / "base.txt").write_text("torch==2.14.0\n")
    dev = tmp_path / "dev.txt"
    dev.write_text("--requirement base.txt\n")
    assert [r.name for r in parse_requirements(dev)] == ["torch"]


def test_parse_requirements_detects_include_cycle(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("-r b.txt\n")
    b.write_text("-r a.txt\n")
    with pytest.raises(ValueError, match="circular"):
        parse_requirements(a)


def test_parse_requirements_rejects_unsupported_option(tmp_path):
    path = tmp_path / "base.txt"
    path.write_text("-e .\n")
    with pytest.raises(ValueError, match="unsupported requirement option"):
        parse_requirements(path)


def test_distribution_status_reports_installed_and_missing():
    def lookup(name: str) -> str:
        if name == "present-dist":
            return "1.2.3"
        raise PackageNotFoundError(name)

    installed, missing = distribution_status(["present-dist", "absent-dist"], lookup)
    assert (installed.installed, installed.version) == (True, "1.2.3")
    assert (missing.installed, missing.version) == (False, None)


def test_distribution_status_against_real_metadata():
    (status,) = distribution_status(["pytest"])
    assert status.installed and status.version


@pytest.mark.parametrize("text,expected", [("3.11\n", (3, 11)), ("3.11.15", (3, 11))])
def test_read_python_pin(tmp_path, text, expected):
    path = tmp_path / ".python-version"
    path.write_text(text)
    assert read_python_pin(path) == expected


@pytest.mark.parametrize("text", ["3\n", "python3.11\n", ""])
def test_read_python_pin_rejects_malformed(tmp_path, text):
    path = tmp_path / ".python-version"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_python_pin(path)


def test_matches_pin():
    assert matches_pin((3, 11, 15), (3, 11))
    assert not matches_pin((3, 14, 7), (3, 11))


def test_cpu_brand_uses_runner_output():
    assert cpu_brand(runner=lambda cmd: "Apple M1 Pro\n", system="Darwin") == "Apple M1 Pro"


def test_cpu_brand_falls_back_when_command_fails():
    def failing_runner(cmd: list[str]) -> str:
        raise OSError("sysctl unavailable")

    assert cpu_brand(runner=failing_runner, system="Darwin")


def test_python_report_matches_running_interpreter():
    report = python_report()
    assert report.version_info == sys.version_info[:3]
    assert report.in_virtualenv == (sys.prefix != sys.base_prefix)


def test_build_report_flags_missing_distribution(tmp_path):
    (tmp_path / "base.txt").write_text("absent-dist\n")
    (tmp_path / ".python-version").write_text("3.11\n")

    def lookup(name: str) -> str:
        raise PackageNotFoundError(name)

    report = build_report(
        [tmp_path / "base.txt"],
        python_version_file=tmp_path / ".python-version",
        lookup=lookup,
    )
    assert not report.ok
    assert [d.name for d in report.missing_distributions] == ["absent-dist"]


def test_build_report_ok_when_everything_declared_is_installed(tmp_path):
    (tmp_path / "base.txt").write_text("present-dist\n")
    pin = f"{sys.version_info.major}.{sys.version_info.minor}\n"
    (tmp_path / ".python-version").write_text(pin)

    report = build_report(
        [tmp_path / "base.txt"],
        python_version_file=tmp_path / ".python-version",
        lookup=lambda name: "1.0.0",
    )
    assert report.ok
    assert report.python_matches_pin is True


def test_build_report_deduplicates_across_files(tmp_path):
    (tmp_path / "base.txt").write_text("torch\n")
    (tmp_path / "dev.txt").write_text("-r base.txt\ntorch\n")
    report = build_report([tmp_path / "base.txt", tmp_path / "dev.txt"], lookup=lambda n: "1.0")
    assert [d.name for d in report.distributions] == ["torch"]


def test_build_report_mismatched_interpreter_is_not_ok(tmp_path):
    (tmp_path / "base.txt").write_text("")
    (tmp_path / ".python-version").write_text("2.7\n")
    report = build_report(
        [tmp_path / "base.txt"],
        python_version_file=tmp_path / ".python-version",
        lookup=lambda name: "1.0.0",
    )
    assert report.python_matches_pin is False
    assert not report.ok


def test_verify_environment_cli_succeeds_on_this_environment():
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "environment satisfies everything the project declares" in result.stdout


def test_verify_environment_cli_emits_valid_json():
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--json"], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["python"]["version_info"][:2] == [3, 11]
    assert {d["name"] for d in payload["distributions"]} >= {"pytest", "ruff", "mypy", "black"}


def test_verify_environment_cli_fails_when_declared_dist_missing(tmp_path):
    missing = tmp_path / "missing.txt"
    missing.write_text("definitely-not-a-real-distribution-xyz\n")
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--requirements", str(missing)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 1
    assert "MISSING" in result.stdout
