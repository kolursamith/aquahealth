"""Layer 1 — device selection policy (`src.device`)."""

from __future__ import annotations

import pytest
import torch

from src import device as device_module
from src.device import (
    DEVICE_PRIORITY,
    TorchReport,
    available_backends,
    resolve_device,
    torch_report,
)


def _fake_backends(monkeypatch, **availability: bool):
    backends = {"cuda": False, "mps": False, "cpu": True, **availability}
    monkeypatch.setattr(device_module, "available_backends", lambda: backends)


def test_priority_order_is_cuda_mps_cpu():
    assert DEVICE_PRIORITY == ("cuda", "mps", "cpu")


def test_available_backends_reports_every_priority_entry():
    backends = available_backends()
    assert set(backends) == set(DEVICE_PRIORITY)
    assert backends["cpu"] is True
    assert backends["cuda"] == torch.cuda.is_available()
    assert backends["mps"] == torch.backends.mps.is_available()


# --- automatic selection --------------------------------------------------------


def test_auto_selects_cuda_first(monkeypatch):
    _fake_backends(monkeypatch, cuda=True, mps=True)
    assert resolve_device().type == "cuda"


def test_auto_selects_mps_when_no_cuda(monkeypatch):
    _fake_backends(monkeypatch, mps=True)
    assert resolve_device().type == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    _fake_backends(monkeypatch)
    assert resolve_device().type == "cpu"


def test_auto_selection_on_this_machine_is_usable():
    device = resolve_device()
    assert device.type in DEVICE_PRIORITY
    assert torch.ones(2, device=device).sum().item() == 2.0


# --- explicit selection -----------------------------------------------------------


def test_explicit_cpu_always_works():
    assert resolve_device("cpu") == torch.device("cpu")


def test_explicit_available_backend_is_returned(monkeypatch):
    _fake_backends(monkeypatch, mps=True)
    assert resolve_device("mps") == torch.device("mps")


def test_explicit_indexed_device_keeps_index(monkeypatch):
    _fake_backends(monkeypatch, cuda=True)
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_explicit_unavailable_backend_raises_instead_of_falling_back(monkeypatch):
    _fake_backends(monkeypatch)
    with pytest.raises(RuntimeError, match="not available on this machine"):
        resolve_device("cuda")


def test_explicit_unsupported_type_raises():
    with pytest.raises(ValueError, match="unsupported device type"):
        resolve_device("xla")


def test_explicit_gibberish_is_rejected_by_torch():
    with pytest.raises(RuntimeError):
        resolve_device("not a device")


# --- report -----------------------------------------------------------------------


def test_torch_report_reflects_runtime():
    report = torch_report()
    assert isinstance(report, TorchReport)
    assert report.version == torch.__version__
    assert report.cuda_built == (torch.version.cuda is not None)
    assert report.cuda_available == torch.cuda.is_available()
    assert report.mps_built == torch.backends.mps.is_built()
    assert report.mps_available == torch.backends.mps.is_available()
    assert report.num_threads == torch.get_num_threads() > 0
    assert report.selected_device == str(resolve_device())
    if not report.cuda_available:
        assert report.cuda_device_count == 0
        assert report.cuda_device_name is None


def test_torch_report_honours_preferred_device():
    assert torch_report("cpu").selected_device == "cpu"
