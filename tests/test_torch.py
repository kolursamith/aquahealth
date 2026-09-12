"""Layer 1 — PyTorch runtime verification.

These tests establish that the installed torch matches what the project
declares and that the primitives every later layer relies on — tensor
creation, arithmetic, autograd, seeding, serialization — behave correctly on
the CPU and on the automatically selected device.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.device import available_backends, resolve_device
from src.environment import parse_requirements
from src.utils import set_seed

ROOT = Path(__file__).resolve().parent.parent
BASE_REQUIREMENTS = ROOT / "requirements" / "base.txt"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_environment.py"


def _public_version(version: str) -> str:
    """Strip a PEP 440 local label such as '+cpu' or '+cu128'."""
    return version.split("+", 1)[0]


def _devices_to_test() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


# --- declaration vs reality ---------------------------------------------------


def test_torch_version_matches_declared_pin():
    declared = {r.name: r.raw for r in parse_requirements(BASE_REQUIREMENTS)}
    assert "torch" in declared, "torch must be declared in requirements/base.txt"
    assert declared["torch"].startswith("torch=="), "torch must be pinned exactly"
    pinned = declared["torch"].split("==", 1)[1]
    assert _public_version(torch.__version__) == pinned


def test_cpu_backend_always_available():
    assert available_backends()["cpu"] is True


# --- tensors ------------------------------------------------------------------


def test_tensor_creation_shape_and_dtype():
    x = torch.zeros(2, 3, 4)
    assert x.shape == (2, 3, 4)
    assert x.dtype == torch.float32
    assert torch.ones(5, dtype=torch.int64).dtype == torch.int64


@pytest.mark.parametrize("device", _devices_to_test())
def test_matmul_is_correct_on_every_available_device(device):
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(device)
    expected = torch.tensor([[5.0, 14.0], [14.0, 50.0]])
    assert torch.equal((a @ a.T).cpu(), expected)


@pytest.mark.parametrize("device", _devices_to_test())
def test_elementwise_ops_match_cpu(device):
    x = torch.linspace(-3, 3, 101)
    on_device = x.to(device)
    for fn in (torch.exp, torch.tanh, torch.sigmoid, torch.relu):
        torch.testing.assert_close(fn(on_device).cpu(), fn(x), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", _devices_to_test())
def test_reduction_ops_on_device(device):
    x = torch.arange(1, 11, dtype=torch.float32).to(device)
    assert x.sum().item() == pytest.approx(55.0)
    assert x.mean().item() == pytest.approx(5.5)
    assert x.max().item() == 10.0
    assert x.argmax().item() == 9


# --- autograd -----------------------------------------------------------------


def test_autograd_scalar_gradient():
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (w**2).sum().backward()
    assert w.grad is not None
    torch.testing.assert_close(w.grad, torch.tensor([2.0, 4.0, 6.0]))


@pytest.mark.parametrize("device", _devices_to_test())
def test_autograd_through_linear_layer(device):
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2).to(device)
    x = torch.randn(3, 4, device=device)
    loss = layer(x).pow(2).mean()
    loss.backward()
    assert layer.weight.grad is not None and layer.weight.grad.shape == (2, 4)
    assert layer.bias.grad is not None and layer.bias.grad.shape == (2,)
    assert torch.isfinite(layer.weight.grad).all()


def test_no_grad_context_disables_graph():
    w = torch.ones(3, requires_grad=True)
    with torch.no_grad():
        y = w * 2
    assert not y.requires_grad


def test_optimizer_step_changes_parameters():
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2)
    before = layer.weight.detach().clone()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    layer(torch.randn(3, 4)).pow(2).mean().backward()
    optimizer.step()
    assert not torch.equal(before, layer.weight.detach())


# --- determinism --------------------------------------------------------------


def test_manual_seed_reproduces_cpu_random_tensors():
    torch.manual_seed(123)
    a = torch.randn(8)
    torch.manual_seed(123)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seed_reproduces_torch_random_tensors():
    set_seed(7)
    a = torch.randn(8)
    set_seed(7)
    b = torch.randn(8)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", _devices_to_test())
def test_manual_seed_reproduces_on_device(device):
    torch.manual_seed(99)
    a = torch.randn(16, device=device)
    torch.manual_seed(99)
    b = torch.randn(16, device=device)
    assert torch.equal(a.cpu(), b.cpu())


# --- serialization --------------------------------------------------------------


def test_state_dict_round_trip(tmp_path):
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 2)
    path = tmp_path / "layer.pt"
    torch.save(layer.state_dict(), path)

    restored = torch.nn.Linear(4, 2)
    restored.load_state_dict(torch.load(path, weights_only=True))

    for key, value in layer.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


@pytest.mark.parametrize("device", _devices_to_test())
def test_tensor_round_trip_via_cpu(device):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    assert torch.equal(x.to(device).cpu(), x)


# --- verify_environment.py integration --------------------------------------------


def test_verify_environment_reports_torch_section():
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PyTorch" in result.stdout
    assert f"version          {torch.__version__}" in result.stdout
    assert "selected device" in result.stdout


def test_verify_environment_json_includes_torch_facts():
    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--json"], capture_output=True, text=True, check=True
    )
    payload = json.loads(result.stdout)
    assert payload["accelerator"]["torch_installed"] is True
    facts = payload["torch"]
    assert facts["version"] == torch.__version__
    assert facts["cuda_available"] == torch.cuda.is_available()
    assert facts["mps_available"] == torch.backends.mps.is_available()
    assert facts["selected_device"] == str(resolve_device())
