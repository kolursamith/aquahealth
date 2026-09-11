"""Layer 2 — TorchVision verification.

Establishes that torchvision is installed at the declared version, is
binary-compatible with the installed torch (its compiled ops run), and that
the three things later layers take from it — image conversion (transforms.v2),
the pretrained-model registry, and image I/O — work on this machine.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from PIL import Image
from torchvision import models
from torchvision.transforms import v2
from torchvision.tv_tensors import Image as TVImage

from src.device import available_backends
from src.environment import parse_requirements
from src.utils import set_seed

ROOT = Path(__file__).resolve().parent.parent
BASE_REQUIREMENTS = ROOT / "requirements" / "base.txt"


def _public_version(version: str) -> str:
    return version.split("+", 1)[0]


def _devices_to_test() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


def _pil_image(width: int = 300, height: int = 200, color=(10, 20, 30)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


# --- declaration vs reality ---------------------------------------------------------


@pytest.mark.parametrize(
    "name,module",
    [("torchvision", torchvision), ("numpy", np), ("pillow", __import__("PIL"))],
)
def test_declared_pin_matches_installed_version(name, module):
    declared = {r.name: r.raw for r in parse_requirements(BASE_REQUIREMENTS)}
    assert name in declared, f"{name} must be declared in requirements/base.txt"
    assert declared[name].startswith(f"{name}=="), f"{name} must be pinned exactly"
    pinned = declared[name].split("==", 1)[1]
    assert _public_version(module.__version__) == pinned


def test_importing_torch_emits_no_warning():
    """Layer 1 documented a numpy-init UserWarning on `import torch`; it must be gone."""
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import torch, torchvision"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --- binary compatibility with torch ---------------------------------------------------


def test_compiled_ops_run_against_installed_torch():
    """`nms` is a C++ op compiled against one exact torch version; a mismatch fails here."""
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]])
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
    assert keep.tolist() == [0, 2]


def test_torch_numpy_bridge_round_trip():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = torch.from_numpy(array)
    assert tensor.shape == (2, 3)
    assert np.array_equal(tensor.numpy(), array)


# --- transforms.v2: PIL -> tensor ----------------------------------------------------


def test_pil_to_tensor_is_chw_uint8():
    tensor = v2.functional.pil_to_tensor(_pil_image())
    assert tensor.shape == (3, 200, 300)
    assert tensor.dtype == torch.uint8
    assert tensor[:, 0, 0].tolist() == [10, 20, 30]


def test_to_image_and_to_dtype_scales_to_unit_range():
    pipeline = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    image = pipeline(_pil_image(color=(0, 128, 255)))
    assert isinstance(image, TVImage)
    assert image.shape == (3, 200, 300)
    assert image.dtype == torch.float32
    torch.testing.assert_close(image[:, 0, 0], torch.tensor([0.0, 128 / 255, 1.0]))


def test_resize_produces_requested_spatial_size():
    resized = v2.Resize((64, 96), antialias=True)(v2.functional.pil_to_tensor(_pil_image()))
    assert resized.shape == (3, 64, 96)


def test_normalize_applies_mean_and_std():
    mean, std = [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]
    image = torch.full((3, 4, 4), 0.75)
    out = v2.Normalize(mean=mean, std=std)(image)
    torch.testing.assert_close(out, torch.full((3, 4, 4), 1.0))


def test_deterministic_transforms_are_deterministic():
    pipeline = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((32, 32), antialias=True)]
    )
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (50, 70, 3), dtype=np.uint8))
    assert torch.equal(pipeline(image), pipeline(image))


def test_random_transform_is_reproducible_under_seed():
    flip = v2.RandomHorizontalFlip(p=0.5)
    image = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    set_seed(3)
    first = [flip(image) for _ in range(8)]
    set_seed(3)
    second = [flip(image) for _ in range(8)]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert any(not torch.equal(f, image) for f in first), "seeded sequence never flipped"


@pytest.mark.parametrize("device", _devices_to_test())
def test_converted_image_moves_to_device(device):
    tensor = v2.functional.pil_to_tensor(_pil_image()).to(device)
    assert tensor.device.type == device
    assert tensor.float().mean().item() == pytest.approx(20.0)


# --- model registry (no download) -------------------------------------------------------


def test_efficientnet_b0_is_registered():
    assert "efficientnet_b0" in models.list_models()


def test_efficientnet_b0_pretrained_weights_are_described():
    weights = models.EfficientNet_B0_Weights
    assert weights.DEFAULT is weights.IMAGENET1K_V1
    meta = weights.DEFAULT.meta
    assert meta["num_params"] == 5_288_548
    assert len(meta["categories"]) == 1000
    assert meta["_metrics"]["ImageNet-1K"]["acc@1"] == pytest.approx(77.692)


def test_efficientnet_b0_weight_transforms_declare_imagenet_stats():
    preset = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    assert preset.crop_size == [224]
    assert preset.resize_size == [256]
    assert preset.mean == [0.485, 0.456, 0.406]
    assert preset.std == [0.229, 0.224, 0.225]


def test_efficientnet_b0_builds_without_weights():
    """Architecture only; pretrained weights are downloaded and validated in Layer 3."""
    model = models.efficientnet_b0(weights=None)
    assert sum(p.numel() for p in model.parameters()) == 5_288_548
    assert model.classifier[-1].out_features == 1000


# --- image I/O: Pillow decodes, torchvision converts -----------------------------------------
# torchvision.io's encode/decode is deprecated as of 0.29 (superseded by torchcodec),
# so the project's I/O path is Pillow -> transforms.v2 and these tests exercise only that.


def test_png_bytes_decode_via_pillow_round_trip():
    rng = np.random.default_rng(1)
    original = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(original).save(buffer, format="PNG")
    buffer.seek(0)
    decoded = v2.functional.pil_to_tensor(Image.open(buffer).convert("RGB"))
    assert torch.equal(decoded, torch.from_numpy(original).permute(2, 0, 1))


def test_jpeg_bytes_decode_via_pillow_has_expected_shape():
    buffer = io.BytesIO()
    _pil_image(width=40, height=30).save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    decoded = v2.functional.pil_to_tensor(Image.open(buffer).convert("RGB"))
    assert decoded.shape == (3, 30, 40)
    assert decoded.dtype == torch.uint8


def test_torchvision_io_is_not_used_by_project_code():
    """Guard against building on an API torchvision has already deprecated."""
    offenders = [
        path for path in (ROOT / "src").rglob("*.py") if "torchvision.io" in path.read_text()
    ]
    assert offenders == [], f"torchvision.io is deprecated; use Pillow: {offenders}"
