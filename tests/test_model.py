"""Layer 3 — pretrained EfficientNet-B0 backbone (`src.model`).

Verifies that the official ImageNet-1K weights are obtained and hash-checked,
that the stock model has the documented size and output, and that its forward
pass behaves correctly (shape, finiteness, determinism, train/eval
distinction, device agreement) on every backend available on this machine.

The weights are general-purpose ImageNet weights. Nothing here measures fish
disease performance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.hub import get_dir as hub_dir
from torchvision.models import EfficientNet

from src.device import available_backends
from src.model import (
    IMAGENET_NUM_CLASSES,
    INPUT_CHANNELS,
    PRETRAINED_WEIGHTS,
    ModelSummary,
    build_efficientnet_b0,
    logits_to_probabilities,
    summarize,
)
from src.utils import set_seed

EXPECTED_PARAMETERS = 5_288_548
CANONICAL_SIZE = 224


def _devices_to_test() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


@pytest.fixture(scope="module")
def pretrained() -> EfficientNet:
    return build_efficientnet_b0(pretrained=True).eval()


@pytest.fixture(scope="module")
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, INPUT_CHANNELS, CANONICAL_SIZE, CANONICAL_SIZE)


# --- weights provenance ---------------------------------------------------------------


def test_pretrained_weights_are_the_official_imagenet1k_v1():
    assert PRETRAINED_WEIGHTS.name == "IMAGENET1K_V1"
    assert PRETRAINED_WEIGHTS.url.startswith("https://download.pytorch.org/models/")
    assert IMAGENET_NUM_CLASSES == 1000


def test_pretrained_weight_file_is_cached_and_hash_verified(pretrained):
    filename = PRETRAINED_WEIGHTS.url.rsplit("/", 1)[-1]
    path = Path(hub_dir()) / "checkpoints" / filename
    assert path.is_file(), "building the pretrained model must populate the hub cache"

    expected_prefix = filename.rsplit("-", 1)[-1].removesuffix(".pth")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest.startswith(expected_prefix), "cached weight file does not match its declared hash"


def test_pretrained_weights_differ_from_random_init(pretrained):
    torch.manual_seed(0)
    random_init = build_efficientnet_b0(pretrained=False)
    first_conv = "features.0.0.weight"
    assert not torch.equal(
        pretrained.state_dict()[first_conv], random_init.state_dict()[first_conv]
    )


def test_pretrained_load_is_deterministic(pretrained):
    again = build_efficientnet_b0(pretrained=True)
    for key, value in pretrained.state_dict().items():
        assert torch.equal(value, again.state_dict()[key]), key


# --- architecture ---------------------------------------------------------------------


def test_builds_torchvision_efficientnet(pretrained):
    assert isinstance(pretrained, EfficientNet)


def test_summary_matches_documented_architecture(pretrained):
    summary = summarize(pretrained)
    assert summary == ModelSummary(
        architecture="EfficientNet",
        total_parameters=EXPECTED_PARAMETERS,
        trainable_parameters=EXPECTED_PARAMETERS,
        output_features=IMAGENET_NUM_CLASSES,
    )
    assert summary.total_parameters == PRETRAINED_WEIGHTS.meta["num_params"]


def test_summary_counts_only_trainable_parameters_as_trainable(pretrained):
    frozen = build_efficientnet_b0(pretrained=False)
    for parameter in frozen.features.parameters():
        parameter.requires_grad_(False)
    summary = summarize(frozen)
    assert summary.total_parameters == EXPECTED_PARAMETERS
    assert summary.trainable_parameters == sum(p.numel() for p in frozen.classifier.parameters())


def test_stock_head_is_dropout_then_linear_1280_to_1000(pretrained):
    dropout, linear = pretrained.classifier
    assert isinstance(dropout, torch.nn.Dropout) and dropout.p == pytest.approx(0.2)
    assert isinstance(linear, torch.nn.Linear)
    assert (linear.in_features, linear.out_features) == (1280, IMAGENET_NUM_CLASSES)


# --- forward pass -------------------------------------------------------------------------


@pytest.mark.parametrize("device", _devices_to_test())
def test_forward_pass_shape_and_finiteness(pretrained, batch, device):
    model = build_efficientnet_b0(pretrained=True).to(device).eval()
    with torch.no_grad():
        logits = model(batch.to(device))
    assert logits.shape == (2, IMAGENET_NUM_CLASSES)
    assert torch.isfinite(logits).all()


def test_devices_agree_on_logits(pretrained, batch):
    devices = _devices_to_test()
    if len(devices) < 2:
        pytest.skip("only one backend available; nothing to compare")
    with torch.no_grad():
        reference = pretrained(batch)
        for device in devices:
            if device == "cpu":
                continue
            model = build_efficientnet_b0(pretrained=True).to(device).eval()
            logits = model(batch.to(device)).cpu()
            torch.testing.assert_close(logits, reference, rtol=1e-4, atol=1e-4)


def test_eval_mode_is_deterministic(pretrained, batch):
    with torch.no_grad():
        assert torch.equal(pretrained(batch), pretrained(batch))


def test_train_mode_is_stochastic_and_differs_from_eval(pretrained, batch):
    model = build_efficientnet_b0(pretrained=True)
    with torch.no_grad():
        model.eval()
        eval_logits = model(batch)
        model.train()
        set_seed(1)
        first = model(batch)
        second = model(batch)
    assert not torch.equal(first, second), "dropout / stochastic depth must be active in train()"
    assert not torch.equal(first, eval_logits)


def test_train_mode_is_reproducible_under_seed(batch):
    model = build_efficientnet_b0(pretrained=True).train()
    with torch.no_grad():
        set_seed(5)
        first = model(batch)
        set_seed(5)
        second = model(batch)
    assert torch.equal(first, second)


def test_single_image_matches_its_batched_logits(pretrained, batch):
    with torch.no_grad():
        alone = pretrained(batch[:1])
        together = pretrained(batch)[:1]
    torch.testing.assert_close(alone, together, rtol=1e-5, atol=1e-5)


def test_probabilities_sum_to_one(pretrained, batch):
    with torch.no_grad():
        probabilities = logits_to_probabilities(pretrained(batch))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))
    assert (probabilities >= 0).all()


# --- input contract -------------------------------------------------------------------------


def test_weight_preset_transform_yields_canonical_input():
    preset = PRETRAINED_WEIGHTS.transforms()
    tensor = preset(Image.new("RGB", (640, 480), (200, 100, 50)))
    assert tensor.shape == (INPUT_CHANNELS, CANONICAL_SIZE, CANONICAL_SIZE)
    assert tensor.dtype == torch.float32


def test_accepts_non_canonical_spatial_size(pretrained):
    with torch.no_grad():
        assert pretrained(torch.randn(1, 3, 256, 320)).shape == (1, IMAGENET_NUM_CLASSES)


def test_rejects_wrong_channel_count(pretrained):
    with pytest.raises(RuntimeError):
        pretrained(torch.randn(1, 1, CANONICAL_SIZE, CANONICAL_SIZE))


def test_rejects_unbatched_input(pretrained):
    with pytest.raises((RuntimeError, ValueError)):
        pretrained(torch.randn(3, CANONICAL_SIZE, CANONICAL_SIZE))


# --- gradients ----------------------------------------------------------------------------------


def test_gradients_reach_every_parameter(batch):
    model = build_efficientnet_b0(pretrained=True).train()
    model(batch).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
