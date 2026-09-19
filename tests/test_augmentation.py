"""Layer 6 — training augmentation (`src.augmentation`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from src.augmentation import AugmentConfig, build_train_transform
from src.config import IMAGE_SIZE
from src.preprocessing import CLAHE, CLAHEConfig, PreprocessConfig, build_eval_transform
from src.utils import set_seed


def _noise_image(width: int = 400, height: int = 300, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


# --- config ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_scale": (0.0, 1.0)},
        {"crop_scale": (0.9, 0.8)},
        {"crop_scale": (0.5, 1.5)},
        {"horizontal_flip": 1.5},
        {"rotation_degrees": -1},
        {"brightness": -0.1},
        {"saturation": -0.1},
    ],
)
def test_augment_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AugmentConfig(**kwargs)


def test_saturation_jitter_is_off_by_default_and_configurable():
    from torchvision.transforms import v2

    def jitter(transform):
        return next(t for t in transform.transforms if isinstance(t, v2.ColorJitter))

    assert AugmentConfig().saturation == 0.0
    assert jitter(build_train_transform()).saturation is None
    lo, hi = jitter(build_train_transform(augment=AugmentConfig(saturation=0.2))).saturation
    assert (lo, hi) == pytest.approx((0.8, 1.2))


# --- output contract ---


@pytest.mark.parametrize("size", [(400, 300), (224, 224), (60, 800)])
def test_train_output_contract_for_any_input_size(size):
    out = build_train_transform()(_noise_image(*size))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_train_and_eval_share_the_tensor_stage():
    train_tail = build_train_transform().transforms[-3:]
    eval_tail = build_eval_transform().transforms[-3:]
    for a, b in zip(train_tail, eval_tail):
        assert type(a) is type(b)
        assert repr(a) == repr(b)


def test_train_output_lives_in_the_same_space_as_eval():
    image = _noise_image()
    set_seed(0)
    train_out = build_train_transform()(image)
    eval_out = build_eval_transform()(image)
    assert train_out.shape == eval_out.shape
    assert abs(train_out.mean().item() - eval_out.mean().item()) < 0.5
    assert abs(train_out.std().item() - eval_out.std().item()) < 0.5


# --- randomness ---


def test_train_transform_is_reproducible_under_seed_and_varies_otherwise():
    transform = build_train_transform()
    image = _noise_image()
    set_seed(3)
    first = transform(image)
    set_seed(3)
    second = transform(image)
    third = transform(image)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_train_transform_with_all_randomness_disabled_is_deterministic():
    still = AugmentConfig(
        crop_scale=(1.0, 1.0),
        crop_ratio=(1.0, 1.0),
        horizontal_flip=0.0,
        rotation_degrees=0.0,
        brightness=0.0,
        contrast=0.0,
    )
    transform = build_train_transform(augment=still)
    image = _noise_image(300, 300)
    assert torch.equal(transform(image), transform(image))


def test_horizontal_flip_probability_one_mirrors_the_image():
    only_flip = AugmentConfig(
        crop_scale=(1.0, 1.0),
        crop_ratio=(1.0, 1.0),
        horizontal_flip=1.0,
        rotation_degrees=0.0,
        brightness=0.0,
        contrast=0.0,
    )
    image = _noise_image(300, 300)
    flipped = build_train_transform(augment=only_flip)(image)
    plain = build_train_transform(
        augment=AugmentConfig(**{**only_flip.__dict__, "horizontal_flip": 0.0})
    )(image)
    torch.testing.assert_close(flipped, plain.flip(-1))


# --- stage composition ---


def test_stage_order_and_clahe_placement():
    without = build_train_transform().transforms
    assert [type(t).__name__ for t in without] == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomRotation",
        "ColorJitter",
        "PILToTensor",
        "ToDtype",
        "Normalize",
    ]
    with_clahe = build_train_transform(PreprocessConfig(clahe=CLAHEConfig())).transforms
    assert isinstance(with_clahe[0], CLAHE)
    assert [type(t).__name__ for t in with_clahe[1:]] == [type(t).__name__ for t in without]


def test_augment_parameters_reach_the_transforms():
    augment = AugmentConfig(crop_scale=(0.5, 0.9), horizontal_flip=0.25, rotation_degrees=30.0)
    crop, flip, rotation, *_ = build_train_transform(augment=augment).transforms
    assert isinstance(crop, v2.RandomResizedCrop) and tuple(crop.scale) == (0.5, 0.9)
    assert isinstance(flip, v2.RandomHorizontalFlip) and flip.p == 0.25
    assert isinstance(rotation, v2.RandomRotation) and rotation.degrees == [-30.0, 30.0]


def test_train_transform_custom_image_size():
    out = build_train_transform(PreprocessConfig(image_size=128, resize_size=144))(_noise_image())
    assert out.shape == (3, 128, 128)
