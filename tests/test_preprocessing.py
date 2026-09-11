"""Layer 6 — canonical preprocessing (`src.preprocessing`)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from src.config import IMAGE_SIZE
from src.device import available_backends
from src.environment import parse_requirements
from src.model import PRETRAINED_WEIGHTS, build_classifier
from src.preprocessing import (
    CLAHE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PRESET_RESIZE_SIZE,
    CLAHEConfig,
    PreprocessConfig,
    build_eval_transform,
    denormalize,
)

ROOT = Path(__file__).resolve().parent.parent


def _noise_image(
    width: int = 400, height: int = 300, seed: int = 0, mode: str = "RGB"
) -> Image.Image:
    rng = np.random.default_rng(seed)
    channels = {"RGB": 3, "RGBA": 4, "L": 1}[mode]
    shape = (height, width, channels) if channels > 1 else (height, width)
    return Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8), mode=mode)


def _flat_image(value: int = 100, spread: int = 20, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(1)
    pixels = np.full((size, size, 3), value, np.uint8) + rng.integers(
        0, spread, (size, size, 3)
    ).astype(np.uint8)
    return Image.fromarray(pixels)


# --- declaration ---


def test_opencv_is_declared_at_installed_version():
    declared = {r.name: r.raw for r in parse_requirements(ROOT / "requirements" / "base.txt")}
    pinned = declared["opencv-python-headless"].split("==", 1)[1]
    assert pinned.startswith(cv2.__version__), (pinned, cv2.__version__)


def test_constants_match_config_and_pretrained_preset():
    preset = PRETRAINED_WEIGHTS.transforms()
    assert PreprocessConfig().image_size == IMAGE_SIZE == preset.crop_size[0]
    assert PRESET_RESIZE_SIZE == preset.resize_size[0]
    assert list(IMAGENET_MEAN) == preset.mean and list(IMAGENET_STD) == preset.std


# --- config validation ---


@pytest.mark.parametrize("kwargs", [{"clip_limit": 0}, {"clip_limit": -1.0}, {"tile_grid_size": 0}])
def test_clahe_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CLAHEConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"image_size": 0}, {"resize_size": 200, "image_size": 224}, {"resize_mode": "stretch"}],
)
def test_preprocess_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)


def test_squash_mode_allows_resize_size_below_image_size():
    PreprocessConfig(resize_mode="squash", resize_size=1)


def test_clahe_is_off_by_default():
    assert PreprocessConfig().clahe is None
    assert not any(isinstance(t, CLAHE) for t in build_eval_transform().transforms)


# --- eval transform ---


@pytest.mark.parametrize("size", [(400, 300), (300, 400), (224, 224), (50, 900), (1000, 1000)])
@pytest.mark.parametrize("mode", ["crop", "squash"])
def test_eval_output_contract_for_any_input_size(size, mode):
    transform = build_eval_transform(PreprocessConfig(resize_mode=mode))
    out = transform(_noise_image(*size))
    assert out.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_eval_crop_mode_matches_torchvision_preset_numerically():
    image = _noise_image()
    ours = build_eval_transform()(image)
    reference = PRETRAINED_WEIGHTS.transforms()(image)
    torch.testing.assert_close(ours, reference, rtol=0, atol=1e-5)


def test_eval_transform_is_bit_deterministic():
    image = _noise_image()
    for config in (PreprocessConfig(), PreprocessConfig(clahe=CLAHEConfig())):
        transform = build_eval_transform(config)
        assert torch.equal(transform(image), transform(image))


def test_eval_normalisation_is_exact():
    grey = Image.new("RGB", (300, 300), (128, 128, 128))
    out = build_eval_transform()(grey)
    expected = (128 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    torch.testing.assert_close(out[:, 0, 0], expected)
    torch.testing.assert_close(out.amax(dim=(1, 2)), expected)
    torch.testing.assert_close(out.amin(dim=(1, 2)), expected)


def test_denormalize_inverts_normalisation():
    from torchvision.transforms import InterpolationMode, v2

    image = _noise_image()
    restored = denormalize(build_eval_transform()(image))
    unnormalised = v2.Compose(
        [
            v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )(image)
    torch.testing.assert_close(restored, unnormalised, rtol=0, atol=1e-5)
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_crop_mode_crops_and_squash_mode_keeps_everything():
    width, height = 400, 200
    pixels = np.zeros((height, width, 3), np.uint8)
    pixels[:, :10] = 255
    image = Image.fromarray(pixels)
    crop = denormalize(build_eval_transform(PreprocessConfig(resize_mode="crop"))(image))
    squash = denormalize(build_eval_transform(PreprocessConfig(resize_mode="squash"))(image))
    assert crop[:, :, :5].max() < 0.1, "the left stripe should be cropped away in crop mode"
    assert squash[:, :, :3].mean() > 0.5, "the left stripe should survive in squash mode"


def test_eval_transform_custom_image_size():
    out = build_eval_transform(PreprocessConfig(image_size=96, resize_size=112))(_noise_image())
    assert out.shape == (3, 96, 96)


# --- CLAHE ---


def test_clahe_increases_contrast_of_a_flat_image():
    flat = _flat_image()
    before = np.asarray(flat).astype(np.float32).std()
    after = np.asarray(CLAHE()(flat)).astype(np.float32).std()
    assert after > before * 2


def test_clahe_preserves_size_mode_and_is_deterministic():
    image = _noise_image(123, 77)
    out = CLAHE()(image)
    assert out.size == image.size and out.mode == "RGB"
    assert np.array_equal(np.asarray(out), np.asarray(CLAHE()(image)))


def test_clahe_rejects_non_rgb():
    with pytest.raises(ValueError, match="RGB"):
        CLAHE()(_noise_image(mode="L"))


def test_clahe_parameters_change_the_result():
    image = _flat_image()
    a = np.asarray(CLAHE(CLAHEConfig(clip_limit=1.0))(image))
    b = np.asarray(CLAHE(CLAHEConfig(clip_limit=8.0))(image))
    c = np.asarray(CLAHE(CLAHEConfig(tile_grid_size=2))(image))
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


def test_clahe_changes_lightness_but_not_chroma_much():
    image = _flat_image(value=90, spread=30)
    before = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.int16)
    after = cv2.cvtColor(np.asarray(CLAHE()(image)), cv2.COLOR_RGB2LAB).astype(np.int16)
    lightness_delta = np.abs(after[:, :, 0] - before[:, :, 0]).mean()
    chroma_delta = np.abs(after[:, :, 1:] - before[:, :, 1:]).mean()
    assert lightness_delta > 5
    assert chroma_delta < lightness_delta / 4


def test_clahe_survives_pickling_and_still_works():
    """DataLoader workers receive the transform by pickling; cv2 handles cannot be pickled."""
    import pickle

    original = CLAHE(CLAHEConfig(clip_limit=3.0, tile_grid_size=4))
    image = _flat_image()
    expected = np.asarray(original(image))
    restored = pickle.loads(pickle.dumps(original))
    assert restored.config == original.config
    assert np.array_equal(np.asarray(restored(image)), expected)
    assert pickle.loads(pickle.dumps(build_eval_transform(PreprocessConfig(clahe=CLAHEConfig()))))


def test_clahe_pipelines_work_in_worker_processes(make_image_folder):
    from src.augmentation import build_train_transform
    from src.dataset import ImageFolderDataset, build_dataloader

    config = PreprocessConfig(clahe=CLAHEConfig())
    root = make_image_folder(("a", "b"), per_class=3, size=(90, 70))
    for transform in (build_eval_transform(config), build_train_transform(config)):
        dataset = ImageFolderDataset(root, transform=transform)
        images, labels = next(iter(build_dataloader(dataset, batch_size=6, num_workers=2)))
        assert images.shape == (6, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert labels.tolist() == [0, 0, 0, 1, 1, 1]


def test_eval_transform_with_clahe_differs_from_without():
    image = _noise_image()
    with_clahe = build_eval_transform(PreprocessConfig(clahe=CLAHEConfig()))(image)
    without = build_eval_transform()(image)
    assert with_clahe.shape == without.shape
    assert not torch.equal(with_clahe, without)


# --- integration with Layers 4–5 ---


def test_dataset_with_eval_transform_batches_and_feeds_classifier(make_image_folder):
    from src.dataset import ImageFolderDataset, build_dataloader

    root = make_image_folder(("a", "b", "c", "d"), per_class=2, size=(90, 70))
    dataset = ImageFolderDataset(root, transform=build_eval_transform())
    images, labels = next(iter(build_dataloader(dataset, batch_size=8, num_workers=0)))
    assert images.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE) and images.dtype == torch.float32

    model = build_classifier(len(dataset.class_names), pretrained=False).eval()
    with torch.no_grad():
        assert model(images).shape == (8, 4)


@pytest.mark.parametrize("device", [n for n, ok in available_backends().items() if ok])
def test_preprocessed_tensor_moves_to_device(device):
    out = build_eval_transform()(_noise_image()).to(device)
    assert out.device.type == device and torch.isfinite(out).all()
