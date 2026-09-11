import pytest

pytest.importorskip("cv2", reason="OpenCV is introduced by Layer 6 (CLAHE)")

import numpy as np  # noqa: E402

from src.config import IMAGE_SIZE  # noqa: E402
from src.preprocessing import apply_clahe, preprocess, resize  # noqa: E402


def _dummy_image(size: int = 300) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_apply_clahe_preserves_shape():
    image = _dummy_image()
    result = apply_clahe(image)
    assert result.shape == image.shape


def test_resize_to_config_image_size():
    image = _dummy_image()
    result = resize(image)
    assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)


def test_preprocess_returns_chw_tensor():
    image = _dummy_image()
    tensor = preprocess(image)
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
