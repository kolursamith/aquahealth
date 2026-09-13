"""Brightness / blur screening used by the inference path and the UI."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from src.image_quality import BLUR_THRESHOLD, DARK_THRESHOLD, assess_quality


def _textured(size=(240, 180), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 220, (size[1] // 6, size[0] // 6, 3), np.uint8)
    return Image.fromarray(base).resize(size, Image.NEAREST)


def test_textured_bright_image_is_ok():
    quality = assess_quality(_textured())
    assert quality.ok and not quality.dark and not quality.blurry
    assert quality.brightness > DARK_THRESHOLD and quality.sharpness > BLUR_THRESHOLD
    assert quality.warnings == []
    assert set(quality.to_dict()) == {"brightness", "sharpness", "dark", "blurry"}


def test_dark_image_is_flagged():
    dark = Image.fromarray((np.asarray(_textured()) * 0.08).astype(np.uint8))
    quality = assess_quality(dark)
    assert quality.dark and any("dark" in w for w in quality.warnings)


def test_blurred_image_is_flagged():
    blurred = _textured().filter(ImageFilter.GaussianBlur(radius=12))
    quality = assess_quality(blurred)
    assert quality.blurry and any("blurry" in w for w in quality.warnings)


def test_large_and_small_inputs_share_the_scale():
    small = assess_quality(_textured((120, 90)))
    large = assess_quality(_textured((2400, 1800)))
    assert small.ok and large.ok
