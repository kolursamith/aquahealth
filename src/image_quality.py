"""Cheap image-quality screening for the inference path (Layer 11 support).

Two heuristics that the frontend turns into guidance, never into a diagnosis:

- **brightness**: mean of the greyscale image in [0, 1]; below `DARK_THRESHOLD`
  the photo is flagged as too dark.
- **sharpness**: variance of the Laplacian of the greyscale image (the classic
  blur detector); below `BLUR_THRESHOLD` the photo is flagged as blurry. The
  Laplacian is computed after resizing the longer side to `ANALYSIS_SIDE` so the
  threshold means the same thing for a 128-px and a 4000-px upload.

Both flags are advisory: prediction still runs, and the result carries the
flags so the UI can tell the farmer to re-shoot. Thresholds are engineering
choices, not learned values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

DARK_THRESHOLD = 0.15
BLUR_THRESHOLD = 25.0
ANALYSIS_SIDE = 512


@dataclass(frozen=True)
class ImageQuality:
    brightness: float
    sharpness: float
    dark: bool
    blurry: bool

    @property
    def ok(self) -> bool:
        return not (self.dark or self.blurry)

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.dark:
            out.append(f"image looks dark (brightness {self.brightness:.2f} < {DARK_THRESHOLD})")
        if self.blurry:
            out.append(f"image looks blurry (sharpness {self.sharpness:.1f} < {BLUR_THRESHOLD})")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_quality(image: Image.Image) -> ImageQuality:
    """Brightness and blur flags for an RGB PIL image."""
    grey = np.asarray(image.convert("L"))
    longest = max(grey.shape)
    if longest > ANALYSIS_SIDE:
        scale = ANALYSIS_SIDE / longest
        size = (max(1, round(grey.shape[1] * scale)), max(1, round(grey.shape[0] * scale)))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    brightness = float(grey.mean() / 255.0)
    sharpness = float(cv2.Laplacian(grey, cv2.CV_64F).var())
    return ImageQuality(
        brightness=round(brightness, 4),
        sharpness=round(sharpness, 2),
        dark=brightness < DARK_THRESHOLD,
        blurry=sharpness < BLUR_THRESHOLD,
    )
