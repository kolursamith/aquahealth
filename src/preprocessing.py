"""Canonical image preprocessing (Layer 6).

There is ONE definition of "an image ready for the model", `build_eval_transform`,
and it is used unchanged for validation, test and inference. Training
augmentation (src/augmentation.py) composes *around* the same resize target
and normalisation so train and inference inputs stay in the same space.

    PIL RGB
      │
      ├─ CLAHE (optional, off by default; configurable)   PIL → PIL
      │
      ├─ resize:  "crop"   shorter side → resize_size, centre-crop image_size   (torchvision preset)
      │           "squash" direct resize to image_size × image_size, no crop
      │
      ├─ uint8 → float32 in [0, 1]
      └─ ImageNet normalisation
                → torch.float32 (3, image_size, image_size)

CLAHE is applied to the lightness channel of the LAB representation so chroma
is untouched. It is an *option*, not a default: whether it helps on the real
dataset is an experiment for later, not an assumption.

Owner: Student 2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2

from src.config import IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRESET_RESIZE_SIZE = 256

ResizeMode = Literal["crop", "squash"]


@dataclass(frozen=True)
class CLAHEConfig:
    clip_limit: float = 2.0
    tile_grid_size: int = 8

    def __post_init__(self) -> None:
        if self.clip_limit <= 0:
            raise ValueError(f"clip_limit must be > 0, got {self.clip_limit}")
        if self.tile_grid_size < 1:
            raise ValueError(f"tile_grid_size must be >= 1, got {self.tile_grid_size}")


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = IMAGE_SIZE
    resize_size: int = PRESET_RESIZE_SIZE
    resize_mode: ResizeMode = "crop"
    clahe: CLAHEConfig | None = None
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __post_init__(self) -> None:
        if self.image_size < 1:
            raise ValueError(f"image_size must be >= 1, got {self.image_size}")
        if self.resize_mode == "crop" and self.resize_size < self.image_size:
            raise ValueError(
                f"resize_size ({self.resize_size}) must be >= image_size ({self.image_size}) "
                "in 'crop' mode"
            )
        if self.resize_mode not in ("crop", "squash"):
            raise ValueError(f"resize_mode must be 'crop' or 'squash', got {self.resize_mode!r}")


class CLAHE:
    """Contrast Limited Adaptive Histogram Equalisation on the LAB lightness channel.

    A callable PIL → PIL transform so it composes with `transforms.v2`. Uses
    OpenCV's reference implementation (tile grid + clip limit + bilinear
    interpolation across tile borders) rather than a re-implementation.
    """

    def __init__(self, config: CLAHEConfig = CLAHEConfig()) -> None:
        self.config = config
        self._clahe: cv2.CLAHE | None = None

    # cv2.CLAHE handles cannot be pickled, and DataLoader workers on macOS are
    # spawned by pickling the dataset (transform included). Only the config
    # crosses the process boundary; each process builds its own handle.
    def __getstate__(self) -> dict[str, CLAHEConfig]:
        return {"config": self.config}

    def __setstate__(self, state: dict[str, CLAHEConfig]) -> None:
        self.config = state["config"]
        self._clahe = None

    def _handle(self) -> cv2.CLAHE:
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.config.clip_limit,
                tileGridSize=(self.config.tile_grid_size, self.config.tile_grid_size),
            )
        return self._clahe

    def __call__(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            raise ValueError(f"CLAHE expects an RGB image, got mode {image.mode!r}")
        rgb = np.asarray(image)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = self._handle().apply(lab[:, :, 0])
        return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))

    def __repr__(self) -> str:
        return f"CLAHE({self.config})"


def resize_transforms(config: PreprocessConfig) -> list[v2.Transform]:
    """Geometric stage shared by eval and train pipelines (train may replace it)."""
    if config.resize_mode == "crop":
        return [
            v2.Resize(config.resize_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(config.image_size),
        ]
    return [
        v2.Resize(
            (config.image_size, config.image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
    ]


def tensor_transforms(config: PreprocessConfig) -> list[v2.Transform]:
    """uint8 PIL/tensor → normalised float32 tensor. The last stage of every pipeline."""
    return [
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(config.mean), std=list(config.std)),
    ]


def build_eval_transform(config: PreprocessConfig = PreprocessConfig()) -> v2.Compose:
    """The canonical, deterministic pipeline for validation, test and inference."""
    stages: list = []
    if config.clahe is not None:
        stages.append(CLAHE(config.clahe))
    stages += resize_transforms(config)
    stages += tensor_transforms(config)
    return v2.Compose(stages)


def denormalize(
    tensor: torch.Tensor, config: PreprocessConfig = PreprocessConfig()
) -> torch.Tensor:
    """Inverse of the normalisation stage; returns float32 in [0, 1] (for inspection only)."""
    mean = torch.tensor(config.mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(config.std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
