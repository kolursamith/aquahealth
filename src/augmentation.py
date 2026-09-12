"""Training-only augmentation (Layer 6).

    PIL RGB
      ├─ CLAHE (if enabled in PreprocessConfig — same stage as eval)
      ├─ RandomResizedCrop(image_size, scale, ratio)   controlled crop / zoom
      ├─ RandomHorizontalFlip(p)
      ├─ RandomRotation(±degrees)
      ├─ ColorJitter(brightness, contrast)
      └─ tensor_transforms(config)                       identical to eval

Only the geometric/photometric middle differs from `build_eval_transform`;
the output space (size, dtype, normalisation) is shared through the same
`PreprocessConfig`, so a model trained with this pipeline receives
identically-scaled inputs at inference.

Owner: Student 2
"""

from __future__ import annotations

from dataclasses import dataclass

from torchvision.transforms import InterpolationMode, v2

from src.preprocessing import CLAHE, PreprocessConfig, tensor_transforms


@dataclass(frozen=True)
class AugmentConfig:
    crop_scale: tuple[float, float] = (0.8, 1.0)
    crop_ratio: tuple[float, float] = (0.9, 1.1)
    horizontal_flip: float = 0.5
    rotation_degrees: float = 15.0
    brightness: float = 0.2
    contrast: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.crop_scale[0] <= self.crop_scale[1] <= 1.0:
            raise ValueError(f"crop_scale must satisfy 0 < lo <= hi <= 1, got {self.crop_scale}")
        if not 0.0 <= self.horizontal_flip <= 1.0:
            raise ValueError(f"horizontal_flip must be a probability, got {self.horizontal_flip}")
        if self.rotation_degrees < 0 or self.brightness < 0 or self.contrast < 0:
            raise ValueError("rotation_degrees, brightness and contrast must be >= 0")


def build_train_transform(
    config: PreprocessConfig = PreprocessConfig(),
    augment: AugmentConfig = AugmentConfig(),
) -> v2.Compose:
    stages: list = []
    if config.clahe is not None:
        stages.append(CLAHE(config.clahe))
    stages += [
        v2.RandomResizedCrop(
            config.image_size,
            scale=augment.crop_scale,
            ratio=augment.crop_ratio,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=augment.horizontal_flip),
        v2.RandomRotation(augment.rotation_degrees, interpolation=InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=augment.brightness, contrast=augment.contrast),
    ]
    stages += tensor_transforms(config)
    return v2.Compose(stages)
