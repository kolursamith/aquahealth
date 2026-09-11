"""Training-only augmentation pipeline.

Training image -> Flip -> Rotation -> Brightness/Contrast -> Controlled crop/zoom

Owner: Student 2
"""

import albumentations as A

from src.config import IMAGE_SIZE


def get_train_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), p=0.5),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE) -> A.Compose:
    """No randomness — resizing/normalization is handled by preprocessing.py."""
    return A.Compose([])
