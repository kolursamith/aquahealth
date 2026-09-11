"""Image -> CLAHE -> Resize -> Normalize -> Tensor.

Owner: Student 2
"""

import cv2
import numpy as np

from src.config import IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE to the luminance channel of an RGB image."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    lightness = clahe.apply(lightness)
    lab = cv2.merge((lightness, green_red, blue_yellow))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def resize(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] and standardize with ImageNet statistics."""
    image = image.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return (image - mean) / std


def preprocess(image: np.ndarray) -> np.ndarray:
    """Full inference-time pipeline: CLAHE -> resize -> normalize -> CHW tensor."""
    image = apply_clahe(image)
    image = resize(image)
    image = normalize(image)
    return np.transpose(image, (2, 0, 1))
