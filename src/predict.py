"""Prediction API (Layer 11): one image in, one structured result out.

    image (PIL / numpy uint8 / bytes / path)
      → validate + coerce to RGB PIL              (this module)
      → build_eval_transform(checkpoint.preprocess) (Layer 6, from the checkpoint)
      → Checkpoint.build_model()                  (Layers 3/4/7, from the checkpoint)
      → softmax → ranked classes                  (src/model.py)
      → risk level                                (src/risk_engine.py)
      → PredictionResult

Nothing about the model or the preprocessing is defined here; both are
rebuilt from the checkpoint so serving matches training exactly. Input
problems never raise out of `Predictor.predict`: they come back as a result
with `status="error"`. Problems that make prediction impossible at all
(missing / incompatible checkpoint) raise `ModelLoadError` when the
`Predictor` is constructed.

Frontend contract (superset of the original one):
    predicted_class, confidence, risk, message   — unchanged keys
    ranked_predictions, model_version, preprocessing_version, api_version,
    status, warnings, error, device, healthy,
    recommendation, quality                      — added
`risk` follows confidence alone (<50% LOW, 50-80% MODERATE, >80% HIGH);
`healthy` is True when the top class is "Healthy Fish", and the message then
says "no disease detected" instead of "disease indication"; `recommendation`
is the farmer-facing next step (src/risk_engine.py); `quality` carries the
brightness/blur screen of the input (src/image_quality.py), advisory only.

Owner: Student 1 + Student 2
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import v2

from src.config import CHECKPOINT_PATH
from src.device import resolve_device
from src.image_quality import assess_quality
from src.model import logits_to_probabilities
from src.preprocessing import build_eval_transform
from src.risk_engine import get_recommendation, get_risk_level, get_risk_message, is_healthy
from src.train import Checkpoint, load_checkpoint
from src.utils import get_logger

logger = get_logger(__name__)

API_VERSION = "1.0"
STATUS_OK = "ok"
STATUS_ERROR = "error"
MIN_IMAGE_SIDE = 8
ACCEPTED_ARRAY_CHANNELS = (1, 3, 4)

ImageInput = Union[Image.Image, np.ndarray, bytes, bytearray, io.BytesIO, str, Path]


class PredictionError(Exception):
    """Base class for this module's errors."""


class InvalidImageError(PredictionError):
    """The input could not be turned into an RGB image the model can consume."""


class ModelLoadError(PredictionError):
    """The checkpoint could not be loaded into a usable model."""


@dataclass(frozen=True)
class RankedPrediction:
    rank: int
    class_name: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    status: str
    api_version: str
    model_version: str | None
    preprocessing_version: str | None
    device: str | None
    predicted_class: str | None = None
    confidence: float | None = None
    risk: str | None = None
    message: str | None = None
    healthy: bool = False
    recommendation: str | None = None
    quality: dict[str, Any] | None = None
    ranked_predictions: list[RankedPrediction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- input handling ------------------------------------------------------------------


def _array_to_image(array: np.ndarray, warnings: list[str]) -> Image.Image:
    if array.size == 0:
        raise InvalidImageError("empty array")
    if array.dtype != np.uint8:
        raise InvalidImageError(f"expected a uint8 array, got dtype {array.dtype}")
    if array.ndim == 2:
        warnings.append("grayscale array converted to RGB")
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim != 3 or array.shape[2] not in ACCEPTED_ARRAY_CHANNELS:
        raise InvalidImageError(
            f"expected an array of shape (H, W, 3), (H, W, 4) or (H, W); got {array.shape}"
        )
    channels = array.shape[2]
    if channels == 1:
        warnings.append("single-channel array converted to RGB")
        return Image.fromarray(array[:, :, 0], mode="L").convert("RGB")
    if channels == 4:
        warnings.append("alpha channel discarded")
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def _decode_bytes(data: bytes, warnings: list[str]) -> Image.Image:
    if not data:
        raise InvalidImageError("empty input")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return _pil_to_rgb(image, warnings)
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "could not decode image: the file is not a readable image (unsupported or corrupt)"
        ) from exc
    except (OSError, ValueError) as exc:
        raise InvalidImageError(f"could not decode image: {exc}") from exc


def _pil_to_rgb(image: Image.Image, warnings: list[str]) -> Image.Image:
    if image.mode != "RGB":
        warnings.append(f"image mode {image.mode} converted to RGB")
    return image.convert("RGB")


def load_input(image: ImageInput) -> tuple[Image.Image, list[str]]:
    """Coerce any accepted input to an RGB PIL image, collecting non-fatal warnings."""
    warnings: list[str] = []
    if image is None:
        raise InvalidImageError("no image provided")
    if isinstance(image, Image.Image):
        rgb = _pil_to_rgb(image, warnings)
    elif isinstance(image, np.ndarray):
        rgb = _array_to_image(image, warnings)
    elif isinstance(image, (bytes, bytearray)):
        rgb = _decode_bytes(bytes(image), warnings)
    elif isinstance(image, io.BytesIO):
        rgb = _decode_bytes(image.getvalue(), warnings)
    elif isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise InvalidImageError(f"file not found: {path}")
        rgb = _decode_bytes(path.read_bytes(), warnings)
    else:
        raise InvalidImageError(f"unsupported input type {type(image).__name__}")

    width, height = rgb.size
    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise InvalidImageError(
            f"image too small: {width}x{height}; each side must be >= {MIN_IMAGE_SIDE}"
        )
    return rgb, warnings


# --- predictor ----------------------------------------------------------------------------


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def preprocessing_version(checkpoint: Checkpoint) -> str:
    payload = json.dumps(asdict(checkpoint.preprocess), sort_keys=True)
    return "preprocess-" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def model_version(checkpoint: Checkpoint) -> str:
    stage = f"/{checkpoint.stage}" if checkpoint.stage else ""
    return f"{checkpoint.path.name}@{_file_digest(checkpoint.path)} epoch {checkpoint.epoch}{stage}"


class Predictor:
    """Loads one checkpoint once and answers `predict(image)` repeatedly."""

    def __init__(self, checkpoint_path: Path | str, device: str | torch.device | None = None):
        path = Path(checkpoint_path)
        if not path.is_file():
            raise ModelLoadError(f"checkpoint not found: {path}")
        try:
            self.checkpoint = load_checkpoint(path)
            self.device = device if isinstance(device, torch.device) else resolve_device(device)
            self.model = self.checkpoint.build_model().to(self.device).eval()
        except ModelLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure mode is surfaced as one type
            raise ModelLoadError(f"could not load checkpoint {path}: {exc}") from exc

        self.class_names: list[str] = list(self.checkpoint.class_names)
        self.transform: v2.Compose = build_eval_transform(self.checkpoint.preprocess)
        self.model_version = model_version(self.checkpoint)
        self.preprocessing_version = preprocessing_version(self.checkpoint)
        logger.info(
            "predictor ready: %s, %d classes, %s, device %s",
            self.model_version,
            len(self.class_names),
            self.preprocessing_version,
            self.device,
        )

    def _error(self, message: str, warnings: list[str] | None = None) -> PredictionResult:
        return PredictionResult(
            status=STATUS_ERROR,
            api_version=API_VERSION,
            model_version=self.model_version,
            preprocessing_version=self.preprocessing_version,
            device=str(self.device),
            warnings=list(warnings or []),
            error=message,
        )

    def probabilities(self, image: Image.Image) -> torch.Tensor:
        """Class probabilities for one RGB PIL image, on the CPU, shape (K,)."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
        return logits_to_probabilities(logits.float())[0].cpu()

    def predict(self, image: ImageInput, top_k: int | None = None) -> PredictionResult:
        try:
            rgb, warnings = load_input(image)
        except InvalidImageError as exc:
            return self._error(str(exc))

        target = self.checkpoint.preprocess.image_size
        if min(rgb.size) < target:
            warnings.append(f"image {rgb.size[0]}x{rgb.size[1]} is smaller than {target}; upscaled")
        quality = assess_quality(rgb)
        warnings.extend(quality.warnings)

        try:
            probs = self.probabilities(rgb)
        except Exception as exc:  # noqa: BLE001 - inference failure is reported, not raised
            logger.exception("inference failed")
            return self._error(f"inference failed: {exc}", warnings)

        order = torch.argsort(probs, descending=True).tolist()
        limit = len(order) if top_k is None else max(1, min(top_k, len(order)))
        ranked = [
            RankedPrediction(
                rank=i + 1, class_name=self.class_names[j], probability=float(probs[j])
            )
            for i, j in enumerate(order[:limit])
        ]
        confidence = ranked[0].probability
        risk = get_risk_level(confidence)
        return PredictionResult(
            status=STATUS_OK,
            api_version=API_VERSION,
            model_version=self.model_version,
            preprocessing_version=self.preprocessing_version,
            device=str(self.device),
            predicted_class=ranked[0].class_name,
            confidence=confidence,
            risk=risk,
            message=get_risk_message(risk, ranked[0].class_name),
            healthy=is_healthy(ranked[0].class_name),
            recommendation=get_recommendation(ranked[0].class_name, risk),
            quality=quality.to_dict(),
            ranked_predictions=ranked,
            warnings=warnings,
        )


@lru_cache(maxsize=4)
def get_predictor(
    checkpoint_path: str | Path = CHECKPOINT_PATH, device: str | None = None
) -> Predictor:
    """Process-wide cached predictor for the frontend; raises ModelLoadError if unusable."""
    return Predictor(Path(checkpoint_path), device)


def predict(image: ImageInput, checkpoint_path: str | Path = CHECKPOINT_PATH) -> dict[str, Any]:
    """Frontend entry point: `predict(image)` → JSON-serialisable dict (see module docstring)."""
    return get_predictor(str(checkpoint_path)).predict(image).to_dict()
