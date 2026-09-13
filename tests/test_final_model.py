"""The frozen final checkpoint (models/final_model.pth) through the production predictor.

Skipped when the checkpoint is not present (it is not committed). Uses only train-split
images from the frozen manifest — never the test split. No training, no weight changes.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import CHECKPOINT_PATH, CLASS_NAMES
from src.manifest import CANONICAL_CLASSES, MANIFEST_PATH
from src.predict import Predictor
from src.preprocessing import build_eval_transform
from src.risk_engine import get_risk_level
from src.train import load_checkpoint

pytestmark = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists() or not MANIFEST_PATH.exists(),
    reason="final checkpoint or frozen manifest not present",
)


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return Predictor(CHECKPOINT_PATH, device="cpu")


@pytest.fixture(scope="module")
def train_images() -> list[tuple[str, str]]:
    """(filepath, class_name) for the first train-split image of each class — never test."""
    seen: dict[str, str] = {}
    with MANIFEST_PATH.open() as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "train" and row["class_name"] not in seen:
                seen[row["class_name"]] = row["filepath"]
    assert set(seen) == set(CANONICAL_CLASSES)
    return [(seen[name], name) for name in CANONICAL_CLASSES]


def test_final_checkpoint_metadata_matches_the_project_contract():
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    assert list(checkpoint.class_names) == list(CANONICAL_CLASSES) == list(CLASS_NAMES)
    assert checkpoint.model_spec["num_classes"] == 8
    pre = checkpoint.preprocess
    assert (pre.image_size, pre.resize_size, pre.resize_mode) == (224, 256, "crop")
    assert pre.clahe is not None and (pre.clahe.clip_limit, pre.clahe.tile_grid_size) == (2.0, 8)
    assert tuple(pre.mean) == (0.485, 0.456, 0.406) and tuple(pre.std) == (0.229, 0.224, 0.225)
    model = checkpoint.build_model().eval()
    assert type(model).__name__ == "EfficientNet"
    assert sum(p.numel() for p in model.parameters()) == 4_017_796
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 8)


def test_predictor_uses_the_checkpoint_preprocessing_and_is_in_eval_mode(predictor):
    assert not predictor.model.training
    stages = [type(t).__name__ for t in predictor.transform.transforms]
    assert stages == [
        type(t).__name__ for t in build_eval_transform(predictor.checkpoint.preprocess).transforms
    ]
    assert stages[:3] == ["CLAHE", "Resize", "CenterCrop"]
    assert not any(s.startswith("Random") or s == "ColorJitter" for s in stages)


def test_real_train_images_predict_with_full_contract(predictor, train_images):
    for path, _ in train_images:
        result = predictor.predict(path)
        assert result.ok, result.error
        assert result.predicted_class in CANONICAL_CLASSES
        assert 0.0 <= result.confidence <= 1.0
        assert result.risk == get_risk_level(result.confidence)
        assert result.healthy == (result.predicted_class == "Healthy Fish")
        assert result.recommendation and result.message
        probs = [r.probability for r in result.ranked_predictions]
        assert len(probs) == 8 and abs(sum(probs) - 1.0) < 1e-4
        assert probs == sorted(probs, reverse=True) and result.confidence == probs[0]


def test_repeated_inference_is_deterministic(predictor, train_images):
    path = train_images[0][0]
    first = predictor.predict(path).to_dict()
    second = predictor.predict(path).to_dict()
    assert first["predicted_class"] == second["predicted_class"]
    assert abs(first["confidence"] - second["confidence"]) < 1e-6


def test_invalid_and_corrupt_inputs_return_error_results(predictor):
    for bad in (b"", b"not an image", np.zeros((2, 2, 3), np.uint8), "no/such/file.jpg"):
        result = predictor.predict(bad)
        assert result.status == "error" and result.error and result.predicted_class is None


def test_large_and_grayscale_inputs_are_handled(predictor):
    big = Image.fromarray(np.random.default_rng(1).integers(0, 255, (1800, 2400, 3), np.uint8))
    assert predictor.predict(big).ok
    grey = np.random.default_rng(2).integers(0, 255, (300, 300), np.uint8)
    result = predictor.predict(grey)
    assert result.ok and any("grayscale" in w for w in result.warnings)
