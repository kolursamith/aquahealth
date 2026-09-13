"""Layer 11 — prediction API (`src.predict`) and its frontend contract."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import ImageFolderDataset, build_dataloader
from src.device import available_backends
from src.model import build_classifier
from src.predict import (
    API_VERSION,
    MIN_IMAGE_SIDE,
    InvalidImageError,
    ModelLoadError,
    PredictionResult,
    Predictor,
    get_predictor,
    load_input,
    predict,
)
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform
from src.risk_engine import get_risk_level
from src.train import CHECKPOINT_FORMAT_VERSION, LAST_CHECKPOINT_NAME, TrainConfig, fit
from src.utils import set_seed
from tests.conftest import class_colour, write_image_folder

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("alpha", "beta", "gamma", "delta")
SMALL = PreprocessConfig(image_size=64, resize_size=72, clahe=CLAHEConfig())
CPU = torch.device("cpu")
CONTRACT_KEYS = {
    "status",
    "api_version",
    "model_version",
    "preprocessing_version",
    "device",
    "predicted_class",
    "confidence",
    "risk",
    "message",
    "healthy",
    "recommendation",
    "quality",
    "ranked_predictions",
    "warnings",
    "error",
}
LEGACY_KEYS = {"predicted_class", "confidence", "risk", "message"}


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory) -> Path:
    return write_image_folder(tmp_path_factory.mktemp("pred") / "train", CLASSES, per_class=8)


@pytest.fixture(scope="module")
def checkpoint_path(fixture_root, tmp_path_factory) -> Path:
    """A checkpoint trained on the colour fixture so predictions are checkable (plumbing only)."""
    dataset = ImageFolderDataset(fixture_root, transform=build_eval_transform(SMALL))
    loader = build_dataloader(dataset, batch_size=8, shuffle=True, seed=0, num_workers=0)
    set_seed(0)
    model = build_classifier(len(CLASSES), freeze_backbone=True)
    ck_dir = tmp_path_factory.mktemp("ck")
    fit(
        model,
        loader,
        config=TrainConfig(epochs=10, learning_rate=1e-3, seed=0),
        class_names=dataset.class_names,
        preprocess=SMALL,
        device=CPU,
        checkpoint_dir=ck_dir,
        stage="head",
    )
    return ck_dir / LAST_CHECKPOINT_NAME


@pytest.fixture(scope="module")
def predictor(checkpoint_path) -> Predictor:
    return Predictor(checkpoint_path, device="cpu")


def _class_image(index: int, size=(120, 90)) -> Image.Image:
    return Image.new("RGB", size, class_colour(index))


# --- input handling ---


def test_load_input_accepts_every_documented_form(tmp_path):
    image = _class_image(0)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    path = tmp_path / "img.png"
    path.write_bytes(buffer.getvalue())

    for candidate in (
        image,
        np.array(image),
        buffer.getvalue(),
        bytearray(buffer.getvalue()),
        io.BytesIO(buffer.getvalue()),
        str(path),
        path,
    ):
        rgb, warnings = load_input(candidate)
        assert rgb.mode == "RGB" and rgb.size == (120, 90)
        assert warnings == []


def test_load_input_converts_and_warns_for_non_rgb():
    grey_array = np.full((40, 50), 90, np.uint8)
    rgba = Image.new("RGBA", (40, 50), (1, 2, 3, 4))
    single = np.full((40, 50, 1), 90, np.uint8)
    four = np.zeros((40, 50, 4), np.uint8)
    for candidate, expected in (
        (grey_array, "grayscale array converted to RGB"),
        (rgba, "image mode RGBA converted to RGB"),
        (single, "single-channel array converted to RGB"),
        (four, "alpha channel discarded"),
    ):
        rgb, warnings = load_input(candidate)
        assert rgb.mode == "RGB" and warnings == [expected]


@pytest.mark.parametrize(
    "candidate,match",
    [
        (None, "no image provided"),
        (b"", "empty input"),
        (b"not an image at all", "could not decode"),
        (np.zeros((0, 0, 3), np.uint8), "empty array"),
        (np.zeros((40, 50, 3), np.float32), "uint8"),
        (np.zeros((40, 50, 2), np.uint8), "shape"),
        (np.zeros((4, 40, 50, 3), np.uint8), "shape"),
        (Image.new("RGB", (4, 4)), "too small"),
        (np.zeros((MIN_IMAGE_SIDE - 1, 100, 3), np.uint8), "too small"),
        (12345, "unsupported input type"),
        ("/definitely/not/here.png", "file not found"),
    ],
)
def test_load_input_rejects_invalid_input(candidate, match):
    with pytest.raises(InvalidImageError, match=match):
        load_input(candidate)


# --- predictor construction ---


def test_missing_checkpoint_raises_model_load_error(tmp_path):
    with pytest.raises(ModelLoadError, match="not found"):
        Predictor(tmp_path / "nope.pt")


def test_incompatible_checkpoint_raises_model_load_error(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION + 1}, bad)
    with pytest.raises(ModelLoadError, match="unsupported checkpoint format"):
        Predictor(bad)
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ModelLoadError, match="could not load"):
        Predictor(garbage)


def test_predictor_exposes_versions_and_classes(predictor, checkpoint_path):
    assert predictor.class_names == sorted(CLASSES)
    assert predictor.model_version.startswith(f"{checkpoint_path.name}@")
    assert "epoch 10" in predictor.model_version and "/head" in predictor.model_version
    assert predictor.preprocessing_version.startswith("preprocess-")
    assert str(predictor.device) == "cpu"


def test_preprocessing_version_changes_with_preprocessing_config(predictor):
    from src.predict import preprocessing_version

    other = Predictor(predictor.checkpoint.path, device="cpu")
    assert preprocessing_version(other.checkpoint) == predictor.preprocessing_version
    object.__setattr__(
        other.checkpoint, "preprocess", PreprocessConfig(image_size=96, resize_size=96)
    )
    assert preprocessing_version(other.checkpoint) != predictor.preprocessing_version


# --- prediction results ---


def test_prediction_result_contract(predictor):
    result = predictor.predict(_class_image(0))
    assert isinstance(result, PredictionResult) and result.ok
    payload = result.to_dict()
    assert set(payload) == CONTRACT_KEYS
    assert payload["status"] == "ok" and payload["api_version"] == API_VERSION
    assert payload["error"] is None
    assert payload["predicted_class"] in predictor.class_names
    assert 0.0 <= payload["confidence"] <= 1.0
    assert payload["risk"] == get_risk_level(payload["confidence"])
    assert payload["message"]
    assert json.dumps(payload)


def test_ranked_predictions_are_complete_sorted_and_normalised(predictor):
    result = predictor.predict(_class_image(2))
    ranked = result.ranked_predictions
    assert [r.rank for r in ranked] == [1, 2, 3, 4]
    assert [r.class_name for r in ranked] != [] and len({r.class_name for r in ranked}) == 4
    probabilities = [r.probability for r in ranked]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) == pytest.approx(1.0, abs=1e-5)
    assert result.predicted_class == ranked[0].class_name
    assert result.confidence == ranked[0].probability


def test_top_k_limits_ranking_but_not_the_top_prediction(predictor):
    full = predictor.predict(_class_image(1))
    top2 = predictor.predict(_class_image(1), top_k=2)
    assert len(top2.ranked_predictions) == 2
    assert top2.ranked_predictions == full.ranked_predictions[:2]
    assert (
        predictor.predict(_class_image(1), top_k=0).ranked_predictions[:1]
        == full.ranked_predictions[:1]
    )


def test_api_matches_the_raw_model_path_on_every_fixture_file(predictor, fixture_root):
    """Plumbing check: file -> decode -> checkpoint preprocessing -> model, via the API,
    equals the same computation done by hand, for every training file.

    The accuracy floor only guards against a garbage checkpoint: on this
    fixture CLAHE equalises away part of the colour signal and the head
    plateaus near 0.9 (measured), so exact fitting is not the claim here.
    """
    from src.dataset import load_image

    dataset = ImageFolderDataset(fixture_root)
    transform = build_eval_transform(predictor.checkpoint.preprocess)
    correct = 0
    for sample in dataset.samples:
        result = predictor.predict(sample.path)
        assert result.ok and result.warnings == ["image 48x40 is smaller than 64; upscaled"]
        with torch.inference_mode():
            expected = torch.softmax(predictor.model(transform(load_image(sample.path))[None]), 1)[
                0
            ]
        for ranked in result.ranked_predictions:
            index = predictor.class_names.index(ranked.class_name)
            assert ranked.probability == pytest.approx(expected[index].item(), abs=1e-6)
        correct += result.predicted_class == dataset.class_names[sample.label]
    assert correct / len(dataset) >= 0.8


def test_prediction_is_deterministic_and_matches_the_eval_transform_path(predictor):
    image = _class_image(3)
    a, b = predictor.predict(image), predictor.predict(image)
    assert a.to_dict() == b.to_dict()

    tensor = build_eval_transform(predictor.checkpoint.preprocess)(image).unsqueeze(0)
    with torch.inference_mode():
        expected = torch.softmax(predictor.model(tensor), dim=1)[0]
    for ranked in a.ranked_predictions:
        index = predictor.class_names.index(ranked.class_name)
        assert ranked.probability == pytest.approx(expected[index].item(), abs=1e-6)


def test_invalid_input_returns_an_error_result_not_an_exception(predictor):
    for bad in (b"", b"garbage", np.zeros((3, 3, 3), np.uint8), None, 3.14):
        result = predictor.predict(bad)
        assert not result.ok and result.status == "error"
        assert result.error and result.predicted_class is None and result.ranked_predictions == []
        assert result.model_version == predictor.model_version
        assert set(result.to_dict()) == CONTRACT_KEYS


def test_small_and_non_rgb_inputs_predict_with_warnings(predictor):
    tiny = np.full((20, 20), 120, np.uint8)
    result = predictor.predict(tiny)
    assert result.ok
    assert any("converted to RGB" in w for w in result.warnings)
    assert any("upscaled" in w for w in result.warnings)


def test_prediction_does_not_modify_the_model(predictor):
    before = {k: v.clone() for k, v in predictor.model.state_dict().items()}
    predictor.predict(_class_image(0))
    for key, value in predictor.model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert not predictor.model.training


@pytest.mark.parametrize("device", [n for n, ok in available_backends().items() if ok])
def test_predictor_runs_on_every_device_and_agrees_with_cpu(checkpoint_path, predictor, device):
    other = Predictor(checkpoint_path, device=device)
    assert str(other.device) == device
    a = predictor.predict(_class_image(2)).ranked_predictions
    b = other.predict(_class_image(2)).ranked_predictions
    assert [r.class_name for r in a] == [r.class_name for r in b]
    for x, y in zip(a, b):
        assert x.probability == pytest.approx(y.probability, abs=1e-4)


def test_unavailable_device_is_rejected_at_construction(checkpoint_path):
    if available_backends()["cuda"]:
        pytest.skip("cuda available here")
    with pytest.raises(ModelLoadError, match="not available on this machine"):
        Predictor(checkpoint_path, device="cuda")


# --- module-level convenience ---


def test_predict_function_uses_a_cached_predictor(checkpoint_path):
    get_predictor.cache_clear()
    first = predict(_class_image(0), checkpoint_path=checkpoint_path)
    second = predict(_class_image(0), checkpoint_path=checkpoint_path)
    assert first == second and first["status"] == "ok"
    assert get_predictor.cache_info().hits >= 1


def test_predict_function_with_missing_default_checkpoint_raises(tmp_path):
    with pytest.raises(ModelLoadError):
        predict(_class_image(0), checkpoint_path=tmp_path / "missing.pt")


def test_healthy_fish_prediction_is_flagged_and_worded_distinctly(predictor, monkeypatch):
    """Whatever the synthetic classes are, a top class named "Healthy Fish" must set
    `healthy` and use the "no disease detected" wording; any other class must not."""
    monkeypatch.setattr(predictor, "class_names", ["Healthy Fish", "beta", "gamma", "delta"])
    monkeypatch.setattr(
        predictor, "probabilities", lambda rgb: torch.tensor([0.9, 0.05, 0.03, 0.02])
    )
    healthy = predictor.predict(_class_image(0))
    assert healthy.ok and healthy.predicted_class == "Healthy Fish" and healthy.healthy is True
    assert healthy.risk == "HIGH" and "no disease detected" in healthy.message
    assert healthy.to_dict()["healthy"] is True

    monkeypatch.setattr(
        predictor, "probabilities", lambda rgb: torch.tensor([0.05, 0.9, 0.03, 0.02])
    )
    disease = predictor.predict(_class_image(0))
    assert disease.predicted_class == "beta" and disease.healthy is False
    assert disease.risk == "HIGH" and "disease indication" in disease.message

    monkeypatch.setattr(predictor, "probabilities", lambda rgb: torch.tensor([0.4, 0.3, 0.2, 0.1]))
    unsure = predictor.predict(_class_image(0))
    assert unsure.healthy is True and unsure.risk == "LOW" and "uncertain" in unsure.message


def test_result_carries_recommendation_and_quality(predictor):
    result = predictor.predict(_class_image(0))
    assert result.ok
    assert isinstance(result.recommendation, str) and result.recommendation
    assert set(result.quality) == {"brightness", "sharpness", "dark", "blurry"}
    payload = result.to_dict()
    assert payload["recommendation"] == result.recommendation
    assert payload["quality"]["dark"] in (True, False)


def test_dark_upload_is_flagged_but_still_predicted(predictor):
    dark = Image.fromarray(np.zeros((120, 90, 3), np.uint8) + 8)
    result = predictor.predict(dark)
    assert result.ok and result.quality["dark"] is True
    assert any("dark" in w for w in result.warnings)


def test_no_mock_predictor_remains():
    import importlib.util

    assert importlib.util.find_spec("app.mock_prediction") is None
