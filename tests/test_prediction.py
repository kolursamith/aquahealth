"""Layer 11 — prediction API (`src.predict`), mock parity, and the Streamlit entry point."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from app.mock_prediction import predict as mock_predict
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


# --- mock parity ---


def test_mock_predictor_matches_the_real_contract(predictor):
    real = predictor.predict(_class_image(0)).to_dict()
    mock = mock_predict(np.zeros((64, 64, 3), np.uint8))
    assert set(mock) == set(real) == CONTRACT_KEYS
    assert LEGACY_KEYS <= set(mock)
    assert mock["status"] == "ok" and mock["risk"] == get_risk_level(mock["confidence"])
    assert mock["model_version"].startswith("mock")
    assert any("mock predictor" in w for w in mock["warnings"])
    assert [r["rank"] for r in mock["ranked_predictions"]] == [1, 2, 3]


def test_mock_predictor_reports_invalid_input_like_the_real_one():
    result = mock_predict(b"")
    assert result["status"] == "error" and "empty input" in result["error"]
    assert set(result) == CONTRACT_KEYS


# --- Streamlit entry point ---


def _run_app(monkeypatch, checkpoint: Path | None):
    from streamlit.testing.v1 import AppTest

    if checkpoint is None:
        monkeypatch.setenv("AQUAHEALTH_CHECKPOINT", str(ROOT / "models" / "does-not-exist.pt"))
    else:
        monkeypatch.setenv("AQUAHEALTH_CHECKPOINT", str(checkpoint))
    return AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=120).run()


def test_app_runs_on_the_mock_path_without_a_checkpoint(monkeypatch):
    at = _run_app(monkeypatch, None)
    assert not at.exception
    assert [t.value for t in at.title] == ["AquaHealth AI"]
    assert any("mock predictor" in c.value for c in at.caption)
    assert at.expander[0].label.startswith("Classes known to the loaded model")


def test_app_loads_the_real_predictor_when_a_checkpoint_exists(monkeypatch, checkpoint_path):
    at = _run_app(monkeypatch, checkpoint_path)
    assert not at.exception
    assert any(checkpoint_path.name in c.value for c in at.caption)
    assert at.expander[0].label == "Classes known to the loaded model (4)"
