"""Grad-CAM on the frozen classifier: shapes, ranges, geometry, model untouched."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from src.explainability import (
    EXPLANATION_NOTE,
    Explanation,
    blend,
    colourise,
    crop_geometry,
    generate_gradcam,
    map_heatmap_to_original,
    target_layer,
)
from src.preprocessing import CLAHEConfig, PreprocessConfig
from tests.test_prediction import (  # noqa: F401 - module-scoped fixtures are re-used here
    CLASSES,
    _class_image,
    checkpoint_path,
    fixture_root,
    predictor,
)


def _state_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_target_layer_is_the_last_conv_block(predictor):  # noqa: F811
    layer = target_layer(predictor.model)
    assert layer is predictor.model.features[-1]
    assert type(layer).__name__ == "Conv2dNormActivation"
    assert isinstance(layer[0], torch.nn.Conv2d) and layer[0].out_channels == 1280


def test_gradcam_output_contract_and_model_untouched(predictor):  # noqa: F811
    before = _state_digest(predictor.model)
    explanation = generate_gradcam(predictor, _class_image(1))
    assert isinstance(explanation, Explanation)
    size = predictor.checkpoint.preprocess.image_size
    assert explanation.heatmap.shape == (size, size)
    assert explanation.heatmap.dtype == np.float32
    assert 0.0 <= explanation.heatmap.min() and explanation.heatmap.max() <= 1.0
    assert explanation.heatmap.max() == pytest.approx(1.0)
    assert explanation.target_class == predictor.class_names[explanation.target_index]
    assert explanation.model_input.size == (size, size)
    assert explanation.overlay_input.size == (size, size)
    assert explanation.overlay_original.size == (120, 90)  # same as the uploaded image
    assert 0.0 <= explanation.peak[0] <= 1.0 and 0.0 <= explanation.peak[1] <= 1.0
    assert _state_digest(predictor.model) == before, "weights must not change"
    assert not predictor.model.training
    assert all(p.grad is None for p in predictor.model.parameters())


def test_gradcam_matches_the_predicted_class_and_is_deterministic(predictor):  # noqa: F811
    image = _class_image(2)
    result = predictor.predict(image)
    a = generate_gradcam(predictor, image)
    b = generate_gradcam(predictor, image)
    assert a.target_class == result.predicted_class
    assert np.allclose(a.heatmap, b.heatmap, atol=1e-5)
    other = generate_gradcam(predictor, image, class_index=(a.target_index + 1) % len(CLASSES))
    assert other.target_index != a.target_index


def test_prediction_after_explanation_is_unchanged(predictor):  # noqa: F811
    image = _class_image(0)
    before = predictor.predict(image).to_dict()
    generate_gradcam(predictor, image)
    after = predictor.predict(image).to_dict()
    assert before["predicted_class"] == after["predicted_class"]
    assert abs(before["confidence"] - after["confidence"]) < 1e-6


def test_colourise_and_blend_shapes():
    heat = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    colours = colourise(heat)
    assert colours.shape == (4, 4, 3) and colours.dtype == np.uint8
    base = Image.fromarray(np.full((4, 4, 3), 120, np.uint8))
    out = blend(base, heat)
    assert out.size == (4, 4) and out.mode == "RGB"
    assert np.array_equal(np.asarray(out)[0, 0], [120, 120, 120]), "zero heat leaves pixels alone"


def test_crop_geometry_and_back_projection():
    pre = PreprocessConfig(image_size=224, resize_size=256, clahe=CLAHEConfig())
    wide = Image.new("RGB", (800, 400))
    rw, rh, left, top = crop_geometry(wide, pre)
    assert rh == 256 and rw == 512 and top == 16 and left == 144
    heat = np.zeros((224, 224), np.float32)
    heat[100:124, 100:124] = 1.0
    mapped = map_heatmap_to_original(heat, wide, pre)
    assert mapped.shape == (400, 800) and mapped.max() > 0.9
    ys, xs = np.nonzero(mapped > 0.5)
    assert 300 < xs.mean() < 500 and 150 < ys.mean() < 250  # centre of the wide image


def test_explanation_note_states_the_limitation():
    assert "not a medical/veterinary diagnosis" in EXPLANATION_NOTE
    assert "activation" in EXPLANATION_NOTE
