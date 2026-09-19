"""Phase 11 — the five hybrid classifiers: import, build, forward/backward, logits shape
for several class counts, device moves, pretrained path, registry metadata. No training."""

from __future__ import annotations

import pytest
import torch

from src.hybrid_models import (
    HYBRID_MODELS,
    SLCAM,
    HybridBase,
    YoloStyleBackbone,
    build_hybrid,
)
from src.model import summarize
from src.preprocessing import PreprocessConfig, build_eval_transform
from tests.conftest import class_colour

KEYS = sorted(HYBRID_MODELS)
EXPECTED = {
    "cnn_vit_lstm",
    "yolo_efficientnet",
    "cnn_bilstm",
    "resnet_attention",
    "yolo_transformer",
}


def test_registry_lists_exactly_the_five_with_provenance():
    assert set(KEYS) == EXPECTED
    for spec in HYBRID_MODELS.values():
        assert spec.architecture and spec.paper_source and spec.adaptation
    assert "P4" in HYBRID_MODELS["resnet_attention"].paper_source
    for key in ("cnn_vit_lstm", "cnn_bilstm", "yolo_efficientnet", "yolo_transformer"):
        assert "none of the supplied papers" in HYBRID_MODELS[key].paper_source
    assert HYBRID_MODELS["yolo_transformer"].pretrained_components == ()


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("num_classes", [8, 5])
def test_forward_and_backward_produce_logits(key, num_classes):
    torch.manual_seed(0)
    model = build_hybrid(key, num_classes, pretrained=False)
    assert isinstance(model, HybridBase)
    images = torch.randn(3, 3, 224, 224)
    model.train()
    logits = model(images)
    assert logits.shape == (3, num_classes) and logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, num_classes - 1]))
    loss.backward()
    head_grads = [p.grad for p in model.classifier.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in head_grads)
    assert summarize(model).output_features == num_classes
    model.eval()
    with torch.no_grad():
        assert model(images[:1]).shape == (1, num_classes)


@pytest.mark.parametrize("key", KEYS)
def test_head_and_backbone_parameters_partition(key):
    model = build_hybrid(key, 8, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    head = sum(p.numel() for p in model.classifier.parameters())
    backbone = sum(p.numel() for p in model.backbone_parameters())
    assert head + backbone == total and head > 0 and backbone > head
    model.set_backbone_trainable(False)
    assert all(not p.requires_grad for p in model.backbone_parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())


@pytest.mark.parametrize("key", KEYS)
def test_models_move_between_devices(key):
    model = build_hybrid(key, 8, pretrained=False)
    cpu = model.to("cpu")
    with torch.no_grad():
        assert cpu(torch.randn(1, 3, 224, 224)).device.type == "cpu"
    if torch.backends.mps.is_available():
        mps = model.to("mps").eval()
        with torch.no_grad():
            assert mps(torch.randn(1, 3, 224, 224, device="mps")).device.type == "mps"
        model.to("cpu")
    if torch.cuda.is_available():
        cuda = model.to("cuda").eval()
        with torch.no_grad():
            assert cuda(torch.randn(1, 3, 224, 224, device="cuda")).device.type == "cuda"


def test_models_accept_the_existing_preprocessing_output(make_image_folder):
    root = make_image_folder(("a", "b"), per_class=1, size=(300, 200))
    image_path = next(root.glob("a/*"))
    from src.dataset import load_image

    tensor = build_eval_transform(PreprocessConfig())(load_image(image_path)).unsqueeze(0)
    assert tensor.shape == (1, 3, 224, 224)
    for key in KEYS:
        with torch.no_grad():
            assert build_hybrid(key, 8, pretrained=False).eval()(tensor).shape == (1, 8)


def test_pretrained_efficientnet_backbone_is_the_projects_weights():
    """The EfficientNet-B0 weights come from the project's existing builder (cached locally)."""
    from src.model import build_efficientnet_b0

    reference = build_efficientnet_b0(pretrained=True).features[0][0].weight
    model = build_hybrid("cnn_bilstm", 8, pretrained=True)
    assert torch.equal(model.cnn[0][0].weight, reference)


def test_yolo_backbone_and_slcam_shapes():
    backbone = YoloStyleBackbone()
    assert backbone(torch.randn(2, 3, 224, 224)).shape == (2, 256, 7, 7)
    attention = SLCAM(64)
    x = torch.randn(2, 64, 14, 14)
    y = attention(x)
    assert y.shape == x.shape and torch.isfinite(y).all()


def test_wrong_input_size_is_reported_not_silently_accepted():
    model = build_hybrid("yolo_transformer", 8, pretrained=False).eval()
    with pytest.raises(ValueError, match="224x224"):
        model(torch.randn(1, 3, 160, 160))
    with pytest.raises(ValueError):
        build_hybrid("does_not_exist", 8)
    with pytest.raises(ValueError):
        build_hybrid("cnn_bilstm", 1)
    with pytest.raises(ValueError):
        build_hybrid("resnet_attention", 8, pretrained=False, depth=34)


def test_resnet50_variant_builds():
    model = build_hybrid("resnet_attention", 8, pretrained=False, depth=50).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 8)


def test_class_colour_fixture_still_available():
    assert class_colour(0) != class_colour(1)
