"""Layer 4 — configurable disease classifier head (`src.model.build_classifier`).

The class count is a required argument throughout: no test relies on
`src.config.NUM_CLASSES`, because that value is a brief-derived placeholder
until the real dataset has been inspected.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from src.device import available_backends
from src.model import (
    IMAGENET_NUM_CLASSES,
    MIN_NUM_CLASSES,
    STOCK_DROPOUT,
    build_classifier,
    build_efficientnet_b0,
    logits_to_probabilities,
    set_backbone_trainable,
    summarize,
)
from src.utils import set_seed

BACKBONE_FEATURES = 1280
STOCK_TOTAL_PARAMETERS = 5_288_548


def _devices_to_test() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


def _head_parameter_count(num_classes: int) -> int:
    return BACKBONE_FEATURES * num_classes + num_classes


@pytest.fixture(scope="module")
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 3, 224, 224)


# --- interface ---


def test_num_classes_is_a_required_positional_argument():
    parameters = inspect.signature(build_classifier).parameters
    assert parameters["num_classes"].default is inspect.Parameter.empty
    assert parameters["num_classes"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("dropout", "freeze_backbone", "pretrained"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("bad", [0, 1, -3])
def test_rejects_fewer_than_two_classes(bad):
    with pytest.raises(ValueError, match="num_classes"):
        build_classifier(bad, pretrained=False)


@pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
def test_rejects_invalid_dropout(bad):
    with pytest.raises(ValueError, match="dropout"):
        build_classifier(3, dropout=bad, pretrained=False)


# --- head structure ---


@pytest.mark.parametrize("num_classes", [2, 5, 8, 13])
def test_head_is_dropout_then_linear_to_num_classes(num_classes):
    model = build_classifier(num_classes, pretrained=False)
    dropout, linear = model.classifier
    assert isinstance(dropout, nn.Dropout) and dropout.p == pytest.approx(STOCK_DROPOUT)
    assert isinstance(linear, nn.Linear)
    assert (linear.in_features, linear.out_features) == (BACKBONE_FEATURES, num_classes)


def test_dropout_is_configurable():
    assert build_classifier(3, dropout=0.0, pretrained=False).classifier[0].p == 0.0
    assert build_classifier(3, dropout=0.5, pretrained=False).classifier[0].p == 0.5


@pytest.mark.parametrize("num_classes", [2, 8])
def test_parameter_count_is_stock_minus_old_head_plus_new_head(num_classes):
    summary = summarize(build_classifier(num_classes, pretrained=False))
    expected = (
        STOCK_TOTAL_PARAMETERS
        - _head_parameter_count(IMAGENET_NUM_CLASSES)
        + _head_parameter_count(num_classes)
    )
    assert summary.total_parameters == expected
    assert summary.output_features == num_classes


def test_head_initialisation_is_fan_in_scaled_with_zero_bias():
    set_seed(0)
    linear = build_classifier(8, pretrained=False).classifier[1]
    bound = 1.0 / BACKBONE_FEATURES**0.5
    assert torch.equal(linear.bias, torch.zeros(8))
    assert linear.weight.abs().max() <= bound
    expected_std = bound / 3**0.5
    assert linear.weight.std().item() == pytest.approx(expected_std, rel=0.05)


@pytest.mark.parametrize("num_classes", [2, 4, 8, 13])
def test_initial_loss_is_close_to_uniform_chance_for_any_class_count(batch, num_classes):
    """Regression for Layer 7's finding: out_features-scaled init gave loss ≈ 3·ln(K)."""
    import math

    set_seed(0)
    model = build_classifier(num_classes, pretrained=True).eval()
    with torch.no_grad():
        logits = model(batch)
        loss = F.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    assert logits.std().item() < 0.5
    assert loss.item() == pytest.approx(math.log(num_classes), abs=0.3)


def test_head_initialisation_is_deterministic_under_seed():
    set_seed(11)
    first = build_classifier(5, pretrained=False).classifier[1].weight.clone()
    set_seed(11)
    second = build_classifier(5, pretrained=False).classifier[1].weight.clone()
    set_seed(12)
    third = build_classifier(5, pretrained=False).classifier[1].weight.clone()
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


# --- backbone preservation ---


def test_pretrained_backbone_is_unchanged_by_head_replacement():
    stock = build_efficientnet_b0(pretrained=True)
    classifier = build_classifier(8, pretrained=True)
    for key, value in stock.features.state_dict().items():
        assert torch.equal(value, classifier.features.state_dict()[key]), key


def test_pretrained_false_gives_a_randomly_initialised_backbone():
    stock = build_efficientnet_b0(pretrained=True)
    scratch = build_classifier(8, pretrained=False)
    assert not torch.equal(stock.features[0][0].weight, scratch.features[0][0].weight)


# --- freezing ---


def test_default_is_fully_trainable():
    summary = summarize(build_classifier(8, pretrained=False))
    assert summary.trainable_parameters == summary.total_parameters


def test_freeze_backbone_leaves_only_the_head_trainable():
    model = build_classifier(8, pretrained=False, freeze_backbone=True)
    assert all(not p.requires_grad for p in model.features.parameters())
    assert all(p.requires_grad for p in model.classifier.parameters())
    assert summarize(model).trainable_parameters == _head_parameter_count(8)


def test_set_backbone_trainable_toggles_both_ways():
    model = build_classifier(4, pretrained=False)
    set_backbone_trainable(model, False)
    assert summarize(model).trainable_parameters == _head_parameter_count(4)
    set_backbone_trainable(model, True)
    assert summarize(model).trainable_parameters == summarize(model).total_parameters


def test_frozen_backbone_receives_no_gradients(batch):
    model = build_classifier(8, pretrained=False, freeze_backbone=True).train()
    model(batch).sum().backward()
    assert all(p.grad is None for p in model.features.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.classifier.parameters()
    )


def test_unfrozen_backbone_receives_gradients(batch):
    model = build_classifier(8, pretrained=False).train()
    model(batch).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


# --- forward pass ---


@pytest.mark.parametrize("device", _devices_to_test())
def test_forward_pass_yields_num_classes_logits(batch, device):
    model = build_classifier(8, pretrained=True).to(device).eval()
    with torch.no_grad():
        logits = model(batch.to(device))
    assert logits.shape == (4, 8)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("num_classes", [2, 5, 8, 13])
def test_output_width_tracks_num_classes(batch, num_classes):
    model = build_classifier(num_classes, pretrained=False).eval()
    with torch.no_grad():
        assert model(batch[:1]).shape == (1, num_classes)


def test_eval_mode_is_deterministic(batch):
    model = build_classifier(8, pretrained=True).eval()
    with torch.no_grad():
        assert torch.equal(model(batch), model(batch))


def test_probabilities_are_a_distribution_over_num_classes(batch):
    model = build_classifier(5, pretrained=True).eval()
    with torch.no_grad():
        probabilities = logits_to_probabilities(model(batch))
    assert probabilities.shape == (4, 5)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(4))


# --- learnability ---


def test_one_sgd_step_on_the_head_reduces_loss_on_a_fixed_batch(batch):
    set_seed(0)
    model = build_classifier(8, pretrained=True, freeze_backbone=True).eval()
    targets = torch.tensor([0, 1, 2, 3])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)

    before = F.cross_entropy(model(batch), targets)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        after = F.cross_entropy(model(batch), targets)
    assert after.item() < before.item()


# --- checkpoint contract ---


def test_state_dict_round_trip_between_same_shaped_classifiers(tmp_path):
    set_seed(0)
    source = build_classifier(8, pretrained=False)
    path = tmp_path / "classifier.pt"
    torch.save(source.state_dict(), path)

    set_seed(1)
    target = build_classifier(8, pretrained=False)
    target.load_state_dict(torch.load(path, weights_only=True))
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key]), key


def test_state_dict_from_different_num_classes_is_rejected(tmp_path):
    path = tmp_path / "classifier.pt"
    torch.save(build_classifier(8, pretrained=False).state_dict(), path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        build_classifier(5, pretrained=False).load_state_dict(torch.load(path, weights_only=True))


def test_min_num_classes_constant_is_binary():
    assert MIN_NUM_CLASSES == 2
