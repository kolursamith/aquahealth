"""Layer 8 — metric arithmetic (`src.metrics`), checked against hand-computed values."""

from __future__ import annotations

import pytest
import torch

from src.metrics import (
    classification_report,
    compute_metrics,
    confusion_matrix,
    metrics_from_confusion,
    normalize_confusion_matrix,
)

NAMES = ["a", "b", "c"]
# true:  a a a a  b b b  c c c
# pred:  a a b c  b b a  c c c
TARGETS = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
PREDICTIONS = torch.tensor([0, 0, 1, 2, 1, 1, 0, 2, 2, 2])
EXPECTED_CONFUSION = torch.tensor([[2, 1, 1], [1, 2, 0], [0, 0, 3]])


def test_confusion_matrix_rows_are_true_columns_are_predicted():
    assert torch.equal(confusion_matrix(TARGETS, PREDICTIONS, 3), EXPECTED_CONFUSION)


def test_confusion_matrix_counts_every_sample_once():
    assert confusion_matrix(TARGETS, PREDICTIONS, 3).sum().item() == len(TARGETS)


def test_confusion_matrix_includes_absent_classes():
    matrix = confusion_matrix(torch.tensor([0, 0]), torch.tensor([0, 1]), 4)
    assert matrix.shape == (4, 4)
    assert matrix[0].tolist() == [1, 1, 0, 0]
    assert matrix[2:].sum().item() == 0


@pytest.mark.parametrize(
    "targets,predictions,num_classes",
    [
        (torch.tensor([0, 1]), torch.tensor([0]), 2),
        (torch.tensor([0, 2]), torch.tensor([0, 1]), 2),
        (torch.tensor([0, -1]), torch.tensor([0, 1]), 2),
        (torch.tensor([0]), torch.tensor([0]), 0),
    ],
)
def test_confusion_matrix_rejects_inconsistent_input(targets, predictions, num_classes):
    with pytest.raises(ValueError):
        confusion_matrix(targets, predictions, num_classes)


def test_normalized_confusion_rows_sum_to_one_or_zero():
    normalized = normalize_confusion_matrix(EXPECTED_CONFUSION)
    torch.testing.assert_close(normalized.sum(dim=1), torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(normalized[0], torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64))
    empty_row = normalize_confusion_matrix(torch.tensor([[2, 0], [0, 0]]))
    assert empty_row[1].tolist() == [0.0, 0.0]


def test_per_class_metrics_match_hand_computation():
    metrics = compute_metrics(TARGETS, PREDICTIONS, NAMES)
    a, b, c = metrics.per_class
    # a: tp=2, predicted as a = 3 (2 true + 1 from b), true a = 4
    assert a.precision == pytest.approx(2 / 3) and a.recall == pytest.approx(0.5)
    assert a.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))
    assert a.support == 4
    # b: tp=2, predicted as b = 3, true b = 3
    assert b.precision == pytest.approx(2 / 3) and b.recall == pytest.approx(2 / 3)
    assert b.f1 == pytest.approx(2 / 3) and b.support == 3
    # c: tp=3, predicted as c = 4, true c = 3
    assert c.precision == pytest.approx(0.75) and c.recall == pytest.approx(1.0)
    assert c.f1 == pytest.approx(2 * 0.75 / 1.75) and c.support == 3


def test_aggregate_metrics_match_hand_computation():
    metrics = compute_metrics(TARGETS, PREDICTIONS, NAMES)
    f1_a = 2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5)
    f1_b = 2 / 3
    f1_c = 2 * 0.75 / 1.75
    assert metrics.samples == 10
    assert metrics.accuracy == pytest.approx(7 / 10)
    assert metrics.precision_macro == pytest.approx((2 / 3 + 2 / 3 + 0.75) / 3)
    assert metrics.recall_macro == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)
    assert metrics.f1_macro == pytest.approx((f1_a + f1_b + f1_c) / 3)
    assert metrics.f1_weighted == pytest.approx((4 * f1_a + 3 * f1_b + 3 * f1_c) / 10)


def test_perfect_predictions_score_one_everywhere():
    metrics = compute_metrics(TARGETS, TARGETS, NAMES)
    assert metrics.accuracy == metrics.f1_macro == metrics.f1_weighted == 1.0
    assert all(c.precision == c.recall == c.f1 == 1.0 for c in metrics.per_class)


def test_never_predicted_class_gets_zero_precision_without_error():
    metrics = compute_metrics(torch.tensor([0, 1, 1]), torch.tensor([0, 0, 0]), ["x", "y"])
    x, y = metrics.per_class
    assert x.precision == pytest.approx(1 / 3) and x.recall == 1.0
    assert y.precision == 0.0 and y.recall == 0.0 and y.f1 == 0.0
    assert metrics.f1_macro == pytest.approx(x.f1 / 2)


def test_absent_class_counts_in_macro_average_but_not_weighted():
    metrics = compute_metrics(torch.tensor([0, 0]), torch.tensor([0, 0]), ["x", "y"])
    assert metrics.f1_macro == pytest.approx(0.5)
    assert metrics.f1_weighted == pytest.approx(1.0)


def test_metrics_from_confusion_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        metrics_from_confusion(EXPECTED_CONFUSION, ["a", "b"])


def test_classification_report_lists_every_class_and_averages():
    report = classification_report(compute_metrics(TARGETS, PREDICTIONS, NAMES))
    lines = report.splitlines()
    assert lines[0].split() == ["precision", "recall", "f1-score", "support"]
    assert [line.split()[0] for line in lines[1:4]] == NAMES
    assert lines[1].split()[1:] == ["0.667", "0.500", "0.571", "4"]
    assert "accuracy" in report and "macro avg" in report and "weighted avg" in report
    assert lines[-2].split()[-2:] == ["0.698", "10"]  # macro f1 = (0.571+0.667+0.857)/3
