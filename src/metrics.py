"""Classification metrics computed from labels and predictions (Layer 8).

Everything derives from one confusion matrix `C[true, pred]` over a fixed,
known class list, so the numbers are exactly reproducible and independent of
which classes happen to appear in a particular batch. Averages are taken over
all K classes with zero-division → 0, which matches
`sklearn.metrics.*(labels=range(K), zero_division=0)`.

Used by validation (Layer 8) and evaluation (Layer 10); no other module
computes metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ClassMetrics:
    name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    samples: int
    per_class: list[ClassMetrics] = field(default_factory=list)


def confusion_matrix(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """K×K int64 matrix with rows = true class, columns = predicted class."""
    targets = torch.as_tensor(targets, dtype=torch.long).flatten()
    predictions = torch.as_tensor(predictions, dtype=torch.long).flatten()
    if targets.shape != predictions.shape:
        raise ValueError(
            f"targets {tuple(targets.shape)} and predictions {tuple(predictions.shape)} differ"
        )
    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")
    for name, values in (("targets", targets), ("predictions", predictions)):
        if values.numel() and (values.min() < 0 or values.max() >= num_classes):
            raise ValueError(f"{name} contain labels outside [0, {num_classes})")
    flat = targets * num_classes + predictions
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def normalize_confusion_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Row-normalised (per true class); rows with no samples become all zeros."""
    matrix = matrix.to(torch.float64)
    row_sums = matrix.sum(dim=1, keepdim=True)
    return torch.where(row_sums > 0, matrix / row_sums.clamp(min=1), torch.zeros_like(matrix))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def metrics_from_confusion(matrix: torch.Tensor, class_names: list[str]) -> ClassificationMetrics:
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError(
            f"confusion matrix {tuple(matrix.shape)} does not match {len(class_names)} classes"
        )
    matrix = matrix.to(torch.int64)
    total = int(matrix.sum().item())
    diagonal = matrix.diagonal()
    true_per_class = matrix.sum(dim=1)
    predicted_per_class = matrix.sum(dim=0)

    per_class: list[ClassMetrics] = []
    for index, name in enumerate(class_names):
        tp = int(diagonal[index].item())
        precision = _safe_div(tp, int(predicted_per_class[index].item()))
        recall = _safe_div(tp, int(true_per_class[index].item()))
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            ClassMetrics(
                name=name,
                precision=precision,
                recall=recall,
                f1=f1,
                support=int(true_per_class[index].item()),
            )
        )

    k = len(class_names)
    return ClassificationMetrics(
        accuracy=_safe_div(int(diagonal.sum().item()), total),
        precision_macro=sum(c.precision for c in per_class) / k,
        recall_macro=sum(c.recall for c in per_class) / k,
        f1_macro=sum(c.f1 for c in per_class) / k,
        f1_weighted=_safe_div(sum(c.f1 * c.support for c in per_class), total),
        samples=total,
        per_class=per_class,
    )


def compute_metrics(
    targets: torch.Tensor, predictions: torch.Tensor, class_names: list[str]
) -> ClassificationMetrics:
    return metrics_from_confusion(
        confusion_matrix(targets, predictions, len(class_names)), class_names
    )


def classification_report(metrics: ClassificationMetrics) -> str:
    width = max(len("weighted avg"), *(len(c.name) for c in metrics.per_class))
    header = f"{'':<{width}}  precision  recall  f1-score  support"
    lines = [header]
    for c in metrics.per_class:
        lines.append(
            f"{c.name:<{width}}  {c.precision:9.3f}  {c.recall:6.3f}  {c.f1:8.3f}  {c.support:7d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':<{width}}  {'':9}  {'':6}  {metrics.accuracy:8.3f}  {metrics.samples:7d}"
    )
    lines.append(
        f"{'macro avg':<{width}}  {metrics.precision_macro:9.3f}  {metrics.recall_macro:6.3f}  "
        f"{metrics.f1_macro:8.3f}  {metrics.samples:7d}"
    )
    weighted_precision = _safe_div(
        sum(c.precision * c.support for c in metrics.per_class), metrics.samples
    )
    weighted_recall = _safe_div(
        sum(c.recall * c.support for c in metrics.per_class), metrics.samples
    )
    lines.append(
        f"{'weighted avg':<{width}}  {weighted_precision:9.3f}  {weighted_recall:6.3f}  "
        f"{metrics.f1_weighted:8.3f}  {metrics.samples:7d}"
    )
    return "\n".join(lines)
