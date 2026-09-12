"""Validation pass (Layer 8): eval mode, no gradients, no optimiser, no augmentation.

    val loader (eval transform) ─► model.eval() ─► inference_mode ─►
        logits ─► loss, probabilities, predictions ─► metrics (src/metrics.py)

`run_validation` never touches an optimiser and refuses a loader whose
transform contains random (training) stages, so validation cannot be run
on augmented data by accident. Parameters and buffers are provably unchanged
by a validation pass (tests assert byte equality before/after).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import ClassificationMetrics, compute_metrics, confusion_matrix

RANDOM_TRANSFORM_MARKERS = ("Random", "ColorJitter", "Gaussian", "Elastic", "AutoAugment")


@dataclass(frozen=True)
class ValidationSummary:
    """Scalar metrics only; what gets logged and stored in checkpoint history."""

    loss: float
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    f1_weighted: float
    samples: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    summary: ValidationSummary
    metrics: ClassificationMetrics
    confusion: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor
    class_names: list[str] = field(default_factory=list)


def transform_stage_names(transform: Any) -> list[str]:
    stages = getattr(transform, "transforms", None)
    if stages is None:
        return [type(transform).__name__] if transform is not None else []
    return [type(stage).__name__ for stage in stages]


def contains_random_transform(transform: Any) -> bool:
    return any(
        name.startswith(RANDOM_TRANSFORM_MARKERS) for name in transform_stage_names(transform)
    )


def assert_deterministic_loader(loader: DataLoader) -> None:
    transform = getattr(loader.dataset, "transform", None)
    if contains_random_transform(transform):
        raise ValueError(
            "validation loader uses a random (training) transform: "
            f"{transform_stage_names(transform)}; use build_eval_transform"
        )


def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> ValidationResult:
    assert_deterministic_loader(loader)
    model.eval()
    started = time.perf_counter()

    total_loss = 0.0
    all_targets: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            all_targets.append(labels.cpu())
            all_probabilities.append(torch.softmax(logits.float(), dim=1).cpu())

    if not all_targets:
        raise ValueError("validation loader yielded no samples")

    targets = torch.cat(all_targets)
    probabilities = torch.cat(all_probabilities)
    predictions = probabilities.argmax(dim=1)
    if len(class_names) != probabilities.shape[1]:
        raise ValueError(
            f"model has {probabilities.shape[1]} outputs but {len(class_names)} class names given"
        )

    metrics = compute_metrics(targets, predictions, class_names)
    summary = ValidationSummary(
        loss=total_loss / metrics.samples,
        accuracy=metrics.accuracy,
        precision_macro=metrics.precision_macro,
        recall_macro=metrics.recall_macro,
        f1_macro=metrics.f1_macro,
        f1_weighted=metrics.f1_weighted,
        samples=metrics.samples,
        seconds=time.perf_counter() - started,
    )
    return ValidationResult(
        summary=summary,
        metrics=metrics,
        confusion=confusion_matrix(targets, predictions, len(class_names)),
        targets=targets,
        predictions=predictions,
        probabilities=probabilities,
        class_names=list(class_names),
    )
