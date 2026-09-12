"""Checkpoint-driven evaluation (Layer 10), independent of training.

    checkpoint ─► Checkpoint.build_model() + its own PreprocessConfig
    image folder ─► ImageFolderDataset(eval transform, checkpoint's class list)
                 ─► run_validation (Layer 8: eval mode, no grad, deterministic)
                 ─► metrics (Layer 8) + per-sample predictions + confidence analysis
                 ─► export: metrics.json, predictions.csv, misclassified.csv,
                            confusion_matrix.csv, confusion_matrix_normalized.csv,
                            classification_report.txt

No metric is computed here that is not computed by src/metrics.py, and no
inference path exists here that is not src/validation.py; this module adds
per-sample bookkeeping, confidence analysis and exports.

The held-out test split must be evaluated once, at the end, and never used
to pick a checkpoint. Evaluating a directory named `test` logs a warning to
that effect.

Runnable as `python -m src.evaluate --checkpoint … --data-root … --out-dir …`.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.models import EfficientNet

from src.config import BATCH_SIZE
from src.dataset import ImageFolderDataset, build_dataloader
from src.device import resolve_device
from src.metrics import ClassificationMetrics, classification_report, normalize_confusion_matrix
from src.preprocessing import build_eval_transform
from src.train import Checkpoint, load_checkpoint
from src.utils import get_logger
from src.validation import ValidationSummary, run_validation

logger = get_logger(__name__)

DEFAULT_CONFIDENCE_BINS = 10
HELD_OUT_SPLIT_NAMES = ("test",)


@dataclass(frozen=True)
class SamplePrediction:
    path: str
    target: str
    predicted: str
    confidence: float
    correct: bool
    probabilities: list[float]


@dataclass(frozen=True)
class ConfidenceBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    mean_confidence: float


@dataclass(frozen=True)
class ConfidenceAnalysis:
    mean_confidence: float
    mean_confidence_correct: float | None
    mean_confidence_incorrect: float | None
    expected_calibration_error: float
    bins: list[ConfidenceBin] = field(default_factory=list)


@dataclass
class EvaluationReport:
    checkpoint: str
    epoch: int
    stage: str | None
    class_names: list[str]
    preprocess: dict[str, Any]
    data_root: str
    summary: ValidationSummary
    metrics: ClassificationMetrics
    confusion: list[list[int]]
    confusion_normalized: list[list[float]]
    confidence: ConfidenceAnalysis
    samples: list[SamplePrediction]

    @property
    def misclassified(self) -> list[SamplePrediction]:
        return [s for s in self.samples if not s.correct]

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        payload = {
            "checkpoint": self.checkpoint,
            "epoch": self.epoch,
            "stage": self.stage,
            "class_names": self.class_names,
            "preprocess": self.preprocess,
            "data_root": self.data_root,
            "summary": self.summary.to_dict(),
            "metrics": asdict(self.metrics),
            "confusion": self.confusion,
            "confusion_normalized": self.confusion_normalized,
            "confidence": asdict(self.confidence),
            "misclassified_count": len(self.misclassified),
        }
        if include_samples:
            payload["samples"] = [asdict(s) for s in self.samples]
        return payload


# --- confidence -------------------------------------------------------------------


def confidence_analysis(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = DEFAULT_CONFIDENCE_BINS,
) -> ConfidenceAnalysis:
    """Top-1 confidence statistics and an equal-width reliability diagram with ECE.

    ECE = Σ_b (n_b / N) · |accuracy_b − mean_confidence_b| over non-empty bins.
    """
    if num_bins < 1:
        raise ValueError("num_bins must be >= 1")
    if probabilities.ndim != 2 or probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probabilities must be (N, K) and targets (N,)")
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted == targets
    n = confidence.shape[0]

    edges = torch.linspace(0.0, 1.0, num_bins + 1)
    bins: list[ConfidenceBin] = []
    ece = 0.0
    for index in range(num_bins):
        lower, upper = edges[index].item(), edges[index + 1].item()
        in_bin = (confidence > lower) & (confidence <= upper)
        if index == 0:
            in_bin |= confidence == 0.0
        count = int(in_bin.sum().item())
        if count:
            accuracy = correct[in_bin].float().mean().item()
            mean_conf = confidence[in_bin].mean().item()
            ece += (count / n) * abs(accuracy - mean_conf)
        else:
            accuracy, mean_conf = 0.0, 0.0
        bins.append(ConfidenceBin(lower, upper, count, accuracy, mean_conf))

    def _mean(mask: torch.Tensor) -> float | None:
        return confidence[mask].mean().item() if bool(mask.any()) else None

    return ConfidenceAnalysis(
        mean_confidence=confidence.mean().item(),
        mean_confidence_correct=_mean(correct),
        mean_confidence_incorrect=_mean(~correct),
        expected_calibration_error=ece,
        bins=bins,
    )


# --- evaluation ------------------------------------------------------------------------


def _assert_sequential(loader: DataLoader) -> None:
    if not isinstance(loader.sampler, SequentialSampler):
        raise ValueError("evaluation loader must be unshuffled so predictions align with files")


def evaluate_model(
    model: EfficientNet,
    dataset: ImageFolderDataset,
    loader: DataLoader,
    device: torch.device,
    *,
    checkpoint: Checkpoint,
    num_bins: int = DEFAULT_CONFIDENCE_BINS,
) -> EvaluationReport:
    _assert_sequential(loader)
    if dataset.class_names != checkpoint.class_names:
        raise ValueError(
            f"dataset classes {dataset.class_names} differ from checkpoint classes "
            f"{checkpoint.class_names}"
        )
    result = run_validation(model, loader, device, checkpoint.class_names)
    if result.targets.shape[0] != len(dataset):
        raise ValueError("validation produced a different number of samples than the dataset")

    names = checkpoint.class_names
    samples = [
        SamplePrediction(
            path=str(sample.path),
            target=names[int(target)],
            predicted=names[int(pred)],
            confidence=float(probs.max()),
            correct=bool(target == pred),
            probabilities=[float(p) for p in probs],
        )
        for sample, target, pred, probs in zip(
            dataset.samples, result.targets, result.predictions, result.probabilities, strict=True
        )
    ]

    return EvaluationReport(
        checkpoint=str(checkpoint.path),
        epoch=checkpoint.epoch,
        stage=checkpoint.stage,
        class_names=list(names),
        preprocess=asdict(checkpoint.preprocess),
        data_root=str(dataset.root),
        summary=result.summary,
        metrics=result.metrics,
        confusion=result.confusion.tolist(),
        confusion_normalized=normalize_confusion_matrix(result.confusion).tolist(),
        confidence=confidence_analysis(result.probabilities, result.targets, num_bins),
        samples=samples,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    *,
    device: torch.device | None = None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    num_bins: int = DEFAULT_CONFIDENCE_BINS,
) -> EvaluationReport:
    """Evaluate a saved checkpoint on an image folder using the checkpoint's own preprocessing."""
    data_root = Path(data_root)
    if data_root.name in HELD_OUT_SPLIT_NAMES:
        logger.warning(
            "evaluating %s: a held-out test split must be evaluated once, at the end, "
            "and never used to choose a checkpoint",
            data_root,
        )
    checkpoint = load_checkpoint(checkpoint_path)
    device = device or resolve_device()
    model = checkpoint.build_model().to(device).eval()
    dataset = ImageFolderDataset(
        data_root,
        transform=build_eval_transform(checkpoint.preprocess),
        class_names=checkpoint.class_names,
    )
    loader = build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
    return evaluate_model(model, dataset, loader, device, checkpoint=checkpoint, num_bins=num_bins)


# --- export --------------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def export_report(report: EvaluationReport, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = report.class_names
    written: dict[str, Path] = {}

    written["metrics"] = out_dir / "metrics.json"
    written["metrics"].write_text(json.dumps(report.to_dict(), indent=2))

    prob_columns = [f"p_{name}" for name in names]
    header = ["path", "target", "predicted", "confidence", "correct", *prob_columns]

    def to_row(s: SamplePrediction) -> list[Any]:
        return [s.path, s.target, s.predicted, s.confidence, s.correct, *s.probabilities]

    written["predictions"] = out_dir / "predictions.csv"
    _write_csv(written["predictions"], header, [to_row(s) for s in report.samples])
    written["misclassified"] = out_dir / "misclassified.csv"
    _write_csv(written["misclassified"], header, [to_row(s) for s in report.misclassified])

    written["confusion"] = out_dir / "confusion_matrix.csv"
    _write_csv(
        written["confusion"],
        ["true\\predicted", *names],
        [[name, *row] for name, row in zip(names, report.confusion)],
    )
    written["confusion_normalized"] = out_dir / "confusion_matrix_normalized.csv"
    _write_csv(
        written["confusion_normalized"],
        ["true\\predicted", *names],
        [[name, *row] for name, row in zip(names, report.confusion_normalized)],
    )

    written["classification_report"] = out_dir / "classification_report.txt"
    written["classification_report"].write_text(classification_report(report.metrics) + "\n")
    return written


# --- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on an image folder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bins", type=int, default=DEFAULT_CONFIDENCE_BINS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    report = evaluate_checkpoint(
        args.checkpoint,
        args.data_root,
        device=resolve_device(args.device),
        batch_size=args.batch_size,
        num_workers=args.workers,
        num_bins=args.bins,
    )
    written = export_report(report, args.out_dir)
    logger.info(
        "evaluated %d samples from %s with %s (epoch %d%s)",
        report.summary.samples,
        report.data_root,
        report.checkpoint,
        report.epoch,
        f", stage {report.stage}" if report.stage else "",
    )
    logger.info(
        "accuracy %.4f  f1_macro %.4f  f1_weighted %.4f  loss %.4f  ECE %.4f  misclassified %d",
        report.summary.accuracy,
        report.summary.f1_macro,
        report.summary.f1_weighted,
        report.summary.loss,
        report.confidence.expected_calibration_error,
        len(report.misclassified),
    )
    print(classification_report(report.metrics))
    for name, path in written.items():
        logger.info("wrote %s -> %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
