"""Layer 10 — checkpoint-driven evaluation (`src.evaluate`). Fixture data only."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.dataset import ImageFolderDataset, build_dataloader
from src.device import available_backends
from src.evaluate import (
    EvaluationReport,
    confidence_analysis,
    evaluate_checkpoint,
    evaluate_model,
    export_report,
)
from src.model import build_classifier
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform
from src.train import LAST_CHECKPOINT_NAME, TrainConfig, fit, load_checkpoint
from src.utils import set_seed
from tests.conftest import write_image_folder

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("alpha", "beta", "gamma", "delta")
SMALL = PreprocessConfig(image_size=64, resize_size=72, clahe=CLAHEConfig(clip_limit=3.0))
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def roots(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("eval")
    return {
        "train": write_image_folder(base / "train", CLASSES, per_class=6, size=(80, 60), seed=0),
        "val": write_image_folder(base / "val", CLASSES, per_class=3, size=(80, 60), seed=1),
        "test": write_image_folder(base / "test", CLASSES, per_class=2, size=(80, 60), seed=2),
    }


@pytest.fixture(scope="module")
def checkpoint_path(roots, tmp_path_factory) -> Path:
    """A real checkpoint produced by the Layer 7 engine (few epochs; plumbing only)."""
    dataset = ImageFolderDataset(roots["train"], transform=build_eval_transform(SMALL))
    loader = build_dataloader(dataset, batch_size=8, shuffle=True, seed=0, num_workers=0)
    set_seed(0)
    model = build_classifier(len(CLASSES), freeze_backbone=True)
    ck_dir = tmp_path_factory.mktemp("ck")
    fit(
        model,
        loader,
        config=TrainConfig(epochs=6, learning_rate=1e-3, seed=0),
        class_names=dataset.class_names,
        preprocess=SMALL,
        device=CPU,
        checkpoint_dir=ck_dir,
        stage="head",
    )
    return ck_dir / LAST_CHECKPOINT_NAME


# --- confidence analysis ---


def test_confidence_analysis_matches_hand_computation():
    probabilities = torch.tensor(
        [
            [0.9, 0.1],  # correct, conf 0.9  -> bin (0.8, 0.9]
            [0.6, 0.4],  # wrong,   conf 0.6  -> bin (0.5, 0.6]
            [0.3, 0.7],  # correct, conf 0.7  -> bin (0.6, 0.7]
            [0.55, 0.45],  # correct, conf 0.55 -> bin (0.5, 0.6]
        ]
    )
    targets = torch.tensor([0, 1, 1, 0])
    analysis = confidence_analysis(probabilities, targets, num_bins=10)

    assert analysis.mean_confidence == pytest.approx((0.9 + 0.6 + 0.7 + 0.55) / 4)
    assert analysis.mean_confidence_correct == pytest.approx((0.9 + 0.7 + 0.55) / 3)
    assert analysis.mean_confidence_incorrect == pytest.approx(0.6)
    assert len(analysis.bins) == 10
    counts = [b.count for b in analysis.bins]
    assert counts == [0, 0, 0, 0, 0, 2, 1, 0, 1, 0]
    bin_5 = analysis.bins[5]
    assert bin_5.accuracy == pytest.approx(0.5) and bin_5.mean_confidence == pytest.approx(0.575)
    expected_ece = (2 / 4) * abs(0.5 - 0.575) + (1 / 4) * abs(1.0 - 0.7) + (1 / 4) * abs(1.0 - 0.9)
    assert analysis.expected_calibration_error == pytest.approx(expected_ece)


def test_confidence_analysis_handles_all_correct_and_all_wrong():
    probabilities = torch.tensor([[0.8, 0.2], [0.75, 0.25]])
    all_correct = confidence_analysis(probabilities, torch.tensor([0, 0]))
    all_wrong = confidence_analysis(probabilities, torch.tensor([1, 1]))
    assert all_correct.mean_confidence_incorrect is None
    assert all_wrong.mean_confidence_correct is None
    assert all_wrong.expected_calibration_error == pytest.approx(0.775)


def test_confidence_analysis_bins_cover_zero_and_one():
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    analysis = confidence_analysis(probabilities, torch.tensor([0, 1]), num_bins=4)
    assert sum(b.count for b in analysis.bins) == 2
    assert analysis.bins[-1].count == 2 and analysis.bins[-1].upper == 1.0


@pytest.mark.parametrize("bad", [0, -3])
def test_confidence_analysis_rejects_bad_bin_count(bad):
    with pytest.raises(ValueError, match="num_bins"):
        confidence_analysis(torch.tensor([[0.5, 0.5]]), torch.tensor([0]), num_bins=bad)


def test_confidence_analysis_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        confidence_analysis(torch.tensor([[0.5, 0.5]]), torch.tensor([0, 1]))


# --- evaluate_checkpoint ---


def test_evaluate_checkpoint_rebuilds_model_and_preprocessing_from_the_file(roots, checkpoint_path):
    report = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU, batch_size=5)
    assert isinstance(report, EvaluationReport)
    assert report.checkpoint == str(checkpoint_path) and report.epoch == 6
    assert report.stage == "head"
    assert report.class_names == sorted(CLASSES)
    assert report.preprocess["image_size"] == 64 and report.preprocess["clahe"] == {
        "clip_limit": 3.0,
        "tile_grid_size": 8,
    }
    assert report.summary.samples == 12 == len(report.samples)
    assert len(report.confusion) == 4 and sum(map(sum, report.confusion)) == 12
    assert all(abs(sum(row) - 1.0) < 1e-9 for row in report.confusion_normalized)


def test_samples_align_with_files_and_labels(roots, checkpoint_path):
    report = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU, batch_size=5)
    dataset = ImageFolderDataset(roots["val"])
    for sample, expected in zip(report.samples, dataset.samples, strict=True):
        assert sample.path == str(expected.path)
        assert sample.target == dataset.class_names[expected.label]
        assert sample.target == Path(sample.path).parent.name
        assert (
            sample.predicted
            == report.class_names[max(range(4), key=sample.probabilities.__getitem__)]
        )
        assert sample.confidence == pytest.approx(max(sample.probabilities))
        assert sample.correct == (sample.target == sample.predicted)
        assert sum(sample.probabilities) == pytest.approx(1.0, abs=1e-5)


def test_misclassified_are_exactly_the_incorrect_samples(roots, checkpoint_path):
    report = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU)
    wrong = [s for s in report.samples if s.target != s.predicted]
    assert report.misclassified == wrong
    assert len(wrong) == 12 - sum(report.confusion[i][i] for i in range(4))
    assert report.to_dict()["misclassified_count"] == len(wrong)


def test_evaluation_is_deterministic_and_batch_size_independent(roots, checkpoint_path):
    a = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU, batch_size=5)
    b = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU, batch_size=5)
    c = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU, batch_size=1)
    assert [s.probabilities for s in a.samples] == [s.probabilities for s in b.samples]
    assert a.confusion == b.confusion == c.confusion
    for x, y in zip(a.samples, c.samples):
        assert x.probabilities == pytest.approx(y.probabilities, abs=1e-5)


def test_metrics_agree_with_the_validation_pass_on_the_same_model(roots, checkpoint_path):
    from src.validation import run_validation

    checkpoint = load_checkpoint(checkpoint_path)
    model = checkpoint.build_model().eval()
    dataset = ImageFolderDataset(
        roots["val"], transform=build_eval_transform(SMALL), class_names=checkpoint.class_names
    )
    loader = build_dataloader(dataset, batch_size=6, num_workers=0)
    validation = run_validation(model, loader, CPU, checkpoint.class_names)
    report = evaluate_model(model, dataset, loader, CPU, checkpoint=checkpoint)
    assert report.summary == validation.summary or report.summary.to_dict() == {
        **validation.summary.to_dict(),
        "seconds": report.summary.seconds,
    }
    assert report.metrics == validation.metrics


def test_evaluation_leaves_the_model_untouched(roots, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    model = checkpoint.build_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    dataset = ImageFolderDataset(
        roots["val"], transform=build_eval_transform(SMALL), class_names=checkpoint.class_names
    )
    evaluate_model(
        model,
        dataset,
        build_dataloader(dataset, batch_size=4, num_workers=0),
        CPU,
        checkpoint=checkpoint,
    )
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("device", [n for n, ok in available_backends().items() if ok])
def test_evaluation_runs_on_every_device(roots, checkpoint_path, device):
    report = evaluate_checkpoint(checkpoint_path, roots["val"], device=torch.device(device))
    assert report.summary.samples == 12


# --- guards ---


def test_shuffled_loader_is_rejected(roots, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    dataset = ImageFolderDataset(
        roots["val"], transform=build_eval_transform(SMALL), class_names=checkpoint.class_names
    )
    loader = build_dataloader(dataset, batch_size=4, shuffle=True, seed=0, num_workers=0)
    with pytest.raises(ValueError, match="unshuffled"):
        evaluate_model(checkpoint.build_model(), dataset, loader, CPU, checkpoint=checkpoint)


def test_class_mismatch_between_data_and_checkpoint_is_rejected(roots, checkpoint_path, tmp_path):
    other = write_image_folder(tmp_path / "other", ("alpha", "zeta"), per_class=1)
    with pytest.raises(ValueError, match="unexpected class directories"):
        evaluate_checkpoint(checkpoint_path, other, device=CPU)


def test_evaluating_a_test_split_logs_a_one_shot_warning(roots, checkpoint_path, caplog):
    with caplog.at_level("WARNING"):
        evaluate_checkpoint(checkpoint_path, roots["test"], device=CPU)
    assert any("held-out test split" in message for message in caplog.messages)
    with caplog.at_level("WARNING"):
        caplog.clear()
        evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU)
    assert not any("held-out" in message for message in caplog.messages)


# --- export ---


def test_export_writes_every_artifact_consistently(roots, checkpoint_path, tmp_path):
    report = evaluate_checkpoint(checkpoint_path, roots["val"], device=CPU)
    written = export_report(report, tmp_path / "out")
    assert set(written) == {
        "metrics",
        "predictions",
        "misclassified",
        "confusion",
        "confusion_normalized",
        "classification_report",
    }
    assert all(path.exists() for path in written.values())

    metrics = json.loads(written["metrics"].read_text())
    assert metrics["summary"]["accuracy"] == report.summary.accuracy
    assert metrics["confusion"] == report.confusion
    assert metrics["confidence"]["expected_calibration_error"] == pytest.approx(
        report.confidence.expected_calibration_error
    )
    assert "samples" not in metrics

    with written["predictions"].open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert rows[0]["path"] == report.samples[0].path
    assert set(rows[0]) >= {"path", "target", "predicted", "confidence", "correct"} | {
        f"p_{name}" for name in report.class_names
    }
    assert float(rows[0]["p_alpha"]) == pytest.approx(report.samples[0].probabilities[0])

    with written["misclassified"].open() as handle:
        wrong_rows = list(csv.DictReader(handle))
    assert len(wrong_rows) == len(report.misclassified)
    assert all(row["correct"] == "False" for row in wrong_rows)

    with written["confusion"].open() as handle:
        confusion_rows = list(csv.reader(handle))
    assert confusion_rows[0] == ["true\\predicted", *report.class_names]
    assert [int(v) for v in confusion_rows[1][1:]] == report.confusion[0]

    text = written["classification_report"].read_text()
    assert "macro avg" in text and all(name in text for name in report.class_names)


def test_cli_evaluates_and_exports(roots, checkpoint_path, tmp_path):
    out = tmp_path / "cli_out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
            "--checkpoint",
            str(checkpoint_path),
            "--data-root",
            str(roots["val"]),
            "--out-dir",
            str(out),
            "--device",
            "cpu",
            "--batch-size",
            "6",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "evaluated 12 samples" in result.stderr
    assert "f1_macro" in result.stderr and "ECE" in result.stderr
    assert "macro avg" in result.stdout
    assert (out / "metrics.json").exists() and (out / "predictions.csv").exists()
