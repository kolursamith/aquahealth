"""Layer 8 — validation pass (`src.validation`) and its integration into `fit`.

Fixture data only; the numbers prove the machinery, not disease performance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.augmentation import build_train_transform
from src.dataset import ImageFolderDataset, build_dataloader
from src.device import available_backends
from src.model import build_classifier
from src.preprocessing import PreprocessConfig, build_eval_transform
from src.train import (
    BEST_CHECKPOINT_NAME,
    LAST_CHECKPOINT_NAME,
    TrainConfig,
    fit,
    is_improvement,
    load_checkpoint,
)
from src.utils import set_seed
from src.validation import (
    ValidationResult,
    ValidationSummary,
    assert_deterministic_loader,
    contains_random_transform,
    run_validation,
)
from tests.conftest import write_image_folder

CLASSES = ("alpha", "beta", "gamma", "delta")
SMALL = PreprocessConfig(image_size=64, resize_size=72)
CPU = torch.device("cpu")


def _devices_to_test() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


@pytest.fixture(scope="module")
def roots(tmp_path_factory) -> dict[str, Path]:
    base = tmp_path_factory.mktemp("val")
    return {
        "train": write_image_folder(base / "train", CLASSES, per_class=8, size=(80, 60), seed=0),
        "val": write_image_folder(base / "val", CLASSES, per_class=3, size=(80, 60), seed=1),
    }


def _eval_loader(root: Path, class_names=None, batch_size: int = 5):
    dataset = ImageFolderDataset(
        root, transform=build_eval_transform(SMALL), class_names=class_names
    )
    return dataset, build_dataloader(dataset, batch_size=batch_size, num_workers=0)


def _model(seed: int = 0):
    set_seed(seed)
    return build_classifier(len(CLASSES), freeze_backbone=True).eval()


# --- guards ---


def test_random_transform_detection():
    assert contains_random_transform(build_train_transform(SMALL))
    assert not contains_random_transform(build_eval_transform(SMALL))
    assert not contains_random_transform(None)


def test_validation_refuses_training_augmentation(roots):
    dataset = ImageFolderDataset(roots["val"], transform=build_train_transform(SMALL))
    loader = build_dataloader(dataset, batch_size=4, num_workers=0)
    with pytest.raises(ValueError, match="random \\(training\\) transform"):
        assert_deterministic_loader(loader)
    with pytest.raises(ValueError, match="random \\(training\\) transform"):
        run_validation(_model(), loader, CPU, list(CLASSES))


def test_validation_rejects_class_name_count_mismatch(roots):
    _, loader = _eval_loader(roots["val"])
    with pytest.raises(ValueError, match="outputs"):
        run_validation(_model(), loader, CPU, ["only", "two"])


# --- result structure and aggregation ---


def test_validation_result_shapes_and_aggregation(roots):
    dataset, loader = _eval_loader(roots["val"], batch_size=5)
    result = run_validation(_model(), loader, CPU, dataset.class_names)
    n, k = len(dataset), len(CLASSES)
    assert isinstance(result, ValidationResult)
    assert result.targets.tolist() == dataset.targets, "aggregation must preserve loader order"
    assert result.predictions.shape == (n,) and result.probabilities.shape == (n, k)
    torch.testing.assert_close(result.probabilities.sum(dim=1), torch.ones(n))
    assert torch.equal(result.predictions, result.probabilities.argmax(dim=1))
    assert result.confusion.shape == (k, k) and result.confusion.sum().item() == n
    assert result.summary.samples == n == result.metrics.samples
    assert result.class_names == sorted(CLASSES)


def test_summary_is_consistent_with_metrics_and_loss(roots):
    dataset, loader = _eval_loader(roots["val"])
    model = _model()
    result = run_validation(model, loader, CPU, dataset.class_names)
    summary = result.summary
    assert isinstance(summary, ValidationSummary)
    assert summary.accuracy == result.metrics.accuracy
    assert summary.f1_macro == result.metrics.f1_macro
    assert summary.f1_weighted == result.metrics.f1_weighted

    with torch.no_grad():
        images = torch.stack([dataset[i][0] for i in range(len(dataset))])
        expected_loss = torch.nn.functional.cross_entropy(
            model(images), torch.tensor(dataset.targets)
        ).item()
    assert summary.loss == pytest.approx(expected_loss, abs=1e-5)
    assert set(summary.to_dict()) == {
        "loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "samples",
        "seconds",
    }


def test_validation_is_deterministic_and_batch_size_independent(roots):
    model = _model()
    dataset, loader_5 = _eval_loader(roots["val"], batch_size=5)
    _, loader_1 = _eval_loader(roots["val"], batch_size=1)
    a = run_validation(model, loader_5, CPU, dataset.class_names)
    b = run_validation(model, loader_5, CPU, dataset.class_names)
    c = run_validation(model, loader_1, CPU, dataset.class_names)
    assert torch.equal(a.probabilities, b.probabilities)
    torch.testing.assert_close(a.probabilities, c.probabilities, rtol=1e-5, atol=1e-6)
    assert a.summary.loss == b.summary.loss


# --- the model must be untouched ---


def test_validation_changes_no_parameter_or_buffer(roots):
    dataset, loader = _eval_loader(roots["val"])
    model = build_classifier(len(CLASSES), freeze_backbone=False).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    grads_before = [p.grad for p in model.parameters()]

    run_validation(model, loader, CPU, dataset.class_names)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), f"validation modified {key}"
    assert [p.grad for p in model.parameters()] == grads_before
    assert all(p.grad is None for p in model.parameters()), "validation must not create grads"


def test_validation_puts_the_model_in_eval_mode(roots):
    dataset, loader = _eval_loader(roots["val"])
    model = build_classifier(len(CLASSES), freeze_backbone=False).train()
    assert model.training
    run_validation(model, loader, CPU, dataset.class_names)
    assert not model.training and not model.features.training


def test_validation_uses_eval_mode_batchnorm_running_stats(roots):
    """In train mode BatchNorm would use batch statistics; validation must not."""
    model = build_classifier(len(CLASSES), freeze_backbone=False)
    dataset, loader = _eval_loader(roots["val"], batch_size=3)
    running_before = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
    run_validation(model.train(), loader, CPU, dataset.class_names)
    for key, value in running_before.items():
        assert torch.equal(value, model.state_dict()[key]), key


@pytest.mark.parametrize("device", _devices_to_test())
def test_validation_runs_on_every_device_with_cpu_side_outputs(roots, device):
    dataset, loader = _eval_loader(roots["val"])
    model = _model().to(device)
    result = run_validation(model, loader, torch.device(device), dataset.class_names)
    assert result.probabilities.device.type == "cpu"
    assert result.predictions.device.type == "cpu"


# --- integration with fit ---


def test_is_improvement_direction_per_metric():
    assert is_improvement("f1_macro", 0.5, None)
    assert is_improvement("f1_macro", 0.6, 0.5) and not is_improvement("f1_macro", 0.5, 0.5)
    assert is_improvement("loss", 0.4, 0.5) and not is_improvement("loss", 0.5, 0.5)


def test_train_config_rejects_unknown_selection_metric():
    with pytest.raises(ValueError, match="selection_metric"):
        TrainConfig(selection_metric="bleu")


def test_fit_with_validation_records_summaries_and_writes_best(roots, tmp_path):
    train_dataset, train_loader = _eval_loader(roots["train"], batch_size=8)
    val_dataset, val_loader = _eval_loader(roots["val"], train_dataset.class_names)
    model = _model()
    result = fit(
        model,
        train_loader,
        config=TrainConfig(epochs=6, learning_rate=1e-3, seed=0),
        class_names=train_dataset.class_names,
        preprocess=SMALL,
        device=CPU,
        checkpoint_dir=tmp_path / "ck",
        val_loader=val_loader,
    )

    assert all(h.validation is not None for h in result.history)
    assert all(h.validation.samples == len(val_dataset) for h in result.history)
    f1s = [h.validation.f1_macro for h in result.history]
    assert result.best_metric == max(f1s)
    assert result.best_epoch == f1s.index(max(f1s)) + 1
    assert result.best_checkpoint == tmp_path / "ck" / BEST_CHECKPOINT_NAME
    assert result.last_checkpoint == tmp_path / "ck" / LAST_CHECKPOINT_NAME

    best = load_checkpoint(result.best_checkpoint)
    last = load_checkpoint(result.last_checkpoint)
    assert best.epoch == result.best_epoch and last.epoch == 6
    assert best.best_epoch == result.best_epoch and last.best_metric == result.best_metric
    assert last.history[-1].validation == result.history[-1].validation

    best_model = best.build_model()
    rerun = run_validation(best_model, val_loader, CPU, train_dataset.class_names)
    assert rerun.summary.f1_macro == pytest.approx(result.best_metric)


def test_best_checkpoint_is_not_overwritten_by_a_worse_epoch(roots, tmp_path):
    train_dataset, train_loader = _eval_loader(roots["train"], batch_size=8)
    _, val_loader = _eval_loader(roots["val"], train_dataset.class_names)
    result = fit(
        _model(),
        train_loader,
        config=TrainConfig(epochs=4, learning_rate=1e-3, seed=0, selection_metric="loss"),
        class_names=train_dataset.class_names,
        preprocess=SMALL,
        device=CPU,
        checkpoint_dir=tmp_path / "ck",
        val_loader=val_loader,
    )
    losses = [h.validation.loss for h in result.history]
    assert result.best_epoch == losses.index(min(losses)) + 1
    assert load_checkpoint(result.best_checkpoint).epoch == result.best_epoch


def test_fit_with_validation_resumes_with_best_tracking_intact(roots, tmp_path):
    train_dataset, train_loader = _eval_loader(roots["train"], batch_size=8)
    _, val_loader = _eval_loader(roots["val"], train_dataset.class_names)
    kwargs = dict(class_names=train_dataset.class_names, preprocess=SMALL, device=CPU)
    ck = tmp_path / "ck"

    full = fit(
        _model(),
        train_loader,
        config=TrainConfig(epochs=3, learning_rate=1e-3, seed=0),
        val_loader=val_loader,
        **kwargs,
    )
    fit(
        _model(),
        _eval_loader(roots["train"], batch_size=8)[1],
        config=TrainConfig(epochs=1, learning_rate=1e-3, seed=0),
        checkpoint_dir=ck,
        val_loader=val_loader,
        **kwargs,
    )
    checkpoint = load_checkpoint(ck / LAST_CHECKPOINT_NAME)
    resumed = fit(
        checkpoint.build_model(),
        _eval_loader(roots["train"], batch_size=8)[1],
        config=TrainConfig(epochs=3, learning_rate=1e-3, seed=0),
        checkpoint_dir=ck,
        resume_from=checkpoint,
        val_loader=val_loader,
        **kwargs,
    )
    assert [h.validation.loss for h in resumed.history] == [h.validation.loss for h in full.history]
    assert (resumed.best_epoch, resumed.best_metric) == (full.best_epoch, full.best_metric)


def test_fit_without_val_loader_tracks_no_best(roots, tmp_path):
    train_dataset, train_loader = _eval_loader(roots["train"], batch_size=8)
    result = fit(
        _model(),
        train_loader,
        config=TrainConfig(epochs=1, learning_rate=1e-3, seed=0),
        class_names=train_dataset.class_names,
        preprocess=SMALL,
        device=CPU,
        checkpoint_dir=tmp_path / "ck",
    )
    assert result.history[0].validation is None
    assert result.best_epoch is None and result.best_checkpoint is None
    assert not (tmp_path / "ck" / BEST_CHECKPOINT_NAME).exists()


def test_cli_with_val_root_logs_validation_and_writes_best(roots, tmp_path):
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    ck = tmp_path / "cli_ck"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--train-root",
            str(roots["train"]),
            "--val-root",
            str(roots["val"]),
            "--checkpoint-dir",
            str(ck),
            "--epochs",
            "2",
            "--image-size",
            "64",
            "--batch-size",
            "8",
            "--freeze-backbone",
            "--device",
            "cpu",
            "--selection-metric",
            "accuracy",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr
    assert "val loss" in result.stderr and "best accuracy" in result.stderr
    assert (ck / BEST_CHECKPOINT_NAME).exists() and (ck / LAST_CHECKPOINT_NAME).exists()
    assert load_checkpoint(ck / LAST_CHECKPOINT_NAME).train_config.selection_metric == "accuracy"
