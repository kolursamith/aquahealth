"""Layer 7 — training loop and checkpointing (`src.train`).

Training here is on the synthetic colour-per-class fixture, which is
linearly separable in pretrained-feature space. Reaching 100 % on it proves
the plumbing (forward → loss → backward → step → checkpoint → resume); it is
not a disease-classification result of any kind.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.dataset import ImageFolderDataset, build_dataloader
from src.device import available_backends
from src.model import build_classifier, summarize
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform
from src.train import (
    CHECKPOINT_FORMAT_VERSION,
    LAST_CHECKPOINT_NAME,
    EpochStats,
    TrainConfig,
    build_optimizer,
    fit,
    is_backbone_frozen,
    load_checkpoint,
    model_spec,
    save_checkpoint,
    set_train_mode,
    train_one_epoch,
)
from src.utils import set_seed
from tests.conftest import write_image_folder

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("alpha", "beta", "gamma", "delta")
SMALL = PreprocessConfig(image_size=64, resize_size=72)
CPU = torch.device("cpu")


def _devices_to_test() -> list[str]:
    return [name for name, ok in available_backends().items() if ok]


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory) -> Path:
    return write_image_folder(
        tmp_path_factory.mktemp("train") / "images", CLASSES, per_class=8, size=(80, 60)
    )


def _loader(root: Path, seed: int = 0, batch_size: int = 8):
    dataset = ImageFolderDataset(root, transform=build_eval_transform(SMALL))
    return dataset, build_dataloader(
        dataset, batch_size=batch_size, shuffle=True, seed=seed, num_workers=0
    )


def _model(seed: int = 0, freeze: bool = True):
    set_seed(seed)
    return build_classifier(len(CLASSES), freeze_backbone=freeze)


def _fit(
    root, epochs, *, device=CPU, seed=0, lr=1e-3, checkpoint_dir=None, resume=None, model=None
):
    dataset, loader = _loader(root, seed=seed)
    model = model if model is not None else _model(seed)
    return fit(
        model,
        loader,
        config=TrainConfig(epochs=epochs, learning_rate=lr, seed=seed),
        class_names=dataset.class_names,
        preprocess=SMALL,
        device=torch.device(device),
        checkpoint_dir=checkpoint_dir,
        resume_from=resume,
    )


# --- config ---


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"learning_rate": 0}, {"weight_decay": -1}])
def test_train_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_train_config_defaults_come_from_config_module():
    from src.config import LEARNING_RATE, NUM_EPOCHS, SEED, WEIGHT_DECAY

    config = TrainConfig()
    assert (config.epochs, config.learning_rate, config.weight_decay, config.seed) == (
        NUM_EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        SEED,
    )


# --- modes and optimiser ---


def test_set_train_mode_keeps_frozen_backbone_in_eval():
    model = build_classifier(4, pretrained=False, freeze_backbone=True)
    set_train_mode(model)
    assert model.training and model.classifier.training
    assert all(not block.training for block in model.features)
    assert is_backbone_frozen(model)


def test_set_train_mode_trains_everything_when_unfrozen():
    model = build_classifier(4, pretrained=False)
    set_train_mode(model)
    assert all(block.training for block in model.features) and model.classifier.training
    assert not is_backbone_frozen(model)


def test_optimizer_only_holds_trainable_parameters():
    model = build_classifier(4, pretrained=False, freeze_backbone=True)
    optimizer = build_optimizer(model, TrainConfig(learning_rate=0.01, weight_decay=0.5))
    held = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
    assert held == summarize(model).trainable_parameters
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.5


def test_optimizer_rejects_fully_frozen_model():
    model = build_classifier(4, pretrained=False, freeze_backbone=True)
    for p in model.classifier.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        build_optimizer(model, TrainConfig())


# --- one epoch ---


def test_train_one_epoch_returns_stats_and_updates_only_the_head(fixture_root):
    _, loader = _loader(fixture_root)
    model = _model()
    optimizer = build_optimizer(model, TrainConfig(learning_rate=1e-3))
    backbone_before = {k: v.clone() for k, v in model.features.state_dict().items()}
    head_before = model.classifier[1].weight.clone()

    stats = train_one_epoch(model, loader, optimizer, CPU, epoch=1)

    assert isinstance(stats, EpochStats)
    assert stats.epoch == 1 and stats.samples == len(CLASSES) * 8
    assert stats.loss > 0 and 0.0 <= stats.train_accuracy <= 1.0 and stats.seconds > 0
    assert not torch.equal(head_before, model.classifier[1].weight)
    for key, value in model.features.state_dict().items():
        assert torch.equal(value, backbone_before[key]), f"frozen backbone changed: {key}"


def test_frozen_backbone_batchnorm_statistics_do_not_drift(fixture_root):
    _, loader = _loader(fixture_root)
    model = _model()
    running = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k}
    assert running, "expected BatchNorm running statistics in the backbone"
    train_one_epoch(model, loader, build_optimizer(model, TrainConfig()), CPU, epoch=1)
    for key, value in running.items():
        assert torch.equal(value, model.state_dict()[key]), key


def test_unfrozen_backbone_does_update(fixture_root):
    _, loader = _loader(fixture_root)
    model = _model(freeze=False)
    first_conv = model.features[0][0].weight.clone()
    train_one_epoch(
        model, loader, build_optimizer(model, TrainConfig(learning_rate=1e-3)), CPU, epoch=1
    )
    assert not torch.equal(first_conv, model.features[0][0].weight)


# --- tiny-fixture training ---


def test_loss_falls_and_fixture_is_fitted(fixture_root):
    model = _model()
    result = _fit(fixture_root, epochs=10, model=model)
    losses = [h.loss for h in result.history]
    assert losses[0] == pytest.approx(1.386, abs=0.3), "initial loss should be near ln(4)"
    assert losses[-1] < losses[0] * 0.75
    assert result.final.train_accuracy >= 0.9
    assert [h.epoch for h in result.history] == list(range(1, 11))

    dataset, _ = _loader(fixture_root)
    model.eval()
    with torch.no_grad():
        images = torch.stack([dataset[i][0] for i in range(len(dataset))])
        predictions = model(images).argmax(dim=1).tolist()
    assert predictions == dataset.targets, "eval-mode predictions should fit the fixture exactly"


def test_fit_returns_the_model_in_eval_mode(fixture_root):
    model = _model()
    _fit(fixture_root, epochs=1, model=model)
    assert not model.training
    assert not model.classifier.training and not model.features.training


def test_epoch_train_accuracy_is_a_train_mode_estimate_not_an_evaluation(fixture_root):
    """Regression: an earlier test judged 'fitted' by train-mode accuracy (dropout on).

    The two measurements are computed differently and must not be conflated:
    the epoch statistic is accumulated mid-update with dropout active, while
    a real evaluation is a separate eval-mode pass over fixed weights.
    """
    model = _model()
    result = _fit(fixture_root, epochs=4, model=model)
    dataset, _ = _loader(fixture_root)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    targets = torch.tensor(dataset.targets)

    with torch.no_grad():
        model.eval()
        eval_predictions = model(images).argmax(dim=1)
        set_train_mode(model)  # bare model.train() would let frozen BatchNorm stats drift
        set_seed(0)
        train_mode_predictions = model(images).argmax(dim=1)
        model.eval()

    assert torch.equal(eval_predictions, model(images).argmax(dim=1)), "eval mode is deterministic"
    assert (
        not torch.equal(eval_predictions, train_mode_predictions)
        or result.final.train_accuracy < 1.0
    )
    assert (eval_predictions == targets).float().mean().item() >= result.final.train_accuracy - 0.2


def test_each_batch_gets_exactly_one_zero_grad_and_one_step(fixture_root):
    _, loader = _loader(fixture_root, batch_size=8)
    model = _model()

    class Counting(torch.optim.AdamW):
        zero_grad_calls = 0
        step_calls = 0

        def zero_grad(self, set_to_none: bool = True) -> None:
            type(self).zero_grad_calls += 1
            super().zero_grad(set_to_none=set_to_none)

        def step(self, closure=None):
            type(self).step_calls += 1
            return super().step(closure)

    optimizer = Counting([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    train_one_epoch(model, loader, optimizer, CPU, epoch=1)
    assert Counting.zero_grad_calls == Counting.step_calls == len(loader) == 4
    assert all(
        p.grad is not None for p in model.classifier.parameters()
    ), "last step's grads remain"


@pytest.mark.parametrize("device", _devices_to_test())
def test_training_runs_on_every_available_device(fixture_root, device):
    result = _fit(fixture_root, epochs=2, device=device)
    assert len(result.history) == 2
    assert all(torch.isfinite(torch.tensor(h.loss)) for h in result.history)


def test_seed_makes_the_loss_curve_reproducible(fixture_root):
    first = _fit(fixture_root, epochs=3, seed=5)
    second = _fit(fixture_root, epochs=3, seed=5)
    third = _fit(fixture_root, epochs=3, seed=6)
    assert [h.loss for h in first.history] == [h.loss for h in second.history]
    assert [h.loss for h in first.history] != [h.loss for h in third.history]


# --- checkpoints ---


def test_checkpoint_round_trip_preserves_everything(fixture_root, tmp_path):
    dataset, loader = _loader(fixture_root)
    model = _model()
    optimizer = build_optimizer(model, TrainConfig(epochs=3, learning_rate=2e-3))
    stats = train_one_epoch(model, loader, optimizer, CPU, epoch=1)
    preprocess = PreprocessConfig(image_size=64, resize_size=72, clahe=CLAHEConfig(clip_limit=3.0))

    path = save_checkpoint(
        tmp_path / "ck" / "last.pt",
        model=model,
        optimizer=optimizer,
        epoch=1,
        history=[stats],
        class_names=dataset.class_names,
        preprocess=preprocess,
        train_config=TrainConfig(epochs=3, learning_rate=2e-3),
        loader=loader,
    )
    assert path.exists() and not path.with_suffix(".pt.tmp").exists()

    checkpoint = load_checkpoint(path)
    assert checkpoint.epoch == 1
    assert checkpoint.class_names == sorted(CLASSES)
    assert checkpoint.preprocess == preprocess
    assert checkpoint.train_config == TrainConfig(epochs=3, learning_rate=2e-3)
    assert checkpoint.history == [stats]
    assert checkpoint.model_spec == {
        "num_classes": 4,
        "dropout": 0.2,
        "freeze_backbone": True,
        "trainable_blocks": 0,
    }

    rebuilt = checkpoint.build_model()
    for key, value in model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key]), key
    assert is_backbone_frozen(rebuilt)

    new_optimizer = build_optimizer(rebuilt, TrainConfig(epochs=3, learning_rate=2e-3))
    checkpoint.restore_optimizer(new_optimizer)
    assert new_optimizer.state_dict()["param_groups"][0]["lr"] == 2e-3
    assert len(new_optimizer.state_dict()["state"]) == len(optimizer.state_dict()["state"])


def test_checkpoint_format_version_is_enforced(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION + 1}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        load_checkpoint(path)


def test_model_spec_reflects_the_model():
    assert model_spec(build_classifier(7, dropout=0.35, pretrained=False)) == {
        "num_classes": 7,
        "dropout": 0.35,
        "freeze_backbone": False,
        "trainable_blocks": 9,
    }


def test_fit_writes_last_checkpoint_each_epoch(fixture_root, tmp_path):
    result = _fit(fixture_root, epochs=2, checkpoint_dir=tmp_path / "ck")
    assert result.last_checkpoint == tmp_path / "ck" / LAST_CHECKPOINT_NAME
    assert load_checkpoint(result.last_checkpoint).epoch == 2


def test_resume_reproduces_the_uninterrupted_trajectory_exactly(fixture_root, tmp_path):
    uninterrupted = _fit(fixture_root, epochs=3)

    ck_dir = tmp_path / "ck"
    _fit(fixture_root, epochs=1, checkpoint_dir=ck_dir)
    checkpoint = load_checkpoint(ck_dir / LAST_CHECKPOINT_NAME)
    resumed = _fit(fixture_root, epochs=3, resume=checkpoint, model=checkpoint.build_model())

    assert [h.epoch for h in resumed.history] == [1, 2, 3]
    assert [h.loss for h in resumed.history] == [h.loss for h in uninterrupted.history]
    assert [h.train_accuracy for h in resumed.history] == [
        h.train_accuracy for h in uninterrupted.history
    ]


def test_resume_at_target_epoch_trains_nothing_more(fixture_root, tmp_path):
    ck_dir = tmp_path / "ck"
    _fit(fixture_root, epochs=2, checkpoint_dir=ck_dir)
    checkpoint = load_checkpoint(ck_dir / LAST_CHECKPOINT_NAME)
    resumed = _fit(
        fixture_root,
        epochs=2,
        resume=checkpoint,
        model=checkpoint.build_model(),
        checkpoint_dir=ck_dir,
    )
    assert len(resumed.history) == 2
    assert resumed.history == checkpoint.history
    assert (
        resumed.last_checkpoint == ck_dir / LAST_CHECKPOINT_NAME
    ), "a zero-epoch resume must still report the checkpoint it was resumed from"


# --- CLI ---


def test_cli_trains_and_resumes_with_worker_processes(fixture_root, tmp_path):
    ck_dir = tmp_path / "cli_ck"
    base = [
        sys.executable,
        "-m",
        "src.train",
        "--train-root",
        str(fixture_root),
        "--checkpoint-dir",
        str(ck_dir),
        "--image-size",
        "64",
        "--batch-size",
        "8",
        "--freeze-backbone",
        "--device",
        "cpu",
        "--workers",
        "2",
    ]
    first = subprocess.run(base + ["--epochs", "1"], capture_output=True, text=True, cwd=ROOT)
    assert first.returncode == 0, first.stderr
    assert "epoch 1/1" in first.stderr
    assert load_checkpoint(ck_dir / LAST_CHECKPOINT_NAME).epoch == 1

    second = subprocess.run(
        base + ["--epochs", "2", "--resume", str(ck_dir / LAST_CHECKPOINT_NAME)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert second.returncode == 0, second.stderr
    assert "resuming from" in second.stderr and "epoch 2/2" in second.stderr
    checkpoint = load_checkpoint(ck_dir / LAST_CHECKPOINT_NAME)
    assert checkpoint.epoch == 2 and len(checkpoint.history) == 2
    assert checkpoint.preprocess.image_size == 64
    assert checkpoint.class_names == sorted(CLASSES)
    shutil.rmtree(ck_dir)


def test_cli_rejects_unavailable_device(fixture_root, tmp_path):
    if available_backends()["cuda"]:
        pytest.skip("cuda is available here; nothing to reject")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--train-root",
            str(fixture_root),
            "--checkpoint-dir",
            str(tmp_path),
            "--epochs",
            "1",
            "--device",
            "cuda",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode != 0
    assert "not available on this machine" in result.stderr
