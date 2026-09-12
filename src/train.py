"""Training loop and checkpointing (Layer 7).

    train loader ─► for each epoch:
                       set_train_mode(model)          (frozen backbone stays in eval)
                       for x, y in loader:
                           logits = model(x)
                           loss = cross_entropy(logits, y)
                           loss.backward(); optimizer.step()
                       save_checkpoint(last.pt)       (model, optimizer, RNG, loader order)

Validation and best-checkpoint selection are Layer 8; fine-tuning schedules
are Layer 9. This layer proves that parameters move in the right direction,
that a run is reproducible from its seed, and that a run can be stopped and
resumed to the identical trajectory.

Runnable as `python -m src.train` — the `__main__` guard is required because
macOS DataLoader workers are spawned by re-importing the entry point.

Owner: Student 1
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet

from src.augmentation import build_train_transform
from src.config import LEARNING_RATE, NUM_EPOCHS, SEED, WEIGHT_DECAY
from src.dataset import ImageFolderDataset, build_dataloader
from src.device import resolve_device
from src.manifest import MANIFEST_PATH, build_split_dataset
from src.model import build_classifier, set_trainable_blocks, trainable_block_count
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform
from src.utils import get_logger, set_seed
from src.validation import ValidationSummary, run_validation

logger = get_logger(__name__)

CHECKPOINT_FORMAT_VERSION = 2
LAST_CHECKPOINT_NAME = "last.pt"
BEST_CHECKPOINT_NAME = "best.pt"
SELECTION_METRICS = ("f1_macro", "accuracy", "loss")
OPTIMIZERS = ("adamw", "adam", "sgd")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    seed: int = SEED
    selection_metric: str = "f1_macro"
    backbone_learning_rate: float | None = None
    amp: bool = False
    optimizer: str = "adamw"
    momentum: float = 0.9
    lr_step_size: int | None = None
    lr_gamma: float = 0.1

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.lr_step_size is not None and self.lr_step_size < 1:
            raise ValueError(f"lr_step_size must be >= 1, got {self.lr_step_size}")
        if not 0.0 < self.lr_gamma <= 1.0:
            raise ValueError(f"lr_gamma must be in (0, 1], got {self.lr_gamma}")
        if self.backbone_learning_rate is not None and self.backbone_learning_rate <= 0:
            raise ValueError(
                f"backbone_learning_rate must be > 0, got {self.backbone_learning_rate}"
            )
        if self.selection_metric not in SELECTION_METRICS:
            raise ValueError(
                f"selection_metric must be one of {SELECTION_METRICS}, "
                f"got {self.selection_metric!r}"
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")


@dataclass(frozen=True)
class EpochStats:
    """Running statistics of one training epoch.

    `train_accuracy` is accumulated batch by batch while the model is in
    train mode (dropout active) and its weights are changing. It is a
    progress signal, not an evaluation; measure real accuracy with the model
    in eval mode (Layer 8).
    """

    epoch: int
    loss: float
    train_accuracy: float
    samples: int
    seconds: float
    validation: ValidationSummary | None = None
    lr_head: float | None = None
    lr_backbone: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation"] = self.validation.to_dict() if self.validation else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpochStats:
        validation = payload.get("validation")
        return cls(
            **{k: v for k, v in payload.items() if k != "validation"},
            validation=ValidationSummary(**validation) if validation else None,
        )


@dataclass
class TrainingResult:
    history: list[EpochStats] = field(default_factory=list)
    last_checkpoint: Path | None = None
    best_checkpoint: Path | None = None
    best_epoch: int | None = None
    best_metric: float | None = None

    @property
    def final(self) -> EpochStats:
        return self.history[-1]


def is_improvement(metric: str, candidate: float, incumbent: float | None) -> bool:
    """Strict improvement; for `loss` lower is better, otherwise higher."""
    if incumbent is None:
        return True
    return candidate < incumbent if metric == "loss" else candidate > incumbent


# --- model mode -------------------------------------------------------------


def is_backbone_frozen(model: EfficientNet) -> bool:
    return all(not p.requires_grad for p in model.features.parameters())


def set_train_mode(model: EfficientNet) -> None:
    """`train()` for what is being trained; frozen backbone blocks stay in `eval()`.

    A frozen block's parameters do not update, but its BatchNorm running
    statistics would still drift in `train()` mode. Keeping frozen blocks in
    `eval()` means "frozen" is true for buffers as well as parameters, at
    block granularity so partial unfreezing (Layer 9) behaves the same way.
    """
    model.train()
    for block in model.features:
        if not any(p.requires_grad for p in block.parameters()):
            block.eval()


# --- one epoch ----------------------------------------------------------------


def build_optimizer(model: EfficientNet, config: TrainConfig) -> torch.optim.Optimizer:
    """Optimizer over trainable parameters, in two groups: `head` and (if any) `backbone`.

    The backbone group uses `config.backbone_learning_rate` when set, so
    fine-tuning can move pretrained weights more gently than the fresh head.
    `config.optimizer` selects AdamW (default), Adam or SGD with momentum.
    """
    head = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    if not head and not backbone:
        raise ValueError("model has no trainable parameters")

    groups: list[dict[str, Any]] = []
    if head:
        groups.append({"params": head, "lr": config.learning_rate, "name": "head"})
    if backbone:
        backbone_lr = config.backbone_learning_rate or config.learning_rate
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.SGD(
        groups, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.StepLR | None:
    """Optional per-stage step decay: lr *= gamma every `lr_step_size` epochs."""
    if config.lr_step_size is None:
        return None
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )


def current_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        group.get("name", f"group{i}"): group["lr"]
        for i, group in enumerate(optimizer.param_groups)
    }


def autocast_dtype(device: torch.device) -> torch.dtype:
    """float16 on CUDA (with a GradScaler), bfloat16 elsewhere (no scaler needed)."""
    return torch.float16 if device.type == "cuda" else torch.bfloat16


def build_scaler(device: torch.device, amp: bool) -> torch.amp.GradScaler:
    """Loss scaler for float16 autocast; a disabled scaler is a transparent no-op."""
    return torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")


def train_one_epoch(
    model: EfficientNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> EpochStats:
    set_train_mode(model)
    scaler = scaler if scaler is not None else build_scaler(device, amp)
    started = time.perf_counter()
    total_loss = 0.0
    correct = 0
    samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype(device), enabled=amp):
            logits = model(images)
            loss = F.cross_entropy(logits.float(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = labels.shape[0]
        total_loss += loss.item() * batch
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += batch

    if samples == 0:
        raise ValueError("train loader yielded no samples")
    return EpochStats(
        epoch=epoch,
        loss=total_loss / samples,
        train_accuracy=correct / samples,
        samples=samples,
        seconds=time.perf_counter() - started,
    )


# --- checkpoints -----------------------------------------------------------------


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"])


def model_spec(model: EfficientNet) -> dict[str, Any]:
    dropout, head = model.classifier
    return {
        "num_classes": head.out_features,
        "dropout": dropout.p,
        "freeze_backbone": is_backbone_frozen(model),
        "trainable_blocks": trainable_block_count(model),
    }


def save_checkpoint(
    path: Path,
    *,
    model: EfficientNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[EpochStats],
    class_names: list[str],
    preprocess: PreprocessConfig,
    train_config: TrainConfig,
    loader: DataLoader | None = None,
    best_epoch: int | None = None,
    best_metric: float | None = None,
    stage: str | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> Path:
    """Everything needed to rebuild the model and continue the exact trajectory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_spec": model_spec(model),
        "class_names": list(class_names),
        "preprocess": asdict(preprocess),
        "train_config": asdict(train_config),
        "history": [h.to_dict() for h in history],
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "stage": stage,
        "scaler_state": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng": _rng_state(),
        "loader_generator": (
            loader.generator.get_state()
            if loader is not None and loader.generator is not None
            else None
        ),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


@dataclass
class Checkpoint:
    path: Path
    epoch: int
    model_spec: dict[str, Any]
    class_names: list[str]
    preprocess: PreprocessConfig
    train_config: TrainConfig
    history: list[EpochStats]
    best_epoch: int | None
    best_metric: float | None
    stage: str | None
    _payload: dict[str, Any]

    @property
    def model_state(self) -> dict[str, torch.Tensor]:
        return self._payload["model_state"]

    def build_model(self, *, pretrained: bool = False) -> EfficientNet:
        """Rebuild the architecture, load the saved weights, restore the freeze pattern."""
        spec = self.model_spec
        model = build_classifier(
            spec["num_classes"],
            dropout=spec["dropout"],
            freeze_backbone=spec["freeze_backbone"],
            pretrained=pretrained,
        )
        model.load_state_dict(self.model_state)
        set_trainable_blocks(model, spec["trainable_blocks"])
        return model

    def restore_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.load_state_dict(self._payload["optimizer_state"])

    def restore_scaler(self, scaler: torch.amp.GradScaler) -> None:
        state = self._payload.get("scaler_state")
        if state is not None and scaler.is_enabled():
            scaler.load_state_dict(state)

    def restore_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler | None) -> None:
        state = self._payload.get("scheduler_state")
        if scheduler is not None and state is not None:
            scheduler.load_state_dict(state)

    def restore_rng(self, loader: DataLoader | None = None) -> None:
        _restore_rng_state(self._payload["rng"])
        state = self._payload["loader_generator"]
        if loader is not None and loader.generator is not None and state is not None:
            loader.generator.set_state(state)


def load_checkpoint(path: Path) -> Checkpoint:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version!r} in {path}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    pre = dict(payload["preprocess"])
    if pre.get("clahe") is not None:
        pre["clahe"] = CLAHEConfig(**pre["clahe"])
    pre["mean"] = tuple(pre["mean"])
    pre["std"] = tuple(pre["std"])
    return Checkpoint(
        path=path,
        epoch=payload["epoch"],
        model_spec=payload["model_spec"],
        class_names=list(payload["class_names"]),
        preprocess=PreprocessConfig(**pre),
        train_config=TrainConfig(**payload["train_config"]),
        history=[EpochStats.from_dict(h) for h in payload["history"]],
        best_epoch=payload.get("best_epoch"),
        best_metric=payload.get("best_metric"),
        stage=payload.get("stage"),
        _payload=payload,
    )


# --- fit ----------------------------------------------------------------------------


def fit(
    model: EfficientNet,
    loader: DataLoader,
    *,
    config: TrainConfig,
    class_names: list[str],
    preprocess: PreprocessConfig,
    device: torch.device,
    checkpoint_dir: Path | None = None,
    resume_from: Checkpoint | None = None,
    val_loader: DataLoader | None = None,
    stage: str | None = None,
) -> TrainingResult:
    """Train for `config.epochs` total epochs, checkpointing after each.

    With `val_loader`, every epoch ends with an eval-mode validation pass
    (src/validation.py); the epoch whose `config.selection_metric` is best
    is written to `best.pt`. With `resume_from`, model/optimizer/RNG/loader
    state are restored and training continues from the checkpoint's epoch.
    """
    model.to(device)
    optimizer = build_optimizer(model, config)
    scaler = build_scaler(device, config.amp)
    scheduler = build_scheduler(optimizer, config)
    result = TrainingResult()
    start_epoch = 1

    if resume_from is None:
        set_seed(config.seed)
    else:
        resume_from.restore_optimizer(optimizer)
        resume_from.restore_scaler(scaler)
        resume_from.restore_scheduler(scheduler)
        resume_from.restore_rng(loader)
        result.history = list(resume_from.history)
        result.best_epoch = resume_from.best_epoch
        result.best_metric = resume_from.best_metric
        start_epoch = resume_from.epoch + 1
        if checkpoint_dir is not None:
            # A resumed run may train zero further epochs; the files written by
            # the interrupted run are still this run's checkpoints.
            last_path = Path(checkpoint_dir) / LAST_CHECKPOINT_NAME
            best_path = Path(checkpoint_dir) / BEST_CHECKPOINT_NAME
            if last_path.exists():
                result.last_checkpoint = last_path
            if result.best_epoch is not None and best_path.exists():
                result.best_checkpoint = best_path
        logger.info("resuming from %s at epoch %d", resume_from.path, start_epoch)

    def checkpoint(name: str) -> Path:
        return save_checkpoint(
            Path(checkpoint_dir) / name,  # type: ignore[arg-type]
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=result.history,
            class_names=class_names,
            preprocess=preprocess,
            train_config=config,
            loader=loader,
            best_epoch=result.best_epoch,
            best_metric=result.best_metric,
            stage=stage,
            scaler=scaler,
            scheduler=scheduler,
        )

    for epoch in range(start_epoch, config.epochs + 1):
        rates = current_learning_rates(optimizer)
        stats = train_one_epoch(
            model, loader, optimizer, device, epoch=epoch, amp=config.amp, scaler=scaler
        )
        stats = dataclasses.replace(
            stats, lr_head=rates.get("head"), lr_backbone=rates.get("backbone")
        )
        if scheduler is not None:
            scheduler.step()
        message = "epoch %d/%d  loss %.4f  train_acc %.3f  (%d samples, %.1fs)" % (
            epoch,
            config.epochs,
            stats.loss,
            stats.train_accuracy,
            stats.samples,
            stats.seconds,
        )

        if val_loader is not None:
            summary = run_validation(model, val_loader, device, class_names).summary
            stats = dataclasses.replace(stats, validation=summary)
            message += "  |  val loss %.4f  acc %.3f  f1_macro %.3f" % (
                summary.loss,
                summary.accuracy,
                summary.f1_macro,
            )
            candidate = getattr(summary, config.selection_metric)
            if is_improvement(config.selection_metric, candidate, result.best_metric):
                result.best_metric = candidate
                result.best_epoch = epoch
                message += "  *"

        result.history.append(stats)
        logger.info(message)

        if checkpoint_dir is not None:
            result.last_checkpoint = checkpoint(LAST_CHECKPOINT_NAME)
            if result.best_epoch == epoch:
                result.best_checkpoint = checkpoint(BEST_CHECKPOINT_NAME)
            elif result.best_epoch is not None:
                result.best_checkpoint = Path(checkpoint_dir) / BEST_CHECKPOINT_NAME

    model.eval()
    return result


# --- CLI ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the classifier on an image folder.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-root", type=Path, help="image folder laid out as <class>/<img>")
    source.add_argument(
        "--manifest",
        type=Path,
        nargs="?",
        const=MANIFEST_PATH,
        help=f"frozen split manifest (default {MANIFEST_PATH.name}); trains on split=train, "
        "validates on split=val",
    )
    parser.add_argument("--val-root", type=Path, default=None, help="validation image folder")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="f1_macro")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=PreprocessConfig().image_size)
    parser.add_argument("--clahe", action="store_true", help="enable CLAHE (off by default)")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", default=None, help="cpu / mps / cuda; default: auto")
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint to continue from")
    parser.add_argument("--amp", action="store_true", help="mixed precision (float16 on CUDA)")
    parser.add_argument("--pin-memory", action="store_true", help="pinned host memory (CUDA)")
    args = parser.parse_args(argv)

    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        selection_metric=args.selection_metric,
        amp=args.amp,
    )
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None:
        preprocess = checkpoint.preprocess
        class_names = checkpoint.class_names
        model = checkpoint.build_model()
    else:
        resize_size = max(PreprocessConfig().resize_size, args.image_size)
        preprocess = PreprocessConfig(
            image_size=args.image_size,
            resize_size=resize_size,
            clahe=CLAHEConfig() if args.clahe else None,
        )
        class_names = None
        model = None

    set_seed(config.seed)
    dataset: Dataset[tuple[torch.Tensor, int]]
    if args.manifest is not None:
        dataset = build_split_dataset(
            "train",
            transform=build_train_transform(preprocess),
            manifest_path=args.manifest,
            class_names=class_names,
        )
    else:
        dataset = ImageFolderDataset(
            args.train_root, transform=build_train_transform(preprocess), class_names=class_names
        )
    if model is None:
        model = build_classifier(
            len(dataset.class_names),
            freeze_backbone=args.freeze_backbone,
            pretrained=not args.no_pretrained,
        )
    loader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        seed=config.seed,
        persistent_workers=True,
        pin_memory=args.pin_memory,
    )
    val_loader = None
    val_dataset: Dataset[tuple[torch.Tensor, int]] | None = None
    if args.manifest is not None:
        val_dataset = build_split_dataset(
            "val", transform=build_eval_transform(preprocess), manifest_path=args.manifest
        )
    elif args.val_root is not None:
        val_dataset = ImageFolderDataset(
            args.val_root,
            transform=build_eval_transform(preprocess),
            class_names=dataset.class_names,
        )
    if val_dataset is not None:
        val_loader = build_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            persistent_workers=True,
            pin_memory=args.pin_memory,
        )

    logger.info(
        "training %d classes %s on %s — %d images, %d epochs",
        len(dataset.class_names),
        dataset.class_names,
        device,
        len(dataset),
        config.epochs,
    )
    result = fit(
        model,
        loader,
        config=config,
        class_names=dataset.class_names,
        preprocess=preprocess,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=checkpoint,
        val_loader=val_loader,
    )
    logger.info(
        "done: final loss %.4f train_acc %.3f", result.final.loss, result.final.train_accuracy
    )
    if result.best_epoch is not None:
        logger.info(
            "best %s %.4f at epoch %d -> %s",
            config.selection_metric,
            result.best_metric,
            result.best_epoch,
            result.best_checkpoint,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
