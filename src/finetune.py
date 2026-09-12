"""Staged transfer learning (Layer 9).

    Stage A  "head"     backbone frozen, train the classifier head
    Stage B  "partial"  unfreeze the last N backbone blocks, smaller backbone LR
    Stage C  "full"     unfreeze everything, smallest LRs

Each stage is one `fit` call (Layer 7) with its own optimiser and its own
checkpoint directory. The weights that enter a stage are the previous
stage's best checkpoint when validation is available, otherwise its last.
Nothing here re-implements the loop, the validation pass, or checkpointing.

The default schedule's epoch counts and learning rates are conventional
starting points for fine-tuning an ImageNet backbone on a small dataset;
they are NOT tuned on fish data and are expected to change once the real
dataset exists.

Runnable as `python -m src.finetune`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet

from src.augmentation import build_train_transform
from src.config import LEARNING_RATE, SEED, WEIGHT_DECAY
from src.dataset import ImageFolderDataset, build_dataloader
from src.device import resolve_device
from src.manifest import MANIFEST_PATH, build_split_dataset
from src.model import backbone_block_count, build_classifier, set_trainable_blocks
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform
from src.train import (
    Checkpoint,
    TrainConfig,
    TrainingResult,
    fit,
    load_checkpoint,
)
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

ALL_BLOCKS = -1


@dataclass(frozen=True)
class Stage:
    name: str
    epochs: int
    trainable_blocks: int
    learning_rate: float
    backbone_learning_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if self.epochs < 1:
            raise ValueError(f"stage {self.name!r}: epochs must be >= 1")
        if self.trainable_blocks < ALL_BLOCKS:
            raise ValueError(f"stage {self.name!r}: trainable_blocks must be >= -1")

    def resolve_blocks(self, model: EfficientNet) -> int:
        total = backbone_block_count(model)
        return total if self.trainable_blocks == ALL_BLOCKS else self.trainable_blocks


DEFAULT_SCHEDULE: tuple[Stage, ...] = (
    Stage("head", epochs=5, trainable_blocks=0, learning_rate=1e-3),
    Stage(
        "partial", epochs=10, trainable_blocks=3, learning_rate=3e-4, backbone_learning_rate=3e-5
    ),
    Stage(
        "full",
        epochs=15,
        trainable_blocks=ALL_BLOCKS,
        learning_rate=LEARNING_RATE,
        backbone_learning_rate=1e-5,
    ),
)


@dataclass
class StageResult:
    stage: Stage
    trainable_blocks: int
    result: TrainingResult
    checkpoint_dir: Path


@dataclass
class ScheduleResult:
    stages: list[StageResult] = field(default_factory=list)

    @property
    def final_checkpoint(self) -> Path | None:
        last = self.stages[-1].result
        return last.best_checkpoint or last.last_checkpoint


def validate_schedule(stages: tuple[Stage, ...] | list[Stage]) -> None:
    if not stages:
        raise ValueError("schedule must contain at least one stage")
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"stage names must be unique, got {names}")


def apply_stage(model: EfficientNet, stage: Stage) -> int:
    """Set the freeze pattern for `stage`; returns the resolved trainable block count."""
    blocks = stage.resolve_blocks(model)
    set_trainable_blocks(model, blocks)
    return blocks


def stage_train_config(
    stage: Stage, *, seed: int, selection_metric: str, amp: bool = False
) -> TrainConfig:
    return TrainConfig(
        epochs=stage.epochs,
        learning_rate=stage.learning_rate,
        weight_decay=WEIGHT_DECAY,
        seed=seed,
        selection_metric=selection_metric,
        backbone_learning_rate=stage.backbone_learning_rate,
        amp=amp,
    )


def run_schedule(
    model: EfficientNet,
    train_loader: DataLoader,
    stages: tuple[Stage, ...] | list[Stage],
    *,
    class_names: list[str],
    preprocess: PreprocessConfig,
    device: torch.device,
    checkpoint_dir: Path,
    val_loader: DataLoader | None = None,
    seed: int = SEED,
    selection_metric: str = "f1_macro",
    resume_from: Checkpoint | None = None,
    amp: bool = False,
) -> ScheduleResult:
    """Run every stage in order; each stage starts from the previous stage's best weights.

    With `resume_from` (a checkpoint written by this function), stages before
    the checkpoint's stage are skipped, that stage is resumed, and the rest run.
    """
    validate_schedule(stages)
    checkpoint_dir = Path(checkpoint_dir)
    schedule = ScheduleResult()

    start_index = 0
    if resume_from is not None:
        names = [s.name for s in stages]
        if resume_from.stage not in names:
            raise ValueError(
                f"checkpoint stage {resume_from.stage!r} is not in the schedule {names}"
            )
        start_index = names.index(resume_from.stage)

    for index, stage in enumerate(stages):
        if index < start_index:
            continue
        resuming = resume_from if index == start_index and resume_from is not None else None
        blocks = apply_stage(model, stage)
        config = stage_train_config(stage, seed=seed, selection_metric=selection_metric, amp=amp)
        stage_dir = checkpoint_dir / stage.name
        logger.info(
            "stage %s: %d epoch(s), %d/%d backbone blocks trainable, lr %g, backbone lr %s",
            stage.name,
            stage.epochs,
            blocks,
            backbone_block_count(model),
            stage.learning_rate,
            stage.backbone_learning_rate if stage.backbone_learning_rate else "same",
        )
        result = fit(
            model,
            train_loader,
            config=config,
            class_names=class_names,
            preprocess=preprocess,
            device=device,
            checkpoint_dir=stage_dir,
            resume_from=resuming,
            val_loader=val_loader,
            stage=stage.name,
        )
        schedule.stages.append(StageResult(stage, blocks, result, stage_dir))

        carry = result.best_checkpoint or result.last_checkpoint
        if carry is not None and index < len(stages) - 1:
            model.load_state_dict(load_checkpoint(carry).model_state)
            logger.info("stage %s: carrying forward weights from %s", stage.name, carry.name)

    return schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Staged fine-tuning on an image folder.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-root", type=Path, help="image folder laid out as <class>/<img>")
    source.add_argument(
        "--manifest",
        type=Path,
        nargs="?",
        const=MANIFEST_PATH,
        help=f"frozen split manifest (default {MANIFEST_PATH.name}); train/val splits",
    )
    parser.add_argument("--val-root", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--head-epochs", type=int, default=DEFAULT_SCHEDULE[0].epochs)
    parser.add_argument("--partial-epochs", type=int, default=DEFAULT_SCHEDULE[1].epochs)
    parser.add_argument("--full-epochs", type=int, default=DEFAULT_SCHEDULE[2].epochs)
    parser.add_argument("--partial-blocks", type=int, default=DEFAULT_SCHEDULE[1].trainable_blocks)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=PreprocessConfig().image_size)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--selection-metric", default="f1_macro")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None, help="stage checkpoint to resume")
    parser.add_argument("--amp", action="store_true", help="mixed precision (float16 on CUDA)")
    parser.add_argument("--pin-memory", action="store_true", help="pinned host memory (CUDA)")
    args = parser.parse_args(argv)

    stages = (
        Stage(
            "head",
            args.head_epochs,
            DEFAULT_SCHEDULE[0].trainable_blocks,
            DEFAULT_SCHEDULE[0].learning_rate,
        ),
        Stage(
            "partial",
            args.partial_epochs,
            args.partial_blocks,
            DEFAULT_SCHEDULE[1].learning_rate,
            DEFAULT_SCHEDULE[1].backbone_learning_rate,
        ),
        Stage(
            "full",
            args.full_epochs,
            ALL_BLOCKS,
            DEFAULT_SCHEDULE[2].learning_rate,
            DEFAULT_SCHEDULE[2].backbone_learning_rate,
        ),
    )
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint is not None:
        preprocess = checkpoint.preprocess
        class_names = checkpoint.class_names
    else:
        preprocess = PreprocessConfig(
            image_size=args.image_size,
            resize_size=max(PreprocessConfig().resize_size, args.image_size),
            clahe=CLAHEConfig() if args.clahe else None,
        )
        class_names = None

    set_seed(args.seed)
    train_dataset: Dataset[tuple[torch.Tensor, int]]
    if args.manifest is not None:
        train_dataset = build_split_dataset(
            "train",
            transform=build_train_transform(preprocess),
            manifest_path=args.manifest,
            class_names=class_names,
        )
    else:
        train_dataset = ImageFolderDataset(
            args.train_root, transform=build_train_transform(preprocess), class_names=class_names
        )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        seed=args.seed,
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
            class_names=train_dataset.class_names,
        )
    if val_dataset is not None:
        val_loader = build_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            persistent_workers=True,
            pin_memory=args.pin_memory,
        )

    model = (
        checkpoint.build_model()
        if checkpoint is not None
        else build_classifier(len(train_dataset.class_names))
    )
    schedule = run_schedule(
        model,
        train_loader,
        stages,
        class_names=train_dataset.class_names,
        preprocess=preprocess,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        val_loader=val_loader,
        seed=args.seed,
        selection_metric=args.selection_metric,
        resume_from=checkpoint,
        amp=args.amp,
    )
    logger.info("schedule complete: final checkpoint %s", schedule.final_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
