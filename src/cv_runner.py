"""Generic cross-validation experiment runner (Phase 12): one model x one fold x one
data arm, on the Phase-8/9 manifests, with hard leakage guards, resumable
checkpoints and full provenance. Model-agnostic: it only needs a module that
maps (N, 3, 224, 224) -> (N, K) logits and exposes the accessors of
src/model_factory.py.

Reused unchanged: src/preprocessing.py (CLAHE + resize + normalise),
src/augmentation.py (train-only augmentation), src/dataset.py (load_image,
build_dataloader), src/metrics.py + src/validation.py (metrics, validation
pass), src/device.py, src/utils.py, and the AMP / RNG helpers of src/train.py.

Output layout: results/v2/experiments/<experiment_id>/
    config.json  history.csv  metrics.json  confusion_matrix.csv  best.pt  latest.pt
    run_summary.json  status.json  logs/train.log

The frozen final test manifest is read ONLY to obtain its image ids and digest
for the guards; its images are never decoded here.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.augmentation import AugmentConfig, build_train_transform
from src.config import ROOT_DIR, SEED
from src.dataset import build_dataloader, load_image
from src.device import resolve_device, torch_report
from src.gan_augmentation import GAN_DIR_NAME
from src.manifest import CANONICAL_CLASSES, file_sha256
from src.metrics import classification_report, compute_metrics
from src.model_factory import (
    MODEL_KEYS,
    backbone_parameters,
    build_model,
    head_parameters,
    set_backbone_trainable,
    set_train_mode,
    trainable_summary,
)
from src.multi_dataset import DATASET_CONFIG_VERSION
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform
from src.split_v2 import CV_DIR_NAME, SPLIT_DIR_NAME
from src.train import _restore_rng_state, _rng_state, autocast_dtype, build_scaler, is_improvement
from src.utils import get_logger, set_seed
from src.validation import run_validation

logger = get_logger(__name__)

DATA_ARMS = ("without_gan", "with_gan")
STATUSES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "INTERRUPTED")
STRATEGIES = ("head_then_full", "full")
SCHEDULERS = ("none", "cosine", "step")
OPTIMIZERS = ("adamw", "adam", "sgd")
RESULTS_V2 = Path("results") / "v2"
EXPERIMENTS_DIR = RESULTS_V2 / "experiments"
HISTORY_COLUMNS = (
    "epoch",
    "stage",
    "train_loss",
    "train_accuracy",
    "train_precision_macro",
    "train_recall_macro",
    "train_f1_macro",
    "val_loss",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "val_f1_weighted",
    "lr_head",
    "lr_backbone",
    "seconds",
    "improved",
)


class LeakageError(RuntimeError):
    """A hard data-isolation violation. Never downgraded to a warning."""


# --- configuration -------------------------------------------------------------------------------


@dataclass(frozen=True)
class CVConfig:
    """Training hyper-parameters shared by every experiment of the matrix. Loaded from
    configs/cv_v2/<name>.json; every value is recorded in the experiment's config.json."""

    epochs: int = 30
    batch_size: int = 32
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    backbone_learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "cosine"
    lr_step_size: int = 5
    lr_gamma: float = 0.1
    strategy: str = "head_then_full"
    head_epochs: int = 3
    early_stopping_patience: int | None = 8
    selection_metric: str = "f1_macro"
    pretrained: bool = True
    dropout: float = 0.2
    amp: bool = True
    num_workers: int = 2
    preprocess_config: str = "configs/preprocess_v2_clahe.json"
    augment: dict[str, Any] = field(default_factory=lambda: asdict(AugmentConfig()))

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}")
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        if self.strategy == "head_then_full" and not 0 <= self.head_epochs < self.epochs:
            raise ValueError("head_epochs must satisfy 0 <= head_epochs < epochs")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be >= 1 or null")
        if self.selection_metric not in ("f1_macro", "accuracy", "loss"):
            raise ValueError("selection_metric must be f1_macro, accuracy or loss")
        AugmentConfig(**self.augment)  # validates


def load_cv_config(path: Path) -> CVConfig:
    payload = json.loads(Path(path).read_text())
    payload.pop("description", None)
    return CVConfig(**payload)


def load_preprocess_config(path: Path) -> PreprocessConfig:
    """The `preprocess` block of configs/preprocess_v2_clahe.json (Phase 7) -> PreprocessConfig,
    built with the existing dataclasses so nothing about CLAHE is redefined here."""
    spec = json.loads(Path(path).read_text())["preprocess"]
    clahe = spec.get("clahe")
    return PreprocessConfig(
        image_size=spec["image_size"],
        resize_size=spec["resize_size"],
        resize_mode=spec["resize_mode"],
        clahe=CLAHEConfig(**clahe) if clahe else None,
    )


# --- manifests -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainRow:
    image_id: str
    source_dataset: str
    filepath: str
    unified_class: str
    label: int
    group_id: str
    fold: int
    synthetic: bool


def read_manifest_rows(path: Path) -> list[TrainRow]:
    """fold_XX_train.csv / fold_XX_validation.csv (FOLD_COLUMNS) or fold_XX_train_gan.csv
    (FOLD_COLUMNS + synthetic)."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_id",
            "source_dataset",
            "filepath",
            "unified_class",
            "label",
            "group_id",
            "fold",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = []
        for raw in reader:
            label = int(raw["label"])
            if CANONICAL_CLASSES[label] != raw["unified_class"]:
                raise ValueError(
                    f"{path}: {raw['image_id']} label {label} != {raw['unified_class']}"
                )
            rows.append(
                TrainRow(
                    image_id=raw["image_id"],
                    source_dataset=raw["source_dataset"],
                    filepath=raw["filepath"],
                    unified_class=raw["unified_class"],
                    label=label,
                    group_id=raw["group_id"],
                    fold=int(raw["fold"]),
                    synthetic=raw.get("synthetic", "False") == "True",
                )
            )
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def final_test_ids(split_dir: Path) -> set[str]:
    """Ids only — the frozen test images are never decoded by this runner."""
    with (Path(split_dir) / "final_test.csv").open(newline="") as handle:
        return {row["image_id"] for row in csv.DictReader(handle)}


def manifest_paths(data_dir: Path, fold: int, data_arm: str) -> tuple[Path, Path]:
    """(training manifest, validation manifest) for a fold and data arm."""
    if data_arm not in DATA_ARMS:
        raise ValueError(f"data_arm must be one of {DATA_ARMS}, got {data_arm!r}")
    cv_dir = Path(data_dir) / "audit" / CV_DIR_NAME
    validation = cv_dir / f"fold_{fold:02d}_validation.csv"
    if data_arm == "without_gan":
        train = cv_dir / f"fold_{fold:02d}_train.csv"
    else:
        train = (
            Path(data_dir) / GAN_DIR_NAME / f"fold_{fold:02d}" / f"fold_{fold:02d}_train_gan.csv"
        )
    return train, validation


# --- hard leakage guards -------------------------------------------------------------------------


def check_isolation(
    *,
    fold: int,
    data_arm: str,
    train_path: Path,
    validation_path: Path,
    train_rows: list[TrainRow],
    validation_rows: list[TrainRow],
    test_ids: set[str],
) -> dict[str, Any]:
    """Every rule the brief lists, as hard errors. Returns the numbers checked."""
    train_name, val_name = Path(train_path).name, Path(validation_path).name
    expected_train = {
        "without_gan": f"fold_{fold:02d}_train.csv",
        "with_gan": f"fold_{fold:02d}_train_gan.csv",
    }[data_arm]
    if train_name != expected_train:
        raise LeakageError(
            f"training manifest {train_name!r} is not the {data_arm} manifest of fold {fold} "
            f"({expected_train!r})"
        )
    if "final_test" in train_name or "final_test" in val_name:
        raise LeakageError("final_test.csv must never be used as a training or validation manifest")
    if val_name != f"fold_{fold:02d}_validation.csv":
        raise LeakageError(
            f"validation manifest {val_name!r} is not fold_{fold:02d}_validation.csv"
        )
    if data_arm == "with_gan" and f"fold_{fold:02d}" not in Path(train_path).parts:
        raise LeakageError(f"WITH-GAN manifest {train_path} is not inside fold_{fold:02d}")

    train_ids = {r.image_id for r in train_rows}
    val_ids = {r.image_id for r in validation_rows}
    if train_ids & val_ids:
        raise LeakageError(
            f"{len(train_ids & val_ids)} image ids are in both training and validation"
        )
    if train_ids & test_ids:
        raise LeakageError(
            f"{len(train_ids & test_ids)} final-test image ids are in the training manifest"
        )
    if val_ids & test_ids:
        raise LeakageError(
            f"{len(val_ids & test_ids)} final-test image ids are in the validation manifest"
        )
    if any(r.fold == fold for r in train_rows if not r.synthetic):
        raise LeakageError(f"training manifest contains real rows whose validation fold is {fold}")
    if any(r.fold != fold for r in validation_rows):
        raise LeakageError(f"validation manifest contains rows of another fold than {fold}")

    synthetic = [r for r in train_rows if r.synthetic]
    if data_arm == "without_gan" and synthetic:
        raise LeakageError("WITHOUT-GAN training manifest contains synthetic rows")
    for r in synthetic:
        parts = Path(r.filepath).parts
        if GAN_DIR_NAME not in parts or f"fold_{fold:02d}" not in parts or "train" not in parts:
            raise LeakageError(
                f"synthetic image {r.image_id} ({r.filepath}) is not under "
                f"data/gan/fold_{fold:02d}/train/"
            )
        if r.image_id in val_ids or r.image_id in test_ids:
            raise LeakageError(f"synthetic image {r.image_id} appears in validation or final test")
    for r in validation_rows:
        if r.synthetic or GAN_DIR_NAME in Path(r.filepath).parts:
            raise LeakageError(f"validation row {r.image_id} is synthetic / under data/gan")
    for r in train_rows:
        if not r.synthetic and GAN_DIR_NAME in Path(r.filepath).parts:
            raise LeakageError(f"real training row {r.image_id} points into data/gan")
    train_groups = {r.group_id for r in train_rows if not r.synthetic}
    val_groups = {r.group_id for r in validation_rows}
    if train_groups & val_groups:
        raise LeakageError(
            f"{len(train_groups & val_groups)} leakage groups straddle train/validation"
        )
    return {
        "train_rows": len(train_rows),
        "train_real": len(train_rows) - len(synthetic),
        "train_synthetic": len(synthetic),
        "validation_rows": len(validation_rows),
        "final_test_ids_checked": len(test_ids),
        "groups_straddling": 0,
    }


# --- dataset -------------------------------------------------------------------------------------


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, rows: list[TrainRow], transform: Any, repo_root: Path) -> None:
        self.rows = rows
        self.transform = transform
        self.repo_root = Path(repo_root)
        missing = [r.filepath for r in rows if not (self.repo_root / r.filepath).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} manifest images missing on disk, e.g. {missing[:3]}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        return self.transform(load_image(self.repo_root / row.filepath)), row.label

    @property
    def targets(self) -> list[int]:
        return [r.label for r in self.rows]


def subset(rows: list[TrainRow], limit: int | None, seed: int) -> list[TrainRow]:
    """Deterministic subset for smoke runs (recorded as such); None = everything."""
    if limit is None or limit >= len(rows):
        return rows
    order = sorted(
        range(len(rows)),
        key=lambda i: hashlib.sha256(f"{seed}:{rows[i].image_id}".encode()).hexdigest(),
    )
    return [rows[i] for i in sorted(order[:limit])]


# --- experiment record ---------------------------------------------------------------------------


def experiment_id(model_key: str, fold: int, data_arm: str) -> str:
    return f"{model_key}_fold{fold:02d}_{data_arm}"


def git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def environment_record(device: torch.device) -> dict[str, Any]:
    report = asdict(torch_report(str(device)))
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": {
            "built": report["cuda_built"],
            "available": report["cuda_available"],
            "version": torch.version.cuda,
            "device_count": report["cuda_device_count"],
            "device_name": report["cuda_device_name"],
            "memory_gb": cuda_memory,
        },
        "mps": {"built": report["mps_built"], "available": report["mps_available"]},
        "device": str(device),
        "num_threads": report["num_threads"],
    }


def config_hash(record: dict[str, Any]) -> str:
    keys = (
        "model_key",
        "fold",
        "data_arm",
        "seed",
        "num_classes",
        "image_size",
        "preprocessing",
        "augmentation",
        "training",
    )
    payload = json.dumps({k: record[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


# --- checkpoints / status ------------------------------------------------------------------------


@dataclass
class Paths:
    out_dir: Path

    @property
    def config(self) -> Path:
        return self.out_dir / "config.json"

    @property
    def history(self) -> Path:
        return self.out_dir / "history.csv"

    @property
    def metrics(self) -> Path:
        return self.out_dir / "metrics.json"

    @property
    def confusion(self) -> Path:
        return self.out_dir / "confusion_matrix.csv"

    @property
    def best(self) -> Path:
        return self.out_dir / "best.pt"

    @property
    def latest(self) -> Path:
        return self.out_dir / "latest.pt"

    @property
    def summary(self) -> Path:
        return self.out_dir / "run_summary.json"

    @property
    def status(self) -> Path:
        return self.out_dir / "status.json"

    @property
    def log(self) -> Path:
        return self.out_dir / "logs" / "train.log"


def write_status(paths: Paths, status: str, **extra: Any) -> None:
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.status.write_text(
        json.dumps(
            {"status": status, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}, indent=2
        )
    )


def read_status(out_dir: Path) -> str:
    path = Paths(Path(out_dir)).status
    if not path.is_file():
        return "PENDING"
    return json.loads(path.read_text()).get("status", "PENDING")


def is_completed(out_dir: Path) -> bool:
    """COMPLETED only when the status says so AND every required file exists."""
    paths = Paths(Path(out_dir))
    required = (
        paths.config,
        paths.history,
        paths.metrics,
        paths.confusion,
        paths.best,
        paths.latest,
        paths.summary,
    )
    return read_status(out_dir) == "COMPLETED" and all(p.is_file() for p in required)


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    epoch: int,
    stage: str,
    record: dict[str, Any],
    history: list[dict[str, Any]],
    best_metric: float | None,
    best_epoch: int | None,
) -> None:
    torch.save(
        {
            "format": "cv_v3",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict(),
            "epoch": epoch,
            "stage": stage,
            "model_key": record["model_key"],
            "fold": record["fold"],
            "data_arm": record["data_arm"],
            "seed": record["seed"],
            "dataset_config_version": record["dataset_config_version"],
            "optimizer": record["training"]["optimizer"],
            "scheduler": record["training"]["scheduler"],
            "manifest_sha256": {
                "train": record["manifests"]["train_sha256"],
                "validation": record["manifests"]["validation_sha256"],
            },
            "preprocessing_sha256": record["preprocessing"]["config_sha256"],
            "git_commit": record["git_commit"],
            "environment": record["environment"],
            "class_names": list(CANONICAL_CLASSES),
            "best_val_f1_macro": (
                best_metric if record["training"]["selection_metric"] == "f1_macro" else None
            ),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": record,
            "history": history,
            "rng": _rng_state(),
        },
        path,
    )


# --- training ------------------------------------------------------------------------------------


def build_optimizer(
    model: nn.Module, config: CVConfig, *, backbone_frozen: bool
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {
            "params": [p for p in head_parameters(model) if p.requires_grad],
            "lr": config.learning_rate,
            "name": "head",
        }
    ]
    if not backbone_frozen:
        groups.append(
            {
                "params": [p for p in backbone_parameters(model) if p.requires_grad],
                "lr": config.backbone_learning_rate,
                "name": "backbone",
            }
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.SGD(
        groups, lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: CVConfig, stage_epochs: int) -> Any:
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(stage_epochs, 1))
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
    return None


def stage_for_epoch(epoch: int, config: CVConfig) -> str:
    if config.strategy == "head_then_full" and epoch <= config.head_epochs:
        return "head"
    return "full"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    amp: bool,
    backbone_frozen: bool,
) -> dict[str, Any]:
    """Cross-entropy training epoch; returns loss and running train metrics computed with
    the project's metric utilities on train-mode predictions (a progress signal)."""
    set_train_mode(model, backbone_frozen)
    started = time.perf_counter()
    total_loss = 0.0
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
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
        total_loss += loss.item() * labels.shape[0]
        targets.append(labels.detach().cpu())
        predictions.append(logits.detach().argmax(dim=1).cpu())
    if not targets:
        raise ValueError("training loader yielded no samples")
    t, p = torch.cat(targets), torch.cat(predictions)
    m = compute_metrics(t, p, list(CANONICAL_CLASSES))
    return {
        "loss": total_loss / m.samples,
        "accuracy": m.accuracy,
        "precision_macro": m.precision_macro,
        "recall_macro": m.recall_macro,
        "f1_macro": m.f1_macro,
        "seconds": time.perf_counter() - started,
    }


@dataclass
class RunArgs:
    model_key: str
    fold: int
    data_arm: str
    config_path: Path
    data_dir: Path
    repo_root: Path = ROOT_DIR
    out_root: Path = ROOT_DIR / EXPERIMENTS_DIR
    experiment_id: str | None = None
    seed: int = SEED
    device: str | None = None
    require_cuda: bool = False
    resume: bool = False
    epochs_override: int | None = None
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    smoke: bool = False


def _select_device(args: RunArgs) -> torch.device:
    device = resolve_device(args.device)
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            f"CUDA required for this run but the selected device is {device}; "
            "aborting instead of falling back (use a GPU runtime)"
        )
    return device


def _attach_file_logger(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def run_experiment(args: RunArgs) -> dict[str, Any]:
    """Train one experiment end to end; returns the run summary."""
    if args.model_key not in MODEL_KEYS:
        raise ValueError(f"model must be one of {list(MODEL_KEYS)}")
    if args.data_arm not in DATA_ARMS:
        raise ValueError(f"data_arm must be one of {DATA_ARMS}")
    exp_id = args.experiment_id or experiment_id(args.model_key, args.fold, args.data_arm)
    paths = Paths(Path(args.out_root) / exp_id)
    if is_completed(paths.out_dir):
        raise FileExistsError(f"{exp_id} is COMPLETED; not retraining (choose a new id to rerun)")
    if paths.latest.is_file() and not args.resume:
        raise FileExistsError(
            f"{exp_id} has a checkpoint ({read_status(paths.out_dir)}); pass --resume"
        )
    if args.resume and not paths.latest.is_file() and paths.status.is_file():
        logger.warning("%s: --resume but no latest.pt; starting from scratch", exp_id)

    config = load_cv_config(args.config_path)
    if args.epochs_override is not None:
        head_epochs = config.head_epochs
        if config.strategy == "head_then_full" and head_epochs >= args.epochs_override:
            head_epochs = max(args.epochs_override - 1, 0)
        config = dataclasses.replace(config, epochs=args.epochs_override, head_epochs=head_epochs)
    preprocess_path = Path(args.repo_root) / config.preprocess_config
    preprocess = load_preprocess_config(preprocess_path)
    augment = AugmentConfig(**config.augment)
    device = _select_device(args)

    # manifests + hard guards (the frozen test file is read for ids only)
    train_path, validation_path = manifest_paths(args.data_dir, args.fold, args.data_arm)
    split_dir = Path(args.data_dir) / "audit" / SPLIT_DIR_NAME
    for p in (train_path, validation_path, split_dir / "final_test.csv"):
        if not p.is_file():
            raise FileNotFoundError(f"required manifest missing: {p}")
    train_rows_full = read_manifest_rows(train_path)
    validation_rows_full = read_manifest_rows(validation_path)
    test_ids = final_test_ids(split_dir)
    isolation = check_isolation(
        fold=args.fold,
        data_arm=args.data_arm,
        train_path=train_path,
        validation_path=validation_path,
        train_rows=train_rows_full,
        validation_rows=validation_rows_full,
        test_ids=test_ids,
    )
    train_rows = subset(train_rows_full, args.max_train_samples, args.seed)
    validation_rows = subset(validation_rows_full, args.max_validation_samples, args.seed)

    record: dict[str, Any] = {
        "experiment_id": exp_id,
        "model": args.model_key,
        "model_key": args.model_key,
        "fold": args.fold,
        "data_arm": args.data_arm,
        "seed": args.seed,
        "num_classes": len(CANONICAL_CLASSES),
        "class_names": list(CANONICAL_CLASSES),
        "image_size": preprocess.image_size,
        "preprocessing": {
            "config_path": str(config.preprocess_config),
            "config_sha256": file_sha256(preprocess_path),
            "resolved": asdict(preprocess),
            "implementation": "src/preprocessing.py (CLAHE on LAB-L, then resize/crop/normalise)",
        },
        "augmentation": asdict(augment),
        "training": {
            **asdict(config),
            "loss": "cross_entropy",
            "early_stopping": {
                "metric": config.selection_metric,
                "patience": config.early_stopping_patience,
            },
        },
        "device": str(device),
        "environment": environment_record(device),
        "git_commit": git_commit(Path(args.repo_root)),
        "dataset_config_version": DATASET_CONFIG_VERSION,
        "manifests": {
            "split_dir": SPLIT_DIR_NAME,
            "cv_dir": CV_DIR_NAME,
            "train_path": str(train_path),
            "train_sha256": file_sha256(train_path),
            "validation_path": str(validation_path),
            "validation_sha256": file_sha256(validation_path),
            "final_test_path": str(split_dir / "final_test.csv"),
            "final_test_sha256": file_sha256(split_dir / "final_test.csv"),
        },
        "isolation": isolation,
        "smoke": args.smoke
        or args.max_train_samples is not None
        or args.max_validation_samples is not None,
        "subset": {"train": len(train_rows), "validation": len(validation_rows)},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reproducibility_note": "seeds set for python/numpy/torch and the loader generator; "
        "exact bitwise repeatability is not guaranteed on CUDA (cuDNN autotuning, atomics) "
        "or with AMP; the same seed reproduces the data order and initialisation",
    }
    record["config_hash"] = config_hash(record)

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    handler = _attach_file_logger(paths.log)
    try:
        return _train(
            args, config, preprocess, augment, device, train_rows, validation_rows, record, paths
        )
    except KeyboardInterrupt:
        write_status(paths, "INTERRUPTED", experiment_id=exp_id)
        raise
    except Exception as exc:
        write_status(paths, "FAILED", experiment_id=exp_id, error=repr(exc))
        raise
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def _train(
    args: RunArgs,
    config: CVConfig,
    preprocess: PreprocessConfig,
    augment: AugmentConfig,
    device: torch.device,
    train_rows: list[TrainRow],
    validation_rows: list[TrainRow],
    record: dict[str, Any],
    paths: Paths,
) -> dict[str, Any]:
    exp_id = record["experiment_id"]
    set_seed(args.seed)
    model = build_model(
        args.model_key, len(CANONICAL_CLASSES), pretrained=config.pretrained, dropout=config.dropout
    ).to(device)
    train_ds = ManifestImageDataset(
        train_rows, build_train_transform(preprocess, augment), args.repo_root
    )
    val_ds = ManifestImageDataset(validation_rows, build_eval_transform(preprocess), args.repo_root)
    train_loader = build_dataloader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    best_epoch: int | None = None
    start_epoch = 1
    stage = stage_for_epoch(1, config)
    set_backbone_trainable(model, stage == "full")
    optimizer = build_optimizer(model, config, backbone_frozen=stage == "head")
    scheduler = build_scheduler(
        optimizer, config, config.head_epochs if stage == "head" else config.epochs
    )
    scaler = build_scaler(device, config.amp)

    if args.resume and paths.latest.is_file():
        ck = torch.load(paths.latest, map_location="cpu", weights_only=False)
        for key in ("model_key", "fold", "data_arm", "seed"):
            if ck[key] != record[key]:
                raise ValueError(
                    f"cannot resume: checkpoint {key}={ck[key]!r} differs from {record[key]!r}"
                )
        if ck["config"]["config_hash"] != record["config_hash"]:
            raise ValueError("cannot resume: configuration hash differs from the checkpoint's")
        model.load_state_dict(ck["model_state"])
        start_epoch = ck["epoch"] + 1
        stage = stage_for_epoch(start_epoch, config)
        set_backbone_trainable(model, stage == "full")
        optimizer = build_optimizer(model, config, backbone_frozen=stage == "head")
        scheduler = build_scheduler(
            optimizer, config, config.head_epochs if stage == "head" else config.epochs
        )
        if ck["stage"] == stage:
            optimizer.load_state_dict(ck["optimizer_state"])
            if scheduler is not None and ck["scheduler_state"] is not None:
                scheduler.load_state_dict(ck["scheduler_state"])
        scaler.load_state_dict(ck["scaler_state"])
        history = list(ck["history"])
        best_metric, best_epoch = ck["best_metric"], ck["best_epoch"]
        _restore_rng_state(ck["rng"])
        logger.info("%s: resumed at epoch %d (stage %s)", exp_id, start_epoch, stage)
    else:
        paths.config.write_text(json.dumps(record, indent=2, default=str))

    write_status(paths, "RUNNING", experiment_id=exp_id, epoch=start_epoch - 1)
    logger.info(
        "%s: %s fold %d %s | train %d (synthetic %d) / val %d | %s | %s",
        exp_id,
        args.model_key,
        args.fold,
        args.data_arm,
        len(train_rows),
        sum(r.synthetic for r in train_rows),
        len(validation_rows),
        device,
        trainable_summary(model),
    )
    since_improvement = 0
    stopped_early = False
    for epoch in range(start_epoch, config.epochs + 1):
        new_stage = stage_for_epoch(epoch, config)
        if new_stage != stage:  # head -> full: unfreeze, rebuild optimiser/scheduler
            stage = new_stage
            set_backbone_trainable(model, True)
            optimizer = build_optimizer(model, config, backbone_frozen=False)
            scheduler = build_scheduler(optimizer, config, config.epochs - config.head_epochs)
            logger.info(
                "%s: stage %s from epoch %d — %s", exp_id, stage, epoch, trainable_summary(model)
            )
        rates = {g.get("name", f"group{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)}
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp=config.amp,
            backbone_frozen=stage == "head",
        )
        if scheduler is not None:
            scheduler.step()
        validation = run_validation(model, val_loader, device, list(CANONICAL_CLASSES))
        s = validation.summary
        candidate = getattr(s, config.selection_metric)
        improved = is_improvement(config.selection_metric, candidate, best_metric)
        if improved:
            best_metric, best_epoch, since_improvement = candidate, epoch, 0
        else:
            since_improvement += 1
        row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": round(train_stats["loss"], 6),
            "train_accuracy": round(train_stats["accuracy"], 6),
            "train_precision_macro": round(train_stats["precision_macro"], 6),
            "train_recall_macro": round(train_stats["recall_macro"], 6),
            "train_f1_macro": round(train_stats["f1_macro"], 6),
            "val_loss": round(s.loss, 6),
            "val_accuracy": round(s.accuracy, 6),
            "val_precision_macro": round(s.precision_macro, 6),
            "val_recall_macro": round(s.recall_macro, 6),
            "val_f1_macro": round(s.f1_macro, 6),
            "val_f1_weighted": round(s.f1_weighted, 6),
            "lr_head": rates.get("head"),
            "lr_backbone": rates.get("backbone"),
            "seconds": round(train_stats["seconds"] + s.seconds, 1),
            "improved": improved,
        }
        history.append(row)
        with paths.history.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(HISTORY_COLUMNS))
            writer.writeheader()
            writer.writerows(history)
        logger.info(
            "%s epoch %d/%d [%s] train loss %.4f f1 %.3f | val loss %.4f acc %.3f f1 %.3f%s",
            exp_id,
            epoch,
            config.epochs,
            stage,
            row["train_loss"],
            row["train_f1_macro"],
            row["val_loss"],
            row["val_accuracy"],
            row["val_f1_macro"],
            " *" if improved else "",
        )
        ckpt = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            stage=stage,
            record=record,
            history=history,
            best_metric=best_metric,
            best_epoch=best_epoch,
        )
        save_checkpoint(paths.latest, **ckpt)
        if improved:
            save_checkpoint(paths.best, **ckpt)
            _write_metrics(paths, validation, epoch, record)
        write_status(paths, "RUNNING", experiment_id=exp_id, epoch=epoch, best_epoch=best_epoch)
        if (
            config.early_stopping_patience is not None
            and since_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            logger.info(
                "%s: early stopping after %d epochs without improvement", exp_id, since_improvement
            )
            break

    summary = {
        "experiment_id": exp_id,
        "model_key": args.model_key,
        "fold": args.fold,
        "data_arm": args.data_arm,
        "seed": args.seed,
        "device": str(device),
        "epochs_run": len(history),
        "epochs_configured": config.epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        f"best_val_{config.selection_metric}": best_metric,
        "final_epoch_validation": (
            {k: v for k, v in history[-1].items() if k.startswith("val_")} if history else None
        ),
        "train_images": len(train_rows),
        "train_synthetic": sum(r.synthetic for r in train_rows),
        "validation_images": len(validation_rows),
        "smoke": record["smoke"],
        "note": "INFRASTRUCTURE SMOKE TEST — NOT A PERFORMANCE RESULT" if record["smoke"] else "",
        "config_hash": record["config_hash"],
        "manifests": record["manifests"],
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    paths.summary.write_text(json.dumps(summary, indent=2))
    write_status(
        paths, "COMPLETED", experiment_id=exp_id, epoch=len(history), best_epoch=best_epoch
    )
    return summary


def _write_metrics(paths: Paths, validation: Any, epoch: int, record: dict[str, Any]) -> None:
    m = validation.metrics
    payload = {
        "experiment_id": record["experiment_id"],
        "epoch": epoch,
        "split": "validation (fold)",
        "summary": asdict(validation.summary),
        "per_class": [asdict(c) for c in m.per_class],
        "confusion": validation.confusion.tolist(),
        "class_names": list(CANONICAL_CLASSES),
        "classification_report": classification_report(m),
    }
    paths.metrics.write_text(json.dumps(payload, indent=2))
    with paths.confusion.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted", *CANONICAL_CLASSES])
        for name, row in zip(CANONICAL_CLASSES, validation.confusion.tolist()):
            writer.writerow([name, *row])


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train one CV experiment (model x fold x data arm)."
    )
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-arm", required=True, choices=DATA_ARMS)
    parser.add_argument("--config", type=Path, default=Path("configs/cv_v2/default.json"))
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--out-root", type=Path, default=ROOT_DIR / EXPERIMENTS_DIR)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true", help="abort unless CUDA is selected")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="override the config's epochs")
    parser.add_argument("--max-train-samples", type=int, default=None, help="smoke runs only")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="smoke runs only")
    parser.add_argument(
        "--smoke", action="store_true", help="flag the run as an infrastructure smoke test"
    )
    a = parser.parse_args(argv)
    config_path = a.config if a.config.is_absolute() else Path(a.repo_root) / a.config
    summary = run_experiment(
        RunArgs(
            model_key=a.model,
            fold=a.fold,
            data_arm=a.data_arm,
            config_path=config_path,
            data_dir=a.data_dir,
            repo_root=a.repo_root,
            out_root=a.out_root,
            experiment_id=a.experiment_id,
            seed=a.seed,
            device=a.device,
            require_cuda=a.require_cuda,
            resume=a.resume,
            epochs_override=a.epochs,
            max_train_samples=a.max_train_samples,
            max_validation_samples=a.max_validation_samples,
            smoke=a.smoke,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
