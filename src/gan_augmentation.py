"""Training-fold-only GAN augmentation (Phase 10).

Architecture: a class-conditional DCGAN (cDCGAN — Radford et al. 2015 generator/
discriminator blocks with the label injected as in Mirza & Osindero 2014). No
supplied paper or project document specifies a GAN architecture (the project's
paper matrix records "cGAN — no support"; the PRD dropped cGAN), so the choice
is ours and is made for these reasons, all recorded in every run record:

- data: a training fold holds ~2.7k images over 8 classes (v3; ~310-500 per
  class) — too few for a class-per-model design; one conditional model per fold
  learns all classes from the fold's pooled data;
- compute: local Apple MPS / Colab T4, torch + torchvision only (no new
  dependency); DCGAN at 64x64 trains in minutes, not hours;
- honesty: at this data size a DCGAN yields low-detail images; they are an
  ablation arm (WITH vs WITHOUT), not a claim of realism. StyleGAN2-ADA would
  fit few-shot data better but needs an external dependency and far more compute.

Data isolation (enforced at runtime, see `FoldTrainingImages` and `generate`):
the GAN reads only fold_XX_train.csv; any image id that belongs to that fold's
validation file or to final_test.csv aborts; synthetic files are written only
under data/gan/fold_XX/train/<class>/ and are listed in a synthetic manifest
that names the fold, class, label, seed, generator checkpoint and config.

The GAN trains on plain resized RGB in [-1, 1] (no CLAHE): synthetic images are
saved as ordinary PNGs and later go through the SAME classifier preprocessing
(src/preprocessing.py, CLAHE included) as real images. The existing
src/augmentation.py pipeline is untouched; GAN is an additional arm.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import SEED
from src.dataset import load_image
from src.manifest import CANONICAL_CLASSES, file_sha256
from src.split_v2 import FINAL_TEST, FoldRow, SplitRow, fold_members
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

GAN_DIR_NAME = "gan"
ARCHITECTURE = "cDCGAN (class-conditional DCGAN)"
OPTIMIZER = "Adam(lr={lr}, betas=({beta1}, {beta2})) for generator and discriminator"
FORBIDDEN_PATH_PARTS = ("validation", "test", "final_test", "val")


@dataclass(frozen=True)
class GANConfig:
    image_size: int = 64  # power of two >= 16; generated images are upsampled by the classifier
    latent_dim: int = 100
    base_channels: int = 64
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    label_smoothing: float = 0.0
    seed: int = SEED
    # how many synthetic images per class; see `plan_synthetic_counts`
    target_per_class: int | None = None  # top every class up to this many real+synthetic
    synthetic_per_class: dict[str, int] = field(default_factory=dict)  # explicit override
    max_synthetic_ratio: float = 1.0  # never more synthetic than this x real, per class
    max_train_images: int | None = None  # smoke runs only; recorded

    def __post_init__(self) -> None:
        if self.image_size < 16 or self.image_size & (self.image_size - 1):
            raise ValueError("image_size must be a power of two >= 16")
        if self.epochs < 1 or self.batch_size < 1 or self.latent_dim < 1:
            raise ValueError("epochs, batch_size and latent_dim must be >= 1")
        if not 0 <= self.label_smoothing < 0.5:
            raise ValueError("label_smoothing must be in [0, 0.5)")
        for name, n in self.synthetic_per_class.items():
            if name not in CANONICAL_CLASSES or n < 0:
                raise ValueError(f"synthetic_per_class: bad entry {name!r}: {n}")


# --- networks ------------------------------------------------------------------------------


class ConditionalGenerator(nn.Module):
    """z (latent) ++ embedded label -> 4x4 -> ... -> image_size, tanh output."""

    def __init__(self, num_classes: int, config: GANConfig) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = config.latent_dim
        steps = int(math.log2(config.image_size)) - 2  # 4x4 -> image_size
        channels = config.base_channels * 2**steps
        self.embed = nn.Embedding(num_classes, config.latent_dim)
        self.project = nn.Sequential(
            nn.ConvTranspose2d(2 * config.latent_dim, channels, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )
        blocks: list[nn.Module] = []
        for _ in range(steps - 1):
            blocks += [
                nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels // 2),
                nn.ReLU(True),
            ]
            channels //= 2
        blocks += [nn.ConvTranspose2d(channels, 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.embed(labels)], dim=1)[:, :, None, None]
        return self.blocks(self.project(h))


class ConditionalDiscriminator(nn.Module):
    """image ++ label map (one channel per class) -> ... -> 1 logit."""

    def __init__(self, num_classes: int, config: GANConfig) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.image_size = config.image_size
        steps = int(math.log2(config.image_size)) - 2
        channels = config.base_channels
        blocks: list[nn.Module] = [
            nn.Conv2d(3 + num_classes, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
        ]
        for _ in range(steps - 1):
            blocks += [
                nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 2
        blocks += [nn.Conv2d(channels, 1, 4, 1, 0, bias=False)]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        maps = F.one_hot(labels, self.num_classes).to(images.dtype)[:, :, None, None]
        maps = maps.expand(-1, -1, images.shape[2], images.shape[3])
        return self.blocks(torch.cat([images, maps], dim=1)).flatten(1).squeeze(1)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


# --- data: training fold only ----------------------------------------------------------------


class FoldTrainingImages(Dataset[tuple[torch.Tensor, int]]):
    """The training rows of one fold, resized to the GAN resolution, in [-1, 1].

    `forbidden_ids` must contain the fold's validation ids and every final-test
    id; any overlap aborts (runtime leakage guard)."""

    def __init__(
        self,
        rows: list[FoldRow],
        *,
        forbidden_ids: set[str],
        repo_root: Path,
        image_size: int,
        max_images: int | None = None,
        seed: int = SEED,
        cache_in_memory: bool = False,
    ) -> None:
        leaked = sorted({r.image_id for r in rows} & forbidden_ids)
        if leaked:
            raise ValueError(f"validation/test images offered to the GAN: {leaked[:5]}")
        if not rows:
            raise ValueError("no training rows")
        rows = sorted(rows, key=lambda r: r.image_id)
        if max_images is not None and max_images < len(rows):
            rng = random.Random(seed)
            rows = sorted(rng.sample(rows, max_images), key=lambda r: r.image_id)
        self.rows = rows
        self.repo_root = Path(repo_root)
        self.image_size = image_size
        # The GAN only ever sees the fixed image_size x image_size resize, so the
        # decode+resize (the dominant cost on 4000x3000 JPEGs) can be done once per
        # fold instead of once per epoch. Same files, same tensors; recorded in the run.
        self.cache_in_memory = cache_in_memory
        self._cache: torch.Tensor | None = None
        if cache_in_memory:
            self._cache = torch.stack([self._decode(r) for r in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def _decode(self, row: FoldRow) -> torch.Tensor:
        image = load_image(self.repo_root / row.filepath)  # raises on corrupt files
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        tensor = self._cache[index] if self._cache is not None else self._decode(row)
        return tensor, row.label

    def class_counts(self) -> dict[str, int]:
        return dict(Counter(CANONICAL_CLASSES[r.label] for r in self.rows))


def forbidden_ids_for_fold(
    folds: list[FoldRow], fold: int, final_test: Iterable[SplitRow]
) -> set[str]:
    _, validation = fold_members(folds, fold)
    test_ids = {s.image_id for s in final_test}
    if any(s.split != FINAL_TEST for s in final_test):
        raise ValueError("final_test rows expected")
    return {r.image_id for r in validation} | test_ids


# --- synthetic amount policy ------------------------------------------------------------------


def plan_synthetic_counts(real_counts: dict[str, int], config: GANConfig) -> dict[str, int]:
    """How many synthetic images per class, from the ACTUAL class distribution.

    Priority: explicit `synthetic_per_class`; else top-up to `target_per_class`
    (default target = the largest real class, i.e. balance the fold); always
    capped at `max_synthetic_ratio` x real. Classes are never given the same
    number automatically — a class already at the target gets 0."""
    if config.synthetic_per_class:
        plan = {c: config.synthetic_per_class.get(c, 0) for c in real_counts}
    else:
        target = config.target_per_class or max(real_counts.values())
        plan = {c: max(0, target - n) for c, n in real_counts.items()}
    return {c: min(plan[c], int(config.max_synthetic_ratio * real_counts[c])) for c in real_counts}


# --- training ----------------------------------------------------------------------------------


def device_facts(device: torch.device) -> dict[str, str]:
    """GPU name and CUDA version for the run record ("" when not CUDA)."""
    if device.type == "cuda":
        return {
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_version": str(torch.version.cuda or ""),
        }
    return {"gpu_name": "", "cuda_version": ""}


@dataclass
class GANRunRecord:
    architecture: str
    fold: int
    config: dict[str, Any]
    seed: int
    device: str
    gpu_name: str  # torch.cuda.get_device_name, "" off CUDA
    cuda_version: str  # torch.version.cuda, "" off CUDA
    optimizer: str  # "Adam(lr, betas)" for both networks
    images_cached_in_memory: bool  # decode+resize once per fold instead of once per epoch
    real_training_images: int
    real_per_class: dict[str, int]
    forbidden_ids_checked: int
    epochs_run: int
    batches_run: int
    final_losses: dict[str, float]
    seconds: float
    checkpoint: str
    checkpoint_sha256: str
    train_manifest: str
    train_manifest_sha256: str
    torch_version: str
    started: str


def train_gan(
    dataset: FoldTrainingImages,
    *,
    fold: int,
    config: GANConfig,
    device: torch.device,
    out_dir: Path,
    train_manifest: Path,
    forbidden_count: int,
    num_workers: int = 0,
) -> tuple[Path, GANRunRecord]:
    """Standard non-saturating GAN training; writes generator.pt + gan_run.json."""
    set_seed(config.seed)
    num_classes = len(CANONICAL_CLASSES)
    generator = ConditionalGenerator(num_classes, config).to(device)
    discriminator = ConditionalDiscriminator(num_classes, config).to(device)
    generator.apply(_init_weights)
    discriminator.apply(_init_weights)
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2)
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        generator=torch.Generator().manual_seed(config.seed),
    )
    started = time.perf_counter()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    losses = {"generator": float("nan"), "discriminator": float("nan")}
    batches = 0
    real_target = 1.0 - config.label_smoothing
    for epoch in range(1, config.epochs + 1):
        sum_g = sum_d = 0.0
        n = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch = images.shape[0]
            # discriminator step
            z = torch.randn(batch, config.latent_dim, device=device)
            fake = generator(z, labels)
            logits_real = discriminator(images, labels)
            logits_fake = discriminator(fake.detach(), labels)
            loss_d = F.binary_cross_entropy_with_logits(
                logits_real, torch.full_like(logits_real, real_target)
            ) + F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            # generator step (non-saturating)
            logits_fake = discriminator(fake, labels)
            loss_g = F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake))
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()
            sum_g += loss_g.item() * batch
            sum_d += loss_d.item() * batch
            n += batch
            batches += 1
        losses = {"generator": sum_g / n, "discriminator": sum_d / n}
        logger.info(
            "fold %d epoch %d/%d  loss_G %.3f  loss_D %.3f",
            fold,
            epoch,
            config.epochs,
            *losses.values(),
        )
    seconds = time.perf_counter() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "generator.pt"
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "config": asdict(config),
            "class_names": list(CANONICAL_CLASSES),
            "fold": fold,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
        },
        checkpoint,
    )
    record = GANRunRecord(
        architecture=ARCHITECTURE,
        fold=fold,
        config=asdict(config),
        seed=config.seed,
        device=str(device),
        **device_facts(device),
        optimizer=OPTIMIZER.format(lr=config.learning_rate, beta1=config.beta1, beta2=config.beta2),
        images_cached_in_memory=dataset.cache_in_memory,
        real_training_images=len(dataset),
        real_per_class=dataset.class_counts(),
        forbidden_ids_checked=forbidden_count,
        epochs_run=config.epochs,
        batches_run=batches,
        final_losses=losses,
        seconds=round(seconds, 1),
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        train_manifest=str(train_manifest),
        train_manifest_sha256=file_sha256(train_manifest),
        torch_version=torch.__version__,
        started=started_at,
    )
    (out_dir / "gan_run.json").write_text(json.dumps(asdict(record), indent=2))
    return checkpoint, record


def load_generator(checkpoint: Path, device: torch.device) -> tuple[ConditionalGenerator, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("architecture") != ARCHITECTURE:
        raise ValueError(f"{checkpoint}: not a {ARCHITECTURE} checkpoint")
    if payload["class_names"] != list(CANONICAL_CLASSES):
        raise ValueError("checkpoint class list does not match CANONICAL_CLASSES")
    config = GANConfig(**payload["config"])
    generator = ConditionalGenerator(len(CANONICAL_CLASSES), config)
    generator.load_state_dict(payload["generator_state"])
    return generator.to(device).eval(), payload


# --- generation ----------------------------------------------------------------------------

SYNTHETIC_COLUMNS = (
    "image_id",
    "fold",
    "unified_class",
    "label",
    "filepath",
    "index",
    "seed",
    "generator_checkpoint",
    "generator_sha256",
    "architecture",
    "image_size",
    "generated_at",
    "synthetic",
)


def synthetic_root(gan_dir: Path, fold: int) -> Path:
    return Path(gan_dir) / f"fold_{fold:02d}" / "train"


def assert_train_only_path(path: Path) -> None:
    parts = {p.lower() for p in Path(path).parts}
    if parts & set(FORBIDDEN_PATH_PARTS) - {"train"} or "train" not in parts:
        raise ValueError(f"synthetic images may only be written under a train directory: {path}")


@torch.no_grad()
def generate(
    checkpoint: Path,
    *,
    fold: int,
    counts: dict[str, int],
    gan_dir: Path,
    repo_root: Path,
    device: torch.device,
    seed: int = SEED,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Write `counts[class]` PNGs per class under data/gan/fold_XX/train/<class>/ and
    return the synthetic manifest rows. Deterministic in `seed`."""
    generator, payload = load_generator(checkpoint, device)
    if payload["fold"] != fold:
        raise ValueError(f"checkpoint was trained on fold {payload['fold']}, not fold {fold}")
    root = synthetic_root(gan_dir, fold)
    assert_train_only_path(root)
    checkpoint_sha = file_sha256(checkpoint)
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    image_size = int(payload["config"]["image_size"])
    rows: list[dict[str, Any]] = []
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for name, n in sorted(counts.items()):
        if name not in CANONICAL_CLASSES:
            raise ValueError(f"unknown class {name!r}")
        label = CANONICAL_CLASSES.index(name)
        class_dir = root / name.replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        produced = 0
        while produced < n:
            batch = min(batch_size, n - produced)
            z = torch.randn(batch, generator.latent_dim, generator=rng).to(device)
            labels = torch.full((batch,), label, dtype=torch.long, device=device)
            images = ((generator(z, labels).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            for k in range(batch):
                index = produced + k
                path = class_dir / f"synthetic_{index:05d}.png"
                Image.fromarray(images[k].permute(1, 2, 0).cpu().numpy()).save(path)
                rel = (
                    path.relative_to(repo_root).as_posix()
                    if path.is_relative_to(repo_root)
                    else str(path)
                )
                rows.append(
                    {
                        "image_id": (
                            f"gan-f{fold:02d}-{hashlib.sha256(rel.encode()).hexdigest()[:12]}"
                        ),
                        "fold": fold,
                        "unified_class": name,
                        "label": label,
                        "filepath": rel,
                        "index": index,
                        "seed": seed,
                        "generator_checkpoint": str(checkpoint),
                        "generator_sha256": checkpoint_sha,
                        "architecture": ARCHITECTURE,
                        "image_size": image_size,
                        "generated_at": generated_at,
                        "synthetic": True,
                    }
                )
            produced += batch
    return rows


def write_synthetic_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SYNTHETIC_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def validate_synthetic_rows(rows: list[dict[str, Any]], fold: int, repo_root: Path) -> None:
    """Every generated file exists, sits under the fold's train directory, carries a
    valid label and belongs to this fold."""
    for r in rows:
        if int(r["fold"]) != fold:
            raise ValueError(f"{r['image_id']} belongs to fold {r['fold']}, not {fold}")
        if r["unified_class"] not in CANONICAL_CLASSES:
            raise ValueError(f"{r['image_id']}: invalid class {r['unified_class']!r}")
        if CANONICAL_CLASSES[int(r["label"])] != r["unified_class"]:
            raise ValueError(f"{r['image_id']}: label/class mismatch")
        path = Path(repo_root) / r["filepath"]
        assert_train_only_path(path)
        if f"fold_{fold:02d}" not in path.parts:
            raise ValueError(f"{path} is not inside fold_{fold:02d}")
        if not path.is_file():
            raise ValueError(f"synthetic file missing: {path}")


# --- ablation manifests ---------------------------------------------------------------------------

AUGMENTED_COLUMNS = tuple(f.name for f in fields(FoldRow)) + ("synthetic",)


def augmented_training_manifest(
    train_rows: list[FoldRow], synthetic: list[dict[str, Any]], fold: int
) -> list[dict[str, Any]]:
    """WITH-GAN training list for one fold: the real fold_XX_train rows (synthetic=False)
    followed by the synthetic rows (synthetic=True). The WITHOUT-GAN arm is
    fold_XX_train.csv itself."""
    if any(r.fold == fold for r in train_rows):
        raise ValueError("training rows contain the validation fold")
    out: list[dict[str, Any]] = [{**asdict(r), "synthetic": False} for r in train_rows]
    for s in synthetic:
        if int(s["fold"]) != fold:
            raise ValueError("synthetic row from another fold")
        out.append(
            {
                "image_id": s["image_id"],
                "source_dataset": "gan",
                "filepath": s["filepath"],
                "original_path": s["filepath"],
                "original_class": s["unified_class"],
                "unified_class": s["unified_class"],
                "label": int(s["label"]),
                "group_id": s["image_id"],
                "specimen_id": "",
                "stratum": f"{s['unified_class']}|gan",
                "fold": -1,  # never a validation sample
                "synthetic": True,
            }
        )
    return out


def write_augmented_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AUGMENTED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


# --- configuration file --------------------------------------------------------------------------


def load_gan_config(path: Path, **overrides: Any) -> GANConfig:
    """configs/gan_v2/<name>.json -> GANConfig. The file's `gan` block holds GANConfig
    fields only; `architecture` and `description` are informational and must match
    this module. CLI overrides win over the file."""
    spec = json.loads(Path(path).read_text())
    if spec.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError(f"{path}: architecture {spec.get('architecture')!r} != {ARCHITECTURE!r}")
    block = dict(spec.get("gan", {}))
    unknown = set(block) - {f.name for f in fields(GANConfig)}
    if unknown:
        raise ValueError(f"{path}: unknown GANConfig fields {sorted(unknown)}")
    block.update({k: v for k, v in overrides.items() if v is not None})
    return GANConfig(**block)


# --- cross-fold registry of generated data ------------------------------------------------------

REGISTRY_COLUMNS = (
    "fold",
    "architecture",
    "latent_dim",
    "resolution",
    "epochs",
    "batch_size",
    "optimizer",
    "learning_rate",
    "beta1",
    "beta2",
    "seed",
    "device",
    "gpu_name",
    "cuda_version",
    "real_training_images",
    "real_per_class",
    "generated_count",
    "generated_per_class",
    "source_training_fold",
    "source_training_fold_sha256",
    "forbidden_ids_checked",
    "generator_checkpoint",
    "generator_sha256",
    "synthetic_manifest",
    "synthetic_manifest_sha256",
    "with_gan_manifest",
    "with_gan_manifest_sha256",
    "torch_version",
    "started",
    "seconds",
    "smoke_run",
)


def registry_row(
    record: GANRunRecord,
    *,
    plan: dict[str, int],
    synthetic_manifest: Path,
    with_gan_manifest: Path,
    smoke_run: bool,
) -> dict[str, Any]:
    """One registry line per (fold, run): everything needed to trace every synthetic
    image back to the training fold it was generated from."""
    config = record.config
    return {
        "fold": record.fold,
        "architecture": record.architecture,
        "latent_dim": config["latent_dim"],
        "resolution": f"{config['image_size']}x{config['image_size']}",
        "epochs": record.epochs_run,
        "batch_size": config["batch_size"],
        "optimizer": record.optimizer,
        "learning_rate": config["learning_rate"],
        "beta1": config["beta1"],
        "beta2": config["beta2"],
        "seed": record.seed,
        "device": record.device,
        "gpu_name": record.gpu_name,
        "cuda_version": record.cuda_version,
        "real_training_images": record.real_training_images,
        "real_per_class": json.dumps(record.real_per_class, sort_keys=True),
        "generated_count": sum(plan.values()),
        "generated_per_class": json.dumps(plan, sort_keys=True),
        "source_training_fold": record.train_manifest,
        "source_training_fold_sha256": record.train_manifest_sha256,
        "forbidden_ids_checked": record.forbidden_ids_checked,
        "generator_checkpoint": record.checkpoint,
        "generator_sha256": record.checkpoint_sha256,
        "synthetic_manifest": str(synthetic_manifest),
        "synthetic_manifest_sha256": file_sha256(synthetic_manifest),
        "with_gan_manifest": str(with_gan_manifest),
        "with_gan_manifest_sha256": file_sha256(with_gan_manifest),
        "torch_version": record.torch_version,
        "started": record.started,
        "seconds": record.seconds,
        "smoke_run": smoke_run,
    }


def update_registry(path: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    """Append-or-replace the line for `row['fold']`; the file stays sorted by fold."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != REGISTRY_COLUMNS:
                raise ValueError(f"{path}: unexpected registry columns {reader.fieldnames}")
            rows = [r for r in reader if int(r["fold"]) != int(row["fold"])]
    rows.append({k: row[k] for k in REGISTRY_COLUMNS})
    rows.sort(key=lambda r: int(r["fold"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REGISTRY_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def read_registry(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REGISTRY_COLUMNS:
            raise ValueError(f"{path}: unexpected registry columns {reader.fieldnames}")
        return list(reader)


# --- consolidated GAN_MANIFEST.csv (every synthetic image of every fold) ------------------------

GAN_MANIFEST_NAME = "GAN_MANIFEST.csv"
GAN_MANIFEST_COLUMNS = (
    "synthetic_id",
    "fold",
    "class",
    "label",
    "generation_seed",
    "generation_config",  # JSON of the GANConfig the generator was trained with
    "config_file",  # configs/gan_v2/<name>.json named on the command line, or "(defaults)"
    "training_source",  # the fold's training manifest the GAN was trained on
    "training_source_sha256",
    "path",  # repository-relative path of the PNG
    "sha256",  # of the PNG
    "width",
    "height",
    "format",
    "source_dataset",  # "gan": the image is synthetic; it descends from training_source only
    "generator_checkpoint",
    "generator_sha256",
    "architecture",
    "device",
    "gpu_name",
    "generated_at",
    "index",
)


N_FOLDS = 10


def parse_folds(spec: str) -> list[int]:
    """'1-10', '1,2,5' or '3' -> sorted fold numbers within 1..N_FOLDS."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    folds = sorted(out)
    if not folds or folds[0] < 1 or folds[-1] > N_FOLDS:
        raise ValueError(f"folds must be within 1..{N_FOLDS}: {spec!r}")
    return folds


def completed_folds(gan_dir: Path) -> list[int]:
    """Folds under `gan_dir` whose run finished (generator, run record, synthetic manifest)."""
    out = []
    for path in sorted(Path(gan_dir).glob("fold_*")):
        if not path.is_dir() or not path.name[5:].isdigit():
            continue
        if all(
            (path / name).is_file()
            for name in ("generator.pt", "gan_run.json", "synthetic_manifest.csv")
        ):
            out.append(int(path.name[5:]))
    return out


def read_synthetic_manifest(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SYNTHETIC_COLUMNS:
            raise ValueError(f"{path}: unexpected synthetic manifest columns {reader.fieldnames}")
        return list(reader)


def _relative(path: str | Path, repo_root: Path) -> str:
    path = Path(path)
    return path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)


def build_gan_manifest(
    gan_dir: Path, repo_root: Path, *, folds: Iterable[int] | None = None, hash_images: bool = True
) -> list[dict[str, Any]]:
    """One row per synthetic image across the completed folds, joined from each fold's
    synthetic_manifest.csv + gan_run.json (+ summary.json for the config file name).
    Every row names the fold, class, seed, config, training source and generator digest
    the image came from."""
    gan_dir, repo_root = Path(gan_dir), Path(repo_root)
    wanted = sorted(folds) if folds is not None else completed_folds(gan_dir)
    rows: list[dict[str, Any]] = []
    for fold in wanted:
        out = gan_dir / f"fold_{fold:02d}"
        record = json.loads((out / "gan_run.json").read_text())
        if int(record["fold"]) != fold:
            raise ValueError(f"{out}: run record is for fold {record['fold']}")
        config_file = "(defaults)"
        if (out / "summary.json").is_file():
            config_file = json.loads((out / "summary.json").read_text()).get("config_file") or (
                "(defaults)"
            )
        config_json = json.dumps(record["config"], sort_keys=True)
        source = _relative(record["train_manifest"], repo_root)
        for s in read_synthetic_manifest(out / "synthetic_manifest.csv"):
            if int(s["fold"]) != fold:
                raise ValueError(f"{out}: synthetic row {s['image_id']} is for fold {s['fold']}")
            if s["generator_sha256"] != record["checkpoint_sha256"]:
                raise ValueError(f"{s['image_id']}: generator digest differs from the run record")
            png = repo_root / s["filepath"]
            rows.append(
                {
                    "synthetic_id": s["image_id"],
                    "fold": fold,
                    "class": s["unified_class"],
                    "label": int(s["label"]),
                    "generation_seed": int(s["seed"]),
                    "generation_config": config_json,
                    "config_file": config_file,
                    "training_source": source,
                    "training_source_sha256": record["train_manifest_sha256"],
                    "path": s["filepath"],
                    "sha256": file_sha256(png) if hash_images else "",
                    "width": int(s.get("image_size") or record["config"]["image_size"]),
                    "height": int(s.get("image_size") or record["config"]["image_size"]),
                    "format": "PNG",
                    "source_dataset": "gan",
                    "generator_checkpoint": _relative(s["generator_checkpoint"], repo_root),
                    "generator_sha256": s["generator_sha256"],
                    "architecture": s["architecture"],
                    "device": record["device"],
                    "gpu_name": record.get("gpu_name", ""),
                    "generated_at": s.get("generated_at") or record["started"],
                    "index": int(s["index"]),
                }
            )
    return rows


def write_gan_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GAN_MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def read_gan_manifest(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != GAN_MANIFEST_COLUMNS:
            raise ValueError(f"{path}: unexpected GAN manifest columns {reader.fieldnames}")
        return list(reader)


# --- verification of generated outputs (runs on Colab after generation and locally) -------------


def _check(report: dict[str, Any], name: str, ok: bool, detail: Any = "") -> None:
    report["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        report["errors"].append(f"{name}: {detail}")


def verify_gan_outputs(
    *,
    gan_dir: Path,
    repo_root: Path,
    audit_dir: Path,
    registry_path: Path,
    gan_manifest_path: Path | None = None,
    folds: Iterable[int] | None = None,
    images: str = "all",
    seed: int = SEED,
    sample_size: int = 200,
) -> dict[str, Any]:
    """Prove, from the files on disk, that the generated data is what the records say
    and that validation / test data never touched it. `images`: 'all' opens and hashes
    every PNG, 'sample' a seeded sample, 'none' only checks the records. Never modifies
    anything; returns a report with `ok`, `checks`, `errors` and `counts`."""
    from src.split_v2 import CV_DIR_NAME, SPLIT_DIR_NAME, read_folds, read_split

    if images not in ("all", "sample", "none"):
        raise ValueError("images must be all|sample|none")
    gan_dir, repo_root, audit_dir = Path(gan_dir), Path(repo_root), Path(audit_dir)
    report: dict[str, Any] = {"ok": False, "checks": [], "errors": [], "folds": [], "counts": {}}

    # the split / fold manifests still verify against their digests (raw-data protection)
    try:
        all_folds = read_folds(audit_dir / CV_DIR_NAME)
        split = read_split(audit_dir / SPLIT_DIR_NAME)
        _check(report, "split and fold manifests verify", True, f"{len(all_folds)} dev rows")
    except (OSError, ValueError) as exc:
        _check(report, "split and fold manifests verify", False, str(exc))
        return report
    final_test = [s for s in split if s.split == FINAL_TEST]
    test_ids = {s.image_id for s in final_test}
    real_ids = {r.image_id for r in all_folds} | test_ids

    wanted = sorted(folds) if folds is not None else completed_folds(gan_dir)
    _check(report, "completed folds", bool(wanted), wanted)
    if not wanted:
        return report
    registry = {int(r["fold"]): r for r in read_registry(registry_path)} if registry_path else {}
    rng = random.Random(seed)
    seen_ids: dict[str, int] = {}
    seen_paths: dict[str, int] = {}
    per_fold_rows: dict[int, list[dict[str, Any]]] = {}
    total_images = 0
    for fold in wanted:
        out = gan_dir / f"fold_{fold:02d}"
        tag = f"fold {fold:02d}"
        record = json.loads((out / "gan_run.json").read_text())
        checkpoint = out / "generator.pt"
        _check(report, f"{tag}: run record is for this fold", int(record["fold"]) == fold)
        _check(
            report,
            f"{tag}: generator digest matches run record",
            file_sha256(checkpoint) == record["checkpoint_sha256"],
            record["checkpoint_sha256"][:12],
        )
        source = audit_dir / CV_DIR_NAME / f"fold_{fold:02d}_train.csv"
        _check(
            report,
            f"{tag}: training source is fold_{fold:02d}_train.csv and unchanged",
            Path(record["train_manifest"]).name == source.name
            and file_sha256(source) == record["train_manifest_sha256"],
            record["train_manifest_sha256"][:12],
        )
        train_rows, validation_rows = fold_members(all_folds, fold)
        train_ids = {r.image_id for r in train_rows}
        val_ids = {r.image_id for r in validation_rows}
        forbidden = forbidden_ids_for_fold(all_folds, fold, final_test)
        _check(
            report,
            f"{tag}: forbidden ids checked == validation + final test",
            int(record["forbidden_ids_checked"]) == len(forbidden),
            f"{record['forbidden_ids_checked']} vs {len(forbidden)}",
        )
        smoke = record["config"].get("max_train_images") is not None
        _check(
            report,
            f"{tag}: real images seen == training fold rows" + (" (smoke subset)" if smoke else ""),
            int(record["real_training_images"]) == len(train_rows)
            or (smoke and int(record["real_training_images"]) <= len(train_rows)),
            f"{record['real_training_images']} vs {len(train_rows)}",
        )
        # synthetic manifest: rows, paths, classes, files
        synthetic = read_synthetic_manifest(out / "synthetic_manifest.csv")
        per_fold_rows[fold] = synthetic
        try:
            validate_synthetic_rows(synthetic, fold, repo_root)
            _check(
                report,
                f"{tag}: synthetic rows valid (fold, class, label, train path, file)",
                True,
                len(synthetic),
            )
        except ValueError as exc:
            _check(
                report,
                f"{tag}: synthetic rows valid (fold, class, label, train path, file)",
                False,
                str(exc),
            )
        _check(
            report,
            f"{tag}: every synthetic row names this fold's generator",
            all(r["generator_sha256"] == record["checkpoint_sha256"] for r in synthetic),
        )
        ids = [r["image_id"] for r in synthetic]
        paths = [r["filepath"] for r in synthetic]
        _check(report, f"{tag}: synthetic ids unique", len(set(ids)) == len(ids))
        _check(
            report, f"{tag}: synthetic ids disjoint from every real id", not (set(ids) & real_ids)
        )
        _check(
            report,
            f"{tag}: class directory matches class",
            all(
                Path(r["filepath"]).parent.name == r["unified_class"].replace(" ", "_")
                for r in synthetic
            ),
        )
        for i in ids:
            seen_ids.setdefault(i, fold)
        for p in paths:
            seen_paths.setdefault(p, fold)
        # nothing on disk under fold_XX that is not train/ + records
        stray = [
            p.relative_to(out).as_posix()
            for p in out.rglob("*.png")
            if "train" not in p.relative_to(out).parts
        ]
        _check(report, f"{tag}: no PNG outside fold_{fold:02d}/train", not stray, stray[:3])
        on_disk = (
            sorted(p for p in (out / "train").rglob("*.png")) if (out / "train").is_dir() else []
        )
        _check(
            report,
            f"{tag}: PNGs on disk == synthetic manifest rows",
            {p.resolve() for p in on_disk} == {(repo_root / p).resolve() for p in paths},
            f"{len(on_disk)} on disk, {len(paths)} rows",
        )
        # image validity
        if images != "none" and synthetic:
            to_open = synthetic
            if images == "sample":
                to_open = rng.sample(synthetic, min(sample_size, len(synthetic)))
            size = int(record["config"]["image_size"])
            bad = []
            for r in to_open:
                try:
                    with Image.open(repo_root / r["filepath"]) as image:
                        image.load()
                        if (
                            image.format != "PNG"
                            or image.mode != "RGB"
                            or image.size != (size, size)
                        ):
                            bad.append(r["filepath"])
                except (OSError, ValueError):
                    bad.append(r["filepath"])
            _check(
                report,
                f"{tag}: PNGs open as RGB {size}x{size} ({images}, {len(to_open)} files)",
                not bad,
                bad[:3],
            )
        # WITH-GAN manifest = exactly the real training rows + these synthetic rows
        with_gan = out / f"fold_{fold:02d}_train_gan.csv"
        with with_gan.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        real = {r["image_id"] for r in rows if r["synthetic"] == "False"}
        synth = {r["image_id"] for r in rows if r["synthetic"] == "True"}
        _check(
            report, f"{tag}: WITH-GAN real rows == fold_{fold:02d}_train.csv ids", real == train_ids
        )
        _check(
            report, f"{tag}: WITH-GAN synthetic rows == synthetic manifest ids", synth == set(ids)
        )
        _check(
            report,
            f"{tag}: no validation id in WITH-GAN manifest",
            not ({r["image_id"] for r in rows} & val_ids),
        )
        _check(
            report,
            f"{tag}: no final-test id in WITH-GAN manifest",
            not ({r["image_id"] for r in rows} & test_ids),
        )
        _check(
            report,
            f"{tag}: synthetic rows carry fold=-1 and data/gan/fold_{fold:02d}/train paths",
            all(
                int(r["fold"]) == -1
                and GAN_DIR_NAME in Path(r["filepath"]).parts
                and f"fold_{fold:02d}" in Path(r["filepath"]).parts
                and "train" in Path(r["filepath"]).parts
                for r in rows
                if r["synthetic"] == "True"
            ),
        )
        # registry line agrees with the files
        line = registry.get(fold)
        _check(report, f"{tag}: registry line present", line is not None)
        if line is not None:
            _check(
                report,
                f"{tag}: registry digests match generator / manifests / source",
                line["generator_sha256"] == file_sha256(checkpoint)
                and line["synthetic_manifest_sha256"] == file_sha256(out / "synthetic_manifest.csv")
                and line["with_gan_manifest_sha256"] == file_sha256(with_gan)
                and line["source_training_fold_sha256"] == record["train_manifest_sha256"],
            )
            _check(
                report,
                f"{tag}: registry count == synthetic rows",
                int(line["generated_count"]) == len(synthetic),
                f"{line['generated_count']} vs {len(synthetic)}",
            )
            _check(
                report,
                f"{tag}: registry device / seed / architecture == run record",
                line["device"] == record["device"]
                and int(line["seed"]) == int(record["seed"])
                and line["architecture"] == record["architecture"] == ARCHITECTURE,
                line["device"],
            )
        total_images += len(synthetic)
        report["folds"].append(
            {
                "fold": fold,
                "device": record["device"],
                "gpu_name": record.get("gpu_name", ""),
                "cuda_version": record.get("cuda_version", ""),
                "epochs": record["epochs_run"],
                "seconds": record["seconds"],
                "real_training_images": record["real_training_images"],
                "synthetic": len(synthetic),
                "per_class": dict(Counter(r["unified_class"] for r in synthetic)),
                "smoke": smoke,
            }
        )
    # cross-fold isolation
    dup_ids = [
        i
        for i, f in seen_ids.items()
        if any(i in {r["image_id"] for r in per_fold_rows[g]} for g in wanted if g != f)
    ]
    dup_paths = [
        p
        for p, f in seen_paths.items()
        if any(p in {r["filepath"] for r in per_fold_rows[g]} for g in wanted if g != f)
    ]
    _check(report, "no synthetic id shared between folds", not dup_ids, dup_ids[:3])
    _check(report, "no synthetic path shared between folds", not dup_paths, dup_paths[:3])
    _check(
        report,
        "class mapping: every label index names its class",
        all(
            CANONICAL_CLASSES[int(r["label"])] == r["unified_class"]
            for rows_ in per_fold_rows.values()
            for r in rows_
        ),
        f"{len(CANONICAL_CLASSES)} classes",
    )
    # consolidated manifest
    if gan_manifest_path is not None:
        gan_manifest_path = Path(gan_manifest_path)
        _check(
            report, "GAN_MANIFEST.csv present", gan_manifest_path.is_file(), str(gan_manifest_path)
        )
        if gan_manifest_path.is_file():
            manifest = read_gan_manifest(gan_manifest_path)
            listed = {
                (int(m["fold"]), m["synthetic_id"], m["path"])
                for m in manifest
                if int(m["fold"]) in wanted
            }
            expected = {
                (f, r["image_id"], r["filepath"])
                for f, rows_ in per_fold_rows.items()
                for r in rows_
            }
            _check(
                report,
                "GAN_MANIFEST.csv rows == union of per-fold synthetic manifests",
                listed == expected,
                f"{len(listed)} listed, {len(expected)} expected",
            )
            _check(
                report,
                "GAN_MANIFEST.csv classes / labels / sources / dimensions consistent",
                all(
                    CANONICAL_CLASSES[int(m["label"])] == m["class"]
                    and m["training_source"].endswith(f"fold_{int(m['fold']):02d}_train.csv")
                    and m["source_dataset"] == "gan"
                    and m["format"] == "PNG"
                    and int(m["width"]) == int(m["height"]) > 0
                    and m["generated_at"] != ""
                    for m in manifest
                ),
            )
            if images == "all":
                bad = [
                    m["path"]
                    for m in manifest
                    if int(m["fold"]) in wanted
                    and m["sha256"]
                    and file_sha256(repo_root / m["path"]) != m["sha256"]
                ]
                _check(report, "GAN_MANIFEST.csv image digests match files", not bad, bad[:3])
    report["counts"] = {
        "folds": len(wanted),
        "synthetic_images": total_images,
        "per_fold": {f["fold"]: f["synthetic"] for f in report["folds"]},
    }
    report["ok"] = not report["errors"]
    return report


def summarize_verification(report: dict[str, Any]) -> str:
    lines = [
        f"[{'ok' if c['ok'] else 'FAIL'}] {c['check']} — {c['detail']}" for c in report["checks"]
    ]
    lines.append("RESULT: " + ("VERIFIED" if report["ok"] else "NOT VERIFIED"))
    return "\n".join(lines)
