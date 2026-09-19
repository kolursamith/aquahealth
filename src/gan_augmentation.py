"""Training-fold-only GAN augmentation (Phase 10).

Architecture: a class-conditional DCGAN (cDCGAN — Radford et al. 2015 generator/
discriminator blocks with the label injected as in Mirza & Osindero 2014). No
supplied paper or project document specifies a GAN architecture (the project's
paper matrix records "cGAN — no support"; the PRD dropped cGAN), so the choice
is ours and is made for these reasons, all recorded in every run record:

- data: a training fold holds ~3.8-4.3k images over 8 classes (340-1,000 per
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = load_image(self.repo_root / row.filepath)  # raises on corrupt files
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous(), row.label

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


@dataclass
class GANRunRecord:
    architecture: str
    fold: int
    config: dict[str, Any]
    seed: int
    device: str
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
