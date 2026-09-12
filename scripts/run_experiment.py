#!/usr/bin/env python
"""Run one controlled experiment on the frozen manifest and record it.

    results/experiments/<ID>/
        config.json               the exact configuration that ran (+ environment facts)
        metrics.csv               one row per epoch across all stages
        training_curves.png       loss / val Macro-F1 per epoch
        confusion_matrix.png      validation confusion of the selected checkpoint
        best_model.pth            the selected checkpoint (best val Macro-F1 over all stages)
        inference_benchmark.json  ms/image, parameter count
        experiment_report.md      human-readable summary
        checkpoints/<stage>/      last.pt / best.pt per stage (resumable)

Training uses src/finetune.run_schedule (Layer 9) on split=train, selects on
split=val, and never touches split=test.

    python scripts/run_experiment.py --id EXP-001 --config configs/exp001.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.augmentation import AugmentConfig, build_train_transform  # noqa: E402
from src.dataset import build_dataloader  # noqa: E402
from src.device import resolve_device, torch_report  # noqa: E402
from src.finetune import ALL_BLOCKS, OptimizerSettings, Stage, run_schedule  # noqa: E402
from src.manifest import CANONICAL_CLASSES, MANIFEST_PATH, build_split_dataset  # noqa: E402
from src.metrics import classification_report, normalize_confusion_matrix  # noqa: E402
from src.model import build_classifier, summarize  # noqa: E402
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform  # noqa: E402
from src.train import BEST_CHECKPOINT_NAME, load_checkpoint  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402
from src.validation import run_validation  # noqa: E402

logger = get_logger("experiment")
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "experiments"
METRIC_COLUMNS = [
    "stage",
    "epoch",
    "global_epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "lr_head",
    "lr_backbone",
    "epoch_seconds",
    "trainable_parameters",
]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {"description", "preprocess", "augment", "stages", "optimizer", "batch_size"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"config {path} is missing {sorted(missing)}")
    return config


def build_preprocess(spec: dict[str, Any]) -> PreprocessConfig:
    clahe = spec.get("clahe")
    return PreprocessConfig(
        image_size=spec.get("image_size", 224),
        resize_size=spec.get("resize_size", 256),
        resize_mode=spec.get("resize_mode", "crop"),
        clahe=(
            CLAHEConfig(**clahe) if isinstance(clahe, dict) else (CLAHEConfig() if clahe else None)
        ),
    )


def build_stages(spec: list[dict[str, Any]]) -> tuple[Stage, ...]:
    stages = []
    for item in spec:
        blocks = item["trainable_blocks"]
        stages.append(
            Stage(
                item["name"],
                item["epochs"],
                ALL_BLOCKS if blocks == "all" else int(blocks),
                item["learning_rate"],
                item.get("backbone_learning_rate"),
            )
        )
    return tuple(stages)


def benchmark_inference(model, dataset, device: torch.device, n: int = 100) -> dict[str, Any]:
    """Single-image latency (batch 1) after warm-up; preprocessing excluded."""
    model.eval()
    images = [dataset[i][0].unsqueeze(0).to(device) for i in range(min(n, len(dataset)))]
    with torch.inference_mode():
        for image in images[:10]:
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        started = time.perf_counter()
        for image in images:
            model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - started
    summary = summarize(model)
    return {
        "device": str(device),
        "images": len(images),
        "ms_per_image_batch1": round(1000 * elapsed / len(images), 3),
        "total_parameters": summary.total_parameters,
        "trainable_parameters": summary.trainable_parameters,
    }


def plot_curves(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [r["global_epoch"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="train loss")
    axes[0].plot(epochs, [r["val_loss"] for r in rows], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [r["val_f1_macro"] for r in rows], label="val Macro-F1")
    axes[1].plot(epochs, [r["val_accuracy"] for r in rows], label="val accuracy")
    axes[1].plot(epochs, [r["train_accuracy"] for r in rows], label="train acc (running)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    boundaries = [
        r["global_epoch"] for i, r in enumerate(rows) if i and r["stage"] != rows[i - 1]["stage"]
    ]
    for ax in axes:
        for b in boundaries:
            ax.axvline(b - 0.5, color="grey", linestyle=":")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(
    matrix: torch.Tensor, class_names: list[str], path: Path, normalized: bool
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = normalize_confusion_matrix(matrix).numpy() if normalized else matrix.numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = data[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}" if normalized else str(int(value)),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > data.max() / 2 else "black",
            )
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None, help="stage checkpoint to resume")
    parser.add_argument("--amp", action="store_true", help="override config: mixed precision on")
    parser.add_argument("--batch-size", type=int, default=None, help="override config batch size")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = args.out_root / args.id
    if out_dir.exists() and args.resume is None and any(out_dir.glob("metrics.csv")):
        print(f"{out_dir} already holds a completed run; choose a new --id or --resume")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    preprocess = build_preprocess(config["preprocess"])
    augment = AugmentConfig(**config["augment"])
    stages = build_stages(config["stages"])
    settings = OptimizerSettings(**{**config["optimizer"], **({"amp": True} if args.amp else {})})
    batch_size = int(args.batch_size or config["batch_size"])
    set_seed(args.seed)

    train_ds = build_split_dataset(
        "train", transform=build_train_transform(preprocess, augment), manifest_path=args.manifest
    )
    val_ds = build_split_dataset(
        "val", transform=build_eval_transform(preprocess), manifest_path=args.manifest
    )
    train_loader = build_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=args.seed,
        num_workers=args.workers,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=batch_size,
        num_workers=args.workers,
        persistent_workers=True,
        pin_memory=device.type == "cuda",
    )

    resolved = {
        "id": args.id,
        "description": config["description"],
        "preprocess": asdict(preprocess),
        "augment": asdict(augment),
        "stages": [asdict(s) for s in stages],
        "optimizer": asdict(settings),
        "batch_size": batch_size,
        "workers": args.workers,
        "seed": args.seed,
        "selection_metric": "f1_macro",
        "manifest": str(args.manifest),
        "manifest_sha256": (args.manifest.with_suffix(".sha256")).read_text().split()[0],
        "train_images": len(train_ds),
        "val_images": len(val_ds),
        "class_names": list(CANONICAL_CLASSES),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": asdict(torch_report(args.device)),
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "config.json").write_text(json.dumps(resolved, indent=2, default=str))
    logger.info("%s: %s", args.id, config["description"])

    checkpoint = load_checkpoint(args.resume) if args.resume else None
    model = checkpoint.build_model() if checkpoint else build_classifier(len(CANONICAL_CLASSES))
    started = time.perf_counter()
    schedule = run_schedule(
        model,
        train_loader,
        stages,
        class_names=list(CANONICAL_CLASSES),
        preprocess=preprocess,
        device=device,
        checkpoint_dir=out_dir / "checkpoints",
        val_loader=val_loader,
        seed=args.seed,
        selection_metric="f1_macro",
        resume_from=checkpoint,
        settings=settings,
    )
    train_seconds = time.perf_counter() - started

    # --- metrics.csv across stages ---
    rows: list[dict[str, Any]] = []
    global_epoch = 0
    for stage_result in schedule.stages:
        trainable = None
        for h in stage_result.result.history:
            global_epoch += 1
            v = h.validation
            rows.append(
                {
                    "stage": stage_result.stage.name,
                    "epoch": h.epoch,
                    "global_epoch": global_epoch,
                    "train_loss": h.loss,
                    "train_accuracy": h.train_accuracy,
                    "val_loss": v.loss if v else None,
                    "val_accuracy": v.accuracy if v else None,
                    "val_precision_macro": v.precision_macro if v else None,
                    "val_recall_macro": v.recall_macro if v else None,
                    "val_f1_macro": v.f1_macro if v else None,
                    "lr_head": h.lr_head,
                    "lr_backbone": h.lr_backbone,
                    "epoch_seconds": h.seconds,
                    "trainable_parameters": trainable,
                }
            )
    # trainable count per stage from the stage checkpoints
    for stage_result in schedule.stages:
        ck = load_checkpoint(stage_result.checkpoint_dir / "last.pt")
        count = summarize(ck.build_model()).trainable_parameters
        for r in rows:
            if r["stage"] == stage_result.stage.name:
                r["trainable_parameters"] = count
    with (out_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # --- best checkpoint over all stages by val Macro-F1 ---
    candidates = [
        (s.result.best_metric, s.stage.name, s.checkpoint_dir / BEST_CHECKPOINT_NAME)
        for s in schedule.stages
        if s.result.best_metric is not None
    ]
    best_f1, best_stage, best_path = max(candidates, key=lambda c: c[0])
    shutil.copy2(best_path, out_dir / "best_model.pth")
    best = load_checkpoint(out_dir / "best_model.pth")
    best_model = best.build_model().to(device).eval()
    validation = run_validation(best_model, val_loader, device, list(CANONICAL_CLASSES))
    plot_curves(rows, out_dir / "training_curves.png")
    plot_confusion(
        validation.confusion, list(CANONICAL_CLASSES), out_dir / "confusion_matrix.png", False
    )
    plot_confusion(
        validation.confusion,
        list(CANONICAL_CLASSES),
        out_dir / "confusion_matrix_normalized.png",
        True,
    )
    bench = benchmark_inference(best_model, val_ds, device)
    bench.update(
        {
            "best_stage": best_stage,
            "best_epoch": best.epoch,
            "train_seconds": round(train_seconds, 1),
        }
    )
    (out_dir / "inference_benchmark.json").write_text(json.dumps(bench, indent=2))

    m = validation.metrics
    per_class = "\n".join(
        f"| {c.name} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.support} |"
        for c in m.per_class
    )
    report = f"""# {args.id} — {config["description"]}

Split: frozen manifest `{args.manifest.name}` (sha256 {resolved["manifest_sha256"][:12]}…),
train {len(train_ds)} / val {len(val_ds)}. The frozen test split was not read.

## Configuration
- preprocess: {json.dumps(resolved["preprocess"])}
- augmentation: {json.dumps(resolved["augment"])}
- stages: {json.dumps(resolved["stages"])}
- optimizer: {json.dumps(resolved["optimizer"])}
- batch size {batch_size}, seed {args.seed}, device {device}, selection metric val Macro-F1

## Selected checkpoint
Stage **{best_stage}**, epoch {best.epoch} (best validation Macro-F1 over all stages)
→ `best_model.pth`

## Validation results (selected checkpoint)
| metric | value |
|---|---|
| accuracy | {m.accuracy:.4f} |
| macro precision | {m.precision_macro:.4f} |
| macro recall | {m.recall_macro:.4f} |
| **macro F1** | **{m.f1_macro:.4f}** |
| weighted F1 | {m.f1_weighted:.4f} |
| val loss | {validation.summary.loss:.4f} |

| class | precision | recall | F1 | support |
|---|---|---|---|---|
{per_class}

```
{classification_report(m)}
```

## Efficiency
- parameters: {bench["total_parameters"]:,} total
- inference: {bench["ms_per_image_batch1"]} ms/image (batch 1, {device}, preprocessing excluded)
- training wall time: {train_seconds / 60:.1f} min for {global_epoch} epochs

## Artifacts
`config.json`, `metrics.csv`, `training_curves.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `best_model.pth`, `inference_benchmark.json`,
`checkpoints/<stage>/`
"""
    (out_dir / "experiment_report.md").write_text(report)
    logger.info(
        "%s done: val macro-F1 %.4f (stage %s, epoch %d) in %.1f min",
        args.id,
        best_f1,
        best_stage,
        best.epoch,
        train_seconds / 60,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
