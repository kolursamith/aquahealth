#!/usr/bin/env python
"""Pre-training smoke test on the real data: one batch through the whole engine.

Verifies, before any long run is started, that the frozen manifest resolves,
the three preprocessing pipelines are the intended ones, the model has 8
outputs, and one forward → loss → backward → optimizer step produces finite
gradients on the selected device (with mixed precision if requested).
Exit code 0 means "safe to start full training"; anything else means stop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.augmentation import build_train_transform  # noqa: E402
from src.dataset import build_dataloader  # noqa: E402
from src.device import resolve_device, torch_report  # noqa: E402
from src.manifest import (  # noqa: E402
    CANONICAL_CLASSES,
    MANIFEST_PATH,
    build_split_dataset,
    read_manifest,
    validate_manifest,
    verify_manifest_digest,
)
from src.model import build_classifier, summarize  # noqa: E402
from src.preprocessing import CLAHEConfig, PreprocessConfig, build_eval_transform  # noqa: E402
from src.train import TrainConfig, autocast_dtype, build_optimizer, build_scaler  # noqa: E402
from src.validation import contains_random_transform, transform_stage_names  # noqa: E402

EXPECTED_STAGES_EVAL = ["CLAHE", "Resize", "CenterCrop", "PILToTensor", "ToDtype", "Normalize"]


def check(condition: bool, message: str, report: dict) -> None:
    report["checks"].append({"check": message, "ok": bool(condition)})
    print(("  [OK]   " if condition else "  [FAIL] ") + message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="write the report here too")
    args = parser.parse_args(argv)

    report: dict = {"checks": []}
    device = resolve_device(args.device)
    facts = torch_report(args.device)
    report["torch"] = facts.__dict__
    print(
        f"torch {facts.version} | device {device} | "
        f"cuda {facts.cuda_available} | mps {facts.mps_available}"
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report["gpu"] = {"name": props.name, "memory_gb": round(props.total_memory / 1024**3, 1)}
        print(f"GPU {props.name}, {report['gpu']['memory_gb']} GB")

    # --- manifest and splits ---
    digest = verify_manifest_digest(args.manifest)
    rows = read_manifest(args.manifest)
    counts = validate_manifest(rows)
    report["manifest"] = {"digest": digest, "rows": len(rows)}
    report["splits"] = {s: dict(c) for s, c in counts.items()}
    print(f"manifest {args.manifest.name} sha256 {digest[:12]}… rows {len(rows)}")
    for split, c in counts.items():
        print(
            f"  {split:<5} {sum(c.values()):>5}  "
            + "  ".join(f"{k[:12]}={v}" for k, v in c.items())
        )
    check(len(counts) == 3, "train/val/test present and disjoint (validate_manifest)", report)
    check(
        list(CANONICAL_CLASSES) == [c for c in CANONICAL_CLASSES],
        "class mapping is canonical 0..7",
        report,
    )

    preprocess = PreprocessConfig(clahe=None if args.no_clahe else CLAHEConfig())
    train_ds = build_split_dataset(
        "train", transform=build_train_transform(preprocess), manifest_path=args.manifest
    )
    val_ds = build_split_dataset(
        "val", transform=build_eval_transform(preprocess), manifest_path=args.manifest
    )
    test_ds = build_split_dataset(
        "test", transform=build_eval_transform(preprocess), manifest_path=args.manifest
    )
    missing = train_ds.missing_files() + val_ds.missing_files() + test_ds.missing_files()
    check(
        not missing,
        f"all {len(rows)} manifest files exist on disk (missing: {len(missing)})",
        report,
    )
    train_paths = {s.path for s in train_ds.samples}
    check(
        not (train_paths & {s.path for s in val_ds.samples})
        and not (train_paths & {s.path for s in test_ds.samples}),
        "no file overlap between train and val/test",
        report,
    )

    # --- preprocessing contract ---
    train_stages = transform_stage_names(train_ds.transform)
    eval_stages = transform_stage_names(val_ds.transform)
    expected_eval = EXPECTED_STAGES_EVAL if not args.no_clahe else EXPECTED_STAGES_EVAL[1:]
    print("train pipeline:", " -> ".join(train_stages))
    print("val/test pipeline:", " -> ".join(eval_stages))
    check(
        eval_stages == expected_eval,
        "val/test: CLAHE -> resize -> crop 224 -> tensor -> ImageNet normalize",
        report,
    )
    check(
        contains_random_transform(train_ds.transform),
        "train pipeline contains augmentation",
        report,
    )
    check(
        not contains_random_transform(val_ds.transform)
        and not contains_random_transform(test_ds.transform),
        "val/test pipelines contain NO augmentation",
        report,
    )
    check(
        train_stages[-3:] == eval_stages[-3:],
        "train and eval share the same tensor/normalize stage",
        report,
    )

    # --- model ---
    model = build_classifier(len(CANONICAL_CLASSES), freeze_backbone=True).to(device)
    summary = summarize(model)
    report["model"] = summary.__dict__
    check(
        summary.output_features == 8,
        f"EfficientNet-B0 head has 8 outputs ({summary.total_parameters:,} params)",
        report,
    )
    check(
        type(model).__name__ == "EfficientNet",
        "torchvision EfficientNet with ImageNet-pretrained backbone",
        report,
    )

    # --- one batch: forward, loss, backward, step ---
    loader = build_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        seed=0,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    t0 = time.perf_counter()
    images, labels = next(iter(loader))
    load_s = time.perf_counter() - t0
    report["batch"] = {
        "images": list(images.shape),
        "labels": list(labels.shape),
        "dtype": str(images.dtype),
        "load_seconds": round(load_s, 2),
    }
    print(
        f"batch images {tuple(images.shape)} {images.dtype}, "
        f"labels {tuple(labels.shape)} in {load_s:.1f}s"
    )
    check(
        images.shape[1:] == (3, 224, 224) and images.dtype == torch.float32,
        "batch tensor is (N, 3, 224, 224) float32",
        report,
    )
    check(int(labels.min()) >= 0 and int(labels.max()) < 8, "labels within 0..7", report)

    config = TrainConfig(learning_rate=1e-3, amp=args.amp)
    optimizer = build_optimizer(model, config)
    scaler = build_scaler(device, config.amp)
    model.train()
    images, labels = images.to(device), labels.to(device)
    t0 = time.perf_counter()
    with torch.autocast(device_type=device.type, dtype=autocast_dtype(device), enabled=config.amp):
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    scaler.scale(loss).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
    step_s = time.perf_counter() - t0
    report["step"] = {
        "loss": float(loss.detach()),
        "seconds": round(step_s, 3),
        "amp": config.amp,
        "logits_device": str(logits.device),
    }
    print(
        f"forward+backward+step {step_s:.2f}s, loss {float(loss.detach()):.4f}, "
        f"logits on {logits.device}"
    )
    check(logits.shape == (images.shape[0], 8), "logits shape (N, 8)", report)
    check(
        bool(torch.isfinite(loss).item()) and abs(float(loss.detach()) - 2.079) < 1.0,
        "loss finite and near ln(8) at init",
        report,
    )
    check(
        all(g is not None and torch.isfinite(g).all() for g in grads),
        "all trainable params have finite gradients",
        report,
    )
    check(logits.device.type == device.type, f"computation ran on {device.type}", report)
    if device.type == "cuda":
        check(
            torch.cuda.max_memory_allocated() > 0,
            f"GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
            report,
        )

    ok = all(c["ok"] for c in report["checks"])
    report["status"] = "PASS" if ok else "FAIL"
    print(f"\nSMOKE TEST {report['status']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
