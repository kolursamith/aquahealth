#!/usr/bin/env python
"""Phase 12 — verify an INTERRUPTED experiment's latest.pt BEFORE it is resumed.

    python scripts/verify_resume_checkpoint.py --model cnn_vit_lstm --fold 4 --data-arm with_gan \
        --config configs/cv_v2/default.json [--require-cuda]

Reads the checkpoint on the CPU (never trains) and compares its metadata with the
manifests and artefacts that the resumed run will use: model key, fold, data arm, seed,
dataset configuration version, train / validation manifest digests, this fold's GAN
provenance (synthetic manifest, GAN_MANIFEST.csv and generator digests), preprocessing
digest, configuration hash integrity, the optimizer / scheduler / scaler / RNG state and
the per-epoch history, plus the CUDA environment the checkpoint was written on and — with
--require-cuda — the CUDA runtime available now. Exit 1 on any mismatch, so the batch
driver / notebook stops instead of silently restarting. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.config import ROOT_DIR, SEED  # noqa: E402
from src.cv_runner import (  # noqa: E402
    DATA_ARMS,
    EXPERIMENTS_DIR,
    config_hash,
    experiment_id,
    gan_provenance,
    manifest_paths,
    read_status,
)
from src.manifest import file_sha256  # noqa: E402
from src.multi_dataset import DATASET_CONFIG_VERSION  # noqa: E402

REQUIRED_STATE = (
    "model_state",
    "optimizer_state",
    "scaler_state",
    "history",
    "rng",
    "epoch",
    "stage",
    "best_metric",
    "best_epoch",
    "config",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data-arm", choices=DATA_ARMS, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cv_v2/default.json"))
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    repo = Path(args.repo_root)
    data_dir = Path(args.data_dir) if args.data_dir else repo / "data"
    exp = experiment_id(args.model, args.fold, args.data_arm)
    out_dir = repo / EXPERIMENTS_DIR / exp
    latest = out_dir / "latest.pt"
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

    print(f"experiment {exp}: status.json = {read_status(out_dir)}")
    if not latest.is_file():
        check("latest.pt present", False, str(latest))
        return 1
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    check("checkpoint format", ck.get("format") == "cv_v3", str(ck.get("format")))
    for key, expected in (
        ("model_key", args.model),
        ("fold", args.fold),
        ("data_arm", args.data_arm),
        ("seed", SEED),
        ("dataset_config_version", DATASET_CONFIG_VERSION),
    ):
        check(key, ck.get(key) == expected, f"{ck.get(key)!r} (expected {expected!r})")
    for key in REQUIRED_STATE:
        check(f"state: {key}", key in ck, "missing" if key not in ck else "")
    if "scheduler_state" not in ck:
        check("state: scheduler_state", False, "missing")
    epoch = int(ck.get("epoch", 0))
    history = ck.get("history") or []
    check(
        "history length == epoch",
        len(history) == epoch,
        f"{len(history)} rows, epoch {epoch}, stage {ck.get('stage')}",
    )
    check(
        "best epoch <= epoch",
        ck.get("best_epoch") is not None and int(ck["best_epoch"]) <= epoch,
        f"best epoch {ck.get('best_epoch')} (val Macro-F1 {ck.get('best_metric')})",
    )

    # manifests the resumed run will read
    train_path, validation_path = manifest_paths(data_dir, args.fold, args.data_arm)
    digests = ck.get("manifest_sha256") or {}
    for name, path in (("train", train_path), ("validation", validation_path)):
        check(
            f"{name} manifest digest",
            path.is_file() and digests.get(name) == file_sha256(path),
            f"{path.name}: {str(digests.get(name))[:16]}",
        )
    if args.data_arm == "with_gan":
        gan = gan_provenance(data_dir, repo, args.fold)  # raises LeakageError on a foreign fold
        check(
            "GAN synthetic manifest digest (this fold)",
            digests.get("gan_synthetic") == gan["synthetic_manifest_sha256"],
            f"{gan['synthetic_rows']} synthetic rows",
        )
        check(
            "GAN_MANIFEST.csv digest",
            digests.get("gan_manifest") == gan["gan_manifest_sha256"],
            str(gan["gan_manifest_sha256"])[:16],
        )
        check(
            "GAN generator digest (fold run record)",
            ck.get("gan_generator_sha256") == gan["generator_sha256"],
            str(gan["generator_sha256"])[:16],
        )

    # preprocessing + configuration
    train_config = json.loads((repo / args.config).read_text())
    preprocess_path = repo / train_config["preprocess_config"]
    check(
        "preprocessing digest",
        ck.get("preprocessing_sha256") == file_sha256(preprocess_path),
        str(ck.get("preprocessing_sha256"))[:16],
    )
    record = ck.get("config") or {}
    check(
        "configuration hash integrity",
        record.get("config_hash") == ck.get("config_hash") == config_hash(record),
        str(ck.get("config_hash"))[:16],
    )
    training = record.get("training", {})
    check(
        "training configuration == requested config",
        all(training.get(k) == v for k, v in train_config.items() if k != "description"),
        f"epochs {training.get('epochs')}, {training.get('optimizer')}/{training.get('scheduler')}"
        f", amp {training.get('amp')}",
    )

    # environment
    env = ck.get("environment") or {}
    cuda = env.get("cuda") or {}
    on_cuda = bool(cuda.get("available")) and str(record.get("device", "")).startswith("cuda")
    env_detail = (
        f"device {record.get('device')} ({cuda.get('device_name')}), torch {env.get('torch')}, "
        f"cuda {cuda.get('version')}"
    )
    if args.require_cuda:
        check("checkpoint written on CUDA", on_cuda, env_detail)
        check("CUDA available now", torch.cuda.is_available(), torch.__version__)
    else:
        print(f"[info] checkpoint environment: {env_detail}; CUDA {'yes' if on_cuda else 'NO'}")

    ok = all(c[1] for c in checks)
    print(
        json.dumps(
            {
                "experiment_id": exp,
                "resume_from_epoch": epoch + 1,
                "checks": len(checks),
                "failed": [c[0] for c in checks if not c[1]],
                "git_commit_at_checkpoint": ck.get("git_commit"),
                "ok": ok,
            },
            indent=2,
        )
    )
    print("RESULT:", "RESUMABLE" if ok else "NOT RESUMABLE — do not restart; report")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
