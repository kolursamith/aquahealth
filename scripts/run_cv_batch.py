#!/usr/bin/env python
"""Phase 12 — run a controlled batch of CV experiments SEQUENTIALLY through the existing
per-experiment runner, resumably, with every gate re-checked before each run.

    python scripts/run_cv_batch.py --models cnn_vit_lstm --folds 1-10 --data-arm with_gan \
        --config configs/cv_v2/default.json --require-cuda
    python scripts/run_cv_batch.py --models cnn_vit_lstm,yolo_efficientnet,cnn_bilstm,\
resnet_attention,yolo_transformer --folds 1-10 --data-arm with_gan --require-cuda

For each (model, fold), in order:

  1. COMPLETED (status + every output file present)  -> skipped, never rerun.
  2. gates: scripts/colab_preflight.py --fold F --data-arm A --model M [--require-cuda]
     (manifests exist and resolve, id/group isolation, final-test ids absent, digests,
     CUDA/GPU, model (N,3,224,224) -> (N,8) logits) and, for with_gan,
     scripts/verify_gan_outputs.py --folds F (this fold's generator digest, training
     source, synthetic rows, WITH-GAN manifest = real + this fold's synthetic only).
     A failing gate marks the experiment FAILED with the gate's output and moves on.
  3. scripts/run_cv_experiment.py ... [--resume when a latest.pt exists]. A failed run
     leaves status.json = FAILED (with the error) and the batch continues.
  4. src/fit_analysis.write_fit_analysis (fit_analysis.json / .md) for a COMPLETED run.
  5. results/v2/experiment_matrix.csv statuses refreshed.

After each model: scripts/summarize_cv_results.py --model M --data-arm A. After the
batch: --all-models report when --report is given. Nothing here trains; the frozen
final test is never read. One batch_log.jsonl line per experiment under
results/v2/batches/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT_DIR  # noqa: E402
from src.cv_runner import (  # noqa: E402
    DATA_ARMS,
    EXPERIMENTS_DIR,
    Paths,
    experiment_id,
    is_completed,
    read_status,
    write_status,
)
from src.fit_analysis import write_fit_analysis  # noqa: E402
from src.gan_augmentation import parse_folds  # noqa: E402
from src.model_factory import MODEL_KEYS  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger("cv_batch")
SCRIPTS = Path(__file__).resolve().parent


def run(cmd: list[str], log: Path) -> int:
    """Run a child process, streaming its output to stdout and appending it to `log`."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", required=True, help="comma-separated model keys, in order")
    parser.add_argument("--folds", default="1-10")
    parser.add_argument("--data-arm", choices=DATA_ARMS, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cv_v2/default.json"))
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--device", default=None, help="forwarded to the runner (tests: cpu)")
    parser.add_argument("--report", action="store_true", help="--all-models report at the end")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_KEYS]
    if unknown:
        parser.error(f"unknown models {unknown}; choose from {list(MODEL_KEYS)}")
    folds = parse_folds(args.folds)
    repo = Path(args.repo_root)
    data_dir = Path(args.data_dir) if args.data_dir else repo / "data"
    experiments = repo / EXPERIMENTS_DIR
    batch_dir = repo / "results" / "v2" / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    batch_log = batch_dir / f"batch_{args.data_arm}_{stamp}.jsonl"
    console = batch_dir / f"batch_{args.data_arm}_{stamp}.console.log"
    py = sys.executable
    common = ["--repo-root", str(repo), "--data-dir", str(data_dir)]
    cuda = ["--require-cuda"] if args.require_cuda else []
    outcomes: list[dict] = []

    def record(**entry) -> None:
        entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        outcomes.append(entry)
        with batch_log.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

    for model in models:
        for fold in folds:
            exp = experiment_id(model, fold, args.data_arm)
            out_dir = experiments / exp
            if is_completed(out_dir):
                logger.info("%s: COMPLETED — skipped", exp)
                if not (out_dir / "fit_analysis.json").is_file():
                    write_fit_analysis(out_dir)
                record(experiment_id=exp, outcome="skipped_completed")
                continue
            t0 = time.perf_counter()
            # gates
            gate = [
                py,
                str(SCRIPTS / "colab_preflight.py"),
                "--fold",
                str(fold),
                "--data-arm",
                args.data_arm,
                "--model",
                model,
                *common,
                *cuda,
            ]
            rc = run(gate, console)
            if rc == 0 and args.data_arm == "with_gan":
                rc = run(
                    [
                        py,
                        str(SCRIPTS / "verify_gan_outputs.py"),
                        "--folds",
                        str(fold),
                        "--images",
                        "none",
                        *common,
                    ],
                    console,
                )
            if rc != 0:
                logger.error("%s: pre-training gate failed (exit %d) — not trained", exp, rc)
                write_status(
                    Paths(out_dir),
                    "FAILED",
                    experiment_id=exp,
                    error=f"pre-training gate failed (exit {rc}); see {console.name}",
                )
                record(experiment_id=exp, outcome="gate_failed", exit_code=rc)
                if args.stop_on_failure:
                    break
                continue
            # train (resume when a checkpoint exists)
            resume = (out_dir / "latest.pt").is_file()
            cmd = [
                py,
                str(SCRIPTS / "run_cv_experiment.py"),
                "--model",
                model,
                "--fold",
                str(fold),
                "--data-arm",
                args.data_arm,
                "--config",
                str(args.config),
                "--out-root",
                str(experiments),
                *common,
                *cuda,
            ]
            if args.device:
                cmd += ["--device", args.device]
            if resume:
                cmd.append("--resume")
                logger.info("%s: resuming from latest.pt", exp)
            rc = run(cmd, console)
            status = read_status(out_dir)
            if rc == 0 and is_completed(out_dir):
                analysis = write_fit_analysis(out_dir)
                record(
                    experiment_id=exp,
                    outcome="completed",
                    resumed=resume,
                    seconds=round(time.perf_counter() - t0, 1),
                    best_epoch=analysis["best_epoch"],
                    val_f1_macro=analysis["best"]["val_f1_macro"],
                    val_accuracy=analysis["best"]["val_accuracy"],
                    verdict=analysis["verdict"],
                    unstable=analysis["unstable"],
                )
            else:
                logger.error("%s: runner exit %d, status %s", exp, rc, status)
                record(
                    experiment_id=exp,
                    outcome="failed",
                    exit_code=rc,
                    status=status,
                    seconds=round(time.perf_counter() - t0, 1),
                )
                if args.stop_on_failure:
                    break
            run(
                [
                    py,
                    str(SCRIPTS / "build_experiment_matrix.py"),
                    "--refresh",
                    "--repo-root",
                    str(repo),
                ],
                console,
            )
        run(
            [
                py,
                str(SCRIPTS / "summarize_cv_results.py"),
                "--model",
                model,
                "--data-arm",
                args.data_arm,
                "--repo-root",
                str(repo),
            ],
            console,
        )
    if args.report:
        run(
            [
                py,
                str(SCRIPTS / "summarize_cv_results.py"),
                "--all-models",
                "--data-arm",
                args.data_arm,
                "--repo-root",
                str(repo),
            ],
            console,
        )
    counts = {}
    for o in outcomes:
        counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
    print(json.dumps({"batch_log": str(batch_log), "outcomes": counts}, indent=2))
    return 0 if counts.get("failed", 0) + counts.get("gate_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
