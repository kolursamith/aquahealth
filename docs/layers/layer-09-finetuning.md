# Layer 9 — Fine-tuning

## 1. Objective

Controlled, staged transfer learning as *machinery*: an explicit
freeze/unfreeze API at backbone-block granularity, per-group learning
rates, and a schedule runner that chains stages through the existing
training engine with checkpoint/resume compatibility. Nothing here claims
that fine-tuning improves disease classification; all runs are on synthetic
fixtures.

## 2. Architecture

```
src/model.py
  backbone_block_count(model) → 9              (EfficientNet-B0 `features` blocks)
  set_trainable_blocks(model, last_n)          freeze all, unfreeze the last N blocks; head always trainable
  trainable_block_count(model)                 inverse; raises if the pattern is not a trailing suffix

src/train.py (extended)
  TrainConfig.backbone_learning_rate           optional, > 0
  build_optimizer                              AdamW param groups: "head" @ lr, "backbone" @ backbone_lr
  set_train_mode                               per block: frozen blocks eval(), trainable blocks train()
  model_spec / Checkpoint.build_model          record and restore `trainable_blocks`
  save_checkpoint(..., stage=) / Checkpoint.stage
  fit(..., stage=)                             stage name written into every checkpoint

src/finetune.py
  Stage(name, epochs, trainable_blocks, learning_rate, backbone_learning_rate)
  DEFAULT_SCHEDULE   head (0 blocks, 1e-3) → partial (3 blocks, 3e-4 / 3e-5) → full (all, 3e-4 / 1e-5)
  run_schedule(model, train_loader, stages, ..., val_loader, resume_from)
      each stage: apply_stage → fit(checkpoint_dir/<stage>) → load best (or last) weights → next stage
  python -m src.finetune --train-root … --val-root … --checkpoint-dir … [--resume <stage ckpt>]
```

Blocks and parameter counts (EfficientNet-B0): 0:928, 1:1.4k, 2:16.7k,
3:46.6k, 4:243k, 5:543k, 6:2.03M, 7:717k, 8:412k. "Last 3 blocks" = 6–8 ≈
3.2M of the 4.0M backbone parameters.

## 3. Files

**Created**: `src/finetune.py`, `tests/test_finetune.py` (30 tests), this document.
**Modified**: `src/model.py`, `src/train.py`, `tests/test_train.py` (contract updates + one regression test), `README.md`.

## 4. Dependencies

None new.

## 5. Implementation Decisions

**D1 — Freezing is a trailing suffix of blocks.** Fine-tuning conventionally
unfreezes the deepest, most task-specific layers first. `set_trainable_blocks`
encodes exactly that and `trainable_block_count` refuses any other pattern,
so a checkpoint's `trainable_blocks` integer fully describes the freeze
state and `Checkpoint.build_model` can restore it.

**D2 — Mode handling generalised to blocks.** Layer 7 kept a fully frozen
backbone in `eval()`. With partial unfreezing the same rule applies per
block: frozen blocks keep their BatchNorm running statistics
(`test_frozen_blocks_keep_batchnorm_stats_while_unfrozen_blocks_update`
shows frozen-block stats byte-identical while unfrozen-block stats move).

**D3 — Two optimiser groups, named.** `head` at the stage LR, `backbone` at
the backbone LR (absent when nothing in the backbone is trainable). A
10⁻³ vs 10⁻⁵ backbone LR moves the last block's weights >10× less after one
epoch — the LR actually reaches the parameters. Optimiser state is per
stage: a stage boundary is a new optimiser (group structure changes), so
resume is supported *within* a stage, and across stages via the schedule.

**D4 — Best-of-stage carries forward.** The weights entering stage *n+1*
are stage *n*'s `best.pt` (by the selection metric) when validation is
present, else `last.pt`. Verified by spying on `fit`: the state entering
"partial" equals "head"'s `best.pt` exactly.

**D5 — Default schedule is conventional, not tuned.** 5/10/15 epochs and
1e-3 → 3e-4/3e-5 → 3e-4/1e-5 are common starting points for fine-tuning an
ImageNet backbone on a small dataset. The CLI exposes epochs and the partial
block count; the module docstring states these are to be revisited with real
data.

**D6 — Stage name in the checkpoint.** Lets `run_schedule(resume_from=…)`
skip completed stages, resume the interrupted one, and run the rest.

## 6. Tests

`tests/test_finetune.py` — 30:

Freeze API — 9 blocks; exact trailing suffix for N ∈ {0,1,3,9} with parameter counts; out-of-range rejected; reversible both ways; non-suffix pattern rejected; manually frozen head re-enabled

Modes — block-granular train/eval; frozen-block BN stats fixed, unfrozen-block stats move

Optimiser — named groups with their LRs and weight decay and exact parameter counts; no backbone group when frozen; backbone LR defaults to head LR; non-positive backbone LR rejected; smaller backbone LR → >10× smaller weight movement

Gradient routing — blocks 0–5: no grads, byte-identical; blocks 6–8 and head: finite grads, changed

Checkpoints — `model_spec.trainable_blocks` and `stage` recorded; `build_model` restores partial freezing with matching trainable counts

Schedule — invalid `Stage` values; default schedule shape and LR ordering; empty/duplicate schedules rejected; `apply_stage` resolves ALL; `stage_train_config` carries LRs/seed/metric; **3-stage run**: stage order, block counts, epoch counts, per-stage `best.pt`/`last.pt` with stage name and freeze state, final checkpoint; **stage starts from previous best**; **resume from a stage checkpoint** reproduces the full run's validation losses; unknown stage rejected; **CLI** runs a tiny schedule end-to-end

`tests/test_train.py` — updated to the extended contract (block-level mode, `trainable_blocks` in spec) plus a regression test for B1.

## 7. Commands Executed

```bash
.venv/bin/python -m pytest -q tests/test_finetune.py
.venv/bin/python -m pytest -q tests/test_train.py tests/test_validation.py tests/test_classifier.py   # regression
.venv/bin/python -m pytest -q -rs -p no:cacheprovider
.venv/bin/ruff check . ; .venv/bin/black --check . ; .venv/bin/mypy src app scripts
.venv/bin/python -m src.finetune --train-root … --val-root … --checkpoint-dir … \
    --head-epochs 2 --partial-epochs 1 --full-epochs 1 --partial-blocks 3 --image-size 64 --workers 2   # smoke, mps
```

## 8. Results

| Check | Result |
|---|---|
| Tests | **349 passed, 0 skipped, 0 warnings** (was 319) |
| Lint / format / types | clean (27 source files) |
| Regression (Layers 0–8) | pass; Layer 7/8 tests updated only where the contract was deliberately extended |
| Smoke | 3-stage CLI on `mps` with 2 persistent workers: correct block counts/LRs logged, best carried forward, 6 checkpoints written |

Fixture losses from the smoke are not reported: 32 augmented images over 4
epochs carry no information about anything.

## 9. Bugs Found and Fixed

**B1 — Zero-epoch resume dropped the carry-forward.** Found by
`test_run_schedule_resumes_from_a_stage_checkpoint`: resuming a *completed*
stage returned a `TrainingResult` with no checkpoint paths (that call wrote
none), so `run_schedule` carried nothing forward and the next stage started
from `last.pt` weights instead of `best.pt`; validation losses diverged from
the uninterrupted run. Root cause in `fit`: on resume it did not
acknowledge checkpoints already on disk. Fix: on resume, `fit` pre-populates
`last_checkpoint`/`best_checkpoint` from existing files. Regression test
added to Layer 7's suite (`test_resume_at_target_epoch_trains_nothing_more`
now asserts the path is reported).

## 10. Known Limitations

1. **Schedule hyper-parameters are placeholders** (D5) until real data
   exists; no LR scheduler within a stage (constant LR per stage).
2. **Block granularity only.** Unfreezing part of a block, or layer-wise LR
   decay, is not offered.
3. **Cross-stage resume restarts the stage's optimiser** — by design, since
   the parameter groups differ; within-stage resume is exact.
4. **Carry-forward loads `best.pt` by re-reading the file** rather than
   keeping the best state in memory — simple and correct, ~16 MB per stage.

## 11. Acceptance Gate

**PASS** — implementation, imports, 30 new tests, integration (schedule +
resume + CLI), smoke, static checks, gradient/freeze proofs, B1 fixed with
regression, Layers 0–8 regression clean, documentation updated.

## 12. Next Layer Prerequisites

Layer 10 (Evaluation) must:

- replace the `src/evaluate.py` scaffold with checkpoint-driven evaluation built on `src/validation.py` + `src/metrics.py` (no metric re-implementation)
- load a checkpoint, rebuild model + preprocessing from it, evaluate a folder with deterministic inference, and export: metrics JSON, per-sample predictions CSV, confusion (raw + normalised), classification report, misclassified samples, and a confidence analysis (correct vs incorrect confidence, calibration bins)
- never touch a `test/` split for model selection; document that final test evaluation is a one-shot action
- re-run Layers 0–9 as regression
