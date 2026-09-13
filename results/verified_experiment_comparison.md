# Verified experiment comparison (Gate 2 — validation only)

Generated 2026-09-13 00:45 IST from `drive_export/experiments/` (copied from `MyDrive/AquaHealth/runs/experiments/`). Every checkpoint was loaded with `src.train.load_checkpoint`, rebuilt with `Checkpoint.build_model()`, run on a zero tensor (output shape (1, 8)), and re-evaluated with `python -m src.evaluate --checkpoint <best_model.pth> --split val --out-dir results/experiments/<ID>/eval_val` on this machine (Apple MPS). All six carry the canonical 8-class mapping, checkpoint format v2, 4,017,796 parameters. **TEST SPLIT ACCESSED = NO** (every command used `--split val`; `src.evaluate` treats `test` as held-out and was never invoked with it).

| Experiment | Best Stage | Best Epoch | Validation Accuracy | Validation Macro Precision | Validation Macro Recall | Validation Macro-F1 | Weighted F1 | ECE | Weakest Class F1 | Inference ms/image | Verification Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | partial | 9 | 0.9004 | 0.9018 | 0.901 | 0.9004 | 0.9002 | 0.0555 | Bacterial Red Disease 0.8293 | 9.004 | VERIFIED |
| EXP-002 | partial | 10 | 0.918 | 0.9207 | 0.9185 | 0.9182 | 0.918 | 0.0316 | EUS Disease 0.8594 | 8.719 | VERIFIED |
| EXP-003 | finetune | 25 | 0.8789 | 0.8798 | 0.8798 | 0.8783 | 0.8781 | 0.0862 | Bacterial Red Disease 0.7874 | 12.099 | VERIFIED |
| EXP-004 | full | 3 | 0.9336 | 0.9345 | 0.9347 | 0.933 | 0.9326 | 0.0215 | EUS Disease 0.8500 | 10.483 | VERIFIED |
| EXP-005 | full | 15 | 0.9336 | 0.9342 | 0.9343 | 0.9335 | 0.9329 | 0.0218 | EUS Disease 0.8455 | 10.941 | VERIFIED |
| EXP-006 | full | 14 | 0.9023 | 0.9047 | 0.9033 | 0.9014 | 0.9012 | 0.033 | Bacterial Red Disease 0.8095 | 9.183 | VERIFIED |

Full columns (preprocessing, augmentation, optimizer, LR, scheduler, strategy, parameters, checkpoint path/SHA-256/size, provenance, notes) are in `results/verified_experiment_comparison.csv`.

## Cross-checks

- For every experiment the three sources agree: `metrics.csv` best row, `best_model.pth` `best_metric`/stage/epoch, and the local `src.evaluate --split val` re-evaluation (differences ≤ 1e-3, caused by CUDA-AMP vs MPS float32 arithmetic).
- **0.9004** = EXP-001, stage partial, epoch 9 (`metrics.csv` line 15; checkpoint best_metric 0.9004223; local re-evaluation 0.9004).
- **0.904** = EXP-004, stage full, **epoch 2**: val accuracy 0.904297, val Macro-F1 0.903504 (`results/experiments/EXP-004/metrics.csv` line 18). It is an intermediate epoch of EXP-004, not EXP-004's best (0.9330 at epoch 3) and **not an EXP-001 value**.
- EXP-004 observed Colab value 0.9330 → artifact 0.932986 (VERIFIED). EXP-005 observed 0.9335 → artifact 0.933463 (VERIFIED).
- EXP-003 completed after all (finetune stage 25/25, `metrics.csv` 30 rows, best_model.pth at finetune/25): the earlier 'interrupted' status was based on a stale Drive listing. Its best F1 0.8783 is at the **last** epoch (still rising) — the P1 recipe at lr 1e-5 is simply too slow for 30 epochs.
- Checkpoint sizes differ because the optimizer state is stored (SGD momentum buffers < Adam/AdamW moments); model weights are identical in size.

## Summary

Best VERIFIED: **EXP-005** (0.9335) and **EXP-004** (0.9330) — within one validation image of each other (512 images ⇒ 0.2 % per image). Then EXP-002 (0.9182, CLAHE off), EXP-006 (0.9014), EXP-001 (0.9004), EXP-003 (0.8783). No OBSERVED_LOG_ONLY or INCOMPLETE entries remain.
