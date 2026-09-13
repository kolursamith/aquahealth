# AquaHealth AI — Experiment timeline (actual timestamps)

Times are IST (UTC+5:30). Colab log timestamps are UTC and are given in brackets. Sources: git
commit dates (`git log --date=iso`), `config.json` `started` fields, Colab log lines observed in the
session, Google Drive folder modification times, and the session record.

| When | Event | Evidence |
|---|---|---|
| 2026-09-11 21:26 | Repository skeleton created (`3be6f1e`) | git |
| 2026-09-12 14:33 | Foundation build merged to `main` (`75d77e7`, PR #1): Layers 0–11 with 401 tests; Layer 12 YOLO decision deferred to real data | git, `docs/layers/` |
| 15:08 | Five research-paper PDFs text-extracted for the technique matrix | scratchpad timestamps |
| 15:13 | `data/original/aquahealth` symlinked to the delivered "New Dataset" (read-only) | `ls -la` |
| 15:38 | `e057405`: dataset audit (3,503 files, 68 exact-dup groups, 334 near-dup groups, 8 label conflicts) and frozen 70/15/15 manifest (3,405 rows, sha256 b7d1fccb…), Colab notebook | git, `results/data_audit_report.md` |
| 16:25 | `b1eb926`: config-driven experiment runner, optimizer/scheduler options | git |
| 16:53 | `9828d05`: `scripts/smoke_train.py` (Gate 1 script; never reads the test split) | git |
| ~16:55–17:05 | Gate 1 on Colab T4: SMOKE TEST PASS (batch 64, AMP; values observed, no artifact) | session record |
| 17:07 [11:37:37 UTC] | **EXP-001 started** on Colab T4 (Python 3.13.15, torch 2.14.0+cu130) | `results/experiments/EXP-001/config.json` |
| ~17:34 [~12:03 UTC] | **EXP-001 finished**: 30 epochs in 1,549.8 s; best val Macro-F1 0.9004 (partial/9); benchmark 9.004 ms/img; val re-evaluation exported | `metrics.csv`, `inference_benchmark.json`, `eval_val/metrics.json` |
| 17:40–17:55 | EXP-001 artifacts copied into the repo; plots re-rendered; `analysis.md` written; paper matrix written; configs EXP-002…006 written; `AugmentConfig.saturation` added (+ test); 435 tests pass | `9447bb4` (17:55) |
| 17:56 | Cell with `subprocess.run` loop started on Colab → no visible output → interrupted at ~17:59 | session |
| ~18:00 | `%%bash` loop started → output buffered → interrupted at ~18:01 (EXP-002 dir removed) | session |
| 18:01 | Launcher cell with `!` lines started; its first line (`pkill … ; rm -rf …; EXP-002`) killed itself via `pkill -f` → EXP-002 not started | Colab output `^C` |
| 18:02 [12:32:40 UTC] | **EXP-003 run 1 started** (P1 two-phase Adam) | Colab log |
| 18:12 | `598273c` committed locally (Gate 7A audit, healthy-fish wording, compare script) — not pushed | git |
| 18:16 | Colab frontend shows "Reconnect": runtime disconnected while EXP-003 was at fine-tune epoch 13/25 (val F1 0.840, [12:47:40 UTC]) | screenshot, log |
| 18:17–20:15 | Kernel most likely kept running: EXP-003 run 1 wrote `metrics.csv` (inferred from the 21:13 refusal); old VM recycled at some point before ~20:15 | inference, see master doc §20 |
| 20:15 | New Colab VM allocated (uptime 13 min at 20:28 [14:58 UTC]); `/content` empty | `uptime` output |
| 20:29–20:36 | "Run all": clone `9447bb4`, pip install (~4 min), PIL `_Ink` ImportError → "Restart session and run all" | outputs |
| 20:37 | Drive mount attempt 1 timed out ("mount failed"); attempt 2 authorised by the user → `Mounted at /content/drive` | outputs |
| ~20:40–20:45 | DATASET.zip unzipped; 30 manifest tests pass; smoke test PASS (second Colab Gate 1 run) | outputs |
| 21:13 [15:43:46 UTC] | Launcher cell: EXP-003 refused (exit 2, directory already holds a completed run); **EXP-004 started** | Colab log |
| 21:38 [16:08:43 UTC] | **EXP-004 finished**: val Macro-F1 0.9330 (full/3), 24.7 min; **EXP-005 started** | Colab log |
| 22:02 [16:32:40 UTC] | **EXP-005 finished**: 0.9335 (full/15), 23.6 min; **EXP-006 started** | Colab log; Drive folder EXP-006 created 22:02 |
| 22:03 | EXP-002 re-run cell (`rm -rf` + run) and EXP-003 re-run cell queued behind the launcher cell | session |
| ~22:20 | Browser pane resized → Colab page reloaded to the GitHub copy of the notebook; frontend stuck in "Connecting… / Resuming execution"; kernel continues | screenshots |
| ~22:28 | EXP-006 exited (inferred); **EXP-002 started** (Drive folder created 22:28) | Drive listing |
| 22:43 | **EXP-003 run 3 started** (Drive folder re-created 22:43 after `rm -rf`) → EXP-002 ended ≤ 22:43 after ≤ 15 min (status UNVERIFIED) | Drive listing |
| 22:32–23:15 | This documentation audit (no training started, no experiment modified) | this file |
| next | Reconnect Colab; inspect EXP-002; copy all Drive artifacts into `results/experiments/`; Gate 6 (optional); Gate 8 selection; download checkpoint; **Gate 9 one-shot frozen test**; Gate 10 inference audit; final audit | plan |
