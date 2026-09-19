#!/usr/bin/env python
"""Phase 12 — train ONE cross-validation experiment: model x fold x data arm.

    python scripts/run_cv_experiment.py --model cnn_vit_lstm --fold 1 --data-arm without_gan \
        --config configs/cv_v2/default.json --require-cuda [--resume]

Models: efficientnet_b0 (the historical baseline architecture) and the five hybrids
(cnn_vit_lstm, yolo_efficientnet, cnn_bilstm, resnet_attention, yolo_transformer),
all through src/model_factory.build_model. Data arms: without_gan
(data/audit/cv_v2/fold_XX_train.csv) / with_gan (data/gan/fold_XX/fold_XX_train_gan.csv);
validation is always data/audit/cv_v2/fold_XX_validation.csv. The frozen
final_test.csv is read for ids only, to enforce the isolation guards.

Output: results/v2/experiments/<experiment_id>/ (config.json, history.csv,
metrics.json, confusion_matrix.csv, best.pt, latest.pt, run_summary.json,
status.json, logs/). --resume continues from latest.pt; a COMPLETED experiment
is never retrained. --require-cuda aborts instead of falling back to CPU/MPS.
Intended to run on Google Colab (see docs/PHASE_12_COLAB_SETUP.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cv_runner import cli_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli_main())
