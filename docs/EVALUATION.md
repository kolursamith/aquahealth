# Evaluation

## Protocol
- **Validation split (512 images)**: used for checkpoint selection inside each run (best Macro-F1 across epochs) and
  for the final comparison between experiments. Reported as VALIDATION RESULTS.
- **Test split (509 images)**: frozen; never read by training, tuning or selection; evaluated **once** with the
  repository evaluator after the final model was chosen. Reported as OFFICIAL TEST RESULTS.
- Evaluator: `python -m src.evaluate --checkpoint <ckpt> --split {val|test} --out-dir <dir>` → `metrics.json`
  (summary, per-class P/R/F1, confusion raw + normalised, 10-bin confidence/ECE, checkpoint provenance),
  `predictions.csv`, `misclassified.csv`, `confusion_matrix*.csv`, `classification_report.txt`.
- Metrics: accuracy, macro precision/recall/F1, weighted F1, cross-entropy, ECE, per-class P/R/F1, confusion matrix.

## Validation results (final model EXP-004; `results/experiments/EXP-004/eval_val/metrics.json`)
Accuracy 0.9336 · macro P 0.9345 · macro R 0.9347 · Macro-F1 0.9330 · weighted F1 0.9335 · ECE 0.0215 ·
34 misclassified / 512. Per-class F1: Bacterial Red 0.906 · Aeromoniasis 0.933 · Bacterial Gill 0.953 · EUS 0.850 ·
Saprolegniasis 0.967 · Parasitic 0.938 · White Tail 0.961 · Healthy 0.955.

## OFFICIAL TEST RESULTS (`results/final_test/`, run 2026-09-13 15:19 IST, TEST RUN COUNT = 1)
| Metric | Value |
|---|---|
| Accuracy | 0.9450 (481 / 509) |
| Macro precision / recall | 0.9470 / 0.9458 |
| Macro-F1 | 0.9453 |
| Weighted F1 | 0.9450 |
| Loss | 0.1916 |
| ECE | 0.0123 (mean confidence 0.9495; correct 0.9678, incorrect 0.6362) |

| Class | P | R | F1 | n |
|---|---|---|---|---|
| Bacterial Red Disease | 0.983 | 0.935 | 0.959 | 62 |
| Aeromoniasis | 0.940 | 0.955 | 0.947 | 66 |
| Bacterial Gill Disease | 0.969 | 0.939 | 0.954 | 66 |
| EUS Disease | 0.952 | 0.894 | 0.922 | 66 |
| Saprolegniasis | 0.967 | 1.000 | 0.983 | 59 |
| Parasitic Disease | 0.966 | 0.889 | 0.926 | 63 |
| White Tail Disease | 0.859 | 1.000 | 0.924 | 61 |
| Healthy Fish | 0.940 | 0.955 | 0.947 | 66 |

Confusion highlights: EUS → Aeromoniasis 3; diseased images predicted Healthy 4 / 443; Healthy predicted diseased 3 / 66.
Plots: `results/final_test/confusion_matrix.png`, `confusion_matrix_normalized.png`.

## Latency
Tesla T4 (training benchmark, batch 1): 10.483 ms/image. Apple-silicon MPS (this Mac, batch 1): 13.5 ms model-only,
16.7 ms including preprocessing and quality check (`results/final_test/inference_benchmark.json`). Grad-CAM adds ~75 ms.

## Reading the two numbers together
Test Macro-F1 (0.9453) is slightly above validation (0.9330); both splits are ~510 images (one image ≈ 0.2 %), so
the difference is within sampling variation. No tuning followed the test run; the model is frozen.
