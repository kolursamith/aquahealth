# Final paper technique matrix (Gate 3 — evidence-based, read-only)

Sources: the five PDFs in `AI_TECHTAHON_DOCUMENTS/Research papers/` (full text extracted with pypdf
and read; per-technique quotations and [A]/[B]/[C]/[D] origin labels are in
`results/paper_technique_matrix.md`), plus the experiment configs and artifacts in this repository.
Nothing below is inferred from a title or from the external "hybrid model" list. Published numbers
are **PUBLISHED PAPER RESULTS — NOT AQUAHEALTH RESULTS**. "NOT STATED" = not in the paper text.

## Per-paper facts (items 1–24)

| # | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| 1 Title | Efficient Fish Disease Classification Using Fine-Tuned MobileNetV3Large for Mobile-Based Aquaculture Diagnosis (Lusiana, Helmud, Sulaiman; Sinkron 10(3), 2026) | Empirical Evaluation of Deep Learning Techniques for Fish Disease Detection in Aquaculture Systems: A Transfer Learning and Fusion-Based Approach (Biswas et al.; IEEE Access, 2024) | Comparison of EfficientNetB1 Model Effectiveness in Identifying Fish Diseases in South Asian Fish Diseases and Salmon Fish Diseases (Afridiansyah, Setiadi; JAIC 8(2), 2024) | SLCAM-AquaNet: an attention-enhanced lightweight deep learning model for accurate classification of aquaculture disease (Athiraja, Gurulakshmi, Gurupriya, Nagarajan; Discover Applied Sciences 8:382, 2026) | DINO-Patch: Unsupervised Fish Disease Detection via DINOv2 with Leave-One-Fish-Out Evaluation (Liu, Zhu, Shi, Dong, Xiao, Liao; 2025) |
| 2 Dataset | Kaggle irfanulhuda fish-disease (the AquaHealth source data) | 3 datasets; own dataset-3 "Freshwater fish disease aquaculture in South Asia" | Freshwater Fish Disease Aquaculture in South Asia (Kaggle) + SalmonScan | own multi-species set (6 species) | MatsyaDx-BD (own, specimen-level) + SalmonScan |
| 3 Classes | 8 (incl. Healthy) | 7 (no Healthy) | 7 / 2 | 5 per species | 4 conditions × 3 species, used as healthy-vs-anomalous |
| 4 Size | 2,400 images (240-image test split) | ~2,450 (350/class) | 700 / 243 | 5,260 | NOT STATED in extracted text |
| 5 Architecture | single CNN classifier | CNN feature extractors + fusion + SVM | single CNN classifier | CNN backbone + SLCAM attention | frozen ViT feature extractor + memory-bank k-NN |
| 6 Backbone | MobileNetV3Large | VGG16 / MobileNetV2 / InceptionV3 | EfficientNetB1 | ResNet18 / ResNet50 / MobileNetV2 | DINOv2 ViT-S/14 |
| 7 Pretraining | ImageNet | ImageNet | ImageNet | ImageNet (backbones) | DINOv2 self-supervised |
| 8 Input size | 224×224 | NOT STATED in text (224 typical) | 150×150 | 224×224 | shorter side 448, crop to multiple of 14 |
| 9 Preprocessing | resize + ImageNet normalisation | resize; augment-before-split | rescale 1/255 | resize; acquisition-run-level split | PCA foreground mask |
| 10 Augmentation | rotation, horizontal flip | flip, shift 0.2, rotation 20/40/60, brightness | none | flip 0.5, crop 0.8–1.0, jitter ±20 % (b/c/s) | none |
| 11 Optimizer | Adam | Adam vs RMSProp/Adadelta/SGD | Adam | SGD momentum 0.9 | none (training-free) |
| 12 LR | 1e-3 (phase 1), 1e-5 (phase 2) | NOT STATED per model | 3e-5 | 0.01 | — |
| 13 Scheduler | NOT STATED | NOT STATED | none; EarlyStopping(5) | step ×0.1 every 20 epochs | — |
| 14 Batch | NOT STATED | NOT STATED | 64 / 32 | 32 | — |
| 15 Epochs | NOT STATED | NOT STATED | 25 | 100 | — |
| 16 Fine-tuning | frozen backbone + head, then last 70 layers | feature extraction, fusion, SVM | full model at 3e-5 | full training | frozen |
| 17 Attention | none | none | none | **SLCAM (horizontal + vertical + spatial + channel)** | ViT self-attention only (frozen) |
| 18 Transformer | none | none | none | none | **DINOv2 ViT-S/14** |
| 19 LSTM/BiLSTM | none | none | none | none | none |
| 20 YOLO/detection | none | none | none | none | none (YOLO mentioned only as the supervised alternative avoided) |
| 21 Fusion/ensemble | none | **three-backbone feature fusion + SVM** | none | none | none |
| 22 Metrics | acc, loss, CM, P/R/F1, ROC-AUC | acc, P/R/F1, CM | acc, P/R (val) | top-1 acc, P/R/F1, params, FLOPs | AUROC (LOFO) |
| 23 Published result | 92.92 % test acc | VGG16 88.82 %; fusion+SVM 99.59 % | 98.14 % / 99.18 % | ResNet50+SLCAM 98.05 % / F1 98.2 | mean AUROC > 0.930 |
| 24 Comparable to AquaHealth? | **NO** — same source data but their split leaks near-duplicates (our audit: 59 near-dup groups straddle the vendor train/test folders) | NO — different data | NO — different data, validation-as-test | NO — different data | NO — different task |

## Items 25–26 per technique (feasibility and execution) — full detail in `results/final_paper_technique_matrix.csv`

| Paper | Actual architecture | Key technique | Executed in AquaHealth? | Feasible? | Reason |
|---|---|---|---|---|---|
| P1 | MobileNetV3Large classifier | two-phase fine-tuning, Adam 1e-3 → 1e-5 | **PARTIALLY EXECUTED** (EXP-003: two runs, both interrupted; no verified metrics) | YES | engine supports it; needs one uninterrupted run |
| P1 | same | no-CLAHE preprocessing | **EXECUTED** — EXP-002 completed on Colab; artifact on Drive, **not yet verified locally** | YES | |
| P1 | same | rotation+flip-only augmentation | NOT EXECUTED | YES | subset of existing augmentation |
| P1 | same | MobileNetV3Large backbone | REQUIRES NEW ARCHITECTURE | YES with new code | PRD fallback only |
| P2 | 3 CNNs + fusion + SVM | fusion + SVM | NOT APPLICABLE | NO (by design) | 3× cost, SVM head, augment-before-split |
| P2 | — | optimizer comparison | PARTIALLY EXECUTED | YES (Adam/SGD) | RMSProp/Adadelta not in engine |
| P3 | EfficientNetB1 classifier | Adam 3e-5 + early stopping | **EXECUTED** — EXP-006 completed; artifact on Drive, **not yet verified locally** | YES | early stopping ≈ best-checkpoint selection |
| P3 | same | BN-Dense128-Dropout head | REQUIRES NEW ARCHITECTURE | YES with new code | |
| P3 | same | 150 px, rescale 1/255 | NOT APPLICABLE | NO | 224 + ImageNet norm are requirements |
| P4 | ResNet/MobileNetV2 + SLCAM | SGD 0.01 + step decay | **EXECUTED** — EXP-004 completed; observed log 0.9330 (**OBSERVED_LOG_ONLY**) | YES | |
| P4 | same | flip/crop/jitter augmentation | **EXECUTED** — EXP-005 completed; observed log 0.9335 (**OBSERVED_LOG_ONLY**) | YES | |
| P4 | same | SLCAM attention module | REQUIRES NEW ARCHITECTURE | YES with new code + GPU run | not executed |
| P4 | same | run-level leakage control | EXECUTED (analogue: near-duplicate-group split) | YES | |
| P5 | DINOv2 ViT-S/14 + k-NN | unsupervised anomaly detection | NOT APPLICABLE | NO | different task |
| P5 | — | Leave-One-Fish-Out | INFEASIBLE WITH CURRENT DATA | NO | no specimen IDs |
| any | — | LSTM / BiLSTM | NOT APPLICABLE | NO | no paper uses one; single image has no sequence |
| any | — | YOLO / detection | INFEASIBLE WITH CURRENT DATA | NO | zero bounding boxes (`results/YOLO12_AUDIT.md`) |

## Conclusion — what is genuinely useful for the final AquaHealth model

1. **P4's optimiser (SGD 0.9 / 0.01 / step decay) and P4's augmentation** are the two techniques
   with the strongest observed effect (+0.033 Macro-F1 each over EXP-001 in the Colab logs). They
   are the only paper techniques that both apply to EfficientNet-B0 unchanged and were completed.
   Their artifacts must be verified from Drive before they count.
2. **P1's two-phase schedule** is worth one clean run (it also tests dropping the full-unfreeze
   stage, which hurt EXP-001), but it has no verified result yet.
3. **P1's no-CLAHE ablation** and **P3's Adam 3e-5** are completed on Drive and only need
   verification; they answer whether CLAHE helps and whether a lower LR is better.
4. Attention (P4 SLCAM), alternative heads (P3), other backbones (P1 MobileNetV3, P3 B1) would need
   new architecture code and GPU time; none has evidence yet and none is required by the papers to
   beat the baseline given the P4 results.
5. LSTM/BiLSTM, Transformers as classifiers, YOLO detection, fusion+SVM and cGAN have **no support**
   in the five papers for this task/data and were deliberately not built.
