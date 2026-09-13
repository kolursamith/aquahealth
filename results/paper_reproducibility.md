# Paper reproducibility assessment (Phase 3)

Basis: the five PDFs in `AI_TECHTAHON_DOCUMENTS/Research papers/` (full text extracted and read;
technique-level detail with [A]/[B]/[C]/[D] labels is in `results/paper_technique_matrix.{csv,md}`).
Nothing here is inferred from a title. "Executed" means an AquaHealth experiment exists with saved
artifacts; see `results/ML_EXPERIMENT_STATUS.md` for run status.

Legend: **DR** = DIRECTLY REPRODUCIBLE on the AquaHealth frozen split with the existing engine;
**PR** = PARTIALLY REPRODUCIBLE (needs an engineering adaptation [D] or a different scope);
**NR** = NOT REPRODUCIBLE (needs data/annotations we do not have, or is a different task).

| Paper | Item | Class | Reason | Compute |
|---|---|---|---|---|
| P1 (Lusiana et al., 2026) | two-phase transfer learning, Adam, head 1e-3 → last layers 1e-5 | **DR** | staged engine already supports it; "last 70 layers" of MobileNetV3 mapped to "last 3 EfficientNet blocks" [D] | 1 T4 run, ~25 min (EXP-003) |
| P1 | flip + rotation augmentation only | DR | subset of the existing `AugmentConfig` | ~25 min — NOT EXECUTED |
| P1 | resize 224 + ImageNet norm, no CLAHE | **DR** | `clahe: null` (EXP-002) | ~15 min (no CLAHE is faster) |
| P1 | MobileNetV3Large backbone | PR | PRD lists it only as fallback; `src/model.py` is EfficientNet-specific (block freezing) [D] | NOT EXECUTED |
| P1 | 92.92 % test accuracy | NR as a comparison | their 240-image test split of the same Kaggle data contains near-duplicates of training images (our audit found 59 groups straddling the vendor split) | — |
| P2 (Biswas et al., 2024) | ImageNet CNN comparison (VGG16 / MobileNetV2 / InceptionV3) | PR | possible but replaces the mandated EfficientNet-B0 [B] | NOT EXECUTED |
| P2 | fusion of three backbones + SVM | PR/NR | triples inference cost; SVM head breaks the softmax/confidence contract [B]; augment-before-split is a leakage practice we will not copy | NOT EXECUTED (deliberately) |
| P2 | optimizer comparison | PR | Adam and SGD exist in the engine (EXP-003/006, EXP-004); RMSProp/Adadelta would be [D] | partly executed |
| P3 (Afridiansyah & Setiadi, 2024) | Adam lr 3e-5 + EarlyStopping(5) | **DR** | lr yes; early stopping replaced by best-checkpoint selection [D] (EXP-006) | ~25 min |
| P3 | BN → Dense(128) → Dropout → Dense(K) head | PR | needs a head variant in `build_classifier` plus checkpoint/predictor plumbing [D] | NOT EXECUTED |
| P3 | 150×150 input, rescale 1/255, no augmentation | NR (by project rule) | 224 and ImageNet normalisation are requirements [B]; the pretrained weights expect ImageNet statistics | — |
| P3 | EfficientNetB1 | PR | B0 is mandated [B]; B1 is a drop-in comparison if time allows | NOT EXECUTED |
| P4 (Athiraja et al., 2026) | SGD 0.9, lr 0.01, step ×0.1 every 20/100 epochs | **DR** (step scaled to every 5 epochs per stage [D]) | EXP-004 | ~25 min |
| P4 | flip 0.5, crop 0.8–1.0, jitter ±20 % b/c/s | **DR** (saturation jitter added to `AugmentConfig` [D], default off) | EXP-005 | ~24 min |
| P4 | SLCAM attention module | PR | equations are given; needs a new module inserted after late EfficientNet blocks, new tests, and predictor support [D]; ~1–2 h engineering + 25 min training | NOT EXECUTED |
| P4 | ResNet18/50 + SLCAM as architecture comparison | PR | different backbone family; would need its own freezing logic [D] | NOT EXECUTED |
| P4 | acquisition-run-level leakage control | DR (already done) | our near-duplicate-group split is the analogue [C] | — |
| P5 (Liu et al., 2025) | DINOv2 ViT-S/14 patch features + PCA mask + k-NN anomaly score | NR for our task | binary healthy-vs-anomalous, training-free; could only serve as an out-of-distribution filter [D]; needs `torch.hub` DINOv2 weights (~90 MB) | NOT EXECUTED |
| P5 | Leave-One-Fish-Out evaluation | NR | requires specimen identity, absent from our data | — |

## Sequence models (LSTM / BiLSTM) — why nothing was added

AquaHealth consumes **one image → one label**. None of the five papers builds a sequence, and an
LSTM needs an ordered sequence of feature vectors. The only ways to manufacture one from a single
image are:

1. **Patch sequence** — split the 7×7×1280 EfficientNet feature map into 49 spatial tokens and run a
   BiLSTM over them in raster order. This imposes an arbitrary left-to-right order on a 2-D image
   and adds ~10 M parameters; a transformer/attention block (P4's SLCAM is the paper-backed version)
   is the principled alternative.
2. **Multi-crop / multi-scale sequence** — feed K crops or K scales of the same image in a fixed
   order. The "sequence" is synthetic; a mean/max pool over the crops is equivalent and cheaper.
3. **Temporal sequence** — frames of a video of the same fish. The dataset has no video and no
   specimen identity, so this is impossible with current data.

Conclusion: CNN + LSTM/BiLSTM is **NOT REPRODUCIBLE / NOT JUSTIFIED** here (no paper support, no
sequence in the data). The "CNN + Vision Transformer + LSTM" and "CNN + BiLSTM" categories in the
brief have no evidence base and were **NOT EXECUTED** by design.

## YOLO — annotation audit

Zero bounding boxes exist (`results/YOLO12_AUDIT.md`). Supervised YOLO12: NR. Pretrained-detector
crop: assessed, not reasonable, NOT EXECUTED.

## Computational requirements actually observed

One 30-epoch EfficientNet-B0 run on a Colab Tesla T4 with 2 dataloader workers: 23.6–25.8 min
with CLAHE (CPU-bound, ~44 s/epoch), ~15 min without CLAHE (EXP-002). Inference 9 ms/image on T4.
Local Apple-silicon (MPS) smoke test: 1.6 s per 32-image step (AMP off).
