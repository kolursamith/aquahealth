# Paper technique matrix (Gate 4)

Source: the five PDFs in `AI_TECHTAHON_DOCUMENTS/Research papers/`, text-extracted with pypdf and read in full. Every value below is quoted or paraphrased from the paper text ([A]); `NOT STATED` means the extracted text does not contain it (some values exist only inside table images). Nothing is inferred from titles.

Origin labels: **[A]** explicitly stated in the paper · **[B]** AquaHealth project requirement · **[C]** existing repository implementation · **[D]** engineering decision required to adapt.

## 0. The five 'hybrid categories' vs. what the papers actually are

| Expected category | Actual paper | Match? |
|---|---|---|
| CNN + Vision Transformer + LSTM | none of the five papers combines a CNN, a ViT and an LSTM | **no such paper** |
| YOLO + EfficientNet | none of the five uses YOLO in its method (P5 discusses YOLO only as the supervised alternative it avoids) | **no such paper** |
| CNN + BiLSTM | no paper uses an LSTM/BiLSTM; all inputs are single images, so there is no sequence to model | **no such paper** |
| ResNet + Attention | P4 SLCAM-AquaNet: ResNet18/50 and MobileNetV2 + a custom spatial-location-channel attention module | **yes (P4)** |
| YOLO + Transformer | P5 uses a frozen DINOv2 ViT for unsupervised anomaly detection; no YOLO | **partial (ViT only, P5)** |

Consequently: no LSTM/BiLSTM experiment is justified by the papers (there is no sequence in single-image classification), and no YOLO experiment is justified by the papers. The actual transferable material is: staged fine-tuning and Adam LRs (P1), augmentation recipes (P1, P2, P4), SGD+step-decay (P4), the SLCAM attention module (P4), a BN-Dense-Dropout head and early stopping (P3). P2's three-backbone fusion + SVM and P5's training-free anomaly detection do not fit the single EfficientNet-B0, 8-class MVP.

## P1 — Efficient Fish Disease Classification Using Fine-Tuned MobileNetV3Large for Mobile-Based Aquaculture Diagnosis

| Field | Value [A unless noted] |
|---|---|
| Paper title | Efficient Fish Disease Classification Using Fine-Tuned MobileNetV3Large for Mobile-Based Aquaculture Diagnosis |
| Authors | Lusiana, Helmud, Sulaiman |
| Year | 2026 (Sinkron 10(3)) |
| PDF | Efficient fish diesase classification using fine tuned mobilenetv3 large.pdf |
| Dataset | Kaggle irfanulhuda/fish-disease-detection-dataset (the AquaHealth source dataset) |
| Number of images | 2,400 |
| Number of classes | 8 (Bacterial Red, Aeromoniasis, Bacterial gill, EUS, Saprolegniasis, Parasitic, White tail, Healthy) |
| Public/private | public (Kaggle) |
| Bounding-box annotations | none (classification labels only) |
| Preprocessing | resize 224x224; ImageNet mean/std normalisation |
| Image resizing | 224x224 |
| CLAHE / enhancement | none (no CLAHE) |
| Augmentation | rotation, horizontal flip (+ scaling mentioned in integrity audit); training set only |
| CNN architecture | MobileNetV3Large |
| Backbone | MobileNetV3Large (ImageNet weights) |
| Attention mechanism | none |
| Transformer | none |
| LSTM/BiLSTM | none |
| YOLO version | none |
| Detection configuration | none |
| Feature extraction | CNN global features -> new classification head |
| Feature fusion | none |
| Sequence construction | n/a (single image) |
| Classifier | new dense head, 8 classes |
| Loss | categorical cross-entropy |
| Optimizer | Adam |
| Learning rate | Phase 1 1e-3 (head only); Phase 2 1e-5 (last 70 layers unfrozen) |
| Scheduler | NOT STATED in paper text |
| Batch size | NOT STATED in paper text |
| Epochs | NOT STATED in paper text (curves shown in figure only) |
| Transfer learning | ImageNet pretrained |
| Fine-tuning | Phase 1 frozen backbone + head; Phase 2 last 70 layers unfrozen, lr 1e-5 |
| Evaluation metrics | accuracy, loss, confusion matrix, precision/recall/F1, ROC-AUC (micro/macro) |
| Reported results | test accuracy 92.92 %, loss 0.2099, val accuracy 92.50 %, AUC 1.00 most classes (EUS 0.99); test set 240 images |
| Additional labels/data required | none |
| Computational requirements | NOT STATED in paper text |
| Not reproducible with our dataset | none of the method; note their split was 80/10/10 on 2,400 images vs our audited 3,405 -> 70/15/15 (numbers not comparable) |

## P2 — Empirical Evaluation of Deep Learning Techniques for Fish Disease Detection in Aquaculture Systems: A Transfer Learning and Fusion-Based Approach

| Field | Value [A unless noted] |
|---|---|
| Paper title | Empirical Evaluation of Deep Learning Techniques for Fish Disease Detection in Aquaculture Systems: A Transfer Learning and Fusion-Based Approach |
| Authors | Biswas, Muduli, Islam, Kanade, Zamani, Kanade, Parveen |
| Year | 2024 (IEEE Access 12) |
| PDF | Empirical evaluation of deep learning.pdf |
| Dataset | three: dataset-1 Fresh Water Fish Disease (7 classes, public), dataset-2 (305 images, 2 classes, public), dataset-3 own 'Freshwater fish disease aquaculture in South Asia' (~2,450 images, 7 classes, 350/class) |
| Number of images | ~2,450 (dataset-3) -> 10,500 after augmentation |
| Number of classes | 7 (dataset-3): Aeromoniasis, Bacterial gill, Bacterial red, EUS, Fungal, Parasitic, White tail (no Healthy class) |
| Public/private | dataset-1/2 public; dataset-3 self-collected |
| Bounding-box annotations | none |
| Preprocessing | pixel normalisation (rescale 1/255) |
| Image resizing | NOT STATED in paper text |
| CLAHE / enhancement | none |
| Augmentation | horizontal flip, width/height shift 0.2, rotation 20/40/60 deg, brightness variation, fill nearest; applied BEFORE the 60/20/20 split (leakage-prone protocol) |
| CNN architecture | VGG-16, VGG-19, MobileNetV2, MobileNetV3, InceptionV3, ResNet-34, ResNet-50, EfficientNet-B7, ConvNeXtXLarge (compared) |
| Backbone | ImageNet-pretrained CNNs |
| Attention mechanism | none |
| Transformer | none |
| LSTM/BiLSTM | none |
| YOLO version | none |
| Detection configuration | none |
| Feature extraction | deep features from VGG-16, MobileNetV2, InceptionV3 |
| Feature fusion | feature concatenation (fusion) of the three backbones |
| Sequence construction | n/a |
| Classifier | SVM on fused features (kernels compared); softmax heads for single CNNs |
| Loss | NOT STATED in paper text (SVM classifier for the fusion model) |
| Optimizer | compared Adam, RMSProp, Adadelta, SGD per backbone |
| Learning rate | NOT STATED in paper text (only in table figures) |
| Scheduler | NOT STATED in paper text |
| Batch size | NOT STATED in paper text |
| Epochs | NOT STATED in paper text (Table 5 as figure) |
| Transfer learning | ImageNet pretrained |
| Fine-tuning | transfer learning; fine-tuning details NOT STATED in paper text |
| Evaluation metrics | accuracy, precision, recall, F1, confusion matrix (macro/weighted avg) |
| Reported results | best single CNN VGG-16 88.82 % acc / 88.20 F1; fusion+SVM 99.59 % acc (dataset-3) |
| Additional labels/data required | none for fusion; three backbones must be run |
| Computational requirements | 3 backbones + SVM: ~3x inference cost of one CNN |
| Not reproducible with our dataset | their dataset-3 is not ours; augmentation-before-split cannot be replicated legitimately |

## P3 — Comparison of EfficientNetB1 Model Effectiveness in Identifying Fish Diseases in South Asian Fish Diseases and Salmon Fish Diseases

| Field | Value [A unless noted] |
|---|---|
| Paper title | Comparison of EfficientNetB1 Model Effectiveness in Identifying Fish Diseases in South Asian Fish Diseases and Salmon Fish Diseases |
| Authors | Afridiansyah, Setiadi |
| Year | 2024 (JAIC 8(2)) |
| PDF | comparision of effieicntnetb1 model effectives in identifying fish disesase.pdf |
| Dataset | Freshwater Fish Disease Aquaculture in South Asia (Kaggle; 7 classes x 100 = 700 images) and SalmonScan (Mendeley; 2 classes, 243 images) |
| Number of images | 700 + 243 |
| Number of classes | 7 (no Healthy) / 2 |
| Public/private | public (Kaggle, Mendeley) |
| Bounding-box annotations | none |
| Preprocessing | rescale 1/255; resize to 150x150 |
| Image resizing | 150x150 |
| CLAHE / enhancement | none |
| Augmentation | none (listed as future work) |
| CNN architecture | EfficientNetB1 |
| Backbone | EfficientNetB1 include_top=False (ImageNet) |
| Attention mechanism | none |
| Transformer | none |
| LSTM/BiLSTM | none |
| YOLO version | none |
| Detection configuration | none |
| Feature extraction | EfficientNetB1 feature map -> BatchNorm -> Dense(128) -> Dropout |
| Feature fusion | none |
| Sequence construction | n/a |
| Classifier | Dense(7, softmax) |
| Loss | categorical cross-entropy |
| Optimizer | Adam |
| Learning rate | 3e-5 |
| Scheduler | none; EarlyStopping(patience 5, monitor val_accuracy) |
| Batch size | 64 (Table I) / 32 (text) - inconsistent in paper |
| Epochs | 25 |
| Transfer learning | ImageNet pretrained |
| Fine-tuning | all 7,953,433 params trainable (Table II) |
| Evaluation metrics | accuracy, precision, recall (val) |
| Reported results | 98.14 % (South Asian, 80/20 validation_split), 99.18 % (Salmon) |
| Additional labels/data required | none |
| Computational requirements | EfficientNetB1 (7.9M params) at 150px |
| Not reproducible with our dataset | none of the method; 150px input and 80/20 'validation as test' protocol are weaker than ours |

## P4 — SLCAM-AquaNet: an attention-enhanced lightweight deep learning model for accurate classification of aquaculture disease

| Field | Value [A unless noted] |
|---|---|
| Paper title | SLCAM-AquaNet: an attention-enhanced lightweight deep learning model for accurate classification of aquaculture disease |
| Authors | Athiraja, Gurulakshmi, Gurupriya, Nagarajan |
| Year | 2026 (Discover Applied Sciences 8:382) |
| PDF | slcam-aquanet.pdf |
| Dataset | own multi-species aquaculture dataset: 6 species (Tilapia, Grass carp, Catla, Rohu, Nile perch, Channel catfish) x 5 classes (Healthy, Bacterial, Viral, Parasitic, Fungal) |
| Number of images | 5,260 |
| Number of classes | 5 per species |
| Public/private | NOT STATED in paper text (described as acquired in controlled aquaculture settings) |
| Bounding-box annotations | none |
| Preprocessing | resize 224x224; acquisition-run-level split to avoid leakage |
| Image resizing | 224x224 |
| CLAHE / enhancement | none |
| Augmentation | random horizontal flip p=0.5, random crop scale 0.8-1.0, colour jitter brightness/contrast/saturation +-20 %; none on val/test |
| CNN architecture | ResNet18, ResNet50, MobileNetV2 (+ SLCAM) |
| Backbone | ResNet18 / ResNet50 / MobileNetV2 |
| Attention mechanism | SLCAM: 4 branches - horizontal, vertical, spatial (7x7 conv), channel (avg+max pool MLP); compared with CBAM and Coordinate Attention |
| Transformer | none |
| LSTM/BiLSTM | none |
| YOLO version | none (YOLO only in related work) |
| Detection configuration | none |
| Feature extraction | attention-refined intermediate feature maps (SLCAM inserted in backbone) |
| Feature fusion | spatial and channel attention paths fused |
| Sequence construction | n/a |
| Classifier | softmax head (5 classes) |
| Loss | NOT STATED in paper text |
| Optimizer | SGD momentum 0.9 |
| Learning rate | 0.01, decayed x0.1 every 20 epochs |
| Scheduler | step decay /10 every 20 epochs |
| Batch size | 32 |
| Epochs | 100 |
| Transfer learning | NOT STATED in paper text (pretraining not explicitly stated in extracted text) |
| Fine-tuning | full training of backbone+SLCAM |
| Evaluation metrics | Top-1 accuracy, precision, recall, F1, params, FLOPs |
| Reported results | ResNet50+SLCAM 98.05 % acc / 98.2 F1 (+3.8-5.7 % over baselines); MobileNetV2+SLCAM 95.25 %; ResNet18+SLCAM 97.55 % |
| Additional labels/data required | none |
| Computational requirements | +0.35 B FLOPs for SLCAM on ResNet50; 100 epochs |
| Not reproducible with our dataset | their multi-species dataset; SLCAM itself is reproducible on our images (attention module in a CNN) |

## P5 — DINO-Patch: Unsupervised Fish Disease Detection via DINOv2 with Leave-One-Fish-Out Evaluation

| Field | Value [A unless noted] |
|---|---|
| Paper title | DINO-Patch: Unsupervised Fish Disease Detection via DINOv2 with Leave-One-Fish-Out Evaluation |
| Authors | Liu, Zhu, Shi, Dong, Xiao, Liao |
| Year | 2026 (SSRN preprint, not peer reviewed) |
| PDF | unsupervised fish disease detection via dniov2.pdf |
| Dataset | MatsyaDx-BD (own): Grass/Silver/Bighead carp, Healthy + BG + BR + EUS, specimen-level organisation; cross-dataset SalmonScan |
| Number of images | 2,145 |
| Number of classes | 4 conditions x 3 species (used as healthy-vs-anomalous) |
| Public/private | NOT STATED in paper text |
| Bounding-box annotations | none (image-level; anomaly maps produced, no GT boxes) |
| Preprocessing | resize shorter side to 448, centre-crop to a multiple of the 14-px patch |
| Image resizing | 448 shorter side |
| CLAHE / enhancement | PCA-based foreground mask from DINOv2 features |
| Augmentation | none |
| CNN architecture | none (ViT) |
| Backbone | DINOv2 ViT-S/14, frozen |
| Attention mechanism | ViT self-attention (inside DINOv2) |
| Transformer | DINOv2 ViT-S/14 patch features |
| LSTM/BiLSTM | none |
| YOLO version | none (YOLO discussed as the supervised alternative) |
| Detection configuration | anomaly localisation via patch k-NN distances (no detector) |
| Feature extraction | ViT patch features of healthy reference images -> memory bank (PatchCore paradigm, optional coreset) |
| Feature fusion | none |
| Sequence construction | n/a |
| Classifier | k-NN distance score vs memory bank -> image-level anomaly score (healthy vs diseased only) |
| Loss | none (training-free) |
| Optimizer | none |
| Learning rate | none |
| Scheduler | none |
| Batch size | n/a |
| Epochs | none |
| Transfer learning | frozen self-supervised DINOv2 |
| Fine-tuning | none |
| Evaluation metrics | AUROC, Leave-One-Fish-Out protocol |
| Reported results | mean AUROC > 0.930 across three carp species; transfers to SalmonScan |
| Additional labels/data required | specimen IDs for LOFO (we have none) |
| Computational requirements | ViT-S/14 at 448px + FAISS memory bank |
| Not reproducible with our dataset | specimen-level LOFO evaluation (no specimen IDs in our data); binary anomaly detection does not give 8-class output |

## Technique classification and applicability

| Paper | Technique | Category | Origin | Applicability to AquaHealth (single-image, 8-class, 3,405 images, no boxes) | Extra data/annotations |
|---|---|---|---|---|---|
| P1 | two-phase transfer learning: frozen backbone + head (lr 1e-3), then unfreeze last layers (lr 1e-5) | fine_tuning | [A] | APPLICABLE - matches existing Layer 9 staged schedule [C]; paper LRs can be tested as a controlled variant | none / none |
| P1 | augmentation: rotation + horizontal flip (train only) | augmentation | [A] | APPLICABLE - subset of existing augmentation [C] (which also has crop/zoom, brightness/contrast) | none / none |
| P1 | resize 224x224 + ImageNet normalisation, no CLAHE | preprocessing | [A] | APPLICABLE - equals our pipeline with CLAHE off; CLAHE-on is a project requirement [B], so CLAHE off/on is the controlled comparison | none / none |
| P1 | Adam + categorical cross-entropy | optimizer | [A] | APPLICABLE - engine supports adam [C]; baseline uses AdamW [C] | none / none |
| P1 | MobileNetV3Large backbone | backbone | [A] | APPLICABLE as fallback backbone [B] (PRD lists MobileNetV3-Large as hot-swap fallback); not part of EfficientNet experiments | none / none |
| P2 | feature fusion of three CNN backbones + SVM | feature fusion | [A] | NOT PURSUED - triples inference cost and replaces the softmax head with an SVM; conflicts with the single EfficientNet-B0 MVP [B] | none / none |
| P2 | augmentation: h-flip, shift 0.2, rotation 20/40/60, brightness | augmentation | [A] | PARTLY APPLICABLE - flip/rotation/brightness exist [C]; width/height shift is absent (RandomAffine translate would be a [D] addition) | none / none |
| P2 | optimizer comparison Adam / RMSProp / Adadelta / SGD | optimizer | [A] | APPLICABLE in principle; only Adam and SGD are supported by the engine [C]; RMSProp/Adadelta would be [D] additions | none / none |
| P3 | Adam lr 3e-5, batch 64/32, 25 epochs, EarlyStopping(patience 5) | training | [A] | APPLICABLE - lr and early stopping are controlled variants; early stopping is not implemented in the engine [D] | none / none |
| P3 | head: BatchNorm -> Dense(128) -> Dropout -> Dense(K) | classifier | [A] | APPLICABLE as a controlled head variant; existing head is Dropout -> Linear(1280,K) [C] | none / none |
| P3 | input 150x150, rescale 1/255, no augmentation | preprocessing | [A] | NOT ADOPTED - 224 is a project requirement [B]; ImageNet normalisation is what the pretrained weights expect [C] | none / none |
| P3 | EfficientNetB1 backbone | backbone | [A] | POSSIBLE B0->B1 comparison, but B0 is the mandated primary model [B] | none / none |
| P4 | SLCAM attention module (horizontal + vertical + spatial + channel attention) inserted into a CNN backbone | attention | [A] | APPLICABLE - can be inserted after late EfficientNet-B0 blocks; requires new module code [D]; paper equations give the design | none / none |
| P4 | SGD momentum 0.9, lr 0.01, step decay /10 every 20 epochs, batch 32, 100 epochs | optimizer | [A] | APPLICABLE - engine supports sgd + StepLR [C]; 100 epochs exceeds hackathon budget, a shortened schedule is [D] | none / none |
| P4 | augmentation: h-flip 0.5, random crop scale 0.8-1.0, colour jitter +-20 % (brightness/contrast/saturation) | augmentation | [A] | APPLICABLE - nearly identical to existing augmentation [C] except saturation jitter (absent) and no rotation | none / none |
| P4 | resize 224x224; acquisition-run-level leakage control | preprocessing | [A] | APPLICABLE - 224 matches [B]; our near-duplicate-group split [C] is the analogue of run-level control | none / none |
| P5 | DINOv2 ViT-S/14 frozen patch features + PCA foreground mask + memory-bank k-NN anomaly score | transformer | [A] | NOT APPLICABLE to 8-class supervised classification - it is binary healthy-vs-anomalous, training-free; could only serve as an out-of-distribution / 'is this a fish' filter [D] | healthy reference set only / none |
| P5 | Leave-One-Fish-Out specimen-level evaluation | evaluation | [A] | NOT REPRODUCIBLE - dataset has no specimen identity; our group-aware split by near-duplicate is the closest proxy [C] | specimen IDs / none |

## Reproducibility assessment

- **P1** — fully reproducible protocol (two-phase fine-tuning, Adam, flip/rotation, 224, ImageNet norm) on our data; their 92.92 % is on their 240-image test split of the 2,400-image Kaggle set and is **not comparable** to our audited 3,405-image, 70/15/15, duplicate-aware split.
- **P2** — the backbone comparison is reproducible in spirit; the fusion+SVM model is a different architecture family and its 99.59 % was obtained after augmenting before splitting, which we will not replicate.
- **P3** — reproducible (Adam 3e-5, early stopping, BN-Dense-Dropout head); their 98.14 % is a 700-image, 80/20 validation-as-test number.
- **P4** — SLCAM is reproducible from the equations as an attention module inserted into EfficientNet-B0 (an engineering adaptation [D]); their dataset (5,260 multi-species images) is not ours.
- **P5** — not reproducible as a classifier: training-free binary anomaly detection with specimen-level LOFO evaluation; our data has no specimen IDs.
- **Bounding boxes**: none of the papers provides or uses them for our dataset, and the AquaHealth audit found no annotation files. No YOLO training is supported by the papers or the data.

Sequence construction (for the LSTM/BiLSTM question): **no paper constructs a sequence**. Every method consumes one image and produces one label (or one anomaly score). An LSTM would have nothing to iterate over.
