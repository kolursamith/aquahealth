# YOLO12 audit (Gate 7A / Phase 5) — bounding-box annotations and detector feasibility

Scope: decide whether a supervised YOLO12 + EfficientNet-B0 hybrid is valid, and if not,
whether a pretrained detector used inference-only (crop → classify) is worth building.
Nothing below was fabricated: no boxes were drawn, no pseudo-labels were produced.

## 1. What was searched

| Location | Method | Result |
|---|---|---|
| `AI_TECHTAHON_DOCUMENTS/DATASET.zip` (delivered archive, 106.4 MB) | `unzip -l`, filtered for non-image entries | 3,504 entries: 3,503 images + **`test.csv` only** |
| `AI_TECHTAHON_DOCUMENTS/New Dataset/` (extracted; `data/original/aquahealth` symlink) | recursive `find` by extension | 3,453 `.jpg`, 30 `.jpeg`, 20 `.png`, 1 `.csv`, 2 `.DS_Store`; **no `.xml`, `.txt`, `.json`, `.yaml`, `labels/`, `annotations/`** |
| `test.csv` | header + rows | two columns `filename,label` — image-level class labels, no coordinates |
| `results/data_audit_report.md` (Gate: data audit) | prior audit line | "YOLO / bounding-box annotations: none found" |
| `PRD_AquaHealthAI_Final.md` | grep yolo/bounding/detect | "Fish detection on pond images" is **Out of scope (final)**; "Fish detector / pond-image path: adds a second model + bounding-box pipeline; single-fish MVP is the whole demo" |
| Five research papers (Gate 4 matrix) | full-text read | none uses or provides bounding boxes; P5 mentions YOLO only as the supervised alternative it avoids |

**Bounding-box annotations available: NONE (0 files, 0 boxes).**

## 2. Consequence for supervised YOLO12 training

Supervised detector training requires per-image box coordinates for every training image.
Zero exist. Producing them would mean either manual annotation of ~2,400 training images
(no annotator, no time in the hackathon budget, and every image would need a disease-region
definition that no paper or PRD provides) or auto-labelling with another model, which is
fabrication of ground truth and is forbidden by the project constraints.

**Supervised YOLO12 training on this dataset is INVALID and was not attempted.**

## 3. Pretrained detector, inference-only crop stage — feasibility

Idea: run an off-the-shelf detector, crop the highest-scoring fish box, classify the crop.
Findings against the actual data (16 random *training* images inspected as a contact sheet;
the test split was not opened):

- **The images are not consistently "whole fish" photographs.** A large fraction are already
  tight crops of a lesion, a gill, a fin base or a hand holding part of a fish; several show
  only skin texture. A fish detector has nothing to detect in those, so the crop stage would
  either fail (no box → fall back to the full image, i.e. no change) or crop away context.
- **Resolution is low and mixed:** sampled sizes 128×128, 224×224 and 640×640. Cropping a
  128 px image and upsampling to 224 loses information relative to the current pipeline.
- **Many images are pre-augmented** (rotated with white/black borders, mirrored duplicates —
  also seen in the near-duplicate audit). Boxes predicted on rotated, letterboxed images are
  unreliable, and the border regions are exactly what a detector would be confused by.
- **COCO-pretrained YOLO weights have no "fish" class** (COCO's 80 classes contain no fish),
  so an inference-only YOLO crop would require an open-vocabulary detector (e.g. YOLO-World,
  OWL-ViT, Grounding-DINO) — new dependencies, model downloads, and an unvalidatable crop
  quality because there are no boxes to measure it against.
- **No evaluation is possible on the detector's own output quality**: with zero ground-truth
  boxes, "did it crop the right region" cannot be measured; only the downstream classifier
  metric could be compared, which is a confounded, unattributable signal.
- **The PRD explicitly scopes detection out** and the current single-model pipeline already
  meets the latency budget (9.0 ms/image on T4, 4.0 M parameters). A detector adds a second
  model, roughly an order of magnitude more parameters, and a second failure mode.

## 4. Decision

| Option | Verdict |
|---|---|
| Supervised YOLO12 training | **INVALID — no annotations exist** |
| Pretrained detector, inference-only crop → EfficientNet-B0 (HYB-YOLO-EFF-001) | **NOT FEASIBLE within this project**: no fish class in standard pretrained weights, no way to validate crops, data are largely lesion close-ups and low resolution, PRD marks detection out of scope |

Gate 7B is therefore skipped: `results/experiments/HYB-YOLO-EFF-001/` was **not created**
and no hybrid metrics exist (NOT MEASURED). The final candidate pool consists of the
paper-informed EfficientNet-B0 experiments only.

Test set read: **NO**.

## 5. Phase-5 status line

YOLO12 SUPERVISED TRAINING: **NOT SCIENTIFICALLY REPRODUCIBLE WITH CURRENT DATA** (0 bounding-box
files, 0 boxes, 0 annotated images, no label format, no class-ID scheme — searched: DATASET.zip
listing, the extracted dataset tree, `test.csv`, the PRD and the five papers).

PRETRAINED DETECTOR INFERENCE-ONLY EXPERIMENT: **NOT EXECUTED** — judged not reasonable for this
data (no fish class in COCO-pretrained YOLO weights; a large share of images are lesion close-ups
with no whole fish to detect; 128-px images would lose information when cropped and re-upsampled;
crop quality cannot be validated without any ground-truth boxes; the PRD scopes detection out and
the single-model pipeline already runs at ~9 ms/image). `results/experiments/HYB-YOLO-EFF-001/`
was therefore not created. No `ultralytics` dependency was added.
