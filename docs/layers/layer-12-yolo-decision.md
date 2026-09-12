# Layer 12 — YOLO Integration: Decision Record

## Status: DEFERRED — blocked on evidence, by rule

The YOLO rule for this project is explicit: do not add fish localisation
unless the real images require it. That requires inspecting the real
images. As of this record the workspace contains **no dataset**:

```
find data -type f ! -name .gitkeep ! -name README.md   → (nothing)
find . -iname "*.jpg" -o -iname "*.png" … (outside .venv) → (nothing)
```

There is therefore nothing to inspect, and the non-hallucination rule
forbids deciding on an assumption. No YOLO code, dependency (`ultralytics`
or otherwise), or weights are added. The classification pipeline (Layers
0–11) is complete and does not depend on this decision.

## What the decision will be based on

When `data/original/` exists, run `scripts/audit_dataset.py` and then
inspect a stratified sample (≥ 30 images per class, drawn by
`random.Random(SEED)`) against these questions, recording counts:

| Question | Evidence to record | Points toward |
|---|---|---|
| Is there exactly one fish per image? | count of images with 0 / 1 / >1 fish | >1 fish common → localisation needed |
| Does the fish fill the frame? | rough fraction of image area occupied by the fish (bucket: <25 %, 25–60 %, >60 %) | mostly <25 % → crop helps |
| Is the fish centred? | would the Layer 6 centre-crop (`resize_mode="crop"`) cut off the fish? count | yes often → `resize_mode="squash"` first, YOLO second |
| Is the background cluttered / variable? | tanks, hands, nets, rulers, other fish, text overlays | strong clutter → crop helps |
| Are lesions localised or whole-body? | lesion on fin / gill / body vs generalised | localised + small fish → crop helps |
| Image resolution | from the audit's `sizes` | tiny images make cropping pointless |

Decision rule (to be applied, not pre-empted):

- **No YOLO** if ≥ 90 % of images contain one fish occupying ≥ 25 % of the
  frame and the centre-crop rarely removes it. Try `resize_mode="squash"`
  before anything else if the crop is the only issue.
- **YOLO justified** if a substantial share (rule of thumb ≥ 20 %) of images
  have multiple fish, small subjects, or heavy clutter, *and* a
  classification baseline (Layers 7–10 on the real split) shows errors
  concentrated in those images (check `misclassified.csv` from Layer 10).

The classifier baseline is measured **first**; YOLO is only worth its cost
if it fixes errors the baseline actually makes.

## If it is justified — the integration plan (not implemented)

```
image
  → detector (YOLO, pretrained COCO "fish"-capable or fine-tuned on boxes)
  → boxes + scores; keep the highest-scoring box above a threshold
  → crop with margin (e.g. 10 %) → PIL RGB
  → existing Layer 11 Predictor.predict(crop)        (unchanged)
  → result gains: bbox, detector_version, detection_confidence, fallback flag
```

Requirements that follow from the existing layers:

- A new `src/detect.py` owning the detector; `Predictor` gains an optional
  `detector` and falls back to the full frame (with a warning) when nothing
  is detected — no behaviour change for callers without a detector.
- Evaluation (Layer 10) must be run *with and without* cropping on the same
  validation split; adopt cropping only if F1-macro improves.
- Box annotations do not exist in a classification dataset; fine-tuning a
  detector would require labelling. A pretrained detector's zero-shot
  "fish" performance must itself be measured on the sample before use.
- The weight file, its licence, and its hash go through the same provenance
  checks as Layer 3.

## If it is not justified

Record the counts above in this file, set the README row to
"not required (evidence: …)", and leave the pipeline as it is.

## Acceptance

Not applicable — no implementation was made. The layer is complete as a
decision record with an executable procedure; it is re-opened by the first
dataset audit.
