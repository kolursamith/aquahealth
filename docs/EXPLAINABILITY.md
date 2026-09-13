# Explainability — Grad-CAM

## What is computed
`src/explainability.generate_gradcam(predictor, image)` runs Grad-CAM (Selvaraju et al., 2017) on the **frozen** final
EfficientNet-B0:

1. Target layer: `model.features[-1]` — torchvision's last `Conv2dNormActivation` (320 → 1280 channels), whose output
   is 1280 × 7 × 7 for a 224 × 224 input, immediately before global average pooling and the linear head. Chosen by
   inspecting the actual model, not assumed.
2. A forward hook stores the layer's activations; a full backward hook stores the gradient of the target-class logit
   with respect to them (the target defaults to the model's own top prediction, computed with the same transform).
3. Weights = gradients averaged over the 7 × 7 map; CAM = ReLU(Σ weight × activation); bilinear upsampling to 224 × 224;
   min–max normalisation to [0, 1].
4. Outputs: the heatmap, the model input (what the classifier saw after CLAHE → resize → crop), the overlay on that
   input, and the overlay mapped back onto the uploaded image through the exact resize/centre-crop geometry.

Only the input tensor requires grad; `model.eval()` is kept; hooks are removed and gradients cleared in a `finally`.
Tests (`tests/test_explainability.py`) verify shape/range, determinism, that model weights and predictions are
unchanged before/after, and the back-projection geometry. Cost ≈ 75 ms per image on Apple-silicon MPS after warm-up.

## How to read it
Warm colours mark regions whose activations in the last convolutional block pushed the score of the predicted class
highest. This is **evidence of model activation**, not a biological finding: the map is coarse (7 × 7 upsampled),
class-discriminative rather than lesion-segmenting, and can highlight context (fins, background, body texture) that the
model has learned to associate with a class.

## What it is not
- Not guaranteed lesion localisation and not a veterinary finding — the UI states this on every result.
- Not proof that a specific symptom (e.g. "red lesion") was detected; the UI never uses such language.
- Not a bounding box; no detection model exists and no boxes are drawn.
- Not a substitute for a professional inspection.
