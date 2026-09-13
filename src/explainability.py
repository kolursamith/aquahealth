"""Post-hoc visual explanation (Grad-CAM) for the frozen EfficientNet-B0 classifier.

Grad-CAM (Selvaraju et al., 2017) weights the activations of the last convolutional feature
map by the gradient of the target class score, producing a coarse map of the regions that
contributed most to that score. For torchvision's EfficientNet-B0 the last convolutional
block is `model.features[-1]` (Conv2dNormActivation 320→1280, output 1280×7×7 for a 224×224
input), immediately before the global average pool and the linear head — the standard
Grad-CAM target for this architecture.

Nothing here changes the model: parameters are never updated, `model.eval()` is kept, hooks
are removed after every call, and only the input tensor requires grad. The heatmap is a
statement about *model activations*, not a lesion localisation or a veterinary finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from src.predict import Predictor, load_input
from src.preprocessing import CLAHE, PreprocessConfig

TARGET_LAYER_NAME = "features.-1"  # model.features[-1]: last Conv2dNormActivation (1280 ch)
EXPLANATION_NOTE = (
    "This visualization shows model activation patterns (Grad-CAM on the last convolutional "
    "block). It is not a medical/veterinary diagnosis or a guaranteed lesion localization."
)


@dataclass(frozen=True)
class Explanation:
    target_class: str
    target_index: int
    heatmap: np.ndarray  # (h, w) float32 in [0, 1], the model-input (224×224) frame
    model_input: Image.Image  # what the classifier actually saw (after resize/crop)
    overlay_input: Image.Image  # heatmap blended onto the model input
    overlay_original: Image.Image  # heatmap mapped back onto the uploaded image
    target_layer: str = TARGET_LAYER_NAME

    @property
    def peak(self) -> tuple[float, float]:
        """(x, y) of the strongest activation as fractions of the model-input frame."""
        y, x = np.unravel_index(int(self.heatmap.argmax()), self.heatmap.shape)
        return x / max(1, self.heatmap.shape[1] - 1), y / max(1, self.heatmap.shape[0] - 1)


def target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """The last convolutional block of torchvision's EfficientNet (`features[-1]`)."""
    features = model.features
    assert isinstance(features, torch.nn.Sequential)
    return features[-1]


def model_input_image(rgb: Image.Image, preprocess: PreprocessConfig) -> Image.Image:
    """The geometric part of the eval pipeline as a PIL image (CLAHE → resize → crop)."""
    image = CLAHE(preprocess.clahe)(rgb) if preprocess.clahe is not None else rgb
    if preprocess.resize_mode == "crop":
        image = F.resize(image, preprocess.resize_size, antialias=True)
        return F.center_crop(image, [preprocess.image_size, preprocess.image_size])
    return F.resize(image, [preprocess.image_size, preprocess.image_size], antialias=True)


def crop_geometry(rgb: Image.Image, preprocess: PreprocessConfig) -> tuple[int, int, int, int]:
    """(resized_w, resized_h, left, top) of the centre crop in resized-image coordinates."""
    if preprocess.resize_mode != "crop":
        return rgb.width, rgb.height, 0, 0
    resized = F.resize(rgb, preprocess.resize_size, antialias=True)
    size = preprocess.image_size
    top = int(round((resized.height - size) / 2.0))
    left = int(round((resized.width - size) / 2.0))
    return resized.width, resized.height, left, top


def gradcam_heatmap(predictor: Predictor, tensor: torch.Tensor, class_index: int) -> np.ndarray:
    """Grad-CAM map in [0, 1] for `class_index`, upsampled to the input resolution."""
    model = predictor.model
    was_training = model.training
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = target_layer(model)
    fwd = layer.register_forward_hook(lambda _m, _i, out: activations.append(out))
    bwd = layer.register_full_backward_hook(
        lambda _m, _gi, gout: gradients.append(gout[0])  # type: ignore[index]
    )
    try:
        with torch.enable_grad():
            # Only the input requires grad: that is enough for gradients to reach the target
            # activations even when every backbone parameter is frozen; weights are never updated.
            inputs = tensor.unsqueeze(0).to(predictor.device).requires_grad_(True)
            logits = model(inputs)
            model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()
    finally:
        fwd.remove()
        bwd.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    act = activations[0].detach()  # (1, C, h, w)
    grad = gradients[0].detach()
    weights = grad.mean(dim=(2, 3), keepdim=True)  # global-average-pooled gradients
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, h, w)
    cam = torch.nn.functional.interpolate(
        cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    cam = cam - cam.min()
    if float(cam.max()) > 0:
        cam = cam / cam.max()
    return cam.float().cpu().numpy()


def colourise(heatmap: np.ndarray) -> np.ndarray:
    """Aqua→amber→coral colour ramp as an (h, w, 3) uint8 array (no matplotlib needed)."""
    stops = np.array([[10, 24, 48], [34, 211, 238], [245, 158, 11], [251, 113, 133]], np.float32)
    pos = np.array([0.0, 0.35, 0.7, 1.0], np.float32)
    h = np.clip(heatmap, 0.0, 1.0)
    out = np.empty((*h.shape, 3), np.float32)
    for c in range(3):
        out[..., c] = np.interp(h, pos, stops[:, c])
    return out.astype(np.uint8)


def blend(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Alpha-blend a colourised heatmap over `base`; the blend strength follows the heatmap."""
    base_arr = np.asarray(base.convert("RGB")).astype(np.float32)
    colour = colourise(heatmap).astype(np.float32)
    weight = (alpha * np.clip(heatmap, 0, 1))[..., None]
    mixed = base_arr * (1 - weight) + colour * weight
    return Image.fromarray(mixed.clip(0, 255).astype(np.uint8))


def map_heatmap_to_original(
    heatmap: np.ndarray, rgb: Image.Image, preprocess: PreprocessConfig
) -> np.ndarray:
    """Place the model-frame heatmap back into the uploaded image's coordinates."""
    rw, rh, left, top = crop_geometry(rgb, preprocess)
    canvas = np.zeros((rh, rw), np.float32)
    size = heatmap.shape[0]
    canvas[top : top + size, left : left + size] = heatmap[: rh - top, : rw - left]
    return (
        np.asarray(
            Image.fromarray((canvas * 255).astype(np.uint8)).resize(
                rgb.size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )


def generate_gradcam(predictor: Predictor, image, class_index: int | None = None) -> Explanation:
    """Grad-CAM explanation for `image` (any input `Predictor.predict` accepts).

    `class_index` defaults to the model's own top prediction, computed here with the same
    transform so the explanation and the prediction refer to the same forward pass.
    """
    rgb, _ = load_input(image)
    tensor = predictor.transform(rgb)
    if class_index is None:
        with torch.inference_mode():
            logits = predictor.model(tensor.unsqueeze(0).to(predictor.device))
        class_index = int(logits.argmax(dim=1).item())
    heatmap = gradcam_heatmap(predictor, tensor, class_index)
    model_input = model_input_image(rgb, predictor.checkpoint.preprocess)
    return Explanation(
        target_class=predictor.class_names[class_index],
        target_index=class_index,
        heatmap=heatmap,
        model_input=model_input,
        overlay_input=blend(model_input, heatmap),
        overlay_original=blend(
            rgb, map_heatmap_to_original(heatmap, rgb, predictor.checkpoint.preprocess)
        ),
    )
