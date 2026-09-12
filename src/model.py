"""EfficientNet-B0 backbone (Layer 3) and configurable classifier head (Layer 4).

Input (N, 3, H, W), ImageNet-normalized -> EfficientNet-B0 -> logits.

The pretrained weights are torchvision's ImageNet-1K general-purpose image
classification weights. They are NOT fish-disease-trained.

`build_efficientnet_b0` returns the stock model with its 1000-way ImageNet
head. `build_classifier` swaps that head for one with a caller-supplied
number of classes; the count is a required argument because the real class
list is only known once the dataset has been inspected.

Owner: Student 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import EfficientNet, EfficientNet_B0_Weights, efficientnet_b0

PRETRAINED_WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
IMAGENET_NUM_CLASSES = len(PRETRAINED_WEIGHTS.meta["categories"])
INPUT_CHANNELS = 3
STOCK_DROPOUT = 0.2
MIN_NUM_CLASSES = 2


@dataclass(frozen=True)
class ModelSummary:
    architecture: str
    total_parameters: int
    trainable_parameters: int
    output_features: int


def build_efficientnet_b0(pretrained: bool = True) -> EfficientNet:
    """Build EfficientNet-B0 on the CPU; the caller moves it to a device.

    `pretrained=True` loads `PRETRAINED_WEIGHTS`, downloading (with hash
    verification) into the torch hub cache on first use.
    """
    weights = PRETRAINED_WEIGHTS if pretrained else None
    return efficientnet_b0(weights=weights)


def build_classifier(
    num_classes: int,
    *,
    dropout: float = STOCK_DROPOUT,
    freeze_backbone: bool = False,
    pretrained: bool = True,
) -> EfficientNet:
    """EfficientNet-B0 with a fresh `Dropout -> Linear(1280, num_classes)` head.

    The backbone (`features`) is left exactly as loaded; only the head is
    replaced. The head is initialised uniform ±1/sqrt(in_features) with zero
    bias (fan-in scaling), deterministic under `set_seed`.

    torchvision's own scheme scales by 1/sqrt(out_features), which is only
    well-conditioned for its 1000-way head: for a handful of classes it
    yields initial logits with std ≈ 4 and an initial loss ≈ 3x ln(K).
    Fan-in scaling keeps the initial logits near zero for any class count.
    """
    if num_classes < MIN_NUM_CLASSES:
        raise ValueError(f"num_classes must be >= {MIN_NUM_CLASSES}, got {num_classes}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")

    model = build_efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[-1].in_features

    head = nn.Linear(in_features, num_classes)
    init_range = 1.0 / math.sqrt(in_features)
    nn.init.uniform_(head.weight, -init_range, init_range)
    nn.init.zeros_(head.bias)
    model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), head)

    if freeze_backbone:
        set_backbone_trainable(model, False)
    return model


def set_backbone_trainable(model: EfficientNet, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad_(trainable)


def summarize(model: nn.Module) -> ModelSummary:
    parameters = list(model.parameters())
    linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
    output_features = linear_layers[-1].out_features
    return ModelSummary(
        architecture=type(model).__name__,
        total_parameters=sum(p.numel() for p in parameters),
        trainable_parameters=sum(p.numel() for p in parameters if p.requires_grad),
        output_features=output_features,
    )


@torch.no_grad()
def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)
