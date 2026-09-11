"""EfficientNet-B0 backbone (Layer 3).

Input (N, 3, H, W), ImageNet-normalized -> EfficientNet-B0 -> logits.

The pretrained weights are torchvision's ImageNet-1K general-purpose image
classification weights. They are NOT fish-disease-trained. At this layer the
model keeps its stock 1000-way ImageNet head; the disease classifier head is
introduced by Layer 4 and is configurable there.

Owner: Student 1
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import EfficientNet, EfficientNet_B0_Weights, efficientnet_b0

PRETRAINED_WEIGHTS = EfficientNet_B0_Weights.IMAGENET1K_V1
IMAGENET_NUM_CLASSES = len(PRETRAINED_WEIGHTS.meta["categories"])
INPUT_CHANNELS = 3


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
