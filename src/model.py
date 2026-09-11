"""Input -> EfficientNet-B0 -> 8-class classifier -> Logits.

Owner: Student 1
"""

import timm
import torch.nn as nn

from src.config import MODEL_NAME, NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=num_classes)
