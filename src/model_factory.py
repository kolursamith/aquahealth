"""One model factory for the CV experiments (Phase 12).

    build_model("efficientnet_b0", 8)   -> the project's existing EfficientNet-B0 classifier
                                            (src/model.py::build_classifier, unchanged)
    build_model("<hybrid key>", 8)      -> one of the five hybrids (src/hybrid_models.py)

Every model returned takes (N, 3, 224, 224) and returns (N, num_classes) logits.
The generic accessors below are the only structural assumptions the CV runner
makes: a `.classifier` head module and a backbone parameter set. For
EfficientNet the backbone is `model.features` (as src/train.py treats it); for
the hybrids it is `HybridBase.backbone_parameters()`.
"""

from __future__ import annotations

from typing import Iterator

from torch import nn
from torchvision.models import EfficientNet

from src.hybrid_models import HYBRID_MODELS, HybridBase, build_hybrid
from src.model import build_classifier

BASELINE_KEY = "efficientnet_b0"
MODEL_KEYS: tuple[str, ...] = (BASELINE_KEY, *HYBRID_MODELS)


def build_model(
    key: str, num_classes: int, *, pretrained: bool = True, dropout: float = 0.2
) -> nn.Module:
    if key == BASELINE_KEY:
        return build_classifier(num_classes, dropout=dropout, pretrained=pretrained)
    if key in HYBRID_MODELS:
        return build_hybrid(key, num_classes, pretrained=pretrained, dropout=dropout)
    raise ValueError(f"unknown model {key!r}; choose from {list(MODEL_KEYS)}")


def head_module(model: nn.Module) -> nn.Module:
    head = getattr(model, "classifier", None)
    if not isinstance(head, nn.Module):
        raise TypeError(f"{type(model).__name__} has no `.classifier` head module")
    return head


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(head_module(model).parameters())


def backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    if isinstance(model, HybridBase):
        return list(model.backbone_parameters())
    if isinstance(model, EfficientNet):
        return list(model.features.parameters())
    head_ids = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head_ids]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in backbone_parameters(model):
        p.requires_grad_(trainable)
    for p in head_parameters(model):
        p.requires_grad_(True)


def set_train_mode(model: nn.Module, backbone_frozen: bool) -> None:
    """`train()` for what is trained. With a frozen backbone the backbone modules stay
    in `eval()` so their BatchNorm statistics do not drift (the rule src/train.py
    applies block-wise to EfficientNet, here applied to the whole backbone)."""
    if backbone_frozen:
        model.eval()
        head_module(model).train()
    else:
        model.train()


def trainable_summary(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "head": sum(p.numel() for p in head_parameters(model)),
    }


def iter_model_keys() -> Iterator[str]:
    return iter(MODEL_KEYS)
