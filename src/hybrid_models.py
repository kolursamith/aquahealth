"""The five hybrid classifiers of the new experiment (Phase 11) — implementation only.

Common interface (`build_hybrid`): every model takes the SAME input the existing
pipeline produces — a float32 batch (N, 3, 224, 224) from
src/preprocessing.py (CLAHE -> resize -> ImageNet normalisation) — and returns
logits (N, num_classes) for the unified classes. Each model exposes
`.classifier` (its head) and `backbone_parameters()` so the existing staged
fine-tuning idea (head first, then backbone) can be applied.

Paper provenance (results/paper_technique_matrix.md, a full-text read of the
five supplied papers):

    ResNet + Attention      P4 SLCAM-AquaNet: ResNet18/50 + SLCAM (4 branches: horizontal,
                            vertical, spatial 7x7 conv, channel avg+max-pool MLP). The exact
                            fusion formula is NOT STATED in the extracted text; the composition
                            below is ours and is marked as such.
    CNN + ViT + LSTM        no supplied paper (P5 uses a frozen DINOv2 ViT for anomaly
                            detection, no CNN/LSTM) -> engineering adaptation, documented.
    CNN + BiLSTM            no supplied paper uses an LSTM (single-image task, no sequence)
                            -> adaptation: the CNN feature map is read as a spatial token
                            sequence (raster order) and modelled by a BiLSTM.
    YOLO + EfficientNet     no supplied paper uses YOLO (P5 names it only as the supervised
                            alternative). Our task is classification and the datasets have no
                            boxes, so "YOLO" here is a YOLOv8-style CSPDarknet feature extractor
                            (C2f + SPPF blocks, built in-repo, randomly initialised: no
                            ultralytics dependency, no detection head, no bounding boxes) whose
                            pooled features are fused with EfficientNet-B0's.
    YOLO + Transformer      same YOLO-style backbone, its 7x7 feature map tokenised and encoded
                            by a Transformer encoder (ViT-style) before the head.

Pretrained components: EfficientNet-B0 (torchvision IMAGENET1K_V1, the project's
existing backbone) and ResNet18/50 (torchvision ImageNet weights) when
`pretrained=True`; the Transformer, LSTM, attention and YOLO-style modules are
trained from scratch (no pretrained weights exist for them here). Nothing is
trained in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from src.model import MIN_NUM_CLASSES, build_efficientnet_b0

EFFICIENTNET_FEATURES = 1280
TOKEN_DIM = 256


def _head(in_features: int, num_classes: int, dropout: float) -> nn.Sequential:
    if num_classes < MIN_NUM_CLASSES:
        raise ValueError(f"num_classes must be >= {MIN_NUM_CLASSES}, got {num_classes}")
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {dropout}")
    linear = nn.Linear(in_features, num_classes)
    nn.init.uniform_(linear.weight, -(in_features**-0.5), in_features**-0.5)
    nn.init.zeros_(linear.bias)
    return nn.Sequential(nn.Dropout(p=dropout), linear)


class HybridBase(nn.Module):
    """Shared contract: forward(images) -> logits; `.classifier`; backbone_parameters()."""

    architecture: str = "hybrid"
    paper_source: str = ""
    adaptation: str = ""
    pretrained_components: tuple[str, ...] = ()
    classifier: nn.Sequential

    def backbone_parameters(self) -> Iterator[nn.Parameter]:
        head_ids = {id(p) for p in self.classifier.parameters()}
        return (p for p in self.parameters() if id(p) not in head_ids)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad_(trainable)


# --- building blocks --------------------------------------------------------------------------


class TokenEncoder(nn.Module):
    """Feature map (N, C, H, W) -> token sequence (N, H*W [+1 cls], d) with learned positions."""

    def __init__(self, in_channels: int, dim: int, tokens: int, cls_token: bool) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim)) if cls_token else None
        self.positions = nn.Parameter(torch.zeros(1, tokens + (1 if cls_token else 0), dim))
        nn.init.trunc_normal_(self.positions, std=0.02)
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        tokens = self.project(feature_map).flatten(2).transpose(1, 2)  # (N, HW, d)
        if self.cls is not None:
            tokens = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        if tokens.shape[1] != self.positions.shape[1]:
            raise ValueError(
                f"expected {self.positions.shape[1]} tokens, got {tokens.shape[1]} "
                "(input must be 224x224)"
            )
        return tokens + self.positions


def transformer_encoder(dim: int, layers: int, heads: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=4 * dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class ConvBNSiLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1) -> None:
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.SiLU(True)
        )


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool) -> None:
        super().__init__()
        self.conv1 = ConvBNSiLU(channels, channels, 3)
        self.conv2 = ConvBNSiLU(channels, channels, 3)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y


class C2f(nn.Module):
    """YOLOv8 'C2f' block: split, n bottlenecks, concatenate all intermediate outputs."""

    def __init__(self, cin: int, cout: int, n: int, shortcut: bool) -> None:
        super().__init__()
        self.hidden = cout // 2
        self.conv1 = ConvBNSiLU(cin, 2 * self.hidden, 1)
        self.blocks = nn.ModuleList(Bottleneck(self.hidden, shortcut) for _ in range(n))
        self.conv2 = ConvBNSiLU((2 + n) * self.hidden, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.conv1(x).chunk(2, dim=1)
        outputs = [a, b]
        for block in self.blocks:
            outputs.append(block(outputs[-1]))
        return self.conv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    """Spatial pyramid pooling - fast (three chained 5x5 max-pools)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = channels // 2
        self.conv1 = ConvBNSiLU(channels, hidden, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.conv2 = ConvBNSiLU(4 * hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))


class YoloStyleBackbone(nn.Module):
    """YOLOv8-n-shaped CSPDarknet backbone (stem + 4 stages of Conv/C2f, SPPF), stride 32:
    (N, 3, 224, 224) -> (N, 256, 7, 7). Used purely as a feature extractor — there is no
    detection neck or head, and no pretrained weights are loaded (none are available
    without the ultralytics package, which is not a project dependency)."""

    out_channels = 256

    def __init__(self, width: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)) -> None:
        super().__init__()
        w0, w1, w2, w3, w4 = width
        self.stem = ConvBNSiLU(3, w0, 3, 2)  # /2
        self.stage1 = nn.Sequential(ConvBNSiLU(w0, w1, 3, 2), C2f(w1, w1, 1, True))  # /4
        self.stage2 = nn.Sequential(ConvBNSiLU(w1, w2, 3, 2), C2f(w2, w2, 2, True))  # /8
        self.stage3 = nn.Sequential(ConvBNSiLU(w2, w3, 3, 2), C2f(w3, w3, 2, True))  # /16
        self.stage4 = nn.Sequential(ConvBNSiLU(w3, w4, 3, 2), C2f(w4, w4, 1, True), SPPF(w4))
        self.out_channels = w4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))


class SLCAM(nn.Module):
    """Spatial-Location-Channel Attention Module after P4 (SLCAM-AquaNet).

    From the paper text [A]: four branches — horizontal, vertical, spatial (7x7 conv),
    channel (avg + max pooling through an MLP) — with the spatial and channel paths fused.
    NOT STATED in the paper text: the exact fusion formula. Our composition [D]: the
    input is multiplied by the channel gate, then by the product of the location gate
    (horizontal x vertical, coordinate-attention style) and the spatial gate."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(True), nn.Conv2d(hidden, channels, 1)
        )
        self.location = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(True)
        )
        self.horizontal = nn.Conv2d(hidden, channels, 1)
        self.vertical = nn.Conv2d(hidden, channels, 1)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        channel_gate = torch.sigmoid(
            self.channel_mlp(x.mean((2, 3), keepdim=True))
            + self.channel_mlp(x.amax((2, 3), keepdim=True))
        )
        x = x * channel_gate
        pooled_h = x.mean(3, keepdim=True)  # (n, c, h, 1)  vertical profile
        pooled_w = x.mean(2, keepdim=True).transpose(2, 3)  # (n, c, w, 1) horizontal profile
        joint = self.location(torch.cat([pooled_h, pooled_w], dim=2))
        gate_h, gate_w = joint.split([h, w], dim=2)
        location_gate = torch.sigmoid(self.vertical(gate_h)) * torch.sigmoid(
            self.horizontal(gate_w.transpose(2, 3))
        )
        spatial_gate = torch.sigmoid(
            self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1))
        )
        return x * location_gate * spatial_gate


# --- the five models ----------------------------------------------------------------------------


class CnnVitLstm(HybridBase):
    """EfficientNet-B0 feature map -> ViT-style Transformer encoder over 49 spatial tokens
    -> LSTM over the encoded token sequence -> last hidden state -> head."""

    architecture = "CNN + Vision Transformer + LSTM"
    paper_source = "none of the supplied papers (P5 has a ViT only); engineering adaptation"
    adaptation = (
        "single images have no temporal sequence: the 7x7 feature map is read as a 49-token "
        "raster sequence; the Transformer contextualises it, the LSTM summarises it"
    )
    pretrained_components = ("EfficientNet-B0 IMAGENET1K_V1 (torchvision)",)

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,
        dropout: float = 0.2,
        layers: int = 2,
        heads: int = 4,
        lstm_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = build_efficientnet_b0(pretrained=pretrained).features
        self.tokens = TokenEncoder(EFFICIENTNET_FEATURES, TOKEN_DIM, tokens=49, cls_token=False)
        self.transformer = transformer_encoder(TOKEN_DIM, layers, heads, dropout)
        self.lstm = nn.LSTM(TOKEN_DIM, lstm_hidden, batch_first=True)
        self.classifier = _head(lstm_hidden, num_classes, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.transformer(self.tokens(self.cnn(images)))
        _, (hidden, _) = self.lstm(tokens)
        return self.classifier(hidden[-1])


class YoloEfficientNet(HybridBase):
    """Two feature extractors on the same image — a YOLOv8-style CSPDarknet backbone
    (from scratch) and EfficientNet-B0 (pretrained) — globally pooled, concatenated, head."""

    architecture = "YOLO + EfficientNet"
    paper_source = "none of the supplied papers uses YOLO; engineering adaptation"
    adaptation = (
        "classification task, no bounding boxes: YOLO contributes a CSPDarknet feature "
        "extractor (no detection neck/head, no pretrained weights); features are fused with "
        "EfficientNet-B0 by concatenation after global average pooling"
    )
    pretrained_components = ("EfficientNet-B0 IMAGENET1K_V1 (torchvision)",)

    def __init__(self, num_classes: int, *, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        self.yolo = YoloStyleBackbone()
        self.efficientnet = build_efficientnet_b0(pretrained=pretrained).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = _head(
            self.yolo.out_channels + EFFICIENTNET_FEATURES, num_classes, dropout
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        a = self.pool(self.yolo(images)).flatten(1)
        b = self.pool(self.efficientnet(images)).flatten(1)
        return self.classifier(torch.cat([a, b], dim=1))


class CnnBiLstm(HybridBase):
    """EfficientNet-B0 feature map -> 49-token raster sequence -> BiLSTM -> time-mean -> head."""

    architecture = "CNN + BiLSTM"
    paper_source = "none of the supplied papers uses an LSTM; engineering adaptation"
    adaptation = "spatial token sequence (raster order) in place of a temporal sequence"
    pretrained_components = ("EfficientNet-B0 IMAGENET1K_V1 (torchvision)",)

    def __init__(
        self, num_classes: int, *, pretrained: bool = True, dropout: float = 0.2, hidden: int = 256
    ) -> None:
        super().__init__()
        self.cnn = build_efficientnet_b0(pretrained=pretrained).features
        self.project = nn.Conv2d(EFFICIENTNET_FEATURES, TOKEN_DIM, 1)
        self.bilstm = nn.LSTM(TOKEN_DIM, hidden, batch_first=True, bidirectional=True)
        self.classifier = _head(2 * hidden, num_classes, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.project(self.cnn(images)).flatten(2).transpose(1, 2)
        out, _ = self.bilstm(tokens)
        return self.classifier(out.mean(1))


class ResNetAttention(HybridBase):
    """torchvision ResNet (18 or 50, ImageNet weights) with SLCAM inserted after layer3 and
    layer4, as P4 inserts its module in the backbone; global pool; head."""

    architecture = "ResNet + Attention (SLCAM)"
    paper_source = "P4 SLCAM-AquaNet (slcam-aquanet.pdf): ResNet18/50 + SLCAM"
    adaptation = (
        "SLCAM fusion formula not stated in the paper text; composed as documented in SLCAM"
    )
    pretrained_components = ("ResNet18/ResNet50 ImageNet weights (torchvision)",)

    def __init__(
        self, num_classes: int, *, pretrained: bool = True, dropout: float = 0.2, depth: int = 18
    ) -> None:
        super().__init__()
        if depth == 18:
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif depth == 50:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            raise ValueError("depth must be 18 or 50 (the depths P4 evaluates)")
        self.depth = depth
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        last3 = resnet.layer3[-1]
        c3 = (last3.conv3 if hasattr(last3, "conv3") else last3.conv2).out_channels
        c4 = resnet.fc.in_features
        self.attention3 = SLCAM(c3)
        self.attention4 = SLCAM(c4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = _head(c4, num_classes, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer2(self.layer1(self.stem(images)))
        x = self.attention3(self.layer3(x))
        x = self.attention4(self.layer4(x))
        return self.classifier(self.pool(x).flatten(1))


class YoloTransformer(HybridBase):
    """YOLOv8-style CSPDarknet backbone -> 49 tokens + CLS -> Transformer encoder -> CLS -> head."""

    architecture = "YOLO + Transformer"
    paper_source = "none of the supplied papers uses YOLO; P5's ViT is a frozen DINOv2; adaptation"
    adaptation = (
        "YOLO = CSPDarknet feature extractor (no detection, no pretrained weights); the "
        "Transformer encoder is ViT-style over the 7x7 feature map with a CLS token"
    )
    pretrained_components = ()

    def __init__(
        self,
        num_classes: int,
        *,
        pretrained: bool = True,  # accepted for interface parity; nothing pretrained exists here
        dropout: float = 0.2,
        layers: int = 4,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.yolo = YoloStyleBackbone()
        self.tokens = TokenEncoder(self.yolo.out_channels, TOKEN_DIM, tokens=49, cls_token=True)
        self.transformer = transformer_encoder(TOKEN_DIM, layers, heads, dropout)
        self.norm = nn.LayerNorm(TOKEN_DIM)
        self.classifier = _head(TOKEN_DIM, num_classes, dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        encoded = self.transformer(self.tokens(self.yolo(images)))
        return self.classifier(self.norm(encoded[:, 0]))


# --- registry ------------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridSpec:
    key: str
    builder: Callable[..., HybridBase]
    architecture: str
    paper_source: str
    adaptation: str
    pretrained_components: tuple[str, ...]


HYBRID_MODELS: dict[str, HybridSpec] = {
    key: HybridSpec(
        key, cls, cls.architecture, cls.paper_source, cls.adaptation, cls.pretrained_components
    )
    for key, cls in (
        ("cnn_vit_lstm", CnnVitLstm),
        ("yolo_efficientnet", YoloEfficientNet),
        ("cnn_bilstm", CnnBiLstm),
        ("resnet_attention", ResNetAttention),
        ("yolo_transformer", YoloTransformer),
    )
}


def build_hybrid(key: str, num_classes: int, *, pretrained: bool = True, **kwargs) -> HybridBase:
    """The one entry point: `build_hybrid("resnet_attention", 8)` -> model whose
    forward((N,3,224,224)) -> (N, 8) logits."""
    if key not in HYBRID_MODELS:
        raise ValueError(f"unknown hybrid {key!r}; choose from {sorted(HYBRID_MODELS)}")
    return HYBRID_MODELS[key].builder(num_classes, pretrained=pretrained, **kwargs)
