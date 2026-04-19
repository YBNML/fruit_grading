#!/usr/bin/env python3
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ResNet50_Weights, DenseNet201_Weights


def build_model(name="resnet50", num_outputs=3, pretrained=True):
    if name == "resnet50":
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_outputs)
    elif name == "densenet201":
        w = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.densenet201(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, num_outputs)
    else:
        raise ValueError(f"unknown model: {name}")
    return m


def _build_feature_extractor(name, pretrained):
    """Return (backbone_without_head, feature_dim)."""
    if name == "resnet50":
        w = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.resnet50(weights=w)
        dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "densenet201":
        w = DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.densenet201(weights=w)
        dim = m.classifier.in_features
        m.classifier = nn.Identity()
    else:
        raise ValueError(f"unknown model: {name}")
    return m, dim


class MultiViewModel(nn.Module):
    """Shared backbone across 3 views with per-view feature concat.

    Backbone stays the same as single-view (ImageNet pretrained ResNet50/DenseNet201),
    so the exact same weights are reused — the only new parameters are the final head
    that maps (3 * feature_dim) -> num_outputs. Because views are fed in a fixed
    order (t1, t2, b1), the concat preserves view identity, letting the head learn
    per-view contribution weights.
    """

    def __init__(self, backbone="resnet50", num_outputs=1, pretrained=True, num_views=3):
        super().__init__()
        self.num_views = num_views
        self.backbone, feat_dim = _build_feature_extractor(backbone, pretrained)
        self.head = nn.Linear(num_views * feat_dim, num_outputs)

    def forward(self, x):
        # x: (B, V, C, H, W)  where V = num_views
        B, V = x.shape[:2]
        assert V == self.num_views, f"expected {self.num_views} views, got {V}"
        flat = x.reshape(B * V, *x.shape[2:])
        feats = self.backbone(flat)                # (B*V, feat_dim)
        feats = feats.reshape(B, V * feats.size(-1))
        return self.head(feats)
