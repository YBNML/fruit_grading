#!/usr/bin/env python3
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
