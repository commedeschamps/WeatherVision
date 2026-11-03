import torch
from torch import nn
from torchvision.models import (ConvNeXt_Tiny_Weights, MobileNet_V2_Weights,
                                convnext_tiny, mobilenet_v2)


class SimpleCNN(nn.Module):
    """Three small convolutional blocks and global pooling; trained from scratch."""
    def __init__(self):
        super().__init__()
        layers = []
        channels = 3
        for out in (16, 32, 64):
            layers.extend([nn.Conv2d(channels, out, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out), nn.ReLU(), nn.MaxPool2d(2)])
            channels = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, 4))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(architecture: str, pretrained=True, freeze_backbone=True) -> nn.Module:
    if architecture == "cnn":
        model = SimpleCNN()
    elif architecture == "mobilenet":
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[1] = nn.Linear(model.last_channel, 4)
    elif architecture == "convnext":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 4)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    if architecture != "cnn" and freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        # Only the new linear head trains; ConvNeXt LayerNorm remains pretrained.
        head = model.classifier[1] if architecture == "mobilenet" else model.classifier[2]
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    return model


def set_training_mode(model, architecture, freeze_backbone):
    model.train()
    if architecture != "cnn" and freeze_backbone:
        # Keep pretrained BatchNorm statistics and stochastic-depth behavior fixed.
        model.features.eval()


def parameter_counts(model):
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
