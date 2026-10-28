"""Compact CNN baseline with lazy PyTorch dependency loading."""

from __future__ import annotations

from typing import Any


def create_model(
    *,
    input_channels: int = 1,
    num_classes: int = 4,
    dropout: float = 0.25,
) -> Any:
    """Create a compact CNN without downloading pretrained parameters."""
    if input_channels not in {1, 3}:
        raise ValueError("input_channels must be 1 or 3.")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1).")
    try:
        import torch.nn as nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required to construct the CNN classifier. See the "
            "README.md platform-specific setup commands."
        ) from error

    class CompactCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes),
            )

        def forward(self, inputs: Any) -> Any:
            return self.classifier(self.features(inputs))

    return CompactCNN()
