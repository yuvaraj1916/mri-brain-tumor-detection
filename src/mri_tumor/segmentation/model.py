"""Compact, configurable U-Net for binary tumor-region segmentation."""

from __future__ import annotations

from typing import Any


def create_unet(*, input_channels: int = 1, base_channels: int = 16) -> Any:
    """Build a U-Net from scratch; output is one unnormalized logit per pixel."""
    if input_channels not in {1, 3}:
        raise ValueError("input_channels must be 1 or 3.")
    if base_channels < 4:
        raise ValueError("base_channels must be at least 4.")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError("PyTorch is required to construct the U-Net.") from error

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, inputs: Any) -> Any:
            return self.layers(inputs)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            widths = [base_channels * (2**index) for index in range(4)]
            self.encoder = nn.ModuleList()
            previous = input_channels
            for width in widths:
                self.encoder.append(DoubleConv(previous, width))
                previous = width
            self.pool = nn.MaxPool2d(kernel_size=2)
            self.bottleneck = DoubleConv(widths[-1], widths[-1] * 2)
            self.upconvs = nn.ModuleList()
            self.decoder = nn.ModuleList()
            previous = widths[-1] * 2
            for width in reversed(widths):
                self.upconvs.append(nn.ConvTranspose2d(previous, width, 2, stride=2))
                self.decoder.append(DoubleConv(width * 2, width))
                previous = width
            self.output = nn.Conv2d(widths[0], 1, kernel_size=1)

        def forward(self, inputs: Any) -> Any:
            skips = []
            current = inputs
            for layer in self.encoder:
                current = layer(current)
                skips.append(current)
                current = self.pool(current)
            current = self.bottleneck(current)
            for upconv, decoder, skip in zip(
                self.upconvs, self.decoder, reversed(skips), strict=True
            ):
                current = upconv(current)
                if current.shape[-2:] != skip.shape[-2:]:
                    current = functional.interpolate(
                        current, size=skip.shape[-2:], mode="bilinear", align_corners=False
                    )
                current = decoder(torch.cat((skip, current), dim=1))
            return self.output(current)

    return UNet()
