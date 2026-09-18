import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),
            nn.GELU(),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),
            nn.GELU(),

            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.block(x)


class Baseline3DCNN(nn.Module):

    def __init__(self, num_classes=12, dropout=0.2):

        super().__init__()

        self.features = nn.Sequential(
            ConvBlock3D(1, 16),
            ConvBlock3D(16, 32),
            ConvBlock3D(32, 64),
            ConvBlock3D(64, 128),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x
