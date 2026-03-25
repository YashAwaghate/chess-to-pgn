"""
CVCHESS-inspired residual CNN for chess piece classification.

13 classes: empty + 6 white pieces + 6 black pieces.
Architecture follows the CVCHESS paper (ArXiv 2511.11522):
  Stem → 3 residual layers (64→128→256→512) → AdaptiveAvgPool → Linear
"""

import torch
import torch.nn as nn

CLASS_NAMES = [
    'empty',
    'P', 'N', 'B', 'R', 'Q', 'K',   # white pieces
    'p', 'n', 'b', 'r', 'q', 'k',   # black pieces
]

# Mapping from FEN character to class index
FEN_TO_CLASS = {name: i for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_FEN = {i: name for i, name in enumerate(CLASS_NAMES)}


class ResidualBlock(nn.Module):
    """Two-conv residual block with optional 1×1 skip projection and max-pool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

        # 1×1 projection for dimension mismatch
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        out = self.pool(out)
        return out


class ChessPieceCNN(nn.Module):
    """
    Residual CNN for 50×50 RGB chess square patches → 13 classes.

    Stem: Conv(3→64) → BN → ReLU → MaxPool
    Layer 1: ResidualBlock(64 → 128)
    Layer 2: ResidualBlock(128 → 256)
    Layer 3: ResidualBlock(256 → 512)
    Head: AdaptiveAvgPool(1) → Flatten → Linear(512 → 13)
    """

    def __init__(self, num_classes=13):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.layer1 = ResidualBlock(64, 128)
        self.layer2 = ResidualBlock(128, 256)
        self.layer3 = ResidualBlock(256, 512)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        return x
