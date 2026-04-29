"""Chess board corner detector.

Heatmap-based regression of 4 board corners (a8, h8, h1, a1) — order encodes
orientation so the warped output is always canonical.

Input  : RGB image, resized to 512×512
Output : 4-channel heatmap (one per corner), soft-argmax → (x, y) per corner
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


CORNER_ORDER = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
# In ChessReD convention:
#   top_left = a8, top_right = h8, bottom_right = h1, bottom_left = a1
# i.e. clockwise from a8.

INPUT_SIZE   = 512
HEATMAP_SIZE = 128


class CornerDetector(nn.Module):
    """ResNet18 encoder + lightweight upsample head → 4 corner heatmaps."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        # Strip avgpool + fc — keep features
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1   # 64 ch  / 4
        self.layer2 = backbone.layer2   # 128 ch / 8
        self.layer3 = backbone.layer3   # 256 ch / 16
        self.layer4 = backbone.layer4   # 512 ch / 32

        # Decoder: upsample to 128×128, output 4 channels
        self.up = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64
            nn.Conv2d(128, 64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128
            nn.Conv2d(64, 4, 1)   # 4 corner heatmaps
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        h = self.up(x)
        return h   # (B, 4, 128, 128)


def soft_argmax_2d(heatmaps: torch.Tensor, temperature: float = 20.0) -> torch.Tensor:
    """Sub-pixel-accurate argmax over the last 2 dims.

    Args:
        heatmaps: (B, C, H, W)

    Returns:
        (B, C, 2)  — (x, y) in [0, 1] for each channel
    """
    B, C, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, C, -1)
    flat = flat - flat.amax(dim=-1, keepdim=True)
    sm = F.softmax(flat * temperature, dim=-1).reshape(B, C, H, W)

    ys = torch.linspace(0, 1, H, device=heatmaps.device)
    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    x_coord = (sm * xx).sum(dim=(2, 3))   # (B, C)
    y_coord = (sm * yy).sum(dim=(2, 3))
    return torch.stack([x_coord, y_coord], dim=-1)   # (B, C, 2)


def make_target_heatmap(corners_norm: torch.Tensor,
                         size: int = HEATMAP_SIZE,
                         sigma: float = 2.0,
                         normalize: bool = False) -> torch.Tensor:
    """Build Gaussian target heatmaps from normalized (x, y) corners.

    Args:
        corners_norm: (B, 4, 2) — (x, y) in [0, 1]
        size: heatmap spatial size

    Returns:
        (B, 4, size, size) — Gaussian peaks at each corner
    """
    B, C, _ = corners_norm.shape
    device = corners_norm.device
    yy, xx = torch.meshgrid(
        torch.arange(size, device=device).float(),
        torch.arange(size, device=device).float(),
        indexing='ij',
    )
    cx = corners_norm[..., 0:1].unsqueeze(-1) * (size - 1)   # (B, C, 1, 1)
    cy = corners_norm[..., 1:2].unsqueeze(-1) * (size - 1)
    d2 = (xx - cx).pow(2) + (yy - cy).pow(2)
    heatmap = torch.exp(-d2 / (2 * sigma ** 2))
    if normalize:
        heatmap = heatmap / heatmap.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)
    return heatmap


@torch.no_grad()
def predict_corners(model: nn.Module, image_bgr, device='cuda'):
    """Run inference on a single BGR image and return corners in image coords.

    Args:
        model: trained CornerDetector
        image_bgr: H×W×3 uint8 BGR (from cv2.imread)
        device: torch device

    Returns:
        dict with keys top_left/top_right/bottom_right/bottom_left,
        each (x, y) in original image pixel space.
    """
    import cv2
    import numpy as np

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rsz = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
    t = torch.from_numpy(rsz.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    # ImageNet normalization (matches ResNet pretraining)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std

    model.eval()
    hm = model(t)
    coords = soft_argmax_2d(hm)[0].cpu().numpy()   # (4, 2) in [0, 1]

    out = {}
    for i, key in enumerate(CORNER_ORDER):
        x = float(coords[i, 0]) * w
        y = float(coords[i, 1]) * h
        out[key] = [x, y]
    return out
