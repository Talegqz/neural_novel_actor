"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        
        self.actvn = nn.LeakyReLU(0.05,inplace=False)
        self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        return self.actvn(self.bn(self.conv(x)))

## Copy from PointNerf

class ImageEncoder(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, in_dims = 4, mode = 'intermediate'):
        super(ImageEncoder, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(in_dims, 8, 3, 1, 1),
                        ConvBnReLU(8, 8, 3, 1, 1))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2),
                        ConvBnReLU(16, 16, 3, 1, 1),
                        ConvBnReLU(16, 16, 3, 1, 1))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2),
                        ConvBnReLU(32, 32, 3, 1, 1),
                        ConvBnReLU(32, 32, 3, 1, 1))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.mode = mode

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2,
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # B, _, H, W = x.shape
        # x = x.reshape(B, 4, H, W)
        x1 = self.conv0(x)  # (B, 8, H, W)
        x2 = self.conv1(x1)  # (B, 16, H//2, W//2)
        x3 = self.conv2(x2)  # (B, 32, H//4, W//4)
        x3 = self.toplayer(x3)  # (B, 32, H//4, W//4)
        if self.mode == 'intermediate':
            return [x1, x2, x3], None
        elif self.mode == 'all':
            return [x, x1, x2, x3], None
        else:
            return [x3], None