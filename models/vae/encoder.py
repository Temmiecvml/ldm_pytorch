import torch
import torch.nn.functional as F
import torch.nn as nn
from decoder import VResidualBlock
from attention import VAttentionBlock


class PadBlock(nn.Module):
    def __init__(self, pad):
        super(PadBlock, self).__init__()
        self.pad = pad  # (left, right, top, bottom)

    def forward(self, x):
        return F.pad(x, self.pad)


class VEncoder(nn.Module):
    def __init__(self):
        super(VEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),  # resolution 1
            VResidualBlock(128, 128),
            VResidualBlock(128, 128),
            # First downsampling layer
            PadBlock((0, 1, 0, 1)),  # Padding (left, right, top, bottom)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),  # resolution 1/2
            VResidualBlock(128, 256),
            VResidualBlock(256, 256),
            # Second downsampling layer
            PadBlock((0, 1, 0, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),  # resolution 1/4
            VResidualBlock(256, 512),
            VResidualBlock(512, 512),
            # Third downsampling layer
            PadBlock((0, 1, 0, 1)),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),  # resolution 1/8
            VResidualBlock(512, 512),
            VResidualBlock(512, 512),
            VResidualBlock(512, 512),
            # Attention and final layers
            VAttentionBlock(512),
            VResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  # resolution 1/8
            nn.Conv2d(8, 8, kernel_size=3, padding=1),  # resolution 1/8
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30, max=20)
        stdev = logvar.exp().sqrt()
        # sample from N(mean, stdev) distribution
        x = mean + stdev * noise
        # rescale by constant
        x = x * 0.18215
