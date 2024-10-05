import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import VAttentionBlock


class VResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(VResidualBlock, self).__init__()
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x

        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        x = x + self.residual_layer(residue)

        return x


class VDecoder(nn.Module):
    def __init__(
        self, in_channels: int, sequence_length: int, groups: int, n_heads: int
    ):
        super(VDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0),
            nn.Conv2d(in_channels, in_channels * 64, kernel_size=3, padding=0),
            VResidualBlock(in_channels * 128, in_channels * 128),
            VAttentionBlock(sequence_length, in_channels * 128, groups=32, n_heads=4),
            VResidualBlock(in_channels * 128, in_channels * 128),
            VResidualBlock(in_channels * 128, in_channels * 128),
            VResidualBlock(in_channels * 128, in_channels * 128),
            VResidualBlock(in_channels * 128, in_channels * 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels * 128, in_channels * 128, kernel_size=3, padding=1),
            VResidualBlock(in_channels * 128, in_channels * 128),
            VResidualBlock(in_channels * 128, in_channels * 128),
            VResidualBlock(in_channels * 128, in_channels * 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels * 128, in_channels * 128, kernel_size=3, padding=1),
            VResidualBlock(in_channels * 128, in_channels * 64),
            VResidualBlock(in_channels * 64, in_channels * 64),
            VResidualBlock(in_channels * 64, in_channels * 64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels * 64, in_channels * 64, kernel_size=3, padding=1),
            VResidualBlock(in_channels * 64, in_channels * 32),
            VResidualBlock(in_channels * 32, in_channels * 32),
            VResidualBlock(in_channels * 32, in_channels * 32),
            nn.GroupNorm(32, in_channels * 32),
            nn.SiLU(),
            nn.Conv2d(in_channels * 32, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 0.18215
        x = self.layers(x)

        return x
