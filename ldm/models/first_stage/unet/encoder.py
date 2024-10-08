import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import make_attn
from ....utils import Normalize


class ResnetBlock(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        """x: input
        temb: temporal embedding

        x -> groupnorm ->silu -> (add temporal emb) -> **
          **-> groupnorm ->silu -> dropout -> outconv -> **
          **-> add skip connection

        """

        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Down(nn.Module):
    def __init__(
        self,
        resnet_blocks,
        attn_blocks,
        i_level,
        num_resolutions,
        resolution,
        block_in,
        resamp_with_conv,
    ):
        super(Down, self).__init__()

        self.block = resnet_blocks
        self.attn = attn_blocks
        self.resolution = resolution

        # if not output
        if i_level != num_resolutions - 1:
            self.downsample = Downsample(block_in, resamp_with_conv)
            self.resolution = self.resolution // 2
        else:
            self.downsample = None  # No downsample if condition not met

    def forward(self, x, temb):
        for blk in self.block:
            x = blk(x, temb)

        if self.attn:
            for blk in self.attn:
                x = self.attn(x)

        if self.downsample:
            h = self.downsample(x)

        return h, x


class Mid(nn.Module):
    """
    MidModule class containing two Resnet blocks and an attention layer in between.

    Args:
        block_in (int): Number of input channels for the Resnet blocks and attention layer.
        temb_ch (int): Number of temporal embedding channels.
        dropout (float): Dropout rate to use in the Resnet blocks.
        attn_type (str): The type of attention to apply in the middle layer.

    Attributes:
        block_1 (ResnetBlock): The first Resnet block.
        attn_1 (nn.Module): The attention layer between the two Resnet blocks.
        block_2 (ResnetBlock): The second Resnet block.
    """

    def __init__(self, block_in, temb_ch, dropout, attn_type):
        super(Mid, self).__init__()
        self.block_in = block_in
        self.temb_ch = temb_ch
        self.dropout = dropout
        self.attn_type = attn_type

        # Define the layers
        self.block_1 = ResnetBlock(
            in_channels=self.block_in,
            out_channels=self.block_in,
            temb_channels=self.temb_ch,
            dropout=self.dropout,
        )

        self.attn_1 = make_attn(self.block_in, attn_type=self.attn_type)

        self.block_2 = ResnetBlock(
            in_channels=self.block_in,
            out_channels=self.block_in,
            temb_channels=self.temb_ch,
            dropout=self.dropout,
        )

    def forward(self, x, temb):
        """
        Forward pass through the midmodule.

        Args:
            x (torch.Tensor): Input tensor.
            temb (torch.Tensor): Temporal embedding tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the two Resnet blocks and attention.
        """
        x = self.block_1(x, temb)  # First Resnet block
        x = self.attn_1(x)  # Attention layer
        x = self.block_2(x, temb)  # Second Resnet block

        return x


class Encoder(nn.Module):

    temb_ch = 0

    def __init__(
        self,
        *,
        base_channels: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z=True,
        use_linear_attn=False,
        num_res_blocks: int = 2,
        attn_resolutions: list = [],
        dropout=0.0,
        resamp_with_conv=True,
        ch_mult: tuple = (1, 2, 4, 8),
        **ignore_kwargs
    ):
        super().__init__()

        self.base_channels = base_channels
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        attn_type = "linear" if use_linear_attn else "vanilla"

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, base_channels, kernel_size=3, stride=1, padding=1
        )

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_channels * in_ch_mult[i_level]
            block_out = base_channels * ch_mult[i_level]

            for _ in range(self.num_res_blocks):

                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )

                if resolution in attn_resolutions:
                    attn.append(make_attn(block_out, attn_type=attn_type))

            down = Down(
                resnet_blocks=block,
                attn_blocks=attn,
                i_level=i_level,
                num_resolutions=self.num_resolutions,
                resolution=resolution,
                block_in=block_in,
                resamp_with_conv=resamp_with_conv,
            )

            self.down.append(down)
            resolution = self.resolution

        # middle
        self.mid = Mid(
            block_in=block_out,
            temb_ch=self.temb_ch,
            dropout=dropout,
            attn_type=attn_type,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]

        for i_level in range(self.num_resolutions):
            block = self.down[i_level]
            h, x = block(x, temb)
            hs.append(x)
            if i_level != self.num_resolutions - 1:
                hs.append(h)

        # middle
        h = hs[-1]
        h = self.mid(h, temb)

        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h
