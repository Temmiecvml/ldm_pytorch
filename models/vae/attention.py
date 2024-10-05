import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        n_heads: int,
        in_proj_bias: bool = False,
        out_proj_bias: bool = True,
    ):
        super(SelfAttention, self).__init__()
        self.qkv = nn.Linear(channels, channels * 3, bias=in_proj_bias)
        self.n_heads = n_heads
        self.d_head = channels // n_heads
        self.proj = nn.Linear(channels, channels, bias=out_proj_bias)

    def forward(self, x, mask):
        bs, sequence_length, _ = x.shape

        # Compute q, k, v
        qkv = self.qkv(x).chunk(3, dim=-1)

        # (bs, n_head, sequence_length,  d_head)
        q, k, v = map(
            lambda w: w.view(bs, sequence_length, self.n_heads, self.d_head).transpose(
                1, 2
            ),
            qkv,
        )

        # (bs, n_head, sl, d_head) @ (bs, n_head, d_head, sl) => (bs, n_head, sl, sl)
        attn_weights = q @ k.transpose(-1, -2)

        if mask:
            causal_mask = torch.triu(
                torch.ones_like(attn_weights, dtype=torch.bool), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, -torch.inf)

        attn_weights = attn_weights / torch.sqrt(self.d_head)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # bs, n_head, sl, sl

        # (bs, n_head, sl, sl) @ (bs, n_head, sl,  d_head) => (bs, n_head, sl, d_head)
        out = attn_weights @ v
        out = (
            out.transpose(1, 2).contiguous().view(bs, sequence_length, -1)
        )  # (bs, sequence_length, channels)

        out = self.proj(out)  # Final projection

        return out


class VAttentionBlock(nn.Module):
    def __init__(self, sequence_length, channels, groups=32, n_heads=4):
        super(VAttentionBlock, self).__init__()
        self.group_norm = nn.GroupNorm(groups, sequence_length)
        self.attention = SelfAttention(channels, n_heads, False, True)

    def forward(self, x):
        residue = x
        n, c, h, w = x.shape
        x = x.view(n, c, -1)  # (bs, channels, height * width)
        x = x.transpose(-1, -2)  # (bs, height * width, channels)
        x = self.group_norm(x)  # (bs, height * width, channels)
        x = self.attention(x)  # (bs, height * width, channels)
        x = x.transpose(-1, -2).view(n, c, h, w)  # (bs, channels, height, width)
        x = x + residue

        return x
