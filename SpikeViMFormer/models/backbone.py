"""E-SpikeFormer backbone used by SpikeViMFormer.

The module hierarchy intentionally follows the official E-SpikeFormer 10M and
19M ImageNet models so their published checkpoints can be loaded directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .neurons import NormalizedIntegerLIF


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels: int,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, x: Tensor) -> Tensor:
        output = self.bn(x)
        if self.pad_pixels <= 0:
            return output
        if self.bn.affine:
            pad_values = self.bn.bias.detach() - (
                self.bn.running_mean
                * self.bn.weight.detach()
                / torch.sqrt(self.bn.running_var + self.bn.eps)
            )
        else:
            pad_values = -self.bn.running_mean / torch.sqrt(
                self.bn.running_var + self.bn.eps
            )
        output = F.pad(output, [self.pad_pixels] * 4)
        pad_values = pad_values.view(1, -1, 1, 1)
        output[:, :, : self.pad_pixels, :] = pad_values
        output[:, :, -self.pad_pixels :, :] = pad_values
        output[:, :, :, : self.pad_pixels] = pad_values
        output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class SepConv_Spike(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_ratio: int = 2,
        bias: bool = False,
        kernel_size: int = 7,
        padding: int = 3,
    ) -> None:
        super().__init__()
        hidden = expansion_ratio * dim
        self.spike1 = NormalizedIntegerLIF()
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=bias), nn.BatchNorm2d(hidden)
        )
        self.spike2 = NormalizedIntegerLIF()
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size,
                padding=padding,
                groups=hidden,
                bias=bias,
            ),
            nn.BatchNorm2d(hidden),
        )
        self.spike3 = NormalizedIntegerLIF()
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(hidden, dim, 1, bias=bias), nn.BatchNorm2d(dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pwconv1(self.spike1(x))
        x = self.dwconv(self.spike2(x))
        return self.pwconv2(self.spike3(x))


class MS_ConvBlock_spike_SepConv(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.Conv = SepConv_Spike(dim=dim)
        self.mlp_ratio = mlp_ratio
        self.spike1 = NormalizedIntegerLIF()
        self.conv1 = nn.Conv2d(dim, dim * mlp_ratio, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)
        self.spike2 = NormalizedIntegerLIF()
        self.conv2 = nn.Conv2d(dim * mlp_ratio, dim, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        x = self.Conv(x) + x
        residual = x
        x = self.bn1(self.conv1(self.spike1(x))).reshape(
            batch, self.mlp_ratio * channels, height, width
        )
        x = self.bn2(self.conv2(self.spike2(x))).reshape(
            batch, channels, height, width
        )
        return residual + x


class MS_MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, 1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_spike = NormalizedIntegerLIF()
        self.fc2_conv = nn.Conv1d(hidden_features, in_features, 1)
        self.fc2_bn = nn.BatchNorm1d(in_features)
        self.fc2_spike = NormalizedIntegerLIF()
        self.c_hidden = hidden_features

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        length = height * width
        x = self.fc1_conv(self.fc1_spike(x.flatten(2)))
        x = self.fc1_bn(x).reshape(batch, self.c_hidden, length)
        x = self.fc2_conv(self.fc2_spike(x))
        return self.fc2_bn(x).reshape(batch, channels, height, width)


class MS_Attention_linear(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, lamda_ratio: int = 4) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.lamda_ratio = lamda_ratio
        value_dim = dim * lamda_ratio

        self.head_spike = NormalizedIntegerLIF()
        self.q_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.q_spike = NormalizedIntegerLIF()
        self.k_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )
        self.k_spike = NormalizedIntegerLIF()
        self.v_conv = nn.Sequential(
            nn.Conv2d(dim, value_dim, 1, bias=False), nn.BatchNorm2d(value_dim)
        )
        self.v_spike = NormalizedIntegerLIF()
        self.attn_spike = NormalizedIntegerLIF()
        self.proj_conv = nn.Sequential(
            nn.Conv2d(value_dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )

    def _heads(self, x: Tensor, head_dim: int) -> Tensor:
        batch, _, height, width = x.shape
        return (
            x.flatten(2)
            .transpose(-1, -2)
            .reshape(batch, height * width, self.num_heads, head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        value_channels = channels * self.lamda_ratio
        x = self.head_spike(x)
        q = self._heads(self.q_spike(self.q_conv(x)), channels // self.num_heads)
        k = self._heads(self.k_spike(self.k_conv(x)), channels // self.num_heads)
        v = self._heads(
            self.v_spike(self.v_conv(x)), value_channels // self.num_heads
        )
        x = (q @ k.transpose(-2, -1) @ v) * (self.scale * 2)
        x = x.transpose(2, 3).reshape(batch, value_channels, height * width)
        x = self.attn_spike(x).reshape(batch, value_channels, height, width)
        return self.proj_conv(x)


class MS_Block_Spike_SepConv(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = SepConv_Spike(dim=dim, kernel_size=3, padding=1)
        self.attn = MS_Attention_linear(dim, num_heads=num_heads, lamda_ratio=4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.mlp = MS_MLP(dim, dim * mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.conv(x)
        x = x + self.attn(x)
        return x + self.mlp(x)


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dims: int,
        kernel_size: int,
        stride: int,
        padding: int,
        first_layer: bool,
    ) -> None:
        super().__init__()
        self.encode_conv = nn.Conv2d(
            in_channels, embed_dims, kernel_size, stride=stride, padding=padding
        )
        self.encode_bn = nn.BatchNorm2d(embed_dims)
        self.first_layer = first_layer
        if not first_layer:
            self.encode_spike = NormalizedIntegerLIF()

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, "encode_spike"):
            x = self.encode_spike(x)
        return self.encode_bn(self.encode_conv(x))


class SpikeTransformerBackbone(nn.Module):
    """Checkpoint-compatible E-SpikeFormer 10M/19M feature extractor."""

    def __init__(
        self,
        channels: Sequence[int],
        depths: Sequence[int] = (1, 1, 6, 2),
        num_heads: int = 8,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4:
            raise ValueError("channels and depths must contain four values")
        self.out_channels = channels[-1]
        rates = torch.linspace(0, drop_path_rate, depths[2] + depths[3]).tolist()

        self.downsample1_1 = MS_DownSampling(
            3, channels[0] // 2, 7, 2, 3, first_layer=True
        )
        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(channels[0] // 2, mlp_ratio)]
        )
        self.downsample1_2 = MS_DownSampling(
            channels[0] // 2, channels[0], 3, 2, 1, first_layer=False
        )
        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(channels[0], mlp_ratio)]
        )
        self.downsample2 = MS_DownSampling(
            channels[0], channels[1], 3, 2, 1, first_layer=False
        )
        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(channels[1], mlp_ratio)]
        )
        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock_spike_SepConv(channels[1], mlp_ratio)]
        )
        self.downsample3 = MS_DownSampling(
            channels[1], channels[2], 3, 2, 1, first_layer=False
        )
        self.block3 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    channels[2], num_heads, mlp_ratio, rates[index]
                )
                for index in range(depths[2])
            ]
        )
        self.downsample4 = MS_DownSampling(
            channels[2], channels[3], 3, 1, 1, first_layer=False
        )
        self.block4 = nn.ModuleList(
            [
                MS_Block_Spike_SepConv(
                    channels[3],
                    num_heads,
                    mlp_ratio,
                    rates[depths[2] + index],
                )
                for index in range(depths[3])
            ]
        )

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        x = self.downsample1_1(x)
        for block in self.ConvBlock1_1:
            x = block(x)
        stage1 = self.downsample1_2(x)
        for block in self.ConvBlock1_2:
            stage1 = block(stage1)
        stage2 = self.downsample2(stage1)
        for block in self.ConvBlock2_1:
            stage2 = block(stage2)
        for block in self.ConvBlock2_2:
            stage2 = block(stage2)
        stage3 = self.downsample3(stage2)
        for block in self.block3:
            stage3 = block(stage3)
        stage4 = self.downsample4(stage3)
        for block in self.block4:
            stage4 = block(stage4)
        return stage4, (stage1, stage2, stage3, stage4)


BACKBONE_VARIANTS = {
    "tiny": (48, 96, 192, 240),   # E-SpikeFormer 10M
    "small": (64, 128, 256, 360),  # E-SpikeFormer 19M
}

PRETRAINED_FILENAMES = {
    "tiny": "V3_10.0M_1x4.pth",
    "small": "V3_19.0M_1x4.pth",
}


def build_backbone(variant: str) -> SpikeTransformerBackbone:
    try:
        channels = BACKBONE_VARIANTS[variant]
    except KeyError as error:
        choices = ", ".join(sorted(BACKBONE_VARIANTS))
        raise ValueError(f"unknown model variant {variant!r}; choose from {choices}") from error
    return SpikeTransformerBackbone(channels=channels)


def load_pretrained_backbone(
    backbone: SpikeTransformerBackbone,
    checkpoint_path: str | Path,
    variant: str,
) -> tuple[list[str], list[str]]:
    """Load the official E-SpikeFormer ImageNet checkpoint.

    Classification-head parameters are intentionally discarded. A shape
    mismatch usually means the 10M checkpoint was paired with ``small`` or the
    19M checkpoint was paired with ``tiny``.
    """

    checkpoint_path = Path(checkpoint_path)
    expected = PRETRAINED_FILENAMES[variant]
    if checkpoint_path.name != expected:
        raise ValueError(
            f"variant {variant!r} expects {expected}, got {checkpoint_path.name}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for key, value in state_dict.items():
        while key.startswith(("module.", "backbone.")):
            key = key.split(".", 1)[1]
        if not key.startswith(("head.", "spike.")):
            cleaned[key] = value
    incompatible = backbone.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "E-SpikeFormer checkpoint is incomplete for this backbone; missing keys: "
            + ", ".join(incompatible.missing_keys[:10])
        )
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
