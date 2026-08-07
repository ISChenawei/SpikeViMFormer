"""Training-time SSA and SHS blocks from SpikeViMFormer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .neurons import NormalizedIntegerLIF


class ConvNorm1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
        zero_init: bool = False,
    ) -> None:
        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        norm = nn.BatchNorm1d(out_channels)
        if zero_init:
            nn.init.zeros_(norm.weight)
        super().__init__(conv, norm)


class ConvNorm2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
        zero_init: bool = False,
    ) -> None:
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        norm = nn.BatchNorm2d(out_channels)
        if zero_init:
            nn.init.zeros_(norm.weight)
        super().__init__(conv, norm)


class SpikeMLP1d(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.network = nn.Sequential(
            NormalizedIntegerLIF(),
            ConvNorm1d(channels, hidden),
            NormalizedIntegerLIF(),
            ConvNorm1d(hidden, channels, zero_init=True),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return self.network(sequence.transpose(1, 2)).transpose(1, 2)


class SpikeDrivenSelectiveAttention(nn.Module):
    """Selectively strengthen discriminative local features (SSA)."""

    def __init__(self, channels: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.cpe = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.refine = nn.Sequential(
            ConvNorm1d(channels, channels),
            ConvNorm1d(channels, channels, 3, padding=1, groups=channels),
            NormalizedIntegerLIF(),
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            NormalizedIntegerLIF(),
        )
        self.query = ConvNorm1d(channels, channels)
        self.query_local = nn.Sequential(
            ConvNorm1d(channels, channels),
            ConvNorm1d(channels, channels, 3, padding=1, groups=channels),
            NormalizedIntegerLIF(),
        )
        self.selection_gate = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            NormalizedIntegerLIF(),
        )
        self.gate_norm = nn.LayerNorm(channels)
        self.output_spike = NormalizedIntegerLIF()
        self.output_projection = nn.Linear(channels, channels, bias=False)
        self.local_residual = nn.Conv1d(
            channels, channels, 3, padding=1, groups=channels
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = SpikeMLP1d(channels, mlp_ratio)

    def forward(self, feature_map: Tensor) -> Tensor:
        batch, channels, height, width = feature_map.shape
        positioned = feature_map + self.cpe(feature_map)
        sequence = positioned.flatten(2).transpose(1, 2)
        normalized = self.norm1(sequence)

        refined = self.refine(normalized.transpose(1, 2)).transpose(1, 2)
        attention = self.attention_gate(normalized)
        query = self.query(refined.transpose(1, 2))
        local_query = self.query_local(query).transpose(1, 2)
        gate = self.selection_gate(local_query)
        selected = self.gate_norm(local_query * gate)
        selected = self.output_projection(self.output_spike(selected) * attention)

        sequence = sequence + selected
        local = self.local_residual(sequence.transpose(1, 2)).transpose(1, 2)
        sequence = sequence + local
        sequence = sequence + self.mlp(self.norm2(sequence))
        return sequence.transpose(1, 2).reshape(batch, channels, height, width)


class HybridStateSpaceMixer(nn.Module):
    """Low-rank state-space mixer with linear sequence complexity."""

    def __init__(self, channels: int, state_dim: int = 64) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.parameters_projection = nn.Conv1d(
            channels, 3 * state_dim, 1, bias=False
        )
        self.parameter_local = nn.Conv2d(
            3 * state_dim,
            3 * state_dim,
            3,
            padding=1,
            groups=3 * state_dim,
            bias=False,
        )
        self.hidden_projection = nn.Conv1d(
            channels, 2 * channels, 1, bias=False
        )
        self.output_projection = nn.Conv1d(channels, channels, 1, bias=False)
        self.transition = nn.Parameter(torch.empty(state_dim).uniform_(1, 16))
        self.skip = nn.Parameter(torch.ones(1))
        self.input_spike = NormalizedIntegerLIF()
        self.gate_spike = NormalizedIntegerLIF()

    def forward(self, sequence: Tensor, spatial_size: tuple[int, int]) -> Tensor:
        batch, _, length = sequence.shape
        height, width = spatial_size
        if height * width != length:
            raise ValueError("spatial size does not match sequence length")
        sequence = self.input_spike(sequence)
        parameters = self.parameters_projection(sequence)
        parameters = self.parameter_local(
            parameters.reshape(batch, -1, height, width)
        ).flatten(2)
        state_in, state_out, delta = parameters.chunk(3, dim=1)
        transition = (delta + self.transition.view(1, -1, 1)).softmax(dim=-1)
        state = sequence @ (transition * state_in).transpose(1, 2)
        hidden, gate = self.hidden_projection(state).chunk(2, dim=1)
        hidden = self.output_projection(hidden * self.gate_spike(gate) + hidden * self.skip)
        return hidden @ state_out


class SpikeFeedForward2d(nn.Module):
    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = channels * expansion
        self.network = nn.Sequential(
            NormalizedIntegerLIF(),
            ConvNorm2d(channels, hidden),
            NormalizedIntegerLIF(),
            ConvNorm2d(hidden, channels, zero_init=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class SpikeDrivenHybridStateSpace(nn.Module):
    """Capture long-range dependencies and restore local detail (SHS)."""

    def __init__(self, channels: int, state_dim: int = 64) -> None:
        super().__init__()
        self.input_spike = NormalizedIntegerLIF()
        self.local_before = ConvNorm2d(
            channels, channels, 3, padding=1, groups=channels, zero_init=True
        )
        self.norm = nn.LayerNorm(channels)
        self.mixer = HybridStateSpaceMixer(channels, state_dim)
        self.output_spike = NormalizedIntegerLIF()
        self.local_after = ConvNorm2d(
            channels, channels, 3, padding=1, groups=channels, zero_init=True
        )
        self.ffn = SpikeFeedForward2d(channels)

    def forward(self, feature_map: Tensor) -> Tensor:
        batch, channels, height, width = feature_map.shape
        feature_map = feature_map + self.local_before(self.input_spike(feature_map))
        sequence = self.norm(feature_map.flatten(2).transpose(1, 2)).transpose(1, 2)
        mixed = self.mixer(sequence, (height, width))
        feature_map = mixed.reshape(batch, channels, height, width)
        feature_map = feature_map + self.local_after(self.output_spike(feature_map))
        return feature_map + self.ffn(feature_map)


class ClassificationHead(nn.Module):
    def __init__(
        self, channels: int, num_classes: int, embedding_dim: int = 512
    ) -> None:
        super().__init__()
        self.spike = NormalizedIntegerLIF()
        self.embedding = nn.Sequential(
            nn.Linear(channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        embedding = self.embedding(self.spike(feature))
        return self.classifier(embedding), embedding
