"""Complete SpikeViMFormer retrieval model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import build_backbone, load_pretrained_backbone
from .blocks import (
    ClassificationHead,
    SpikeDrivenHybridStateSpace,
    SpikeDrivenSelectiveAttention,
)


@dataclass
class ModelOutput:
    """Outputs used by the three training objectives."""

    descriptor: Tensor
    ssa_descriptor: Tensor | None = None
    shs_logits: Tensor | None = None
    shs_embedding: Tensor | None = None


class SpikeViMFormer(nn.Module):
    """Weight-shared dual-stream SpikeViMFormer.

    SSA and SHS are evaluated only in training mode. Calling ``eval()`` leaves
    the lightweight shared backbone as the inference path described by the
    paper.
    """

    def __init__(self, num_classes: int, variant: str = "small") -> None:
        super().__init__()
        self.variant = variant
        self.backbone = build_backbone(variant)
        channels = self.backbone.out_channels
        self.ssa = SpikeDrivenSelectiveAttention(channels)
        self.shs = SpikeDrivenHybridStateSpace(channels)
        self.classifier = ClassificationHead(channels, num_classes)
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

    @property
    def descriptor_dim(self) -> int:
        return self.backbone.out_channels

    def load_backbone_checkpoint(self, checkpoint_path: str) -> None:
        """Initialize the backbone from the matching E-SpikeFormer checkpoint."""

        load_pretrained_backbone(self.backbone, checkpoint_path, self.variant)

    def encode(self, images: Tensor) -> ModelOutput:
        feature_map, _ = self.backbone(images)
        descriptor = F.normalize(feature_map.mean(dim=(-2, -1)), dim=-1)
        if not self.training:
            return ModelOutput(descriptor=descriptor)

        ssa_map = self.ssa(feature_map)
        ssa_descriptor = F.normalize(ssa_map.mean(dim=(-2, -1)), dim=-1)
        shs_map = self.shs(feature_map)
        shs_logits, shs_embedding = self.classifier(shs_map.mean(dim=(-2, -1)))
        return ModelOutput(
            descriptor=descriptor,
            ssa_descriptor=ssa_descriptor,
            shs_logits=shs_logits,
            shs_embedding=shs_embedding,
        )

    def forward(
        self, query: Tensor, gallery: Tensor | None = None
    ) -> ModelOutput | tuple[ModelOutput, ModelOutput]:
        query_output = self.encode(query)
        if gallery is None:
            return query_output
        return query_output, self.encode(gallery)


def build_model(num_classes: int, variant: str = "small") -> SpikeViMFormer:
    """Build ``SpikeViMFormer-T`` (tiny) or ``SpikeViMFormer-S`` (small)."""

    return SpikeViMFormer(num_classes=num_classes, variant=variant)
