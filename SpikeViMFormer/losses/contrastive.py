"""Contrastive losses for paired cross-view features."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def symmetric_info_nce(query: Tensor, gallery: Tensor, scale: Tensor) -> Tensor:
    """Symmetric InfoNCE for aligned query-gallery pairs in a batch."""

    query = F.normalize(query, dim=-1)
    gallery = F.normalize(gallery, dim=-1)
    logits = scale.clamp(max=100) * query @ gallery.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
    )


def cosine_alignment_loss(query: Tensor, gallery: Tensor) -> Tensor:
    """Cosine embedding objective used to supervise the SSA block."""

    return 1.0 - F.cosine_similarity(query, gallery, dim=-1).mean()
