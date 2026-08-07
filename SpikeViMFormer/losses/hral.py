"""Hierarchical Re-ranking Alignment Learning (HRAL)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def rerank_features(
    query: Tensor,
    gallery: Tensor,
    top_k: int = 15,
    query_expansion: int = 6,
    alpha: float = 0.7,
) -> tuple[Tensor, Tensor]:
    """Refine a batch using k-reciprocal affinity and residual diffusion."""

    features = F.normalize(torch.cat((query, gallery), dim=0).float(), dim=-1)
    count = features.shape[0]
    query_count = query.shape[0]
    if count < 2:
        return query.detach(), gallery.detach()

    top_k = min(max(1, top_k), count - 1)
    query_expansion = min(max(1, query_expansion), top_k)
    distance = (2.0 - 2.0 * features @ features.t()).clamp_min_(0)
    rank = distance.argsort(dim=1)
    affinity = torch.zeros_like(distance)

    for index in range(count):
        forward = rank[index, : top_k + 1]
        backward = rank[forward, : top_k + 1]
        reciprocal = forward[(backward == index).any(dim=1)]
        expanded = [reciprocal]
        half_k = max(1, top_k // 2)
        for candidate in reciprocal.tolist():
            candidate_forward = rank[candidate, : half_k + 1]
            candidate_backward = rank[candidate_forward, : half_k + 1]
            candidate_reciprocal = candidate_forward[
                (candidate_backward == candidate).any(dim=1)
            ]
            overlap = torch.isin(candidate_reciprocal, reciprocal).sum()
            if overlap * 3 > candidate_reciprocal.numel() * 2:
                expanded.append(candidate_reciprocal)
        neighbors = torch.unique(torch.cat(expanded))
        weights = torch.exp(-distance[index, neighbors])
        affinity[index, neighbors] = weights / weights.sum().clamp_min(1e-12)

    smoothed = torch.stack(
        [affinity[rank[i, :query_expansion]].mean(dim=0) for i in range(count)]
    )
    refined = F.normalize(alpha * features + (1.0 - alpha) * smoothed @ features, dim=-1)
    return refined[:query_count], refined[query_count:]


def _alignment(original: Tensor, refined: Tensor) -> Tensor:
    cosine = 1.0 - F.cosine_similarity(original, refined, dim=-1).mean()
    log_prob = F.log_softmax(original.float(), dim=-1)
    target_prob = F.softmax(refined.float(), dim=-1)
    divergence = F.kl_div(log_prob, target_prob, reduction="batchmean")
    return cosine + divergence


def hral_alignment_loss(
    query: Tensor,
    gallery: Tensor,
    *,
    top_k: int = 15,
    query_expansion: int = 6,
    alpha: float = 0.7,
) -> Tensor:
    """Align original descriptors with detached HRAL-refined descriptors."""

    refined_query, refined_gallery = rerank_features(
        query,
        gallery,
        top_k=top_k,
        query_expansion=query_expansion,
        alpha=alpha,
    )
    return 0.5 * (
        _alignment(query, refined_query.detach())
        + _alignment(gallery, refined_gallery.detach())
    )


def hral_alignment_loss_with_history(
    query: Tensor,
    gallery: Tensor,
    historical_query: Tensor | None,
    historical_gallery: Tensor | None,
    *,
    top_k: int = 15,
    query_expansion: int = 6,
    alpha: float = 0.7,
) -> Tensor:
    """Refine with historical context while supervising the current batch."""

    if historical_query is None or historical_gallery is None:
        return hral_alignment_loss(
            query,
            gallery,
            top_k=top_k,
            query_expansion=query_expansion,
            alpha=alpha,
        )
    current_count = query.shape[0]
    all_query = torch.cat((query, historical_query.detach()), dim=0)
    all_gallery = torch.cat((gallery, historical_gallery.detach()), dim=0)
    refined_query, refined_gallery = rerank_features(
        all_query,
        all_gallery,
        top_k=top_k,
        query_expansion=query_expansion,
        alpha=alpha,
    )
    return 0.5 * (
        _alignment(query, refined_query[:current_count].detach())
        + _alignment(gallery, refined_gallery[:current_count].detach())
    )


class FeatureQueue:
    """Fixed-size FIFO queue for historical HRAL descriptors."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._features: Tensor | None = None

    def __len__(self) -> int:
        return 0 if self._features is None else self._features.shape[0]

    @torch.no_grad()
    def enqueue(self, features: Tensor) -> None:
        features = features.detach()
        if self._features is None:
            self._features = features[-self.capacity :]
        else:
            self._features = torch.cat((self._features, features), dim=0)[
                -self.capacity :
            ]

    def get(self, max_features: int | None = None) -> Tensor | None:
        if self._features is None or max_features is None:
            return self._features
        return self._features[-max_features:]
