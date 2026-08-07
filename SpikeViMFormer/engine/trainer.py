"""End-to-end SpikeViMFormer training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

from spikevimformer.losses import (
    FeatureQueue,
    cosine_alignment_loss,
    hral_alignment_loss_with_history,
    symmetric_info_nce,
)


@dataclass(frozen=True)
class LossWeights:
    info_nce: float = 1.0
    ssa: float = 0.6
    shs: float = 0.54
    hral: float = 1.0


def _compute_loss(
    model: nn.Module,
    query: Tensor,
    gallery: Tensor,
    labels: Tensor,
    weights: LossWeights,
    query_queue: FeatureQueue,
    gallery_queue: FeatureQueue,
    hral_top_k: int,
    history_samples: int,
) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
    query_output, gallery_output = model(query, gallery)
    if (
        query_output.ssa_descriptor is None
        or gallery_output.ssa_descriptor is None
        or query_output.shs_logits is None
        or gallery_output.shs_logits is None
    ):
        raise RuntimeError("training outputs are missing; call model.train() first")

    raw_model = model.module if hasattr(model, "module") else model
    losses = {
        "info_nce": symmetric_info_nce(
            query_output.descriptor,
            gallery_output.descriptor,
            raw_model.logit_scale.exp(),
        ),
        "ssa": cosine_alignment_loss(
            query_output.ssa_descriptor, gallery_output.ssa_descriptor
        ),
        "shs": 0.5
        * (
            F.cross_entropy(query_output.shs_logits, labels)
            + F.cross_entropy(gallery_output.shs_logits, labels)
        ),
        "hral": hral_alignment_loss_with_history(
            query_output.descriptor,
            gallery_output.descriptor,
            query_queue.get(history_samples),
            gallery_queue.get(history_samples),
            top_k=hral_top_k,
        ),
    }
    total = (
        weights.info_nce * losses["info_nce"]
        + weights.ssa * losses["ssa"]
        + weights.shs * losses["shs"]
        + weights.hral * losses["hral"]
    )
    return total, losses, query_output.descriptor, gallery_output.descriptor


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler=None,
    scheduler=None,
    weights: LossWeights = LossWeights(),
    hral_top_k: int = 15,
    queue_size: int = 4096,
    history_samples: int = 512,
    gradient_clip: float | None = 100.0,
) -> dict[str, float]:
    model.train()
    query_queue = FeatureQueue(queue_size)
    gallery_queue = FeatureQueue(queue_size)
    totals = {name: 0.0 for name in ("total", "info_nce", "ssa", "shs", "hral")}
    steps = 0
    use_amp = scaler is not None and device.type == "cuda"

    progress = tqdm(loader, desc="train", leave=False)
    for query, gallery, labels, _ in progress:
        query = query.to(device, non_blocking=True)
        gallery = gallery.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            total, losses, query_features, gallery_features = _compute_loss(
                model,
                query,
                gallery,
                labels,
                weights,
                query_queue,
                gallery_queue,
                hral_top_k,
                history_samples,
            )

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        query_queue.enqueue(query_features)
        gallery_queue.enqueue(gallery_features)
        totals["total"] += total.detach().item()
        for name, value in losses.items():
            totals[name] += value.detach().item()
        steps += 1
        progress.set_postfix(loss=f"{totals['total'] / steps:.4f}")

    if steps == 0:
        raise RuntimeError("training loader yielded no batches")
    return {name: value / steps for name, value in totals.items()}
