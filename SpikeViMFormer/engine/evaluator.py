"""Cross-view retrieval evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm


@torch.inference_mode()
def extract_features(
    model: nn.Module, loader, device: torch.device
) -> tuple[Tensor, list[str]]:
    model.eval()
    features: list[Tensor] = []
    identities: list[str] = []
    for images, batch_identities in tqdm(loader, desc="features", leave=False):
        output = model(images.to(device, non_blocking=True))
        features.append(output.descriptor.cpu())
        identities.extend(batch_identities)
    if not features:
        raise RuntimeError("evaluation loader yielded no images")
    return F.normalize(torch.cat(features), dim=-1), identities


def _average_precision(ranked_matches: Tensor) -> float:
    relevant = int(ranked_matches.sum().item())
    if relevant == 0:
        return 0.0
    precision = ranked_matches.cumsum(0) / torch.arange(
        1, ranked_matches.numel() + 1, dtype=torch.float32
    )
    return float((precision * ranked_matches).sum().item() / relevant)


@torch.inference_mode()
def evaluate_retrieval(
    model: nn.Module,
    query_loader,
    gallery_loader,
    device: torch.device,
    recall_at: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    query_features, query_ids = extract_features(model, query_loader, device)
    gallery_features, gallery_ids = extract_features(model, gallery_loader, device)
    similarity = query_features @ gallery_features.t()
    ranking = similarity.argsort(dim=1, descending=True)
    recalls = {k: 0 for k in recall_at}
    average_precisions: list[float] = []

    for query_index, query_id in enumerate(query_ids):
        ordered_ids = [gallery_ids[i] for i in ranking[query_index].tolist()]
        matches = torch.tensor([identity == query_id for identity in ordered_ids])
        average_precisions.append(_average_precision(matches))
        for k in recall_at:
            recalls[k] += int(matches[: min(k, len(matches))].any().item())

    count = max(1, len(query_ids))
    result = {f"R@{k}": 100.0 * value / count for k, value in recalls.items()}
    result["AP"] = 100.0 * sum(average_precisions) / count
    return result
