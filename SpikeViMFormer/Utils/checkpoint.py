"""Checkpoint I/O with DataParallel prefix compatibility."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


def _strip_module_prefix(state_dict: dict) -> dict:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(_strip_module_prefix(state_dict), strict=strict)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    metrics: dict[str, float],
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
        },
        path,
    )
