"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int = 1, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
