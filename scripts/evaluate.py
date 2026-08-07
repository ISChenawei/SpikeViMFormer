"""Evaluate a SpikeViMFormer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spikevimformer.data import GeoImageDataset, build_dataset_paths, build_transforms
from spikevimformer.engine import evaluate_retrieval
from spikevimformer.models import build_model
from spikevimformer.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--dataset", choices=("University-1652", "SUES-200"), default="University-1652"
    )
    parser.add_argument(
        "--direction",
        choices=("drone2satellite", "satellite2drone"),
        default="drone2satellite",
    )
    parser.add_argument("--altitude", type=int, choices=(150, 200, 250, 300), default=150)
    parser.add_argument("--variant", choices=("tiny", "small"), default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    metadata = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    variant = args.variant or config.get("variant", "small")
    num_classes = args.num_classes or config.get("num_classes")
    if num_classes is None:
        raise ValueError("--num-classes is required for a checkpoint without metadata")

    model = build_model(int(num_classes), variant).to(device)
    load_checkpoint(args.checkpoint, model)
    transform = build_transforms(args.image_size, training=False)
    paths = build_dataset_paths(
        args.data_root, args.dataset, args.direction, args.altitude
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    query_loader = DataLoader(GeoImageDataset(paths.test_query, transform), **loader_options)
    gallery_loader = DataLoader(
        GeoImageDataset(paths.test_gallery, transform), **loader_options
    )
    metrics = evaluate_retrieval(model, query_loader, gallery_loader, device)
    print("  ".join(f"{name}: {value:.2f}" for name, value in metrics.items()))


if __name__ == "__main__":
    main()
