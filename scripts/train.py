"""Train SpikeViMFormer on University-1652 or SUES-200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from spikevimformer.data import (
    GeoImageDataset,
    PairedGeoDataset,
    build_dataset_paths,
    build_transforms,
)
from spikevimformer.engine import LossWeights, evaluate_retrieval, train_one_epoch
from spikevimformer.models import build_model
from spikevimformer.utils import load_checkpoint, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--variant", choices=("tiny", "small"), default="small")
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        help="matching official E-SpikeFormer 10M/19M ImageNet checkpoint",
    )
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hral-top-k", type=int, default=15)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument("--history-samples", type=int, default=512)
    parser.add_argument("--ssa-weight", type=float, default=0.6)
    parser.add_argument("--shs-weight", type=float, default=0.54)
    parser.add_argument("--hral-weight", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/default"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    if args.resume is None and args.backbone_checkpoint is None:
        parser.error(
            "--backbone-checkpoint is required for a new training run; "
            "use V3_10.0M_1x4.pth with tiny or V3_19.0M_1x4.pth with small"
        )
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    paths = build_dataset_paths(
        args.data_root, args.dataset, args.direction, args.altitude
    )
    train_transform = build_transforms(args.image_size, training=True)
    eval_transform = build_transforms(args.image_size, training=False)
    train_dataset = PairedGeoDataset(
        paths.train_query,
        paths.train_gallery,
        train_transform,
        train_transform,
    )
    query_dataset = GeoImageDataset(paths.test_query, eval_transform)
    gallery_dataset = GeoImageDataset(paths.test_gallery, eval_transform)

    loader_options = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch_size,
        **loader_options,
    )
    query_loader = DataLoader(
        query_dataset, batch_size=args.eval_batch_size, **loader_options
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=args.eval_batch_size, **loader_options
    )

    model = build_model(train_dataset.num_classes, args.variant).to(device)
    if args.resume is None:
        model.load_backbone_checkpoint(str(args.backbone_checkpoint))
        print(f"Loaded E-SpikeFormer backbone: {args.backbone_checkpoint}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    scaler = None
    if device.type == "cuda" and not args.no_amp:
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "data_root": str(args.data_root),
            "output": str(args.output),
            "resume": str(args.resume) if args.resume else None,
            "backbone_checkpoint": (
                str(args.backbone_checkpoint) if args.backbone_checkpoint else None
            ),
            "num_classes": train_dataset.num_classes,
        }
    )
    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    weights = LossWeights(ssa=args.ssa_weight, shs=args.shs_weight, hral=args.hral_weight)
    best_recall = float("-inf")

    for epoch in range(start_epoch, args.epochs + 1):
        losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            scheduler=scheduler,
            weights=weights,
            hral_top_k=args.hral_top_k,
            queue_size=args.queue_size,
            history_samples=args.history_samples,
        )
        metrics = evaluate_retrieval(model, query_loader, gallery_loader, device)
        print(
            f"epoch={epoch} loss={losses['total']:.4f} "
            f"R@1={metrics['R@1']:.2f} AP={metrics['AP']:.2f}"
        )
        save_checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
        if metrics["R@1"] > best_recall:
            best_recall = metrics["R@1"]
            save_checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )


if __name__ == "__main__":
    main()
