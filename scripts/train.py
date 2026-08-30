#!/usr/bin/env python
"""CLI entry point for training AIGCClassifier.

Usage:
    python scripts/train.py --epochs 2 --batch_size 32
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigc_detect.train import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AIGC image classifier.")
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/cifake"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier.pt")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint_path=args.checkpoint_path,
    )
    train(config)


if __name__ == "__main__":
    main()
