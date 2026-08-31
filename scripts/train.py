#!/usr/bin/env python
"""CLI entry point for training AIGCClassifier.

Usage:
    python scripts/train.py --epochs 2 --batch_size 32
    python scripts/train.py --data_root data/raw/sid_set --val_split validation --epochs 3
    python scripts/train.py --resume_from models/checkpoints/latest_aigc_classifier.pt --epochs 5
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
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="If set (e.g. 'validation'), load a dedicated pre-split validation "
        "folder instead of carving val_fraction out of train (use for SID_Set).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--no_noise_branch",
        action="store_true",
        help="Disable the noise-residual branch (CLIP features only) - for ablation runs.",
    )
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier.pt")
    )
    parser.add_argument(
        "--latest_checkpoint_path",
        type=Path,
        default=None,
        help="Defaults to 'latest_' + checkpoint_path's name.",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Path to a 'latest' checkpoint to resume from (model + optimizer state + epoch).",
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
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        use_noise_branch=not args.no_noise_branch,
        checkpoint_path=args.checkpoint_path,
        latest_checkpoint_path=args.latest_checkpoint_path,
        resume_from=args.resume_from,
    )
    train(config)


if __name__ == "__main__":
    main()
