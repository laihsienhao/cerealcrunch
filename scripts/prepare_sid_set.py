#!/usr/bin/env python
"""Extract filtered (non-tampered) SID_Set images into CIFAKE-style REAL/FAKE folders.

Usage:
    python scripts/prepare_sid_set.py --target_real 50000 --target_fake 50000

Processes already-downloaded local Parquet files first (deleting each once
its images are extracted), then falls back to network streaming until the
train targets are met. Validation is processed in full (no target - takes
whatever's available).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigc_detect.sid_data import export_split_to_folders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract filtered SID_Set images.")
    parser.add_argument("--output_root", type=Path, default=Path("data/raw/sid_set"))
    parser.add_argument(
        "--local_parquet_dir", type=Path, default=Path("data/raw/sid_set/data")
    )
    parser.add_argument("--target_real", type=int, default=50000)
    parser.add_argument("--target_fake", type=int, default=50000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== train ===", flush=True)
    train_counts = export_split_to_folders(
        split="train",
        output_root=args.output_root / "train",
        target_per_label={0: args.target_real, 1: args.target_fake},
        local_parquet_dir=args.local_parquet_dir,
    )
    print(f"train done: real={train_counts[0]}, fake={train_counts[1]}", flush=True)

    print("=== validation ===", flush=True)
    val_counts = export_split_to_folders(
        split="validation",
        output_root=args.output_root / "validation",
        target_per_label=None,
        local_parquet_dir=args.local_parquet_dir,
    )
    print(f"validation done: real={val_counts[0]}, fake={val_counts[1]}", flush=True)


if __name__ == "__main__":
    main()
