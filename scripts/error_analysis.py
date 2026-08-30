#!/usr/bin/env python
"""CLI entry point for error analysis (representative false positives/negatives).

Usage:
    python scripts/error_analysis.py --checkpoint_path models/checkpoints/aigc_classifier.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from aigc_detect.error_analysis import find_errors, save_error_examples, save_error_summary_csv
from aigc_detect.evaluate import sample_test_subset
from aigc_detect.model import AIGCClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find representative false positives/negatives.")
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier.pt")
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/cifake"))
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/error_analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    samples = sample_test_subset(args.data_root, n_per_class=args.n_per_class, seed=args.seed)
    print(f"Finding errors among {len(samples)} clean test images...")

    result = find_errors(model, samples, device, batch_size=args.batch_size)
    summary = result["summary"]
    print(
        f"\nFalse positives: {summary['n_false_positives']} "
        f"(FPR={summary['false_positive_rate']:.4f})"
    )
    print(
        f"False negatives: {summary['n_false_negatives']} "
        f"(FNR={summary['false_negative_rate']:.4f})"
    )

    save_error_summary_csv(
        result["false_positives"], result["false_negatives"], args.output_dir / "errors.csv"
    )
    save_error_examples(
        result["false_positives"],
        args.output_dir / "false_positives",
        prefix="fp",
        n=args.n_examples,
    )
    save_error_examples(
        result["false_negatives"],
        args.output_dir / "false_negatives",
        prefix="fn",
        n=args.n_examples,
    )

    print(f"\nWrote {args.output_dir}/errors.csv and top-{args.n_examples} example images per category")


if __name__ == "__main__":
    main()
