#!/usr/bin/env python
"""CLI entry point for the robustness evaluation harness.

Usage:
    python scripts/evaluate.py --checkpoint_path models/checkpoints/aigc_classifier.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from aigc_detect.evaluate import plot_robustness, run_robustness_eval, sample_test_subset, save_results_csv
from aigc_detect.model import AIGCClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the robustness evaluation harness.")
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier.pt")
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/cifake"))
    parser.add_argument("--n_per_class", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/eval"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AIGCClassifier().to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    samples = sample_test_subset(args.data_root, n_per_class=args.n_per_class, seed=args.seed)
    print(f"Evaluating on {len(samples)} test images across 15 conditions...")

    results = run_robustness_eval(model, samples, device, batch_size=args.batch_size)

    csv_path = args.output_dir / "robustness_results.csv"
    plot_path = args.output_dir / "robustness_plot.png"
    save_results_csv(results, csv_path)
    plot_robustness(results, plot_path)

    print(f"\n{'condition':<20}{'accuracy':>10}{'f1':>10}{'roc_auc':>10}")
    for row in results:
        print(f"{row['condition']:<20}{row['accuracy']:>10.4f}{row['f1']:>10.4f}{row['roc_auc']:>10.4f}")
    print(f"\nWrote {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
