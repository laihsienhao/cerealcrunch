#!/usr/bin/env python
"""CLI entry point for the robustness evaluation harness.

Usage:
    python scripts/evaluate.py --checkpoint_path models/checkpoints/aigc_classifier.pt
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from aigc_detect.calibration import load_temperature
from aigc_detect.evaluate import (
    compute_final_score,
    plot_robustness,
    run_robustness_eval,
    sample_test_subset,
    save_results_csv,
)
from aigc_detect.model import AIGCClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the robustness evaluation harness.")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=Path("models/checkpoints/cereal_crunch.pt"),
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/artifact_full"))
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

    temperature = load_temperature(args.checkpoint_path)
    samples = sample_test_subset(args.data_root, n_per_class=args.n_per_class, seed=args.seed)
    print(f"Evaluating on {len(samples)} test images across 15 conditions "
          f"(temperature={temperature:.4f})...")

    results = run_robustness_eval(model, samples, device, batch_size=args.batch_size, temperature=temperature)

    csv_path = args.output_dir / "robustness_results.csv"
    plot_path = args.output_dir / "robustness_plot.png"
    score_path = args.output_dir / "final_score.json"
    save_results_csv(results, csv_path)
    plot_robustness(results, plot_path)

    final_score = compute_final_score(results)
    with open(score_path, "w") as f:
        json.dump(final_score, f, indent=2)

    print(f"\n{'condition':<20}{'accuracy':>10}{'f1':>10}{'roc_auc':>10}")
    for row in results:
        print(f"{row['condition']:<20}{row['accuracy']:>10.4f}{row['f1']:>10.4f}{row['roc_auc']:>10.4f}")
    print(
        f"\nAUC_clean={final_score['auc_clean']:.4f}  AUC_robust={final_score['auc_robust']:.4f}  "
        f"Final score (0.5*clean + 0.5*robust) = {final_score['final_score']:.4f}"
    )
    print(f"\nWrote {csv_path}, {plot_path}, and {score_path}")


if __name__ == "__main__":
    main()
