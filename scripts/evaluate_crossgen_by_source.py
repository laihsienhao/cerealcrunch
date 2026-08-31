#!/usr/bin/env python
"""Per-fake-generator breakdown on the ArtiFact cross-generator eval-only set.

Usage:
    python scripts/evaluate_crossgen_by_source.py \\
        --checkpoint_path models/checkpoints/aigc_classifier_sid.pt

Each of ArtiFact's fake generators is paired against the *same* shared real
pool (AUC needs both classes present - see aigc_detect.evaluate.evaluate_by_fake_source
for why), so results are directly comparable across generators.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from aigc_detect.crossgen_data import list_samples_by_source
from aigc_detect.evaluate import evaluate_by_fake_source, save_results_csv
from aigc_detect.model import AIGCClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-fake-generator breakdown on the ArtiFact cross-generator set."
    )
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier_sid.pt")
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/artifact_crossgen"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/eval_artifact_crossgen"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    split_root = args.data_root / args.split
    real_by_source = list_samples_by_source(split_root / "REAL", label=0)
    fake_by_source = list_samples_by_source(split_root / "FAKE", label=1)

    real_samples = [s for group in real_by_source.values() for s in group]
    print(
        f"Real pool: {len(real_samples)} images across {len(real_by_source)} sources "
        f"({sorted(real_by_source)})"
    )
    print(f"Evaluating {len(fake_by_source)} fake generators against that shared real pool...")

    results = evaluate_by_fake_source(model, real_samples, fake_by_source, device, args.batch_size)

    csv_path = args.output_dir / "by_generator_results.csv"
    save_results_csv(results, csv_path)

    print(f"\n{'generator':<28}{'n_fake':>8}{'accuracy':>10}{'recall':>10}{'roc_auc':>10}")
    for row in sorted(results, key=lambda r: r["roc_auc"]):
        print(
            f"{row['source']:<28}{row['n_fake']:>8}{row['accuracy']:>10.4f}"
            f"{row['recall']:>10.4f}{row['roc_auc']:>10.4f}"
        )
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
