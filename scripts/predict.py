#!/usr/bin/env python
"""Required deliverable CLI: image directory in -> JSON predictions out.

Usage:
    python scripts/predict.py --input_dir path/to/images --output_json outputs/predictions.json

Output format:
    [{"image_path": "...", "pred": 0.87}, ...]

`pred` is a confidence score in [0, 1]: the probability the image is AI-generated.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from aigc_detect.calibration import load_temperature
from aigc_detect.model import AIGCClassifier
from aigc_detect.predict import predict_directory, save_predictions_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AIGC-image detector over a directory of images."
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, default=Path("outputs/predictions.json"))
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier.pt")
    )
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise NotADirectoryError(f"--input_dir does not exist: {args.input_dir}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    temperature = load_temperature(args.checkpoint_path)
    results = predict_directory(
        model, args.input_dir, device, batch_size=args.batch_size, temperature=temperature
    )
    save_predictions_json(results, args.output_json)

    print(f"Wrote {len(results)} predictions to {args.output_json} (temperature={temperature:.4f})")


if __name__ == "__main__":
    main()
