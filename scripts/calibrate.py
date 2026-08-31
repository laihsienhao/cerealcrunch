#!/usr/bin/env python
"""Fit and save a temperature-scaling calibration for a trained checkpoint.

Usage:
    python scripts/calibrate.py --checkpoint_path models/checkpoints/aigc_classifier_sid.pt \\
        --data_root data/raw/sid_set --val_split validation

Writes a `<checkpoint_stem>_temperature.json` sidecar next to the checkpoint
(aigc_detect.calibration.temperature_path_for) - scripts/predict.py and
scripts/evaluate.py auto-load it if present, so this only needs to be run
once per checkpoint.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from aigc_detect.calibration import fit_temperature, save_temperature, temperature_path_for
from aigc_detect.data import list_samples, split_train_val
from aigc_detect.evaluate import predict_logits
from aigc_detect.model import AIGCClassifier
from aigc_detect.transforms import strip_source_artifacts

CHUNK_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a temperature-scaling calibration on a checkpoint's validation logits."
    )
    parser.add_argument(
        "--checkpoint_path", type=Path, default=Path("models/checkpoints/aigc_classifier.pt")
    )
    parser.add_argument("--data_root", type=Path, default=Path("data/raw/cifake"))
    parser.add_argument(
        "--val_split",
        type=str,
        default=None,
        help="If set (e.g. 'validation'), load that dedicated split - use for SID_Set. "
        "Defaults to carving 10%% off 'train' the way CIFAKE needs to.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Stratified (by label) cap on how many validation images to use - a single-scalar "
        "fit doesn't need the full validation set, and capping keeps memory use bounded.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def _load_val_samples(args: argparse.Namespace) -> list[tuple[Path, int]]:
    if args.val_split is not None:
        return list_samples(args.data_root, args.val_split)
    samples = list_samples(args.data_root, "train")
    _, val_samples = split_train_val(samples, val_fraction=args.val_fraction, seed=args.seed)
    return val_samples


def _stratified_cap(samples: list[tuple[Path, int]], n_total: int, seed: int) -> list[tuple[Path, int]]:
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    n_per_label = n_total // len(by_label)
    capped: list[tuple[Path, int]] = []
    for label_samples in by_label.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)
        capped.extend(shuffled[:n_per_label])
    rng.shuffle(capped)
    return capped


def _compute_logits_in_chunks(
    model: torch.nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    """Processes samples in small chunks so at most CHUNK_SIZE decoded images
    are held in memory at once - loading tens of thousands of full-resolution
    images into a single list upfront is enough to OOM-kill the process."""
    rng = random.Random(seed)
    all_logits = []
    for start in range(0, len(samples), CHUNK_SIZE):
        chunk = samples[start : start + CHUNK_SIZE]
        images = [
            strip_source_artifacts(Image.open(path).convert("RGB"), rng) for path, _ in chunk
        ]
        all_logits.append(predict_logits(model, images, device, batch_size))
    return np.concatenate(all_logits)


def main() -> None:
    args = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))

    val_samples = _load_val_samples(args)
    val_samples = _stratified_cap(val_samples, args.n_samples, args.seed)
    print(f"Fitting temperature on {len(val_samples)} validation images...")

    labels = np.array([label for _, label in val_samples], dtype=np.float32)
    logits = _compute_logits_in_chunks(model, val_samples, device, args.batch_size, args.seed)

    temperature = fit_temperature(logits, labels)
    save_temperature(args.checkpoint_path, temperature)

    print(f"Fitted temperature = {temperature:.4f}")
    print(f"Saved to {temperature_path_for(args.checkpoint_path)}")


if __name__ == "__main__":
    main()
