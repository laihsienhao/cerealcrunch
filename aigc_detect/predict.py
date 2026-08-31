"""Required deliverable: image directory in -> JSON predictions out.

"A script that takes an image directory as input and outputs a confidence
score for each image, indicating the likelihood that it is AIGC-generated.
The output should be a JSON file containing image_path and pred for each
image." (PROBLEM.md)

Unlike the internal eval/error-analysis scripts (which assume the known
CIFAKE folder layout), this runs against an arbitrary directory of images,
so it's the one place in the codebase that tolerates unpredictable external
input (a corrupt/unreadable file is skipped with a warning, not a crash).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from aigc_detect.evaluate import predict_probs

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(input_dir: Path) -> list[Path]:
    """Recursively find image files under input_dir, sorted for deterministic ordering."""
    return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def predict_directory(
    model: torch.nn.Module,
    input_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    temperature: float = 1.0,
) -> list[dict]:
    """Run predictions over every image in input_dir, skipping unreadable files."""
    paths = list_images(input_dir)

    valid_paths = []
    images = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"Warning: skipping unreadable image {path}: {exc}", file=sys.stderr)

    if not images:
        return []

    probs = predict_probs(model, images, device, batch_size, temperature=temperature)
    return [
        {"image_path": path.as_posix(), "pred": float(prob)}
        for path, prob in zip(valid_paths, probs)
    ]


def save_predictions_json(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
