"""CIFAKE dataset loading (data/raw/cifake, see data/README.md).

Label convention: 0 = REAL (authentic), 1 = FAKE (AI-generated), matching
the `pred` semantics required by PROBLEM.md (pred = P(image is AIGC)).
"""
from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from aigc_detect.transforms import apply_random_robustness

CLASS_TO_LABEL = {"REAL": 0, "FAKE": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_samples(root: Path, split: str) -> list[tuple[Path, int]]:
    """List (image_path, label) pairs for a CIFAKE split ('train' or 'test')."""
    samples = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = root / split / class_name
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


def split_train_val(
    samples: list[tuple[Path, int]], val_fraction: float = 0.1, seed: int = 42
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Shuffle and split samples into (train, val), stratified by label."""
    rng = random.Random(seed)
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    train, val = [], []
    for label_samples in by_label.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)
        n_val = round(len(shuffled) * val_fraction)
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CIFAKEDataset(Dataset):
    """Yields (PIL.Image, label) pairs, with optional robustness augmentation."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        augment: bool = False,
        seed: int | None = None,
    ):
        self.samples = samples
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = apply_random_robustness(image, rng=self._rng)
        return image, label
