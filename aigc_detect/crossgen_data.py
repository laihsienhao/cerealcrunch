"""Loading utilities for the ArtiFact cross-generator eval-only set.

data/raw/artifact_crossgen/test/{REAL,FAKE}/<source>__<filename>.jpg - see
data/README.md. The `<source>__` prefix (added at extraction time) records
which of ArtiFact's 25 fake generators or 8 real sources each image came
from, so results can be broken down per-source rather than only in
aggregate.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_samples_by_source(
    label_dir: Path, label: int
) -> dict[str, list[tuple[Path, int]]]:
    """Group one label folder's files by the source/generator name encoded
    in their `<source>__<filename>` prefix."""
    grouped: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for path in sorted(label_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source = path.name.split("__", 1)[0]
        grouped[source].append((path, label))
    return dict(grouped)


def split_by_source_stratified(
    groups: dict[str, list[tuple[Path, int]]],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 42,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Partition every source/generator group into train/val/test, so every
    group is represented in all three splits rather than some being absent
    from validation/test by chance.

    `n_train + n_val + n_test` must equal the total sample count across all
    groups (an exact partition, not a subsample) - val/test targets are
    distributed as evenly as possible across groups (base + remainder, so
    the totals land exactly on n_val/n_test even when group count doesn't
    divide evenly), and train gets whatever's left per group - which sums to
    exactly n_train in aggregate regardless of group size.
    """
    total_available = sum(len(samples) for samples in groups.values())
    if n_train + n_val + n_test != total_available:
        raise ValueError(
            f"n_train+n_val+n_test ({n_train + n_val + n_test}) must equal "
            f"the total available samples ({total_available}) - this partitions "
            f"the full pool, it doesn't subsample from it"
        )

    sources = sorted(groups.keys())
    n_groups = len(sources)
    val_base, val_remainder = divmod(n_val, n_groups)
    test_base, test_remainder = divmod(n_test, n_groups)

    rng = random.Random(seed)
    train_samples, val_samples, test_samples = [], [], []
    for i, source in enumerate(sources):
        shuffled = groups[source][:]
        rng.shuffle(shuffled)

        n_val_i = val_base + (1 if i < val_remainder else 0)
        n_test_i = test_base + (1 if i < test_remainder else 0)
        if n_val_i + n_test_i > len(shuffled):
            raise ValueError(
                f"source {source!r} has only {len(shuffled)} samples, "
                f"fewer than its val+test share ({n_val_i + n_test_i})"
            )

        val_samples.extend(shuffled[:n_val_i])
        test_samples.extend(shuffled[n_val_i : n_val_i + n_test_i])
        train_samples.extend(shuffled[n_val_i + n_test_i :])

    return train_samples, val_samples, test_samples
