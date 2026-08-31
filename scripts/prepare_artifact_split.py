#!/usr/bin/env python
"""Split the extracted ArtiFact cross-generator pool into train/validation/test.

Usage:
    python scripts/prepare_artifact_split.py

Reads data/raw/artifact_crossgen/test/{REAL,FAKE}/<source>__<name>.jpg (the
full stratified pool: 200/fake-generator x 25, ~625/real-source x 8) and
copies (not moves - keeps the original pool intact) into:
    data/raw/artifact_full/{train,validation,test}/{REAL,FAKE}/
matching the exact CIFAKEDataset/list_samples folder convention, stratified
by source/generator so every one of the 25 generators and 8 real sources is
represented in all three splits (aigc_detect.crossgen_data.split_by_source_stratified).
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aigc_detect.crossgen_data import list_samples_by_source, split_by_source_stratified

SOURCE_ROOT = Path("data/raw/artifact_crossgen/test")
DEST_ROOT = Path("data/raw/artifact_full")

N_TRAIN = 5000
N_VAL = 1000
N_TEST = 1000
SEED = 42


def main() -> None:
    if DEST_ROOT.exists():
        shutil.rmtree(DEST_ROOT)  # avoid stale files from a previous split's different group sizes

    real_by_source = list_samples_by_source(SOURCE_ROOT / "REAL", label=0)
    fake_by_source = list_samples_by_source(SOURCE_ROOT / "FAKE", label=1)
    print(
        f"Real: {sum(len(v) for v in real_by_source.values())} images across "
        f"{len(real_by_source)} sources"
    )
    print(
        f"Fake: {sum(len(v) for v in fake_by_source.values())} images across "
        f"{len(fake_by_source)} generators"
    )

    real_train, real_val, real_test = split_by_source_stratified(
        real_by_source, N_TRAIN, N_VAL, N_TEST, seed=SEED
    )
    fake_train, fake_val, fake_test = split_by_source_stratified(
        fake_by_source, N_TRAIN, N_VAL, N_TEST, seed=SEED
    )

    splits = {
        "train": real_train + fake_train,
        "validation": real_val + fake_val,
        "test": real_test + fake_test,
    }
    label_to_folder = {0: "REAL", 1: "FAKE"}

    for split_name, samples in splits.items():
        for label_dir in label_to_folder.values():
            (DEST_ROOT / split_name / label_dir).mkdir(parents=True, exist_ok=True)

        for path, label in samples:
            dest = DEST_ROOT / split_name / label_to_folder[label] / path.name
            shutil.copy2(path, dest)

        n_real = sum(1 for _, label in samples if label == 0)
        n_fake = sum(1 for _, label in samples if label == 1)
        print(f"{split_name}: {n_real} REAL, {n_fake} FAKE -> {DEST_ROOT / split_name}")


if __name__ == "__main__":
    main()
