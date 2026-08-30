from pathlib import Path

import pytest

from aigc_detect.data import CIFAKEDataset, list_samples, split_train_val

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cifake"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="CIFAKE dataset not found at data/raw/cifake - see data/README.md",
)


def test_list_samples_train_counts():
    samples = list_samples(DATA_ROOT, "train")
    labels = [label for _, label in samples]
    assert labels.count(0) == 50000
    assert labels.count(1) == 50000


def test_list_samples_test_counts():
    samples = list_samples(DATA_ROOT, "test")
    labels = [label for _, label in samples]
    assert labels.count(0) == 10000
    assert labels.count(1) == 10000


def test_split_train_val_is_disjoint_and_covers_all_samples():
    samples = list_samples(DATA_ROOT, "train")
    train, val = split_train_val(samples, val_fraction=0.1, seed=0)

    assert set(train).isdisjoint(set(val))
    assert len(train) + len(val) == len(samples)
    assert set(train) | set(val) == set(samples)


def test_split_train_val_is_stratified():
    samples = list_samples(DATA_ROOT, "train")
    train, val = split_train_val(samples, val_fraction=0.1, seed=0)

    val_labels = [label for _, label in val]
    assert val_labels.count(0) == pytest.approx(len(val) / 2, abs=5)
    assert val_labels.count(1) == pytest.approx(len(val) / 2, abs=5)

    train_labels = [label for _, label in train]
    assert train_labels.count(0) == pytest.approx(len(train) / 2, abs=5)
    assert train_labels.count(1) == pytest.approx(len(train) / 2, abs=5)


def test_dataset_returns_image_and_label():
    samples = list_samples(DATA_ROOT, "test")[:4]
    dataset = CIFAKEDataset(samples, augment=False)

    assert len(dataset) == 4
    image, label = dataset[0]
    assert image.size == (32, 32)
    assert image.mode == "RGB"
    assert label == samples[0][1]
    assert label in (0, 1)


def test_dataset_augmented_sample_is_valid_image():
    samples = list_samples(DATA_ROOT, "test")[:1]
    dataset = CIFAKEDataset(samples, augment=True, seed=0)

    image, label = dataset[0]
    assert image.size == (32, 32)
    assert image.mode == "RGB"
    assert label == samples[0][1]
