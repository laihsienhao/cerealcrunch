import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aigc_detect.data import CIFAKEDataset, list_samples
from aigc_detect.model import IMAGE_SIZE, AIGCClassifier
from aigc_detect.train import (
    _collate,
    evaluate,
    make_dataloaders,
    make_presplit_dataloaders,
    train_one_epoch,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cifake"
SID_SET_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "sid_set"

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason="CIFAKE dataset not found at data/raw/cifake - see data/README.md",
)


def _build_model_or_skip() -> AIGCClassifier:
    try:
        return AIGCClassifier()
    except OSError as exc:
        pytest.skip(f"Could not download pretrained CLIP weights (network/cert issue?): {exc}")


def test_make_dataloaders_batch_shapes():
    train_loader, val_loader = make_dataloaders(DATA_ROOT, batch_size=8, val_fraction=0.1)

    pixel_values, labels = next(iter(train_loader))
    assert pixel_values.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert labels.shape == (8,)
    assert labels.dtype == torch.float32

    pixel_values, labels = next(iter(val_loader))
    assert pixel_values.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.skipif(
    not SID_SET_ROOT.exists(),
    reason="SID_Set data not found at data/raw/sid_set - see data/README.md",
)
def test_make_presplit_dataloaders_uses_dedicated_splits_no_carving():
    train_loader, val_loader = make_presplit_dataloaders(
        SID_SET_ROOT, train_split="train", val_split="validation", batch_size=8
    )

    pixel_values, labels = next(iter(train_loader))
    assert pixel_values.shape == (8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert labels.dtype == torch.float32

    # Full pre-split pools, not a 90/10 carve of "train" alone.
    assert len(train_loader.dataset) == 100_000
    assert len(val_loader.dataset) == 20_000


def test_train_one_epoch_runs_and_produces_finite_loss():
    samples = list_samples(DATA_ROOT, "train")
    tiny_samples = samples[:8] + samples[-8:]  # list_samples orders REAL then FAKE
    dataset = CIFAKEDataset(tiny_samples, augment=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=_collate)

    model = _build_model_or_skip()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    loss, accuracy = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"))
    assert math.isfinite(loss)
    assert loss > 0
    assert 0.0 <= accuracy <= 1.0


class _FixedLogitModel(nn.Module):
    """Test double: returns the value embedded in pixel_values[:, 0, 0, 0] as the logit."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return pixel_values[:, 0, 0, 0]


def test_evaluate_accuracy_matches_known_labels():
    logits = torch.tensor([5.0, -5.0, 5.0, -5.0])  # -> predicted labels [1, 0, 1, 0]
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])  # 3rd prediction is wrong -> accuracy 3/4

    pixel_values = torch.zeros(4, 3, 4, 4)
    pixel_values[:, 0, 0, 0] = logits

    loader = DataLoader(TensorDataset(pixel_values, labels), batch_size=2)
    _, accuracy = evaluate(
        _FixedLogitModel(), loader, nn.BCEWithLogitsLoss(), torch.device("cpu")
    )

    assert accuracy == pytest.approx(0.75)
