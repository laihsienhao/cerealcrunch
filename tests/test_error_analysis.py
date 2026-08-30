from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from aigc_detect.data import list_samples
from aigc_detect.error_analysis import find_errors
from aigc_detect.model import AIGCClassifier

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cifake"
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "aigc_classifier.pt"


class _ConstantLogitModel(nn.Module):
    """Test double: ignores pixel content, returns a fixed logit per batch position."""

    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = torch.tensor(logits)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.logits[: pixel_values.shape[0]]


def _make_sample_images(tmp_path: Path, n: int) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        path = tmp_path / f"img{i}.png"
        Image.fromarray(array, mode="RGB").save(path)
        paths.append(path)
    return paths


def test_find_errors_classifies_and_ranks_correctly(tmp_path):
    # idx0: real, predicted fake -> FP. idx1: real, predicted real -> TN.
    # idx2: fake, predicted real -> FN. idx3: fake, predicted fake -> TP.
    labels = [0, 0, 1, 1]
    logits = [5.0, -5.0, -5.0, 5.0]

    paths = _make_sample_images(tmp_path, 4)
    samples = list(zip(paths, labels))
    model = _ConstantLogitModel(logits)

    result = find_errors(model, samples, torch.device("cpu"), batch_size=8)

    assert len(result["false_positives"]) == 1
    assert result["false_positives"][0]["path"] == paths[0]
    assert len(result["false_negatives"]) == 1
    assert result["false_negatives"][0]["path"] == paths[2]

    summary = result["summary"]
    assert summary["n_total"] == 4
    assert summary["false_positive_rate"] == pytest.approx(0.5)  # 1 FP / 2 real
    assert summary["false_negative_rate"] == pytest.approx(0.5)  # 1 FN / 2 fake


def test_find_errors_ranks_most_confident_first(tmp_path):
    # Two false positives (both real, predicted fake), differing confidence.
    labels = [0, 0]
    logits = [1.0, 5.0]  # idx1 is more confidently wrong than idx0

    paths = _make_sample_images(tmp_path, 2)
    samples = list(zip(paths, labels))
    model = _ConstantLogitModel(logits)

    result = find_errors(model, samples, torch.device("cpu"), batch_size=8)

    assert len(result["false_positives"]) == 2
    assert result["false_positives"][0]["path"] == paths[1]  # highest pred_prob first
    assert result["false_positives"][1]["path"] == paths[0]


@pytest.mark.skipif(
    not DATA_ROOT.exists() or not CHECKPOINT_PATH.exists(),
    reason="Requires CIFAKE test data and a trained checkpoint - see data/README.md and scripts/train.py",
)
def test_find_errors_end_to_end_is_internally_consistent():
    samples = list_samples(DATA_ROOT, "test")
    tiny_samples = samples[:4] + samples[-4:]  # list_samples orders REAL then FAKE

    device = torch.device("cpu")
    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    result = find_errors(model, tiny_samples, device, batch_size=8)
    summary = result["summary"]

    assert summary["n_total"] == 8
    assert summary["n_false_positives"] == len(result["false_positives"])
    assert summary["n_false_negatives"] == len(result["false_negatives"])
    for row in result["false_positives"] + result["false_negatives"]:
        assert 0.0 <= row["pred_prob"] <= 1.0
