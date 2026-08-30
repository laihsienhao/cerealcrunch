import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from aigc_detect.data import list_samples
from aigc_detect.model import AIGCClassifier
from aigc_detect.predict import list_images, predict_directory, save_predictions_json

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cifake"
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "aigc_classifier.pt"


def _make_synthetic_image(path: Path) -> None:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)


def test_list_images_finds_expected_extensions_recursively(tmp_path):
    _make_synthetic_image(tmp_path / "a.jpg")
    _make_synthetic_image(tmp_path / "b.PNG")  # extension casing shouldn't matter
    (tmp_path / "notes.txt").write_text("not an image")
    nested = tmp_path / "subdir"
    nested.mkdir()
    _make_synthetic_image(nested / "c.jpeg")

    found = list_images(tmp_path)

    assert found == sorted(found)
    assert {p.name for p in found} == {"a.jpg", "b.PNG", "c.jpeg"}


class _ConstantLogitModel(nn.Module):
    """Test double: ignores pixel content, returns a fixed logit per batch position."""

    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = torch.tensor(logits)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.logits[: pixel_values.shape[0]]


def test_predict_directory_skips_unreadable_files(tmp_path):
    _make_synthetic_image(tmp_path / "img_a.jpg")
    _make_synthetic_image(tmp_path / "img_b.jpg")
    (tmp_path / "corrupt.jpg").write_bytes(b"not a real image")

    model = _ConstantLogitModel([2.0, -2.0])
    results = predict_directory(model, tmp_path, torch.device("cpu"), batch_size=8)

    assert len(results) == 2
    paths = {r["image_path"] for r in results}
    assert paths == {(tmp_path / "img_a.jpg").as_posix(), (tmp_path / "img_b.jpg").as_posix()}
    for r in results:
        assert 0.0 <= r["pred"] <= 1.0


def test_save_predictions_json_roundtrip(tmp_path):
    results = [
        {"image_path": "some/dir/a.jpg", "pred": 0.87},
        {"image_path": "some/dir/b.jpg", "pred": 0.12},
    ]
    path = tmp_path / "predictions.json"
    save_predictions_json(results, path)

    with open(path) as f:
        loaded = json.load(f)

    assert loaded == results


@pytest.mark.skipif(
    not DATA_ROOT.exists() or not CHECKPOINT_PATH.exists(),
    reason="Requires CIFAKE test data and a trained checkpoint - see data/README.md and scripts/train.py",
)
def test_predict_directory_end_to_end(tmp_path):
    samples = list_samples(DATA_ROOT, "test")[:4]
    for path, _ in samples:
        shutil.copy(path, tmp_path / path.name)

    device = torch.device("cpu")
    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    results = predict_directory(model, tmp_path, device, batch_size=8)

    assert len(results) == 4
    for r in results:
        assert set(r.keys()) == {"image_path", "pred"}
        assert 0.0 <= r["pred"] <= 1.0
