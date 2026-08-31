from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from aigc_detect.data import list_samples
from aigc_detect.evaluate import (
    EVAL_CONDITIONS,
    compute_final_score,
    evaluate_condition,
    run_robustness_eval,
)
from aigc_detect.model import AIGCClassifier
from aigc_detect.transforms import TRANSFORM_GRID

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cifake"
CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "models" / "checkpoints" / "aigc_classifier.pt"


def test_eval_conditions_cover_full_grid():
    assert len(EVAL_CONDITIONS) == 15
    assert EVAL_CONDITIONS[0] == {"name": "clean", "transform": None, "param": None}

    for transform_name, params in TRANSFORM_GRID.items():
        conditions = [c for c in EVAL_CONDITIONS if c["transform"] == transform_name]
        list_valued = [value for value in params.values() if isinstance(value, list)]
        if list_valued:
            assert [c["param"] for c in conditions] == list_valued[0]
        else:
            assert len(conditions) == 1
            assert conditions[0]["param"] is None


class _ConstantLogitModel(nn.Module):
    """Test double: ignores pixel content, returns a fixed logit per batch position."""

    def __init__(self, logits: list[float]):
        super().__init__()
        self.logits = torch.tensor(logits)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.logits[: pixel_values.shape[0]]


def test_evaluate_condition_metrics_on_synthetic_data(tmp_path):
    # predicted (logit > 0): [1, 0, 1, 0]; true labels: [1, 0, 0, 0] -> 3/4 correct.
    logits = [5.0, -5.0, 5.0, -5.0]
    labels = [1, 0, 0, 0]

    rng = np.random.default_rng(0)
    paths = []
    for i in range(4):
        array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        path = tmp_path / f"img{i}.png"
        Image.fromarray(array, mode="RGB").save(path)
        paths.append(path)

    samples = list(zip(paths, labels))
    condition = {"name": "clean", "transform": None, "param": None}
    result = evaluate_condition(
        _ConstantLogitModel(logits), samples, condition, torch.device("cpu"), batch_size=8
    )

    assert result["n"] == 4
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))
    assert 0.0 <= result["roc_auc"] <= 1.0


def test_compute_final_score_averages_correctly():
    results = [{"transform": "clean", "roc_auc": 0.9}]
    results += [{"transform": name, "roc_auc": auc} for name, auc in zip(
        ["jpeg_compression"] * 4 + ["gaussian_blur"] * 3 + ["resize"] * 2
        + ["gaussian_noise"] * 3 + ["color_jitter", "center_crop"],
        [0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.8, 0.7, 0.5, 0.6, 0.5, 0.6, 0.7],
    )]

    score = compute_final_score(results)

    assert score["auc_clean"] == pytest.approx(0.9)
    robust_values = [r["roc_auc"] for r in results if r["transform"] != "clean"]
    assert score["auc_robust"] == pytest.approx(sum(robust_values) / len(robust_values))
    assert score["final_score"] == pytest.approx(0.5 * 0.9 + 0.5 * score["auc_robust"])


@pytest.mark.skipif(
    not DATA_ROOT.exists() or not CHECKPOINT_PATH.exists(),
    reason="Requires CIFAKE test data and a trained checkpoint - see data/README.md and scripts/train.py",
)
def test_run_robustness_eval_end_to_end():
    samples = list_samples(DATA_ROOT, "test")
    tiny_samples = samples[:4] + samples[-4:]  # list_samples orders REAL then FAKE

    device = torch.device("cpu")
    model = AIGCClassifier().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    results = run_robustness_eval(model, tiny_samples, device, batch_size=8)

    assert len(results) == 15
    for row in results:
        assert row["n"] == 8
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["f1"] <= 1.0
