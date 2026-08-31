from pathlib import Path

import numpy as np
import pytest

from aigc_detect.calibration import (
    _bce_loss,
    fit_temperature,
    load_temperature,
    save_temperature,
    temperature_path_for,
)


def _make_overconfident_logits(
    seed: int = 0, n: int = 400, error_rate: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic logits with realistic overconfidence: very large magnitude
    (near-certain), but wrong `error_rate` of the time - a model that's
    confidently wrong often enough is exactly what temperature scaling
    should soften (T > 1), unlike a model that's simply always correct."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n).astype(np.float32)
    label_sign = np.where(labels > 0.5, 1.0, -1.0)
    is_wrong = rng.uniform(size=n) < error_rate
    predicted_sign = np.where(is_wrong, -label_sign, label_sign)
    logits = predicted_sign * rng.uniform(5.0, 15.0, size=n)
    return logits, labels


def test_fit_temperature_reduces_overconfidence():
    logits, labels = _make_overconfident_logits()

    temperature = fit_temperature(logits, labels)
    loss_at_fitted = _bce_loss(logits, labels, temperature)
    loss_at_one = _bce_loss(logits, labels, 1.0)

    assert temperature > 1.0  # should soften the overconfident logits
    assert loss_at_fitted < loss_at_one


def test_load_temperature_defaults_to_one_for_missing_file(tmp_path):
    checkpoint_path = tmp_path / "does_not_exist.pt"
    assert load_temperature(checkpoint_path) == pytest.approx(1.0)


def test_save_and_load_temperature_round_trips(tmp_path):
    checkpoint_path = tmp_path / "aigc_classifier.pt"
    checkpoint_path.write_bytes(b"fake checkpoint bytes")

    save_temperature(checkpoint_path, 2.345)
    loaded = load_temperature(checkpoint_path)

    assert loaded == pytest.approx(2.345)
    assert temperature_path_for(checkpoint_path).name == "aigc_classifier_temperature.json"
