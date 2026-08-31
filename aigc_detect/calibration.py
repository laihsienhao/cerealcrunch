"""Temperature scaling (Guo et al. 2017): rescale logits before the sigmoid
so reported probabilities actually match empirical frequency, without
changing which side of the decision boundary any prediction falls on.

Webinar guidance: "output a calibrated probability, not just a label - you
will need it for thresholding and error analysis." A raw sigmoid(logit) from
a plain BCE-trained model is commonly overconfident.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def _bce_loss(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits / temperature))
    probs = np.clip(probs, _EPS, 1 - _EPS)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single scalar temperature minimizing BCE of sigmoid(logits/T)
    against true labels. Bounded to a sane range - within that range this is
    a smooth, well-behaved 1D optimization (standard temperature scaling)."""
    result = minimize_scalar(
        lambda t: _bce_loss(logits, labels, t), bounds=(0.05, 20.0), method="bounded"
    )
    return float(result.x)


def temperature_path_for(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_temperature.json")


def save_temperature(checkpoint_path: Path, temperature: float) -> None:
    path = temperature_path_for(checkpoint_path)
    path.write_text(json.dumps({"temperature": temperature}))


def load_temperature(checkpoint_path: Path) -> float:
    """Returns 1.0 (a no-op) if no calibration has been fit for this checkpoint."""
    path = temperature_path_for(checkpoint_path)
    if not path.exists():
        return 1.0
    return float(json.loads(path.read_text())["temperature"])
