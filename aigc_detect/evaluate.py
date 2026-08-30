"""Robustness evaluation harness: clean vs. each PROBLEM.md transform/severity.

Produces the required deliverable #4 - "a compact table or visual summary
comparing performance on clean images versus transformed images" - by
running the trained classifier against the CIFAKE test split under 1 clean
condition + 14 individual (transform, severity) conditions from
TRANSFORM_GRID (one transform applied at a time, not stacked - unlike
training-time augmentation, this must isolate each condition to be
attributable).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from aigc_detect.data import list_samples
from aigc_detect.model import preprocess
from aigc_detect.transforms import TRANSFORM_FUNCTIONS, TRANSFORM_GRID


def _build_eval_conditions() -> list[dict]:
    conditions: list[dict] = [{"name": "clean", "transform": None, "param": None}]
    for transform_name, params in TRANSFORM_GRID.items():
        list_valued_keys = [key for key, value in params.items() if isinstance(value, list)]
        if list_valued_keys:
            for value in params[list_valued_keys[0]]:
                conditions.append({"name": transform_name, "transform": transform_name, "param": value})
        else:
            conditions.append({"name": transform_name, "transform": transform_name, "param": None})
    return conditions


EVAL_CONDITIONS: list[dict] = _build_eval_conditions()


def _condition_label(condition: dict) -> str:
    if condition["transform"] is None:
        return "clean"
    if condition["param"] is not None:
        return f"{condition['transform']}_{condition['param']}"
    return condition["transform"]


def sample_test_subset(
    data_root: Path, n_per_class: int = 500, seed: int = 42
) -> list[tuple[Path, int]]:
    """Stratified subset of the test split, balanced across labels."""
    samples = list_samples(data_root, "test")
    by_label: dict[int, list[tuple[Path, int]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    rng = random.Random(seed)
    subset: list[tuple[Path, int]] = []
    for label_samples in by_label.values():
        shuffled = label_samples[:]
        rng.shuffle(shuffled)
        subset.extend(shuffled[:n_per_class])
    rng.shuffle(subset)
    return subset


def apply_condition(image: Image.Image, condition: dict, rng: random.Random) -> Image.Image:
    """Apply a single eval condition (or pass through unchanged for "clean")."""
    name = condition["transform"]
    if name is None:
        return image

    params = TRANSFORM_GRID[name]
    func = TRANSFORM_FUNCTIONS[name]
    if name == "jpeg_compression":
        return func(image, quality=condition["param"])
    if name == "gaussian_blur":
        return func(image, sigma=condition["param"])
    if name == "resize":
        return func(image, scale=condition["param"])
    if name == "gaussian_noise":
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        return func(image, sigma=condition["param"], rng=noise_rng)
    if name == "color_jitter":
        return func(
            image,
            brightness=params["brightness"],
            contrast=params["contrast"],
            saturation=params["saturation"],
            rng=rng,
        )
    if name == "center_crop":
        return func(image, fraction=params["fraction"])
    raise ValueError(f"Unknown transform: {name}")


def predict_probs(
    model: torch.nn.Module, images: list[Image.Image], device: torch.device, batch_size: int = 32
) -> np.ndarray:
    """Batched inference: PIL images -> P(image is AI-generated) per image."""
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            pixel_values = torch.stack([preprocess(image) for image in batch]).to(device)
            logits = model(pixel_values)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


def evaluate_condition(
    model: torch.nn.Module,
    samples: list[tuple[Path, int]],
    condition: dict,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """Apply one condition to every sample, predict, and score against true labels."""
    rng = random.Random(seed)
    images = []
    labels = []
    for path, label in samples:
        image = apply_condition(Image.open(path).convert("RGB"), condition, rng)
        images.append(image)
        labels.append(label)

    labels_array = np.array(labels)
    probs = predict_probs(model, images, device, batch_size)
    preds = (probs > 0.5).astype(int)
    has_both_classes = len(set(labels)) > 1

    return {
        "condition": _condition_label(condition),
        "transform": condition["transform"] or "clean",
        "param": condition["param"],
        "n": len(labels),
        "accuracy": accuracy_score(labels_array, preds),
        "precision": precision_score(labels_array, preds, zero_division=0),
        "recall": recall_score(labels_array, preds, zero_division=0),
        "f1": f1_score(labels_array, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels_array, probs) if has_both_classes else float("nan"),
    }


def run_robustness_eval(
    model: torch.nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int = 32,
) -> list[dict]:
    """Evaluate the model across all 15 conditions (1 clean + 14 transformed)."""
    return [
        evaluate_condition(model, samples, condition, device, batch_size)
        for condition in EVAL_CONDITIONS
    ]


def save_results_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


_SURFACE = "#fcfcfb"
_BAR_COLOR = "#2a78d6"
_BASELINE_COLOR = "#52514e"
_PRIMARY_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"
_MUTED_INK = "#898781"
_GRID_COLOR = "#e1e0d9"

_FAMILY_AXIS_LABELS = {
    "jpeg_compression": "JPEG quality",
    "gaussian_blur": "Blur σ",
    "resize": "Resize scale",
    "gaussian_noise": "Noise σ",
    "color_jitter": "Color jitter",
    "center_crop": "Center crop",
}


def _condition_x_label(family: str, param) -> str:
    if param is not None:
        return str(param)
    params = TRANSFORM_GRID[family]
    if family == "color_jitter":
        return f"±{int(params['brightness'] * 100)}%"
    if family == "center_crop":
        return f"{int(params['fraction'] * 100)}%"
    return "applied"


def plot_robustness(results: list[dict], path: Path) -> None:
    """Bar chart per transform family: accuracy at each severity vs. the clean baseline."""
    clean_accuracy = next(r["accuracy"] for r in results if r["transform"] == "clean")
    families = list(TRANSFORM_GRID.keys())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=_SURFACE)
    fig.suptitle("Robustness: accuracy vs. clean baseline", color=_PRIMARY_INK, fontsize=14, y=0.98)

    for ax, family in zip(axes.flat, families):
        rows = [r for r in results if r["transform"] == family]
        labels = [_condition_x_label(family, r["param"]) for r in rows]
        accuracies = [r["accuracy"] for r in rows]

        ax.set_facecolor(_SURFACE)
        ax.grid(axis="y", color=_GRID_COLOR, linewidth=0.7, zorder=0)
        bars = ax.bar(labels, accuracies, color=_BAR_COLOR, width=0.5, zorder=3)
        ax.axhline(clean_accuracy, color=_BASELINE_COLOR, linestyle="--", linewidth=1.5, zorder=2)

        # Labels sit just inside the top of each bar (not above it) so they
        # never collide with the clean-baseline reference line.
        for bar, accuracy in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                accuracy - 0.03,
                f"{accuracy:.2f}",
                ha="center",
                va="top",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(_FAMILY_AXIS_LABELS[family], color=_PRIMARY_INK, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.tick_params(colors=_MUTED_INK)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_MUTED_INK)

    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=_BAR_COLOR),
            plt.Line2D([0], [0], color=_BASELINE_COLOR, linestyle="--", linewidth=1.5),
        ],
        labels=["Accuracy", "Clean baseline"],
        loc="lower center",
        ncol=2,
        frameon=False,
        labelcolor=_SECONDARY_INK,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
