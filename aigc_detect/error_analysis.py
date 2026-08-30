"""Error analysis: representative false positives/negatives (PROBLEM.md deliverable #5).

Runs the trained classifier on clean test images and highlights the most
confidently wrong predictions in each direction, plus overall false-positive/
false-negative rates for the trade-off discussion (false-flagging a real
photo vs. missing a fake).
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import torch
from PIL import Image

from aigc_detect.evaluate import predict_probs


def find_errors(
    model: torch.nn.Module,
    samples: list[tuple[Path, int]],
    device: torch.device,
    batch_size: int = 32,
) -> dict:
    """Run clean-condition predictions and split them into ranked FP/FN plus a rate summary.

    False positives are sorted most-confidently-wrong first (highest pred_prob);
    false negatives likewise (lowest pred_prob).
    """
    images = [Image.open(path).convert("RGB") for path, _ in samples]
    probs = predict_probs(model, images, device, batch_size)

    rows = []
    for (path, label), prob in zip(samples, probs):
        rows.append(
            {
                "path": path,
                "true_label": label,
                "pred_label": int(prob > 0.5),
                "pred_prob": float(prob),
            }
        )

    false_positives = [r for r in rows if r["true_label"] == 0 and r["pred_label"] == 1]
    false_negatives = [r for r in rows if r["true_label"] == 1 and r["pred_label"] == 0]
    false_positives.sort(key=lambda r: r["pred_prob"], reverse=True)
    false_negatives.sort(key=lambda r: r["pred_prob"])

    n_real = sum(1 for r in rows if r["true_label"] == 0)
    n_fake = sum(1 for r in rows if r["true_label"] == 1)
    summary = {
        "n_total": len(rows),
        "n_false_positives": len(false_positives),
        "n_false_negatives": len(false_negatives),
        "false_positive_rate": len(false_positives) / n_real if n_real else float("nan"),
        "false_negative_rate": len(false_negatives) / n_fake if n_fake else float("nan"),
    }

    return {"false_positives": false_positives, "false_negatives": false_negatives, "summary": summary}


def save_error_examples(errors: list[dict], output_dir: Path, prefix: str, n: int = 10) -> None:
    """Copy the top-n ranked error images for visual inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank, row in enumerate(errors[:n], start=1):
        src: Path = row["path"]
        dest = output_dir / f"{prefix}_{rank:02d}_prob{row['pred_prob']:.3f}{src.suffix}"
        shutil.copy(src, dest)


def save_error_summary_csv(
    false_positives: list[dict], false_negatives: list[dict], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "error_type": "false_positive",
            "path": str(r["path"]),
            "true_label": r["true_label"],
            "pred_prob": r["pred_prob"],
        }
        for r in false_positives
    ] + [
        {
            "error_type": "false_negative",
            "path": str(r["path"]),
            "true_label": r["true_label"],
            "pred_prob": r["pred_prob"],
        }
        for r in false_negatives
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["error_type", "path", "true_label", "pred_prob"])
        writer.writeheader()
        writer.writerows(rows)
