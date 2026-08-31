"""Training loop for AIGCClassifier over CIFAKEDataset.

Only the model's trainable parameters (the noise-residual branch, if
enabled, and fusion head) are optimized - the frozen CLIP branch is
excluded from the optimizer entirely so its (nonexistent) gradients don't
consume optimizer memory/state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from aigc_detect.data import CIFAKEDataset, list_samples, split_train_val
from aigc_detect.model import AIGCClassifier, preprocess


def _collate(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = zip(*batch)
    pixel_values = torch.stack([preprocess(image) for image in images])
    return pixel_values, torch.tensor(labels, dtype=torch.float32)


def _build_loaders(
    train_samples: list, val_samples: list, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CIFAKEDataset(train_samples, augment=True)
    val_dataset = CIFAKEDataset(val_samples, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
    )
    return train_loader, val_loader


def make_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Carve train/val out of a single "train" split - for datasets with no
    dedicated validation split of their own (e.g. CIFAKE)."""
    samples = list_samples(data_root, "train")
    train_samples, val_samples = split_train_val(samples, val_fraction=val_fraction, seed=seed)
    return _build_loaders(train_samples, val_samples, batch_size, num_workers)


def make_presplit_dataloaders(
    data_root: Path,
    train_split: str = "train",
    val_split: str = "validation",
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Load pre-split train/validation folders directly, no carving - for
    datasets that already ship a dedicated validation split (e.g. SID_Set)."""
    train_samples = list_samples(data_root, train_split)
    val_samples = list_samples(data_root, val_split)
    return _build_loaders(train_samples, val_samples, batch_size, num_workers)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    n_correct = 0
    n_batches = len(loader)
    log_every = max(1, min(20, n_batches // 20))
    for batch_idx, (pixel_values, labels) in enumerate(loader, start=1):
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        # Accumulated from each batch's own forward pass, so - like Keras'
        # running train accuracy - this reflects a mix of weights across the
        # epoch rather than a fixed final snapshot; treat it as a training
        # trend indicator, not a clean-eval-equivalent number.
        n_correct += ((logits.detach() > 0) == (labels > 0.5)).sum().item()
        if batch_idx % log_every == 0 or batch_idx == n_batches:
            print(f"    batch {batch_idx}/{n_batches} - loss={loss.item():.4f}", flush=True)
    n = len(loader.dataset)
    return total_loss / n, n_correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    n_correct = 0
    for pixel_values, labels in loader:
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        logits = model(pixel_values)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        n_correct += ((logits > 0) == (labels > 0.5)).sum().item()

    n = len(loader.dataset)
    return total_loss / n, n_correct / n


@dataclass
class TrainConfig:
    data_root: Path
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    val_fraction: float = 0.1
    val_split: str | None = None  # set (e.g. "validation") to use make_presplit_dataloaders instead
    seed: int = 42
    num_workers: int = 0
    use_noise_branch: bool = True
    checkpoint_path: Path = Path("models/checkpoints/aigc_classifier.pt")
    latest_checkpoint_path: Path | None = None  # defaults to "latest_" + checkpoint_path name
    resume_from: Path | None = None
    device: str | None = None


def train(config: TrainConfig) -> AIGCClassifier:
    device = torch.device(
        config.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    if config.val_split is not None:
        train_loader, val_loader = make_presplit_dataloaders(
            config.data_root,
            val_split=config.val_split,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
    else:
        train_loader, val_loader = make_dataloaders(
            config.data_root,
            batch_size=config.batch_size,
            val_fraction=config.val_fraction,
            seed=config.seed,
            num_workers=config.num_workers,
        )

    model = AIGCClassifier(use_noise_branch=config.use_noise_branch).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    latest_checkpoint_path = config.latest_checkpoint_path or config.checkpoint_path.with_name(
        f"latest_{config.checkpoint_path.name}"
    )

    start_epoch = 1
    best_val_accuracy = -1.0
    if config.resume_from is not None and config.resume_from.exists():
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint["best_val_accuracy"]
        print(
            f"resumed from {config.resume_from}: starting at epoch {start_epoch} "
            f"(best_val_accuracy so far={best_val_accuracy:.4f})",
            flush=True,
        )

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f"epoch {epoch}/{config.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), config.checkpoint_path)

        # Saved every epoch (regardless of val improvement) so an
        # interruption loses at most the current epoch's progress, not the
        # whole run - includes optimizer state so resuming isn't just
        # reloading weights, but genuinely continuing training.
        latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
            },
            latest_checkpoint_path,
        )

    return model
