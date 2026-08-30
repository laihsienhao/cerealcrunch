"""Dual-branch AIGC classifier: frozen CLIP RGB branch + a small SRM-inspired
noise-residual branch, fused into a single fake-vs-real logit.

Rationale (see PROBLEM.md / project notes, and arXiv:2502.15176): CNN
backbones fine-tuned end-to-end tend to overfit to their training
generator's low-level artifacts and generalize poorly to unseen generators;
a frozen CLIP encoder's broader, more semantic features transfer much
better. The noise-residual branch adds a complementary signal - fine
edge/texture detail - that's most informative on clean images and is the
first thing degraded by blur/compression, so the two branches are meant to
give the small trainable fusion head more than one kind of evidence to
weigh per example.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import nn
from transformers import CLIPVisionModelWithProjection

IMAGE_SIZE = 224
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def preprocess(image: Image.Image) -> torch.Tensor:
    """Resize a PIL image to the model's input size, as a [0, 1] float tensor (3, H, W)."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


# 4 standard, unambiguous high-pass/edge kernels (Sobel X/Y, two Laplacian
# variants). Inspired by the SRM (Spatial Rich Model) forensics technique of
# using a diverse bank of fixed filters rather than one hand-picked residual
# - NOT a reproduction of the classical 30-filter SRM basis, whose exact
# published coefficients we could not reliably verify from available
# sources.
_FILTER_BANK = torch.tensor(
    [
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  # Sobel X
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],  # Sobel Y
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],  # Laplacian (4-neighbor)
        [[1, 1, 1], [1, -8, 1], [1, 1, 1]],  # Laplacian (8-neighbor)
    ],
    dtype=torch.float32,
).unsqueeze(1)  # (4, 1, 3, 3)


class NoiseResidualBranch(nn.Module):
    """Shallow CNN over a small fixed high-pass filter bank, applied per RGB channel."""

    NUM_FILTERS = _FILTER_BANK.shape[0]
    NUM_CHANNELS = 3

    def __init__(self, out_dim: int = 64):
        super().__init__()
        # Same NUM_FILTERS filters applied independently to each of R, G, B
        # (not collapsed to grayscale first) so per-channel/color artifacts
        # aren't discarded before this branch even starts.
        self.register_buffer("filter_bank", _FILTER_BANK.repeat(self.NUM_CHANNELS, 1, 1, 1))

        in_channels = self.NUM_FILTERS * self.NUM_CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, H, W) float tensor in [0, 1]."""
        filtered = F.conv2d(x, self.filter_bank, padding=1, groups=self.NUM_CHANNELS)
        features = self.conv(filtered)
        return self.pool(features).flatten(1)


class ClipRgbBranch(nn.Module):
    """Frozen CLIP vision encoder, returning the projected image embedding."""

    def __init__(self, vision_model: CLIPVisionModelWithProjection | None = None):
        super().__init__()
        self.clip = vision_model or CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME)
        self.clip.requires_grad_(False)
        self.clip.eval()
        self.out_dim = self.clip.visual_projection.out_features

        self.register_buffer("clip_mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()  # keep the frozen backbone in eval mode regardless
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.clip_mean) / self.clip_std
        with torch.no_grad():
            outputs = self.clip(pixel_values=normalized)
        return outputs.image_embeds


class AIGCClassifier(nn.Module):
    """Frozen CLIP branch + trainable noise-residual branch, fused into one fake/real logit."""

    def __init__(self, clip_branch: ClipRgbBranch | None = None, residual_dim: int = 64):
        super().__init__()
        self.rgb_branch = clip_branch or ClipRgbBranch()
        self.noise_branch = NoiseResidualBranch(out_dim=residual_dim)

        fused_dim = self.rgb_branch.out_dim + residual_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 3, H, W) float tensor in [0, 1]. Returns (batch,) logits."""
        rgb_features = self.rgb_branch(x)
        noise_features = self.noise_branch(x)
        fused = torch.cat([rgb_features, noise_features], dim=1)
        return self.head(fused).squeeze(1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
