"""Robustness transforms matching the exact grid specified in PROBLEM.md.

Each transform takes and returns an RGB `PIL.Image.Image` of the same size,
so they compose with any torchvision/Dataset-style pipeline. They're used
both as training-time augmentation and as the test-time corruption suite
for the clean-vs-transformed robustness evaluation.
"""
from __future__ import annotations

import io
import math
import random
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance

TRANSFORM_GRID = {
    "jpeg_compression": {"quality": [90, 70, 50, 30]},
    "gaussian_blur": {"sigma": [0.5, 1.0, 2.0]},
    "resize": {"scale": [0.5, 0.25]},
    "gaussian_noise": {"sigma": [0.02, 0.05, 0.10]},
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
    "center_crop": {"fraction": 0.8},
}


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Re-encode through an in-memory JPEG buffer at `quality` (1-95)."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Blur with kernel standard deviation `sigma` (kernel size auto-derived)."""
    bgr = _pil_to_bgr(image)
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return _bgr_to_pil(blurred)


def resize_degrade(image: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` then upscale back to the original size."""
    width, height = image.size
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    bgr = _pil_to_bgr(image)
    downscaled = cv2.resize(bgr, small_size, interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(downscaled, (width, height), interpolation=cv2.INTER_LINEAR)
    return _bgr_to_pil(upscaled)


def gaussian_noise(
    image: Image.Image, sigma: float, rng: np.random.Generator | None = None
) -> Image.Image:
    """Add N(0, sigma) noise on a [0, 1]-normalized image, then clip."""
    rng = rng or np.random.default_rng()
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = np.clip(array + rng.normal(0.0, sigma, size=array.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).round().astype(np.uint8))


def color_jitter(
    image: Image.Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    rng: random.Random | None = None,
) -> Image.Image:
    """Randomly scale brightness/contrast/saturation, each by +/- the given fraction."""
    rng = rng or random.Random()
    result = image.convert("RGB")
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(1 - brightness, 1 + brightness))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(1 - contrast, 1 + contrast))
    result = ImageEnhance.Color(result).enhance(rng.uniform(1 - saturation, 1 + saturation))
    return result


def center_crop(image: Image.Image, fraction: float = 0.8) -> Image.Image:
    """Crop the centered `fraction` of width/height, then resize back to the original size."""
    width, height = image.size
    crop_width, crop_height = round(width * fraction), round(height * fraction)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.BILINEAR)


# Range for strip_source_artifacts' random-aspect-ratio crop (width/height).
_ASPECT_RATIO_RANGE = (0.75, 1.33)
# Range for its random-resolution resize (shorter side, in px).
_RESOLUTION_RANGE = (256, 800)
# Range for its random-quality JPEG re-encode.
_QUALITY_RANGE = (40, 95)


def _random_aspect_crop(image: Image.Image, rng: random.Random) -> Image.Image:
    """Crop the largest possible region at a randomly drawn aspect ratio."""
    width, height = image.size
    log_low, log_high = math.log(_ASPECT_RATIO_RANGE[0]), math.log(_ASPECT_RATIO_RANGE[1])
    target_ratio = math.exp(rng.uniform(log_low, log_high))  # width / height

    if target_ratio > width / height:
        crop_width, crop_height = width, round(width / target_ratio)
    else:
        crop_width, crop_height = round(height * target_ratio), height
    crop_width = max(1, min(crop_width, width))
    crop_height = max(1, min(crop_height, height))

    left = rng.randint(0, width - crop_width)
    top = rng.randint(0, height - crop_height)
    return image.crop((left, top, left + crop_width, top + crop_height))


def _random_resolution_resize(image: Image.Image, rng: random.Random) -> Image.Image:
    """Resize (aspect-preserving) so the shorter side equals a randomly drawn length."""
    width, height = image.size
    target_short = rng.randint(*_RESOLUTION_RANGE)
    scale = target_short / min(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.BILINEAR)


def strip_source_artifacts(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Randomize aspect ratio, resolution, and JPEG quality.

    Removes any dataset-level shortcut tied to a class's native framing or
    compression history (e.g. one class always being exactly square at a
    fixed resolution with one fixed encoder signature) rather than genuine
    content. Applied unconditionally to every image, both classes, so
    neither retains a distinguishing source fingerprint. Distinct from
    apply_random_robustness, which simulates post-hoc real-world
    degradation for robustness training/eval rather than equalizing
    source-level distributions.
    """
    rng = rng or random.Random()
    result = image.convert("RGB")
    result = _random_aspect_crop(result, rng)
    result = _random_resolution_resize(result, rng)
    quality = rng.randint(*_QUALITY_RANGE)
    return jpeg_compression(result, quality=quality)


TRANSFORM_FUNCTIONS: dict[str, Callable[..., Image.Image]] = {
    "jpeg_compression": jpeg_compression,
    "gaussian_blur": gaussian_blur,
    "resize": resize_degrade,
    "gaussian_noise": gaussian_noise,
    "color_jitter": color_jitter,
    "center_crop": center_crop,
}


# P(exactly k of the 6 transforms are applied), k=0..6. Weighted toward 0-1
# transforms to match how the robustness eval measures performance (clean vs.
# ONE transform at a time), with 2+ stacked kept as a minority case for
# realistic compound degradation (e.g. resize then re-compress).
_STACK_WEIGHTS = [0.30, 0.40, 0.15, 0.08, 0.04, 0.02, 0.01]


def _sample_num_transforms(rng: random.Random) -> int:
    """Draw how many transforms to apply, per _STACK_WEIGHTS."""
    return rng.choices(range(len(TRANSFORM_GRID) + 1), weights=_STACK_WEIGHTS, k=1)[0]


def apply_random_robustness(image: Image.Image, rng: random.Random | None = None) -> Image.Image:
    """Apply a random subset of PROBLEM.md transforms, sized per _STACK_WEIGHTS.

    Severity for each applied transform is drawn from its own value(s) in
    TRANSFORM_GRID, so training-time augmentation matches the exact
    distribution the robustness eval will test against.
    """
    rng = rng or random.Random()
    num_transforms = _sample_num_transforms(rng)
    chosen_names = rng.sample(list(TRANSFORM_GRID), num_transforms)

    result = image
    for name in chosen_names:
        params = TRANSFORM_GRID[name]
        func = TRANSFORM_FUNCTIONS[name]
        if name == "jpeg_compression":
            result = func(result, quality=rng.choice(params["quality"]))
        elif name == "gaussian_blur":
            result = func(result, sigma=rng.choice(params["sigma"]))
        elif name == "resize":
            result = func(result, scale=rng.choice(params["scale"]))
        elif name == "gaussian_noise":
            noise_rng = np.random.default_rng(rng.randrange(2**32))
            result = func(result, sigma=rng.choice(params["sigma"]), rng=noise_rng)
        elif name == "color_jitter":
            result = func(
                result,
                brightness=params["brightness"],
                contrast=params["contrast"],
                saturation=params["saturation"],
                rng=rng,
            )
        elif name == "center_crop":
            result = func(result, fraction=params["fraction"])
    return result
