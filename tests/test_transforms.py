import random

import numpy as np
import pytest
from PIL import Image

from aigc_detect.transforms import (
    _STACK_WEIGHTS,
    TRANSFORM_GRID,
    _sample_num_transforms,
    apply_random_robustness,
    center_crop,
    color_jitter,
    gaussian_blur,
    gaussian_noise,
    jpeg_compression,
    resize_degrade,
)

# Transcribed independently from PROBLEM.md so a future edit to TRANSFORM_GRID
# can't silently drift from the brief without this test catching it.
EXPECTED_TRANSFORM_GRID = {
    "jpeg_compression": {"quality": [90, 70, 50, 30]},
    "gaussian_blur": {"sigma": [0.5, 1.0, 2.0]},
    "resize": {"scale": [0.5, 0.25]},
    "gaussian_noise": {"sigma": [0.02, 0.05, 0.10]},
    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2},
    "center_crop": {"fraction": 0.8},
}


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_transform_grid_matches_spec():
    assert TRANSFORM_GRID == EXPECTED_TRANSFORM_GRID


def _assert_same_size_and_changed(original: Image.Image, transformed: Image.Image):
    assert transformed.size == original.size
    assert transformed.mode == original.mode
    assert not np.array_equal(np.asarray(transformed), np.asarray(original))


@pytest.mark.parametrize("quality", TRANSFORM_GRID["jpeg_compression"]["quality"])
def test_jpeg_compression(sample_image, quality):
    _assert_same_size_and_changed(sample_image, jpeg_compression(sample_image, quality))


@pytest.mark.parametrize("sigma", TRANSFORM_GRID["gaussian_blur"]["sigma"])
def test_gaussian_blur(sample_image, sigma):
    _assert_same_size_and_changed(sample_image, gaussian_blur(sample_image, sigma))


@pytest.mark.parametrize("scale", TRANSFORM_GRID["resize"]["scale"])
def test_resize_degrade(sample_image, scale):
    _assert_same_size_and_changed(sample_image, resize_degrade(sample_image, scale))


@pytest.mark.parametrize("sigma", TRANSFORM_GRID["gaussian_noise"]["sigma"])
def test_gaussian_noise(sample_image, sigma):
    result = gaussian_noise(sample_image, sigma, rng=np.random.default_rng(1))
    _assert_same_size_and_changed(sample_image, result)


def test_color_jitter(sample_image):
    grid = TRANSFORM_GRID["color_jitter"]
    result = color_jitter(
        sample_image,
        brightness=grid["brightness"],
        contrast=grid["contrast"],
        saturation=grid["saturation"],
        rng=random.Random(0),
    )
    _assert_same_size_and_changed(sample_image, result)


def test_center_crop_removes_border_content(sample_image):
    fraction = TRANSFORM_GRID["center_crop"]["fraction"]
    result = center_crop(sample_image, fraction)
    _assert_same_size_and_changed(sample_image, result)

    width, height = sample_image.size
    crop_width, crop_height = round(width * fraction), round(height * fraction)
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    reference = sample_image.crop(
        (left, top, left + crop_width, top + crop_height)
    ).resize((width, height), Image.BILINEAR)

    assert np.array_equal(np.asarray(result), np.asarray(reference))


def test_stack_weights_sum_to_one():
    assert sum(_STACK_WEIGHTS) == pytest.approx(1.0)


def test_num_transforms_distribution_matches_weights():
    rng = random.Random(0)
    n_trials = 4000
    counts = [0] * len(_STACK_WEIGHTS)
    for _ in range(n_trials):
        counts[_sample_num_transforms(rng)] += 1

    for k, expected in enumerate(_STACK_WEIGHTS):
        observed = counts[k] / n_trials
        assert observed == pytest.approx(expected, abs=0.03)


def test_apply_random_robustness_can_return_clean_image(sample_image):
    for seed in range(200):
        result = apply_random_robustness(sample_image, rng=random.Random(seed))
        if np.array_equal(np.asarray(result), np.asarray(sample_image)):
            return
    pytest.fail("apply_random_robustness never returned a clean (untransformed) image in 200 seeds")
