import random

import numpy as np
import pytest
from PIL import Image

from aigc_detect.transforms import (
    _RESOLUTION_RANGE,
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
    strip_source_artifacts,
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


@pytest.fixture
def square_image() -> Image.Image:
    """A larger, exactly-square source image - mirrors SID_Set's FAKE images
    (always 1024x1024), the specific shape strip_source_artifacts must not
    leave distinguishably square."""
    rng = np.random.default_rng(1)
    array = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_strip_source_artifacts_output_in_expected_ranges(square_image):
    result = strip_source_artifacts(square_image, rng=random.Random(0))
    assert result.mode == "RGB"
    assert _RESOLUTION_RANGE[0] <= min(result.size) <= _RESOLUTION_RANGE[1] + 1


def test_strip_source_artifacts_randomizes_aspect_ratio_of_square_input(square_image):
    """A native 1:1 image must not stay 1:1 forever - that's exactly the
    residual shortcut this function exists to remove."""
    saw_non_square = False
    for seed in range(20):
        result = strip_source_artifacts(square_image, rng=random.Random(seed))
        if result.size[0] != result.size[1]:
            saw_non_square = True
            break
    assert saw_non_square


def test_strip_source_artifacts_different_seeds_give_different_sizes(square_image):
    """Guards against accidentally hardcoding a fixed resolution/quality,
    which would just replace one fixed source fingerprint with another."""
    sizes = {strip_source_artifacts(square_image, rng=random.Random(seed)).size for seed in range(15)}
    assert len(sizes) > 1
