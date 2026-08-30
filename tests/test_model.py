import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from aigc_detect.model import (
    IMAGE_SIZE,
    AIGCClassifier,
    ClipRgbBranch,
    NoiseResidualBranch,
    preprocess,
)

MAX_PARAMS = 2_000_000_000  # PROBLEM.md hard constraint
MAX_TRAINABLE_PARAMS = 500_000  # sanity bound: catches an accidental "CLIP isn't actually frozen" bug


@pytest.fixture(scope="module")
def offline_model() -> AIGCClassifier:
    # Randomly-initialized (not .from_pretrained) CLIP vision model: same
    # architecture/size as the real one, but no network access needed.
    random_clip = CLIPVisionModelWithProjection(CLIPVisionConfig())
    return AIGCClassifier(clip_branch=ClipRgbBranch(vision_model=random_clip))


def test_forward_pass_shape(offline_model):
    batch = torch.rand(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits = offline_model(batch)
    assert logits.shape == (4,)
    assert logits.dtype == torch.float32


def test_parameter_count_within_problem_statement_limit(offline_model):
    assert 0 < offline_model.num_parameters() < MAX_PARAMS


def test_clip_branch_is_frozen(offline_model):
    n_trainable = offline_model.num_trainable_parameters()
    assert 0 < n_trainable < MAX_TRAINABLE_PARAMS
    assert n_trainable < offline_model.num_parameters()


def test_noise_residual_branch_output_shape():
    branch = NoiseResidualBranch(out_dim=64)
    batch = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    features = branch(batch)
    assert features.shape == (2, 64)


def test_preprocess_output_shape_and_range():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")

    tensor = preprocess(image)

    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_pretrained_clip_branch_loads_real_weights():
    """Separate, slower test: actually downloads pretrained CLIP weights."""
    try:
        model = AIGCClassifier()
    except OSError as exc:
        pytest.skip(f"Could not download pretrained CLIP weights (network/cert issue?): {exc}")

    batch = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    logits = model(batch)
    assert logits.shape == (1,)
