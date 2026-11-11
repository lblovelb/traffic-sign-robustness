import numpy as np
import pytest

from tsr import corruptions
from tsr.corruptions import CORRUPTIONS, SEVERITIES


@pytest.fixture
def images() -> np.ndarray:
    """Smooth random images (blur needs structure to have an effect)."""
    rng = np.random.default_rng(0)
    coarse = rng.uniform(0.1, 0.9, size=(8, 8, 8, 3)).astype(np.float32)
    return np.repeat(np.repeat(coarse, 4, axis=1), 4, axis=2)  # (8, 32, 32, 3)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
@pytest.mark.parametrize("severity", SEVERITIES)
def test_shape_dtype_and_range(images, name, severity):
    out = corruptions.apply(images, name, severity, seed=1)
    assert out.shape == images.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_deterministic_given_seed(images, name):
    a = corruptions.apply(images, name, 3, seed=42)
    b = corruptions.apply(images, name, 3, seed=42)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_does_not_modify_input(images, name):
    before = images.copy()
    corruptions.apply(images, name, 5, seed=0)
    np.testing.assert_array_equal(images, before)


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_severity_increases_distortion(images, name):
    dist = [np.abs(corruptions.apply(images, name, s, seed=7) - images).mean() for s in SEVERITIES]
    assert dist[0] > 0
    assert dist[-1] > dist[0]
    assert all(b >= a * 0.95 for a, b in zip(dist[:-1], dist[1:], strict=True))  # ~monotonic


def test_rejects_bad_input(images):
    with pytest.raises(ValueError):
        corruptions.apply(images, "fog", 0, seed=0)
    with pytest.raises(ValueError):
        corruptions.apply(images[0], "fog", 1, seed=0)
    with pytest.raises(KeyError):
        corruptions.apply(images, "snow", 1, seed=0)


def test_random_augment_probability(images):
    names = ["fog", "darkness"]
    unchanged = corruptions.random_augment(images, names, 3, prob=0.0, seed=0)
    np.testing.assert_array_equal(unchanged, images)

    changed = corruptions.random_augment(images, names, 3, prob=1.0, seed=0)
    per_image_diff = np.abs(changed - images).reshape(len(images), -1).mean(axis=1)
    assert (per_image_diff > 0).all()
