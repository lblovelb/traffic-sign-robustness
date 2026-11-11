"""Simulated weather and sensor corruptions for traffic-sign images.

Pure NumPy so it runs (and is tested) without TensorFlow.

All functions take float32 images in [0, 1] with shape (N, H, W, 3) and a
severity level 1-5, and return a new array of the same shape and dtype.
Randomness always comes from an explicit ``numpy.random.Generator`` so every
experiment is reproducible from its seed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

SEVERITIES = (1, 2, 3, 4, 5)


def _check(images: np.ndarray, severity: int) -> None:
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) images, got shape {images.shape}")
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * sigma**2))
    return k / k.sum()


def _separable_blur(images: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Blur along H and W with a 1-D kernel, reflect padding, batched."""
    r = len(kernel) // 2
    out = images
    for axis in (1, 2):
        pad = [(0, 0)] * 4
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="reflect")
        size = out.shape[axis]
        acc = np.zeros_like(out)
        for i, w in enumerate(kernel):
            acc += w * np.take(padded, np.arange(i, i + size), axis=axis)
        out = acc
    return out


def gaussian_noise(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Sensor noise, e.g. a camera in low light."""
    _check(images, severity)
    std = (0.04, 0.08, 0.12, 0.18, 0.26)[severity - 1]
    noisy = images + rng.normal(0.0, std, size=images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def blur(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Defocus or motion-like blur (Gaussian)."""
    _check(images, severity)
    sigma = (0.5, 0.9, 1.3, 1.8, 2.5)[severity - 1]
    return np.clip(_separable_blur(images, _gaussian_kernel(sigma)), 0.0, 1.0).astype(np.float32)


def darkness(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Dusk / night: lower brightness and crush shadows with a gamma curve."""
    _check(images, severity)
    scale = (0.7, 0.5, 0.35, 0.25, 0.15)[severity - 1]
    gamma = (1.2, 1.4, 1.6, 1.8, 2.0)[severity - 1]
    return np.clip(scale * images**gamma, 0.0, 1.0).astype(np.float32)


def fog(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Fog: blend towards a bright, uneven haze layer and lose contrast."""
    _check(images, severity)
    alpha = (0.25, 0.4, 0.55, 0.7, 0.82)[severity - 1]
    n, h, w, _ = images.shape
    # Low-frequency haze: a coarse random grid upsampled and smoothed.
    grid = rng.uniform(0.7, 0.95, size=(n, 4, 4, 1)).astype(np.float32)
    haze = np.repeat(np.repeat(grid, -(-h // 4), axis=1), -(-w // 4), axis=2)[:, :h, :w, :]
    haze = _separable_blur(np.repeat(haze, 3, axis=3), _gaussian_kernel(max(h, w) / 8))
    return np.clip((1 - alpha) * images + alpha * haze, 0.0, 1.0).astype(np.float32)


def rain(images: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Rain: bright slanted streaks, slight blur and a darker, flatter scene."""
    _check(images, severity)
    n_streaks = (8, 16, 28, 42, 60)[severity - 1]
    length = (3, 4, 5, 6, 7)[severity - 1]
    n, h, w, _ = images.shape
    streaks = np.zeros((n, h, w), dtype=np.float32)
    ys = rng.integers(0, h, size=(n, n_streaks))
    xs = rng.integers(0, w, size=(n, n_streaks))
    idx = np.broadcast_to(np.arange(n)[:, None], ys.shape)
    for step in range(length):
        yy, xx = ys + step, xs + step // 2  # slight slant
        inside = (yy < h) & (xx < w)  # streaks leaving the image are cut off
        streaks[idx[inside], yy[inside], xx[inside]] = 1.0
    dimmed = 0.8 * images + 0.05
    out = np.maximum(dimmed, 0.85 * streaks[..., None])
    return np.clip(_separable_blur(out, _gaussian_kernel(0.5)), 0.0, 1.0).astype(np.float32)


CORRUPTIONS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "darkness": darkness,
    "fog": fog,
    "rain": rain,
}


def apply(images: np.ndarray, name: str, severity: int, seed: int) -> np.ndarray:
    """Apply one named corruption deterministically."""
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; choose from {sorted(CORRUPTIONS)}")
    return CORRUPTIONS[name](images, severity, np.random.default_rng(seed))


def random_augment(
    images: np.ndarray,
    names: list[str],
    max_severity: int,
    prob: float,
    seed: int,
    chunk: int = 2048,
) -> np.ndarray:
    """Corrupt a random subset of a training set (the mitigation condition).

    Each image is corrupted with probability ``prob`` by a corruption drawn
    from ``names`` at a severity drawn from 1..``max_severity``.
    """
    rng = np.random.default_rng(seed)
    out = images.copy()
    chosen = rng.random(len(images)) < prob
    kinds = rng.integers(0, len(names), size=len(images))
    sevs = rng.integers(1, max_severity + 1, size=len(images))
    for k, name in enumerate(names):
        for sev in range(1, max_severity + 1):
            idx = np.flatnonzero(chosen & (kinds == k) & (sevs == sev))
            for start in range(0, len(idx), chunk):
                part = idx[start : start + chunk]
                out[part] = CORRUPTIONS[name](images[part], sev, rng)
    return out
