"""Evaluation metrics.

Accuracy alone hides which classes break, so macro-F1 (every class counts the
same) is reported next to it. Latency matters because a model in a vehicle has
a time budget per frame.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy (share correct) and macro-F1 (F1 per class, averaged with equal weight)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def latency_ms_per_image(
    predict: Callable[[np.ndarray], np.ndarray], x: np.ndarray, n: int = 512, repeats: int = 3
) -> float:
    """Median wall-clock time per image over a few repeats (after one warm-up)."""
    batch = x[:n]
    # The first call is slower (memory allocation, graph tracing), so it is not timed.
    predict(batch)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(batch)
        times.append(time.perf_counter() - start)
    # Median is robust to one slow repeat (e.g. the OS doing something else).
    # Seconds per batch -> milliseconds per image.
    return float(np.median(times) / len(batch) * 1000)


def relative_drop(clean: float, corrupted: float) -> float:
    """Share of clean performance lost under corruption (0 = no loss, 1 = all lost)."""
    return 0.0 if clean == 0 else float((clean - corrupted) / clean)
