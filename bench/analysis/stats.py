"""95% percentile bootstrap. Seeded, so the same data always yields the same CI (results.json is committed)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

B = 2000
SEED = 412


def bootstrap_ci(x: Sequence[float], b: int = B, seed: int = SEED) -> tuple[float, float]:
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    idx = np.random.default_rng(seed).integers(0, a.size, size=(b, a.size))
    lo, hi = np.percentile(a[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def paired_diff_ci(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """CI of mean(b − a) over paired items. Significant ⇔ the interval excludes 0."""
    return bootstrap_ci(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))


def significant(ci: tuple[float, float]) -> bool:
    return ci[0] > 0 or ci[1] < 0
