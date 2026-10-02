"""Idle baseline: total GPU draw with no model loaded, sampled every 0.5 s. Subtracted from every call."""

from __future__ import annotations

import statistics
import time

from bench.measure.energy import Meter


def measure_idle(seconds: float = 60.0, every: float = 0.5) -> tuple[float, float]:
    """Returns (mean W, std W) summed over all GPUs. Caller must unload every model first."""
    m = Meter(idle_w=0.0)
    if not m.available:
        raise SystemExit("idle: NVML unavailable — is the notebook accelerator set to GPU T4 x2?")
    samples = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        samples.append(m.power_w())
        time.sleep(every)
    return statistics.fmean(samples), statistics.pstdev(samples)
