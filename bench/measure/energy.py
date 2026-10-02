"""GPU energy per call via NVML, summed over every GPU (Kaggle 2×T4 splits big models across both).

Primary: the hardware energy counter (nvmlDeviceGetTotalEnergyConsumption, Volta+; T4 is Turing).
Fallback: power sampled every 20 ms and integrated. Either way the idle baseline × duration is
subtracted. The result is not clamped at 0: short calls can read slightly below idle, and clamping
would bias every mean upward.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import TypeVar

from bench.core.schemas import EnergySource

T = TypeVar("T")
SAMPLE_S = 0.02


def idle_subtracted_wh(gross_j: float, seconds: float, idle_w: float) -> float:
    return (gross_j - idle_w * seconds) / 3600


class Meter:
    def __init__(self, idle_w: float) -> None:
        self.idle_w = idle_w
        self.handles: list[object] = []
        self.counter = False
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nv = pynvml
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self._read_mj()
            self.counter = True
        except Exception:  # noqa: BLE001 — no NVML or no counter: fall back below
            pass

    @property
    def available(self) -> bool:
        return bool(self.handles)

    def _read_mj(self) -> float:
        return float(sum(self.nv.nvmlDeviceGetTotalEnergyConsumption(h) for h in self.handles))

    def power_w(self) -> float:
        return sum(self.nv.nvmlDeviceGetPowerUsage(h) for h in self.handles) / 1000

    def measure(self, fn: Callable[[], T]) -> tuple[T, float | None, EnergySource]:
        """Run fn; return (its result, idle-subtracted Wh or None, source)."""
        if not self.available:
            return fn(), None, "unavailable"
        if self.counter:
            e0, t0 = self._read_mj(), time.perf_counter()
            out = fn()
            gross_j, dt = (self._read_mj() - e0) / 1000, time.perf_counter() - t0
            return out, idle_subtracted_wh(gross_j, dt, self.idle_w), "nvml"

        joules, stop = [0.0], threading.Event()

        def sample() -> None:
            last_t, last_p = time.perf_counter(), self.power_w()
            while not stop.is_set():
                time.sleep(SAMPLE_S)
                t, p = time.perf_counter(), self.power_w()
                joules[0] += (p + last_p) / 2 * (t - last_t)
                last_t, last_p = t, p

        th = threading.Thread(target=sample, daemon=True)
        t0 = time.perf_counter()
        th.start()
        try:
            out = fn()
        finally:
            stop.set()
            th.join()
        return out, idle_subtracted_wh(joules[0], time.perf_counter() - t0, self.idle_w), "nvml"
