"""Host fingerprint + idle baseline, stored in the DB so every later step and the export can read it.

    python -m bench.core.host --db /kaggle/working/rightsize.db [--idle-seconds 60]
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import time
from pathlib import Path

from bench.core.schemas import HostInfo
from bench.core.store import Store
from bench.measure.idle import measure_idle
from bench.runners import ollama_runner


def _proc(path: str, key: str) -> str:
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def collect(idle_seconds: float) -> HostInfo:
    import pynvml

    pynvml.nvmlInit()
    hs = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    names = [pynvml.nvmlDeviceGetName(h) for h in hs]
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    driver = pynvml.nvmlSystemGetDriverVersion()
    driver = driver.decode() if isinstance(driver, bytes) else driver
    gpu = f"{len(names)}× {names[0]}" if len(names) > 1 else names[0]
    mem = sum(pynvml.nvmlDeviceGetMemoryInfo(h).total for h in hs) / 1e9
    cpu = _proc("/proc/cpuinfo", "model name")
    ram = int(_proc("/proc/meminfo", "MemTotal").split()[0]) / 1e6 if _proc("/proc/meminfo", "MemTotal") != "unknown" else 0.0

    ollama_runner.unload_all()
    time.sleep(5)  # let clocks settle after unloading
    idle_w, idle_std = measure_idle(idle_seconds)
    ident = f"{gpu}|{driver}|{cpu}|{ram:.0f}"
    return HostInfo(
        fingerprint=hashlib.sha256(ident.encode()).hexdigest()[:12], gpu=gpu, gpu_mem_gb=round(mem, 1),
        driver=driver, cpu=cpu, ram_gb=round(ram, 1), os=platform.platform(),
        ollama_version=ollama_runner.version(), idle_w=round(idle_w, 3), idle_w_std=round(idle_std, 3), idle_s=idle_seconds,
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.core.host")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--idle-seconds", type=float, default=60.0)
    a = ap.parse_args(argv)
    store = Store(a.db)
    try:
        old = HostInfo(**store.get_meta("host"))
    except KeyError:
        old = None
    host = collect(a.idle_seconds)
    # Resumed Kaggle sessions can land on a different CPU; only the GPU model must match.
    # ponytail: one GPU type per DB; mixing hardware would blend two power profiles into one mean
    if old and old.gpu != host.gpu:
        raise SystemExit(f"host: this DB was measured on {old.gpu}, this session has {host.gpu}. "
                         "Set the accelerator to the same GPU, or use a fresh --db.")
    store.put_meta("host", host.model_dump())
    print(f"host: {host.gpu} · {host.cpu} · idle {host.idle_w:.1f} ± {host.idle_w_std:.1f} W over {host.idle_s:.0f} s")


if __name__ == "__main__":
    main()
