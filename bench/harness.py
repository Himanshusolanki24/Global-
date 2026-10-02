"""The benchmark loop. Resumable: every call is one committed row, and finished rows are skipped.

    python -m bench.harness --db /kaggle/working/rightsize.db --limit 5        # smoke run
    python -m bench.harness --db /kaggle/working/rightsize.db                  # everything
    python -m bench.harness --db /kaggle/working/rightsize.db --phase conf     # router confidence samples

Repeats: every item runs once at temperature 0 (that is all accuracy needs). The first
--energy-items items per task run --repeats times, for timing/energy spread.
Each local model is pulled just before it runs and deleted after (Kaggle's disk cannot hold all 15).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path
from typing import NamedTuple

from bench.core.schemas import (
    HostInfo, ModelConfig, Pricing, RunRecord, TaskItem, TaskSpec,
    load_configs, load_items, load_judge, load_pricing, load_router_spec, load_tasks,
)
from bench.core.store import Store
from bench.measure.cost import api_cost_inr, local_cost_inr
from bench.measure.energy import Meter
from bench.router.labeler import split
from bench.runners import ollama_runner
from bench.runners.api_runner import ApiRunner
from bench.runners.base import Runner
from bench.runners.ollama_runner import OllamaRunner
from bench.scoring.registry import score

IST = timezone(timedelta(hours=5, minutes=30))
RUN_ID = f"RUN-{datetime.now(IST):%Y%m%d-%H%M}-{uuid.uuid4().hex[:4]}"
API_KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
OFFLOAD_REASON = "model partly offloaded to CPU; the GPU-only meter would undercount"


@functools.cache
def judge() -> Runner:
    return ApiRunner(**load_judge().model_dump())


def log(event: str, **kw: object) -> None:
    print(json.dumps({"ts": datetime.now(IST).isoformat(timespec="seconds"), "run_id": RUN_ID, "event": event, **kw},
                     ensure_ascii=False), flush=True)


class Job(NamedTuple):
    config: ModelConfig
    task: TaskSpec
    item: TaskItem
    repeat: int
    kind: str
    temperature: float
    seed: int

    @property
    def key(self) -> tuple[str, str, str, str, int]:
        return (self.kind, self.config.id, self.task.id, self.item.id, self.repeat)


def plan(configs: list[ModelConfig], tasks: list[TaskSpec], items: dict[str, list[TaskItem]], a: argparse.Namespace) -> list[Job]:
    spec = load_router_spec()
    jobs = []
    for c in configs:
        for t in tasks:
            for idx, it in enumerate(items[t.id][: a.limit]):
                if a.phase == "bench":
                    reps = a.repeats if idx < a.energy_items else 1
                    jobs += [Job(c, t, it, r, "bench", 0.0, 0) for r in range(reps)]
                elif c.id == spec.small and split(it.id) != "train":
                    jobs += [Job(c, t, it, k, "conf", spec.conf_temperature, k + 1) for k in range(spec.samples)]
    return jobs


class Session:
    """Everything that lives for one config: runner, whether energy is trustworthy, cost function."""

    def __init__(self, c: ModelConfig, store: Store, meter: Meter, pricing: Pricing, keep: bool) -> None:
        self.c, self.store, self.keep = c, store, keep
        self.metered = False
        if c.backend == "ollama":
            t0 = time.perf_counter()
            ollama_runner.pull(c.tag)
            log("pulled", config=c.id, tag=c.tag, seconds=round(time.perf_counter() - t0), ollama_list=ollama_runner.local_tags())
            self.runner: Runner = OllamaRunner(c.tag)
            self.runner.warm_up()
            share, vram = ollama_runner.placement(c.tag)
            mem = store.get_meta("mem_gb", {})
            store.put_meta("mem_gb", {**mem, c.id: round(vram, 2)})
            reasons = store.get_meta("energy_reasons", {})
            if share < 0.999:
                log("warn_offload", config=c.id, gpu_share=round(share, 3), note=OFFLOAD_REASON)
                store.put_meta("energy_reasons", {**reasons, c.id: OFFLOAD_REASON})
            else:
                reasons.pop(c.id, None)
                store.put_meta("energy_reasons", reasons)
                self.metered = meter.available
            self.cost = lambda wh, m: local_cost_inr(wh, pricing.tariff_inr_per_kwh)  # noqa: E731
        else:
            self.runner = ApiRunner(c.backend, c.tag)
            self.runner.warm_up()
            price = pricing.api[c.tag]
            self.cost = lambda wh, m: api_cost_inr(m.input_tokens or 0, m.output_tokens, price, pricing.usd_inr)  # noqa: E731

    def close(self) -> None:
        if self.c.backend == "ollama":
            ollama_runner.unload_all()
            if not self.keep:
                ollama_runner.remove(self.c.tag)


def run_job(j: Job, s: Session, meter: Meter, fingerprint: str) -> RunRecord:
    base = dict(run_id=RUN_ID, kind=j.kind, temperature=j.temperature, config_id=j.config.id, task_id=j.task.id,
                item_id=j.item.id, repeat_index=j.repeat, host_fingerprint=fingerprint)
    try:
        call = lambda: s.runner.generate(j.item.prompt, j.task.max_tokens, j.temperature, j.seed)  # noqa: E731
        (text, m), wh, src = meter.measure(call) if s.metered else (call(), None, "unavailable")
        sc, judged = score(j.task, j.item, text, judge) if j.kind == "bench" else (None, None)
        return RunRecord(
            **base, output=text, score=sc, correct=None if sc is None else sc >= j.task.pass_score,
            ttft_ms=round(m.ttft_ms, 2), total_ms=round(m.total_ms, 2), input_tokens=m.input_tokens,
            output_tokens=m.output_tokens, tokens_per_sec=None if m.tokens_per_sec is None else round(m.tokens_per_sec, 2),
            energy_wh=None if wh is None else round(wh, 7), energy_source=src, cost_inr=s.cost(wh, m),
            timestamp=datetime.now(IST).isoformat(timespec="seconds"), judge_output=judged,
        )
    except Exception as e:  # noqa: BLE001 — stored as an error row and retried on the next run
        return RunRecord(**base, output=None, score=None, correct=None, ttft_ms=None, total_ms=None, input_tokens=None,
                         output_tokens=None, tokens_per_sec=None, energy_wh=None, energy_source="unavailable",
                         cost_inr=None, timestamp=datetime.now(IST).isoformat(timespec="seconds"),
                         error=f"{type(e).__name__}: {e}"[:500])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.harness", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--phase", choices=["bench", "conf"], default="bench")
    ap.add_argument("--limit", type=int, default=None, help="items per task (smoke run)")
    ap.add_argument("--configs", nargs="*", help="config ids to run (default: all in models.yaml)")
    ap.add_argument("--tasks", nargs="*", help="task ids to run (default: all)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--energy-items", type=int, default=50, help="items per task that get --repeats runs")
    ap.add_argument("--tasks-dir", type=Path, default=None)
    ap.add_argument("--deadline-hours", type=float, default=None, help="stop cleanly after this long (Kaggle caps at 12 h)")
    ap.add_argument("--skip-api", action="store_true", help="skip API configs instead of failing when a key is missing")
    ap.add_argument("--keep-models", action="store_true", help="do not delete each Ollama model after its run")
    a = ap.parse_args(argv)

    configs = [c for c in load_configs() if not a.configs or c.id in a.configs]
    unknown = set(a.configs or []) - {c.id for c in configs}
    if unknown:
        raise SystemExit(f"harness: unknown config ids {sorted(unknown)}")
    tasks = [t for t in load_tasks() if not a.tasks or t.id in a.tasks]
    items = {t.id: load_items(t, a.tasks_dir) if a.tasks_dir else load_items(t) for t in tasks}
    pricing, store = load_pricing(), Store(a.db)
    try:
        host = HostInfo(**store.get_meta("host"))
    except KeyError:
        raise SystemExit("harness: no host baseline in this DB — run: python -m bench.core.host --db ...") from None

    for c in [c for c in configs if c.backend != "ollama"]:
        if not os.environ.get(API_KEYS[c.backend]):
            if not a.skip_api:
                raise SystemExit(f"harness: {API_KEYS[c.backend]} not set for {c.id} (add it as a Kaggle Secret, or pass --skip-api)")
            log("skip_config", config=c.id, reason=f"{API_KEYS[c.backend]} not set")
            configs.remove(c)
        elif c.tag not in pricing.api:
            raise SystemExit(f"harness: no price for {c.tag} in pricing.yaml")
    if a.phase == "bench" and any(t.scorer == "llm_judge" for t in tasks):
        jb = load_judge().backend
        if not os.environ.get(API_KEYS[jb]):
            raise SystemExit(f"harness: the summarise judge needs {API_KEYS[jb]} (or pass --tasks without summarise)")
    if ollama_runner.version() is None:
        raise SystemExit("harness: Ollama is not answering on localhost:11434 — start `ollama serve` first")
    ollama_runner.verify_tags([c.tag for c in configs if c.backend == "ollama"])

    done = store.done()
    jobs = [j for j in plan(configs, tasks, items, a) if j.key not in done]
    log("plan", phase=a.phase, todo=len(jobs), already_done=len(done), configs=len({j.config.id for j in jobs}))
    meter = Meter(idle_w=host.idle_w)
    if not meter.available:
        log("warn_no_nvml", note="energy will be stored as unavailable")

    t0, n, errors = time.perf_counter(), 0, 0
    for _, grp in groupby(jobs, key=lambda j: j.config.id):
        group = list(grp)
        try:
            session = Session(group[0].config, store, meter, pricing, a.keep_models)
        except Exception as e:  # noqa: BLE001 — one broken model must not end a 10-hour run
            log("config_failed", config=group[0].config.id, error=f"{type(e).__name__}: {e}"[:500], skipped=len(group))
            continue
        try:
            for j in group:
                if a.deadline_hours and time.perf_counter() - t0 > a.deadline_hours * 3600:
                    log("deadline", done=n, remaining=len(jobs) - n, note="re-run the same command to resume")
                    store.close()
                    return
                rec = run_job(j, session, meter, host.fingerprint)
                store.write(rec)
                n += 1
                errors += rec.error is not None
                if rec.error:
                    log("call_error", key=list(j.key), error=rec.error)
                if n % 25 == 0 or n == len(jobs):
                    el = time.perf_counter() - t0
                    log("progress", done=n, total=len(jobs), pct=round(100 * n / len(jobs), 1), errors=errors,
                        config=j.config.id, per_call_s=round(el / n, 2), eta_min=round(el / n * (len(jobs) - n) / 60, 1))
        finally:
            session.close()
    log("finished", calls=n, errors=errors, minutes=round((time.perf_counter() - t0) / 60, 1),
        note="errored calls are retried on the next run" if errors else "")
    store.close()


if __name__ == "__main__":
    main()
