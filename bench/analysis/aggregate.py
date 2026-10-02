"""Rows → per config × task cells. Accuracy, energy and cost are averaged per item first (repeats of
one item are not independent), then bootstrapped over items. Timing stats are over calls."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from bench.analysis.stats import bootstrap_ci
from bench.core.schemas import (
    CLOUD_ENERGY_REASON, NO_NVML_REASON, AggregateRow, ModelConfig, Pricing, RunRecord, TaskSpec,
)

ItemKey = tuple[str, str, str]  # config_id, task_id, item_id


@dataclass(frozen=True)
class Item:
    acc: float  # fraction of repeats correct
    wh: float | None  # mean over repeats with an NVML reading; never filled in otherwise
    inr: float | None


def rd(x, n: int = 4):  # noqa: ANN001, ANN201 — rounds floats, tuples and None alike
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(round(v, n) for v in x)
    return round(float(x), n)


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _scored(rows: list[RunRecord]) -> list[RunRecord]:
    return [r for r in rows if r.kind == "bench" and r.error is None and r.correct is not None]


def items(rows: list[RunRecord]) -> dict[ItemKey, Item]:
    g: dict[ItemKey, list[RunRecord]] = defaultdict(list)
    for r in _scored(rows):
        g[(r.config_id, r.task_id, r.item_id)].append(r)
    return {
        k: Item(
            acc=sum(bool(r.correct) for r in rs) / len(rs),
            wh=_mean([r.energy_wh for r in rs if r.energy_source == "nvml" and r.energy_wh is not None]),
            inr=_mean([r.cost_inr for r in rs if r.cost_inr is not None]),
        )
        for k, rs in g.items()
    }


def aggregate(rows: list[RunRecord], configs: list[ModelConfig], tasks: list[TaskSpec], pricing: Pricing,
              energy_reasons: dict[str, str] | None = None) -> list[AggregateRow]:
    """energy_reasons: config_id → why it has no energy (e.g. partly offloaded to CPU), recorded by the harness."""
    reasons = energy_reasons or {}
    by_item = items(rows)
    cell_items: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for (c, t, _), v in sorted(by_item.items()):
        cell_items[(c, t)].append(v)
    calls: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    errors: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        if r.kind != "bench":
            continue
        if r.error is not None:
            errors[(r.config_id, r.task_id)] += 1
        elif r.correct is not None:
            calls[(r.config_id, r.task_id)].append(r)

    out: list[AggregateRow] = []
    for c in configs:
        for t in tasks:
            its, cs = cell_items.get((c.id, t.id)), calls[(c.id, t.id)]
            if not its:
                continue
            acc = [x.acc for x in its]
            wh = [x.wh for x in its if x.wh is not None]
            inr = [x.inr for x in its if x.inr is not None]
            ttft = [r.ttft_ms for r in cs if r.ttft_ms is not None]
            total = [r.total_ms for r in cs if r.total_ms is not None]
            tps = [r.tokens_per_sec for r in cs if r.tokens_per_sec is not None]
            metered = [r for r in cs if r.energy_source == "nvml" and r.energy_wh is not None]
            out_tok = sum(r.output_tokens or 0 for r in metered)

            wh_q = _mean(wh)
            inr_q = _mean(inr)
            acc_pct = 100 * float(np.mean(acc))
            out.append(AggregateRow(
                config=c.id, task=t.id, lang=t.lang,
                n_items=len(its), n_calls=len(cs), n_errors=errors[(c.id, t.id)],
                acc=rd(acc_pct, 2), acc_ci=rd(tuple(100 * v for v in bootstrap_ci(acc)), 2),
                ttft_ms_mean=rd(_mean(ttft), 1),
                ttft_ms_median=rd(np.median(ttft), 1) if ttft else None,
                ttft_p95_ms=rd(np.percentile(ttft, 95), 1) if ttft else None,
                total_ms_mean=rd(np.mean(total), 1),
                total_ms_median=rd(np.median(total), 1),
                total_ms_p95=rd(np.percentile(total, 95), 1),
                tok_s_mean=rd(_mean(tps), 2),
                wh_q=rd(wh_q, 5),
                wh_ci=rd(bootstrap_ci(wh), 5) if wh else None,
                mwh_tok=rd(sum(r.energy_wh for r in metered) / out_tok * 1000, 4) if out_tok else None,  # type: ignore[misc]
                energy_n=len(metered),
                energy_reason=None if wh else reasons.get(c.id, CLOUD_ENERGY_REASON if c.estimated else NO_NVML_REASON),
                inr_1k=rd(inr_q * 1000 if inr_q is not None else None, 3),
                inr_ci=rd(tuple(1000 * v for v in bootstrap_ci(inr)), 3) if inr else None,
                co2_g_1k=rd(wh_q * 1000 * pricing.grid_g_per_wh if wh_q is not None else None, 2),
                ei=rd(acc_pct / (wh_q * 1000) if wh_q else None, 3),
            ))
    return out
