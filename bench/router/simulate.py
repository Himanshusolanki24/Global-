"""Router policy and its counterfactuals. One rule, used by the simulation, the replay and the tests:

    classifier says "big"                      → big only
    classifier says "small", conf ≥ τ          → small (its `samples` calls measured conf; one answer is used)
    classifier says "small", conf < τ          → small × samples, then escalate to big
"""

from __future__ import annotations

from collections.abc import Callable

from bench.core.schemas import (
    CLOUD_ENERGY_REASON, AggregateRow, Outcome, PolicyTotals, RouteDecision, RouterQuery, RouterSim, RunRecord,
)


def routes_small(q: RouterQuery, tau: float) -> bool:
    return q.pred_small and q.conf >= tau


def escalates(q: RouterQuery, tau: float) -> bool:
    return q.pred_small and q.conf < tau


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


Cells = dict[str, AggregateRow]  # task_id → that config's cell; each query is costed on its own task


def _totals(qs: list[RouterQuery], calls: Callable[[RouterQuery], list[tuple[AggregateRow, int]]],
            ok: Callable[[RouterQuery], bool], small: Callable[[RouterQuery], bool]) -> PolicyTotals:
    n = len(qs) or 1
    inr: float | None = 0.0
    wh: float | None = 0.0
    for q in qs:
        for cell, k in calls(q):
            inr = _add(inr, None if cell.inr_1k is None else cell.inr_1k * k / 1000)
            wh = _add(wh, None if cell.wh_q is None else cell.wh_q * k)
    return PolicyTotals(
        acc=round(100 * sum(map(ok, qs)) / n, 2), pct_small=round(100 * sum(map(small, qs)) / n, 2),
        inr_1k=None if inr is None else round(inr / n * 1000, 4), wh_1k=None if wh is None else round(wh / n * 1000, 4),
    )


def simulate(qs: list[RouterQuery], tau: float, small: Cells, big: Cells, samples: int, big_id: str) -> RouterSim:
    def router_calls(q: RouterQuery) -> list[tuple[AggregateRow, int]]:
        return ([(small[q.task], samples)] if q.pred_small else []) + ([] if routes_small(q, tau) else [(big[q.task], 1)])

    r = _totals(qs, router_calls, lambda q: q.small_ok if routes_small(q, tau) else q.ok[big_id], lambda q: routes_small(q, tau))
    s = _totals(qs, lambda q: [(small[q.task], 1)], lambda q: q.small_ok, lambda q: True)
    b = _totals(qs, lambda q: [(big[q.task], 1)], lambda q: q.ok[big_id], lambda q: False)
    wh_saved = None if b.wh_1k is None or r.wh_1k is None else round(b.wh_1k - r.wh_1k, 4)
    reason = next((c.energy_reason for c in [*big.values(), *small.values()] if c.energy_reason), None)
    return RouterSim(
        router=r, always_small=s, always_big=b,
        quality_retained_pct=round(100 * r.acc / b.acc, 2) if b.acc else 0.0,
        inr_saved_1k=None if b.inr_1k is None or r.inr_1k is None else round(b.inr_1k - r.inr_1k, 4),
        wh_saved_1k=wh_saved, wh_reason=None if wh_saved is not None else reason,
    )


QUALITY_FLOOR = 0.98  # τ must keep ≥ 98% of always-big accuracy on the validation split


def tune_tau(val: list[RouterQuery], small: Cells, big: Cells, samples: int, big_id: str) -> float:
    """Cheapest τ that holds the quality floor. Cheapest = Wh when measured, else ₹."""
    taus = [0.0, *sorted({q.conf for q in val}), 1.01]  # 0 never escalates, 1.01 always does
    sims = [(t, simulate(val, t, small, big, samples, big_id)) for t in taus]
    ok = [(t, x) for t, x in sims if x.router.acc >= QUALITY_FLOOR * x.always_big.acc]
    if not ok:
        return max(sims, key=lambda p: p[1].router.acc)[0]
    cost = (lambda x: x.router.wh_1k) if all(x.router.wh_1k is not None for _, x in ok) else (lambda x: x.router.inr_1k or 0.0)
    return min(ok, key=lambda p: (cost(p[1]), -p[0]))[0]


def pick_replay(qs: list[RouterQuery], tau: float, n: int = 10) -> list[RouterQuery]:
    """A deterministic mix for the demo: answered small, escalated, and sent straight to big."""
    small = [q for q in qs if routes_small(q, tau)]
    esc = [q for q in qs if escalates(q, tau)]
    big = [q for q in qs if not q.pred_small]
    k = n // 3
    picks = esc[:k] + big[:k]
    picks += small[: n - len(picks)]
    picks += [q for q in qs if q not in picks][: n - len(picks)]
    return sorted(picks, key=lambda q: q.id)


def _outcome(model: str, calls: list[RunRecord], correct: bool, reason: str | None) -> Outcome:
    e = [c.energy_wh for c in calls]
    cost = [c.cost_inr for c in calls]
    return Outcome(
        model=model,
        latency_ms=round(sum(c.total_ms or 0 for c in calls), 1),  # sequential: conf samples, then escalation
        energy_wh=None if None in e else round(sum(e), 5),  # type: ignore[arg-type]
        energy_reason=reason if None in e else None,
        cost_inr=None if None in cost else round(sum(cost), 5),  # type: ignore[arg-type]
        correct=correct,
    )


def decision(q: RouterQuery, tau: float, small_id: str, big_id: str,
             small_calls: list[RunRecord], big_call: RunRecord, energy_reason: str = CLOUD_ENERGY_REASON) -> RouteDecision:
    """Replay one query from its recorded calls. small_calls are the `samples` confidence calls."""
    s, esc = routes_small(q, tau), escalates(q, tau)
    calls = (small_calls if q.pred_small else []) + ([] if s else [big_call])
    big_ok = q.ok[big_id]
    return RouteDecision(
        big=big_id, tau=tau, query_id=q.id, task=q.task, lang=q.lang, text=q.text,
        predicted_tier="small" if q.pred_small else "big", confidence=q.conf, escalated=esc,
        routed=_outcome(small_id if s else big_id, calls, q.small_ok if s else big_ok, energy_reason),
        counterfactual=_outcome(big_id, [big_call], big_ok, energy_reason),
    )
