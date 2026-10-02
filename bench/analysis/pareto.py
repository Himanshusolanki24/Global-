"""Pareto frontier per task: lowest x (Wh or ₹ per 1k) for the highest accuracy.
Configs with no measured x are left out, never placed with a guessed value."""

from __future__ import annotations

from collections import defaultdict

from bench.core.schemas import AggregateRow, ParetoPoint


def frontier(pts: list[tuple[str, float, float]]) -> set[str]:
    """pts = (id, x, y). Same rule as frontend/src/metrics.ts pareto()."""
    best, out = float("-inf"), set()
    for pid, _, y in sorted(pts, key=lambda p: (p[1], -p[2])):
        if y > best:
            out.add(pid)
            best = y
    return out


def pareto_points(cells: list[AggregateRow]) -> list[ParetoPoint]:
    groups: dict[tuple[str, str], list[AggregateRow]] = defaultdict(list)
    for c in cells:
        groups[(c.task, c.lang)].append(c)
    out: list[ParetoPoint] = []
    for (task, lang), cs in groups.items():
        for axis, x_of in (("energy", lambda c: None if c.wh_q is None else c.wh_q * 1000),
                           ("cost", lambda c: c.inr_1k)):
            pts = [(c.config, x, c.acc) for c in cs if (x := x_of(c)) is not None]
            on = frontier(pts)
            out += [ParetoPoint(task=task, lang=lang, axis=axis, config=i, x=round(x, 4), y=y, on_frontier=i in on)
                    for i, x, y in pts]
    return out
