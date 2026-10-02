"""Labels and splits for the router. A config "answered" an item when it was right on most repeats."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Literal

from bench.analysis.aggregate import items
from bench.core.schemas import RunRecord

Split = Literal["train", "val", "test"]


def split(item_id: str) -> Split:
    """Stable 60/20/20 by hash, so the harness (conf samples for val+test) and the trainer always agree."""
    b = int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if b < 60 else "val" if b < 80 else "test"


def verdicts(rows: list[RunRecord]) -> dict[tuple[str, str], dict[str, bool]]:
    """(task, item) → {config_id: answered correctly}."""
    out: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for (config, task, item), v in items(rows).items():
        out[(task, item)][config] = v.acc >= 0.5
    return out


def smallest_ok(ok: dict[str, bool], order: list[str]) -> str | None:
    """order = models.yaml order (smallest → largest)."""
    return next((c for c in order if ok.get(c)), None)
