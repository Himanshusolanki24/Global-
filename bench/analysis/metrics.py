"""Cost-of-Quality pairs and the quantization comparison. Both use a paired bootstrap over shared items,
so "Q4 is worse than FP16" is only claimed when the per-item difference says so."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from bench.analysis.aggregate import Item, ItemKey, _mean, rd
from bench.analysis.stats import bootstrap_ci, paired_diff_ci, significant
from bench.core.schemas import AggregateRow, CoQPair, ModelConfig, QuantRow


def _index(by_item: dict[ItemKey, Item]) -> dict[tuple[str, str], set[str]]:
    idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c, t, i in by_item:
        idx[(c, t)].add(i)
    return idx


def coq_pairs(cells: list[AggregateRow], by_item: dict[ItemKey, Item]) -> list[CoQPair]:
    idx = _index(by_item)
    groups: dict[tuple[str, str], list[AggregateRow]] = defaultdict(list)
    for c in cells:
        groups[(c.task, c.lang)].append(c)
    out: list[CoQPair] = []
    for (task, lang), cs in groups.items():
        for a, b in combinations(cs, 2):
            lo, hi = (a, b) if a.acc <= b.acc else (b, a)
            shared = sorted(idx[(lo.config, task)] & idx[(hi.config, task)])
            ci = paired_diff_ci([by_item[(lo.config, task, i)].acc for i in shared],
                                [by_item[(hi.config, task, i)].acc for i in shared])
            d_acc = hi.acc - lo.acc
            d_inr = None if lo.inr_1k is None or hi.inr_1k is None else hi.inr_1k - lo.inr_1k
            d_wh = None if lo.wh_q is None or hi.wh_q is None else (hi.wh_q - lo.wh_q) * 1000
            out.append(CoQPair(
                task=task, lang=lang, lo=lo.config, hi=hi.config,
                d_acc=rd(d_acc, 2), d_acc_ci=rd((100 * ci[0], 100 * ci[1]), 2), significant=significant(ci),
                d_inr_1k=rd(d_inr, 3), d_wh_1k=rd(d_wh, 3),
                inr_per_pt=rd(d_inr / d_acc, 4) if d_inr is not None and d_acc > 0 else None,
                wh_per_pt=rd(d_wh / d_acc, 4) if d_wh is not None and d_acc > 0 else None,
            ))
    return out


def quant_rows(configs: list[ModelConfig], cells: list[AggregateRow], by_item: dict[ItemKey, Item]) -> list[QuantRow]:
    """Per small model × language: Q4/Q8/FP16 pooled over every task in that language, ΔAcc vs FP16."""
    lang_of = {c.task: c.lang for c in cells}
    models = list(dict.fromkeys(c.model for c in configs if c.kind == "small"))
    out: list[QuantRow] = []
    for model in models:
        for lang in sorted(set(lang_of.values())):
            pooled = {q: {(t, i): v for (c, t, i), v in by_item.items() if c == f"{model}/{q}" and lang_of.get(t) == lang}
                      for q in ("Q4", "Q8", "FP16")}
            base = pooled["FP16"]
            for q, its in pooled.items():
                if not its:
                    continue
                acc = [v.acc for v in its.values()]
                wh = [v.wh for v in its.values()]
                inr = [v.inr for v in its.values()]
                shared = sorted(its.keys() & base.keys())
                ci = None if q == "FP16" or not shared else paired_diff_ci([base[k].acc for k in shared], [its[k].acc for k in shared])
                out.append(QuantRow(
                    model=model, lang=lang, quant=q, n_items=len(its),
                    acc=rd(100 * float(np.mean(acc)), 2), acc_ci=rd(tuple(100 * v for v in bootstrap_ci(acc)), 2),
                    d_acc_vs_fp16=None if ci is None else rd(100 * (np.mean([its[k].acc for k in shared]) - np.mean([base[k].acc for k in shared])), 2),
                    d_acc_ci=None if ci is None else rd((100 * ci[0], 100 * ci[1]), 2),
                    significant=None if ci is None else significant(ci),
                    # only when every pooled item was metered; a partial mean would silently reweight tasks
                    wh_q=rd(_mean(wh), 5) if None not in wh else None,  # type: ignore[arg-type]
                    inr_1k=rd(_mean(inr) * 1000, 3) if None not in inr else None,  # type: ignore[arg-type,operator]
                ))
    return out
