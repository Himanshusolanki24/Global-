"""Exact match with three gold shapes, chosen from the gold itself:

- number  ("18")      → the last number in the output (models show working before "Answer: 18")
- letter  ("B")       → the first standalone A–D in the output (MCQ; "उत्तर: B" works)
- label   ("Sports")  → normalised output equals the label, or starts with it ("Sports." / "sports news")
"""

from __future__ import annotations

import re

from bench.scoring.token_f1 import tokens

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def exact_match(output: str, gold: str) -> float:
    if (g := _num(gold)) is not None:
        found = _NUM.findall(output)
        return float(bool(found) and _num(found[-1]) == g)
    if re.fullmatch(r"[A-D]", gold):
        m = _LETTER.search(output)
        return float(bool(m) and m.group(1) == gold)
    out, ref = tokens(output), tokens(gold)
    return float(bool(ref) and out[: len(ref)] == ref)
