"""SQuAD-style token F1. Normalisation keeps Devanagari intact (no ASCII-only regex)."""

from __future__ import annotations

import string
import unicodedata
from collections import Counter

_PUNCT = set(string.punctuation) | {"।", "॥", "“", "”", "‘", "’"}


def tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s).lower()
    return "".join(" " if ch in _PUNCT else ch for ch in s).split()


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)
