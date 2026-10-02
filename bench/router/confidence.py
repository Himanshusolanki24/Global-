"""Self-consistency confidence: the share of samples that agree with the most-agreed-with answer.
Two answers agree when their token F1 ≥ AGREE, so "₹1,800" and "Rs 1800." can match and free-text
summaries are not scored as total disagreement just because the wording differs."""

from __future__ import annotations

from bench.scoring.token_f1 import token_f1

AGREE = 0.8


def agreement(outputs: list[str]) -> float:
    """3 samples → 1/3, 2/3 or 1."""
    if not outputs:
        raise ValueError("agreement needs at least one sample")
    return max(sum(token_f1(a, b) >= AGREE for b in outputs) for a in outputs) / len(outputs)
