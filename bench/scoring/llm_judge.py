"""Summary judge: temperature 0, fixed rubric, reference-based, 1–5 → 0–1. The raw reply is stored.
The judge (tasks.yaml `judge`) is not one of the benchmarked configs, so it never grades itself."""

from __future__ import annotations

import re

from bench.runners.base import Runner

RUBRIC = """You are grading a one-sentence summary of a news article against a reference summary.

Reference summary:
{gold}

Candidate summary:
{output}

Score the candidate from 1 to 5:
5 = same key facts as the reference, nothing invented
4 = main point correct, a minor detail missing or vague
3 = partly correct, misses the main point or adds an unsupported claim
2 = mostly wrong or mostly unsupported
1 = unrelated, empty, or not a summary

Reply with the single digit only."""


class JudgeError(RuntimeError):
    """Unparseable judge reply. The harness stores it as an error row, so the call is retried."""


def judge(runner: Runner, output: str, gold: str) -> tuple[float, str]:
    reply, _ = runner.generate(RUBRIC.format(gold=gold, output=output.strip() or "(empty)"),
                               max_tokens=4, temperature=0.0, seed=0)
    m = re.search(r"[1-5]", reply)
    if not m:
        raise JudgeError(f"judge reply not a 1-5 score: {reply!r}")
    return (int(m.group()) - 1) / 4, reply
