"""task.scorer → score in [0, 1] (+ the judge's raw reply when there is one). Multiple golds: best match."""

from __future__ import annotations

from collections.abc import Callable

from bench.core.schemas import TaskItem, TaskSpec
from bench.runners.base import Runner
from bench.scoring.exact_match import exact_match
from bench.scoring.llm_judge import judge
from bench.scoring.token_f1 import token_f1


def score(task: TaskSpec, item: TaskItem, output: str, judge_runner: Callable[[], Runner]) -> tuple[float, str | None]:
    golds = item.gold if isinstance(item.gold, list) else [item.gold]
    if task.scorer == "exact_match":
        return max(exact_match(output, g) for g in golds), None
    if task.scorer == "token_f1":
        return max(token_f1(output, g) for g in golds), None
    return judge(judge_runner(), output, golds[0])
