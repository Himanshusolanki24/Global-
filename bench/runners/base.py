"""One interface for local and cloud models. Every runner streams, so TTFT is the first token's arrival."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bench.core.schemas import Metrics


class Runner(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> tuple[str, Metrics]: ...

    def warm_up(self) -> None:
        """Load weights / open connections. The call is made and its result discarded."""
        self.generate("Hi", max_tokens=1, temperature=0.0, seed=0)
