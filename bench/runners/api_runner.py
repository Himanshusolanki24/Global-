"""Frontier APIs. Streams for TTFT; token counts come from the provider's usage object, never estimated.
Retry with exponential backoff is the SDKs' own (429, 5xx, connection errors); a failure that survives
it is stored as an error row and retried on the next harness run."""

from __future__ import annotations

import os
import time

from bench.core.schemas import Backend, Metrics
from bench.runners.base import Runner

RETRIES = 6  # the SDK backs off on 429, which the Mistral free tier (≈1 request/s) will send
MISTRAL_URL = "https://api.mistral.ai/v1"


class ApiRunner(Runner):
    def __init__(self, backend: Backend, tag: str) -> None:
        self.backend, self.tag = backend, tag
        if backend in ("openai", "mistral"):
            from openai import OpenAI

            # Mistral's chat API is OpenAI-compatible; only the URL, key and two parameter names differ.
            self.client = (OpenAI(max_retries=RETRIES, timeout=120) if backend == "openai" else
                           OpenAI(base_url=MISTRAL_URL, api_key=os.environ.get("MISTRAL_API_KEY"), max_retries=RETRIES, timeout=120))
        elif backend == "anthropic":
            from anthropic import Anthropic

            self.client = Anthropic(max_retries=RETRIES, timeout=120)
        else:
            raise ValueError(f"ApiRunner cannot serve backend {backend!r}")

    def generate(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> tuple[str, Metrics]:
        parts: list[str] = []
        ttft: float | None = None
        t0 = time.perf_counter()

        def mark(piece: str | None) -> None:
            nonlocal ttft
            if piece:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(piece)

        if self.backend in ("openai", "mistral"):
            usage = None
            extra = ({"seed": seed, "stream_options": {"include_usage": True}} if self.backend == "openai"
                     else {"extra_body": {"random_seed": seed}})  # Mistral sends usage on its last chunk by default
            for chunk in self.client.chat.completions.create(
                model=self.tag, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
                temperature=temperature, stream=True, **extra,
            ):
                if chunk.choices:
                    mark(chunk.choices[0].delta.content)
                if chunk.usage:
                    usage = chunk.usage
            if usage is None:
                raise RuntimeError(f"{self.backend}: stream ended without a usage object")
            in_tok, out_tok = usage.prompt_tokens, usage.completion_tokens
        else:
            with self.client.messages.stream(
                model=self.tag, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    mark(text)
                final = stream.get_final_message()
            in_tok, out_tok = final.usage.input_tokens, final.usage.output_tokens

        total = (time.perf_counter() - t0) * 1000
        first = ttft if ttft is not None else total
        decode_s = (total - first) / 1000
        return "".join(parts), Metrics(ttft_ms=first, total_ms=total, input_tokens=in_tok, output_tokens=out_tok,
                                       tokens_per_sec=out_tok / decode_s if decode_s > 0 else None)
