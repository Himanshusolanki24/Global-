"""Local models through Ollama's streaming HTTP API at localhost:11434."""

from __future__ import annotations

import json
import time

import requests

from bench.core.schemas import Metrics
from bench.runners.base import Runner

URL = "http://localhost:11434"
REGISTRY = "https://registry.ollama.ai/v2/library"


class OllamaRunner(Runner):
    def __init__(self, tag: str) -> None:
        self.tag = tag

    def generate(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> tuple[str, Metrics]:
        body = {
            "model": self.tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": seed},
        }
        parts: list[str] = []
        ttft: float | None = None
        final: dict = {}
        t0 = time.perf_counter()
        with requests.post(f"{URL}/api/chat", json=body, stream=True, timeout=600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"ollama: {chunk['error']}")
                piece = chunk.get("message", {}).get("content", "")
                if piece and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                parts.append(piece)
                if chunk.get("done"):
                    final = chunk
        total = (time.perf_counter() - t0) * 1000
        out_tok = int(final.get("eval_count", 0))
        decode_s = final.get("eval_duration", 0) / 1e9
        return "".join(parts), Metrics(
            ttft_ms=ttft if ttft is not None else total,  # empty reply: first (only) chunk is the end
            total_ms=total,
            input_tokens=final.get("prompt_eval_count"),
            output_tokens=out_tok,
            tokens_per_sec=out_tok / decode_s if decode_s else None,
        )


def _get(path: str) -> dict:
    r = requests.get(f"{URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def version() -> str | None:
    try:
        return _get("/api/version")["version"]
    except requests.RequestException:
        return None


def verify_tags(tags: list[str]) -> None:
    """Fail fast, before any download, if a tag does not exist on the Ollama registry."""
    missing = []
    for tag in tags:
        name, _, variant = tag.partition(":")
        r = requests.head(f"{REGISTRY}/{name}/manifests/{variant or 'latest'}", timeout=30,
                          headers={"Accept": "application/vnd.docker.distribution.manifest.v2+json"})
        if r.status_code != 200:
            missing.append(f"{tag} (HTTP {r.status_code})")
    if missing:
        raise SystemExit("ollama tags not found on the registry — fix bench/config/models.yaml:\n  " + "\n  ".join(missing))


def pull(tag: str) -> None:
    with requests.post(f"{URL}/api/pull", json={"model": tag, "stream": True}, stream=True, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line and "error" in (msg := json.loads(line)):
                raise RuntimeError(f"ollama pull {tag}: {msg['error']}")


def remove(tag: str) -> None:
    requests.delete(f"{URL}/api/delete", json={"model": tag}, timeout=60)


def unload_all() -> None:
    for m in _get("/api/ps").get("models", []):
        requests.post(f"{URL}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)


def placement(tag: str) -> tuple[float, float]:
    """(share of weights in VRAM, VRAM GB) for a loaded model. share < 1 ⇒ part runs on CPU."""
    for m in _get("/api/ps").get("models", []):
        if m["name"] == tag or m.get("model") == tag:
            return (m["size_vram"] / m["size"] if m["size"] else 0.0), m["size_vram"] / 1e9
    raise RuntimeError(f"{tag} is not loaded")


def local_tags() -> list[str]:
    return [m["name"] for m in _get("/api/tags").get("models", [])]
