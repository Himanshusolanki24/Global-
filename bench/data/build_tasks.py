"""Build the task bank: a seeded sample of public test sets, written as {id, prompt, gold, display} JSONL.
Prompts are fixed here, so every model sees byte-identical input.

    python -m bench.data.build_tasks [--n 200] [--out bench/data/tasks]

Hindi Q&A uses MILU (ai4bharat/MILU), which is gated: accept its terms on huggingface.co, then set
HF_TOKEN. Parquet is fetched through the HF datasets-server, so no `datasets` package is needed.
"""

from __future__ import annotations

import argparse
import io
import os
import random
from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd
import requests

from bench.core.schemas import TASKS_DIR, TaskItem

SEED = 412
DOC_CHARS = 2500  # keeps summarise prompts inside every model's context and the API bill bounded


def _hub_urls(dataset: str, config: str, split: str, headers: dict) -> list[str]:
    """Parquet files straight from the dataset repo (<config>/<split>-*.parquet), for datasets the
    viewer does not serve, such as gated MILU."""
    r = requests.get(f"https://huggingface.co/api/datasets/{dataset}/tree/main/{config}", headers=headers, timeout=60)
    r.raise_for_status()
    return [f"https://huggingface.co/datasets/{dataset}/resolve/main/{f['path']}" for f in r.json()
            if f["path"].rsplit("/", 1)[-1].startswith(f"{split}-") and f["path"].endswith(".parquet")]


def parquet(dataset: str, config: str, split: str) -> pd.DataFrame:
    tok = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    r = requests.get("https://datasets-server.huggingface.co/parquet", params={"dataset": dataset, "config": config},
                     headers=headers, timeout=60)
    if r.ok:
        urls = [f["url"] for f in r.json()["parquet_files"] if f["split"] == split]
    else:
        urls = _hub_urls(dataset, config, split, headers)
    if not urls:
        raise SystemExit(f"{dataset}/{config} has no '{split}' parquet files")
    frames = []
    for u in urls:
        b = requests.get(u, headers=headers, timeout=300)
        if b.status_code in (401, 403):
            raise SystemExit(f"{dataset} is gated: open huggingface.co/datasets/{dataset}, click 'Agree and access', "
                             f"and make sure the HF_TOKEN secret is attached (HTTP {b.status_code})")
        b.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(b.content)))
    return pd.concat(frames, ignore_index=True)


def label_names(dataset: str, config: str, split: str, column: str) -> list[str]:
    """ClassLabel names, which /first-rows keeps (and /info sometimes drops)."""
    r = requests.get("https://datasets-server.huggingface.co/first-rows",
                     params={"dataset": dataset, "config": config, "split": split}, timeout=60)
    r.raise_for_status()
    feat = next(f for f in r.json()["features"] if f["name"] == column)["type"]
    return (feat.get("feature") or feat)["names"]


# ── one builder per task: yields (prompt, gold, display) ─────────────────────


def classify() -> Iterator[tuple[str, str, str]]:
    names = ["World", "Sports", "Business", "Sci/Tech"]
    for row in parquet("fancyzhx/ag_news", "default", "test").itertuples():
        yield (f"Classify the news article into exactly one category: World, Sports, Business, Sci/Tech.\n\n"
               f"Article: {row.text}\n\nReply with the category name only.", names[row.label], row.text[:140])


def extract() -> Iterator[tuple[str, str, str]]:
    tags = label_names("lhoestq/conll2003", "default", "test", "ner_tags")
    for row in parquet("lhoestq/conll2003", "default", "test").itertuples():
        words, labels = list(row.tokens), [tags[i] for i in row.ner_tags]
        ents: list[str] = []
        cur: list[str] = []
        kind = ""
        for w, t in zip(words, labels):
            typ = t[2:]
            if t == "O" or typ == "MISC":
                if cur:
                    ents.append(" ".join(cur))
                cur, kind = [], ""
                continue
            if t.startswith("B-") or typ != kind:
                if cur:
                    ents.append(" ".join(cur))
                cur, kind = [w], typ
            else:
                cur.append(w)
        if cur:
            ents.append(" ".join(cur))
        sent = " ".join(words)
        if len(ents) >= 2 and len(words) >= 8:
            yield (f"List every person, organisation and location named in the sentence, separated by '; '. "
                   f"Write nothing else.\n\nSentence: {sent}", "; ".join(ents), sent[:140])


def summarise() -> Iterator[tuple[str, str, str]]:
    for row in parquet("EdinburghNLP/xsum", "default", "test").itertuples():
        if 300 < len(row.document) <= DOC_CHARS:
            yield (f"Summarise the article in one sentence.\n\nArticle: {row.document}", row.summary,
                   f"Summarise: {row.document[:110]}…")


def qa() -> Iterator[tuple[str, list[str], str]]:
    for row in parquet("rajpurkar/squad", "plain_text", "validation").itertuples():
        golds = sorted(set(row.answers["text"]))
        yield (f"Answer the question using the passage. Reply with the shortest exact phrase from the passage.\n\n"
               f"Passage: {row.context}\n\nQuestion: {row.question}", golds, row.question)


def math() -> Iterator[tuple[str, str, str]]:
    for row in parquet("openai/gsm8k", "main", "test").itertuples():
        gold = row.answer.split("####")[-1].strip().replace(",", "")
        yield (f"Solve the problem. End your reply with 'Answer: <number>'.\n\n{row.question}", gold, row.question)


def hindi_qa() -> Iterator[tuple[str, str, str]]:
    df = parquet("ai4bharat/MILU", "Hindi", "test")
    need = {"question", "option1", "option2", "option3", "option4", "target"}
    if not need <= set(df.columns):
        raise SystemExit(f"MILU columns changed: expected {sorted(need)}, got {sorted(df.columns)}")
    for row in df.itertuples():
        opts = [row.option1, row.option2, row.option3, row.option4]
        gold = "ABCD"[int(str(row.target).removeprefix("option")) - 1]
        body = "\n".join(f"{k}. {o}" for k, o in zip("ABCD", opts))
        yield (f"निम्नलिखित प्रश्न का सही विकल्प चुनें। केवल विकल्प का अक्षर (A, B, C या D) लिखें।\n\n"
               f"प्रश्न: {row.question}\n{body}\n\nउत्तर:", gold, row.question)


BUILDERS: dict[str, tuple[str, Callable[[], Iterator]]] = {
    "classify": ("classify_en.jsonl", classify), "extract": ("extract_en.jsonl", extract),
    "summarise": ("summarise_en.jsonl", summarise), "qa": ("qa_en.jsonl", qa),
    "math": ("math_en.jsonl", math), "hindi_qa": ("qa_hi.jsonl", hindi_qa),
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.data.build_tasks", description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=200, help="items per task")
    ap.add_argument("--out", type=Path, default=TASKS_DIR)
    ap.add_argument("--only", nargs="*", default=list(BUILDERS), choices=list(BUILDERS))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for task in a.only:
        file, build = BUILDERS[task]
        pool = list(build())
        if len(pool) < a.n:
            raise SystemExit(f"{task}: only {len(pool)} usable rows, need {a.n}")
        picks = random.Random(f"{SEED}:{task}").sample(pool, a.n)
        lines = [TaskItem(id=f"{task}-{i:03d}", prompt=p, gold=g, display=d).model_dump_json() for i, (p, g, d) in enumerate(picks)]
        (a.out / file).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"build_tasks: {task:<9} {a.n} items → {a.out / file}")


if __name__ == "__main__":
    main()
