"""SAMPLE data for building the frontend before a real run exists. Every number here is INVENTED.

Synthetic per-call rows are written to a throwaway SQLite file and pushed through the exact same
aggregate → pareto → CoQ → router → export path as a real run, so the sample has the real shape.
The output carries sample: true and the UI keeps its sample banner up.
"""

from __future__ import annotations

import math
import random

from bench.core.schemas import (
    CLOUD_ENERGY_REASON, Classifier, HostInfo, RouterQuery, RunRecord, load_configs, load_pricing, load_router_spec, load_tasks,
)
from bench.core.store import Store
from bench.measure.cost import api_cost_inr, local_cost_inr
from bench.router.confidence import agreement
from bench.router.labeler import smallest_ok
from bench.router.simulate import decision, pick_replay

ITEMS_PER_TASK = 40
REPEATS = 3
RUN_ID = "RUN-SAMPLE"

QUALITY = {"qwen2.5-0.5b": 0.40, "llama3.2-1b": 0.50, "gemma2-2b": 0.61, "phi3.5-mini": 0.69,
           "llama3.1-8b": 0.76, "mistral-large": 0.85, "claude-sonnet": 0.89}
QUANT = {"Q4": (-0.028, 0.55, 0.70, 0.62), "Q8": (-0.006, 0.72, 0.80, 1.10), "FP16": (0.0, 1.0, 1.0, 2.05)}  # Δq, energy, latency, GB per B params
TASK = {"classify": (0.13, 0.4, 8, 1.0), "extract": (0.05, 0.8, 60, 1.3), "summarise": (0.03, 1.6, 140, 1.8),
        "qa": (0.0, 1.0, 90, 1.0), "math": (-0.10, 1.8, 180, 1.0), "hindi_qa": (-0.06, 1.2, 100, 1.1)}  # Δq, energy, out tokens, prompt
TEXTS = {
    "classify": ['Is this complaint about billing or delivery? "Order came 4 days late"', 'वर्गीकरण करें: "मेरा राशन कार्ड अब तक नहीं बना"'],
    "extract": ['Pull the PAN and invoice total from: "PAN ABCDE1234F, total ₹14,560"', 'Get the train number and date: "12951 Rajdhani on 14 Oct"'],
    "qa": ["Which article of the Constitution abolishes untouchability?", "What is the repo rate set by RBI used for?"],
    "math": ["A tank fills in 6 h and drains in 9 h. Both open, how long to fill?", "Simple interest on ₹8,000 at 7.5% for 3 years?"],
    "hindi_qa": ["भारत का सबसे लंबा बाँध कौन-सा है?", "पंचायती राज किस संविधान संशोधन से आया?"],
    "summarise": ["Summarise this 400-word RTI reply in two lines.", "इस सरकारी परिपत्र का सार तीन पंक्तियों में लिखें।"],
}
HOST = HostInfo(fingerprint="sample-t4", gpu="Tesla T4", gpu_mem_gb=15.0, driver="550.54.15", cpu="Intel Xeon @ 2.00GHz",
                ram_gb=31.4, os="Linux 6.6 (Kaggle)", ollama_version="0.12.3", idle_w=9.8, idle_w_std=0.4, idle_s=60.0)


def populate(store: Store) -> None:
    rng = random.Random(412)
    configs, tasks, pricing, spec = load_configs(), load_tasks(), load_pricing(), load_router_spec()
    difficulty = {(t.id, i): rng.random() for t in tasks for i in range(ITEMS_PER_TASK)}
    correct: dict[tuple[str, str, int], bool] = {}
    minute = 0

    for c in configs:
        q0 = QUALITY[c.model]
        dq, e_q, lat_q, _ = QUANT.get(c.quant, (0.0, 1.0, 1.0, 0.0))
        for t in tasks:
            t_dq, t_e, t_tok, t_prompt = TASK[t.id]
            q = q0 + t_dq - (0.35 * (1 - q0) if t.id == "math" else 0) - (0.3 * (1 - q0) if t.lang == "hi" else 0) + dq * (1.3 - q0) * 1.6
            hi = 1.5 if t.lang == "hi" else 1.0
            for i in range(ITEMS_PER_TASK):
                ok = rng.random() < 1 / (1 + math.exp(-10 * (q - difficulty[(t.id, i)])))
                correct[(c.id, t.id, i)] = ok
                for rep in range(REPEATS):  # temperature 0: same verdict every repeat, timing/energy jitter
                    jit = math.exp(rng.gauss(0, 0.06))
                    out_tok = max(1, round(t_tok * (1.7 if t.lang == "hi" else 1) * math.exp(rng.gauss(0, 0.15))))
                    in_tok = round(120 * t_prompt * hi)
                    if c.backend == "ollama":
                        p = c.params_b or 1
                        ttft = (28 + 14 * p) * lat_q * t_prompt * hi * jit
                        tps = 140 / (0.4 + 0.3 * p) / lat_q * jit
                        wh: float | None = (0.012 + 0.03 * p) * e_q * t_e * hi * jit
                        cost = local_cost_inr(wh, pricing.tariff_inr_per_kwh)
                    else:
                        ttft = (540 if c.model == "mistral-large" else 780) * t_prompt * jit
                        tps = 80 * jit
                        wh = None
                        cost = api_cost_inr(in_tok, out_tok, pricing.api[c.tag], pricing.usd_inr)
                    minute += 1
                    store.write(RunRecord(
                        run_id=RUN_ID, config_id=c.id, task_id=t.id, item_id=f"{t.id}-{i:03d}", repeat_index=rep,
                        output="(sample)", score=1.0 if ok else 0.0, correct=ok,
                        ttft_ms=round(ttft, 1), total_ms=round(ttft + out_tok / tps * 1000, 1),
                        input_tokens=in_tok, output_tokens=out_tok, tokens_per_sec=round(tps, 2),
                        energy_wh=None if wh is None else round(wh, 6), energy_source="unavailable" if wh is None else "nvml",
                        cost_inr=None if cost is None else round(cost, 6), host_fingerprint=HOST.fingerprint,
                        timestamp=f"2026-09-18T{8 + minute // 3600 % 12:02d}:{minute // 60 % 60:02d}:{minute % 60:02d}+05:30",
                    ))

    store.put_meta("host", HOST.model_dump())
    store.put_meta("mem_gb", {c.id: round((c.params_b or 0) * QUANT[c.quant][3] + 0.4, 1) for c in configs if c.backend == "ollama"})

    # router: every 4th item of each task is the held-out test split.
    # Confidence comes from `samples` extra small-model calls at conf_temperature, stored as kind="conf"
    # rows and scored by agreement(), exactly as the real run will do it.
    order = [c.id for c in configs]
    bench_rows = {(r.config_id, r.item_id, r.repeat_index): r for r in store.rows()}
    answers = {3: ["A", "A", "A"], 2: ["A", "A", "B"], 1: ["A", "B", "C"]}  # sample answers by how many agree
    queries: list[RouterQuery] = []
    for t in tasks:
        for i in range(0, ITEMS_PER_TASK, 4):
            item = f"{t.id}-{i:03d}"
            ok = {cid: correct[(cid, t.id, i)] for cid in order}
            s_ok = ok[spec.small]
            agree = rng.choices([3, 2, 1], [0.75, 0.2, 0.05] if s_ok else [0.2, 0.35, 0.45])[0]
            outs = [f"answer {a}" for a in answers[agree]]
            for k, out in enumerate(outs):
                src = bench_rows[(spec.small, item, k % REPEATS)]
                store.write(src.model_copy(update={"kind": "conf", "temperature": spec.conf_temperature,
                                                   "repeat_index": k, "output": out, "score": None, "correct": None}))
            queries.append(RouterQuery(
                id=f"Q-{len(queries) + 1:03d}", item_id=item, task=t.id, lang=t.lang, text=TEXTS[t.id][(i // 4) % 2],
                conf=round(agreement(outs), 3), pred_small=s_ok if rng.random() < 0.8 else not s_ok, small_ok=s_ok, big_ok=ok[spec.bigs[0]],
                ok=ok, smallest_ok=smallest_ok(ok, order),
            ))
    tau = 0.67  # sample value; the real run tunes τ per big model on the validation split
    heldout = sum(q.pred_small == q.small_ok for q in queries) / len(queries)
    clf = Classifier(encoder="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                     heldout_acc=round(100 * heldout, 2), n_train=144, n_val=48, n_test=len(queries))
    store.put_meta("router", {"small": spec.small, "samples": spec.samples, "conf_temperature": spec.conf_temperature,
                              "classifier": clf.model_dump(), "queries": [q.model_dump() for q in queries],
                              "variants": [{"big": b, "tau": tau} for b in spec.bigs]})

    rows = {(r.kind, r.config_id, r.item_id, r.repeat_index): r for r in store.rows()}
    replay = []
    for big in spec.bigs:
        reason = CLOUD_ENERGY_REASON if big.endswith("/API") else "not measured"
        for q in pick_replay(queries, tau):
            small_calls = [rows[("conf", spec.small, q.item_id, r)] for r in range(spec.samples)]
            replay.append(decision(q, tau, spec.small, big, small_calls, rows[("bench", big, q.item_id, 0)], reason).model_dump())
    store.put_meta("replay", replay)
