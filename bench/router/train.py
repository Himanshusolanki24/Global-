"""Train the router and record everything export.py needs.

    python -m bench.router.train --db /kaggle/working/rightsize.db

Classifier: multilingual sentence embedding of the prompt → logistic regression → "does the small
model suffice?". Confidence: agreement of the small model's conf samples (harness --phase conf).
τ is tuned per big model on the validation split; everything reported is on the test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.analysis.aggregate import aggregate
from bench.core.schemas import (
    Classifier, RouterQuery, RunRecord, load_configs, load_items, load_pricing, load_router_spec, load_tasks,
)
from bench.core.store import Store
from bench.router.confidence import agreement
from bench.router.labeler import smallest_ok, split, verdicts
from bench.router.simulate import decision, pick_replay, tune_tau

ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # covers Hindi
SEED = 412


def queries_for(split_name: str, keyed: dict, pred: dict, conf: dict, spec, order: list[str], text: dict) -> list[RouterQuery]:  # noqa: ANN001
    out = []
    for (task, item), ok in sorted(keyed.items()):
        if split(item) != split_name or (task, item) not in conf or spec.small not in ok or any(b not in ok for b in spec.bigs):
            continue
        out.append(RouterQuery(
            id=f"Q-{len(out) + 1:03d}", item_id=item, task=task, lang=text[(task, item)][1], text=text[(task, item)][0],
            conf=round(conf[(task, item)], 3), pred_small=pred[(task, item)], small_ok=ok[spec.small], big_ok=ok[spec.bigs[0]],
            ok=ok, smallest_ok=smallest_ok(ok, order),
        ))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.router.train")
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--tasks-dir", type=Path, default=None)
    a = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    store = Store(a.db)
    rows = store.rows()
    configs, tasks, pricing, spec = load_configs(), load_tasks(), load_pricing(), load_router_spec()
    order = [c.id for c in configs]
    text, prompt = {}, {}
    for t in tasks:
        for it in (load_items(t, a.tasks_dir) if a.tasks_dir else load_items(t)):
            text[(t.id, it.id)] = (it.display, t.lang)
            prompt[(t.id, it.id)] = it.prompt

    keyed = {k: v for k, v in verdicts(rows).items() if spec.small in v}
    if not keyed:
        raise SystemExit(f"router: no benchmark rows for the small model {spec.small} — run the harness first")
    conf_calls: dict[tuple[str, str], list[RunRecord]] = {}
    for r in rows:
        if r.kind == "conf" and r.config_id == spec.small and r.error is None:
            conf_calls.setdefault((r.task_id, r.item_id), []).append(r)
    conf = {k: agreement([c.output or "" for c in cs]) for k, cs in conf_calls.items() if len(cs) == spec.samples}
    missing = [k for k in keyed if split(k[1]) != "train" and k not in conf]
    if missing:
        raise SystemExit(f"router: {len(missing)} val/test items lack {spec.samples} conf samples — run: "
                         f"python -m bench.harness --db {a.db} --phase conf")

    keys = sorted(keyed)
    enc = SentenceTransformer(ENCODER)
    X = enc.encode([prompt[k] for k in keys], batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    y = [keyed[k][spec.small] for k in keys]
    tr = [i for i, k in enumerate(keys) if split(k[1]) == "train"]
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED).fit(X[tr], [y[i] for i in tr])
    pred = dict(zip(keys, (bool(p) for p in clf.predict(X))))

    val = queries_for("val", keyed, pred, conf, spec, order, text)
    test = queries_for("test", keyed, pred, conf, spec, order, text)
    heldout = sum(q.pred_small == q.small_ok for q in test) / max(len(test), 1)
    print(f"router: classifier held-out accuracy {100 * heldout:.1f}% on {len(test)} test items")

    reasons = store.get_meta("energy_reasons", {})
    cells = aggregate(rows, configs, tasks, pricing, reasons)
    cell = {(c.config, c.task): c for c in cells}
    by_task = lambda cid: {t.id: cell[(cid, t.id)] for t in tasks if (cid, t.id) in cell}  # noqa: E731
    calls = {(r.kind, r.config_id, r.item_id, r.repeat_index): r for r in rows if r.error is None}

    variants, replay = [], []
    for big in spec.bigs:
        tau = tune_tau(val, by_task(spec.small), by_task(big), spec.samples, big)
        variants.append({"big": big, "tau": tau})
        print(f"router: τ = {tau:.2f} for big = {big}")
        reason = reasons.get(big) or next((c.energy_reason for c in by_task(big).values() if c.energy_reason), None)
        for q in pick_replay(test, tau):
            small_calls = [calls[("conf", spec.small, q.item_id, k)] for k in range(spec.samples)]
            replay.append(decision(q, tau, spec.small, big, small_calls, calls[("bench", big, q.item_id, 0)],
                                   reason or "not measured").model_dump())

    store.put_meta("router", {
        "small": spec.small, "samples": spec.samples, "conf_temperature": spec.conf_temperature,
        "classifier": Classifier(encoder=ENCODER, heldout_acc=round(100 * heldout, 2), n_train=len(tr),
                                 n_val=len(val), n_test=len(test)).model_dump(),
        "queries": [q.model_dump() for q in test], "variants": variants,
    })
    store.put_meta("replay", replay)
    store.close()


if __name__ == "__main__":
    main()
