"""SQLite → results.json + replay.json, both validated against bench/core/schemas.py.

    python -m bench.export --db /kaggle/working/rightsize.db --out /kaggle/working
    python -m bench.export --fixtures            # SAMPLE data → frontend/public/, no GPU needed

Output is deterministic: same DB in, byte-identical files out (seeded bootstrap, no wall-clock values).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from bench.analysis.aggregate import aggregate, items
from bench.analysis.metrics import coq_pairs, quant_rows
from bench.analysis.pareto import pareto_points
from bench.core.schemas import (
    AggregateRow, Classifier, HostInfo, Replay, Results, RouteDecision, Router, RouterQuery, RouterVariant, Run,
    load_configs, load_pricing, load_tasks,
)
from bench.core.store import Store
from bench.router.simulate import simulate

REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "frontend" / "public"


def build(store: Store, sample: bool) -> tuple[Results, Replay]:
    rows = store.rows()
    if not rows:
        raise SystemExit("export: the database has no rows — run the harness first")
    pricing, tasks = load_pricing(), load_tasks()
    mem = store.get_meta("mem_gb")
    configs = [c.model_copy(update={"mem_gb": mem.get(c.id)}) for c in load_configs()]
    host = HostInfo(**store.get_meta("host"))

    by_item = items(rows)
    cells = aggregate(rows, configs, tasks, pricing, store.get_meta("energy_reasons", {}))
    present = {c.config for c in cells}
    configs = [c for c in configs if c.id in present]  # configs never run are absent, not zero-filled
    n_items = {t.id: len({i for (_, tid, i) in by_item if tid == t.id}) for t in tasks}

    r = store.get_meta("router")
    queries = [RouterQuery(**q) for q in r["queries"]]
    cell = {(c.config, c.task): c for c in cells}
    def by_task(cid: str) -> dict[str, AggregateRow]:
        return {t.id: cell[(cid, t.id)] for t in tasks if (cid, t.id) in cell}

    variants = [RouterVariant(big=v["big"], tau=v["tau"],
                              sim=simulate(queries, v["tau"], by_task(r["small"]), by_task(v["big"]), r["samples"], v["big"]))
                for v in r["variants"]]

    latest = max(rows, key=lambda x: x.timestamp)
    bench_rows = [x for x in rows if x.kind == "bench"]
    results = Results(
        sample=sample,
        run=Run(
            id=latest.run_id,
            hardware=f"{host.gpu} {host.gpu_mem_gb:g} GB + {host.cpu}",
            boundary="GPU only (NVML), idle-subtracted",
            grid_g_per_wh=pricing.grid_g_per_wh, grid_source=pricing.grid_source,
            tariff_inr_per_kwh=pricing.tariff_inr_per_kwh, usd_inr=pricing.usd_inr,
            n_per_cell=max(n_items.values()), repeats=max(x.repeat_index for x in bench_rows) + 1,
            measured_at=latest.timestamp, host=host,
        ),
        tasks=[t.id for t in tasks if n_items[t.id]],
        task_meta=[t.model_copy(update={"n_items": n_items[t.id]}) for t in tasks if n_items[t.id]],
        configs=configs,
        results=cells,
        pareto=pareto_points(cells),
        coq=coq_pairs(cells, by_item),
        quant=quant_rows(configs, cells, by_item),
        router=Router(
            small=r["small"], big=variants[0].big, samples=r["samples"], conf_temperature=r["conf_temperature"],
            tau=variants[0].tau, classifier=Classifier(**r["classifier"]), queries=queries, variants=variants,
        ),
        external_estimates=pricing.external_energy_estimates,
    )
    replay = Replay(sample=sample, run_id=latest.run_id,
                    decisions=[RouteDecision(**d) for d in store.get_meta("replay")])
    return results, replay


def write(results: Results, replay: Replay, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, model in (("results.json", results), ("replay.json", replay)):
        text = json.dumps(model.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        (out / name).write_text(text + "\n", encoding="utf-8")
        print(f"export: wrote {out / name} ({len(text) / 1024:.0f} KB)")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, help="SQLite file written by the harness")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT.relative_to(REPO)})")
    ap.add_argument("--fixtures", action="store_true", help="write SAMPLE data (sample: true) instead of reading --db")
    a = ap.parse_args(argv)

    if a.fixtures:
        from bench.fixtures import populate

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "sample.db")
            populate(store)
            write(*build(store, sample=True), a.out)
            store.close()
        return
    if a.db is None or not a.db.exists():
        sys.exit(f"export: --db {a.db} not found (or pass --fixtures for sample data)")
    store = Store(a.db)
    write(*build(store, sample=False), a.out)
    store.close()


if __name__ == "__main__":
    main()
