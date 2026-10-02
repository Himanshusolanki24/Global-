from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.analysis.stats import bootstrap_ci, paired_diff_ci, significant
from bench.core.schemas import AggregateRow, ApiPrice, RouterQuery, RunRecord
from bench.core.store import Store
from bench.measure.cost import api_cost_inr, local_cost_inr
from bench.measure.energy import idle_subtracted_wh
from bench.scoring.exact_match import exact_match
from bench.scoring.token_f1 import token_f1
from bench.router.confidence import agreement
from bench.router.labeler import split
from bench.router.simulate import escalates, routes_small, simulate, tune_tau


def rec(item: str, rep: int, error: str | None = None) -> RunRecord:
    return RunRecord(run_id="R", config_id="m/Q4", task_id="qa", item_id=item, repeat_index=rep, output="x",
                     score=1.0, correct=True, ttft_ms=10, total_ms=50, input_tokens=5, output_tokens=7,
                     tokens_per_sec=140, energy_wh=0.01, energy_source="nvml", cost_inr=0.0001,
                     host_fingerprint="h", timestamp="2026-09-18T10:00:00+05:30", error=error)


def test_cost() -> None:
    assert local_cost_inr(1000.0, 8.5) == pytest.approx(8.5)  # 1 kWh
    assert local_cost_inr(None, 8.5) is None  # unmeasured energy stays unmeasured
    price = ApiPrice(in_usd_per_m=2.5, out_usd_per_m=10.0)
    assert api_cost_inr(1_000_000, 1_000_000, price, 88.0) == pytest.approx(12.5 * 88)


def test_bootstrap_ci() -> None:
    lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < 0.5 < hi and hi - lo < 0.25
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    assert bootstrap_ci([0, 1, 1, 0]) == bootstrap_ci([0, 1, 1, 0])  # seeded
    assert significant(paired_diff_ci([0] * 30, [1] * 30))
    assert not significant(paired_diff_ci([0, 1] * 15, [1, 0] * 15))


def cell(inr: float, wh: float | None) -> AggregateRow:
    return AggregateRow(config="c", task="qa", lang="en", n_items=1, n_calls=1, n_errors=0, acc=50, acc_ci=(40, 60),
                        ttft_ms_mean=1, ttft_ms_median=1, ttft_p95_ms=1, total_ms_mean=1, total_ms_median=1,
                        total_ms_p95=1, tok_s_mean=1, wh_q=wh, wh_ci=None if wh is None else (wh, wh), mwh_tok=None,
                        energy_n=1, energy_reason=None if wh is not None else "cloud inference, not measurable locally",
                        inr_1k=inr, inr_ci=(inr, inr), co2_g_1k=None, ei=None)


def q(conf: float, pred_small: bool, small_ok: bool = True, big_ok: bool = True) -> RouterQuery:
    return RouterQuery(id="Q", item_id="i", task="qa", lang="en", text="t", conf=conf, pred_small=pred_small,
                       small_ok=small_ok, big_ok=big_ok, ok={"s": small_ok, "b": big_ok}, smallest_ok=None)


def test_router_escalation() -> None:
    assert routes_small(q(1.0, True), 0.67) and not escalates(q(1.0, True), 0.67)
    assert escalates(q(1 / 3, True), 0.67) and not routes_small(q(1 / 3, True), 0.67)
    assert not routes_small(q(1.0, False), 0.67) and not escalates(q(1.0, False), 0.67)  # classifier said big

    small, big = {"qa": cell(1.0, 0.1)}, {"qa": cell(100.0, 1.0)}
    qs = [q(1.0, True), q(1 / 3, True, small_ok=False), q(1.0, False)]
    sim = simulate(qs, 0.67, small, big, samples=3, big_id="b")
    # ₹ per query: 3×small · 3×small + big · big  → (0.003 + 0.103 + 0.1) / 3 per query
    assert sim.router.inr_1k == pytest.approx((0.003 + 0.103 + 0.1) / 3 * 1000)
    assert sim.router.pct_small == pytest.approx(100 / 3, abs=0.01)
    assert sim.router.acc == 100.0  # escalated query answered by big
    assert sim.wh_saved_1k is not None

    sim = simulate(qs, 0.67, small, {"qa": cell(100.0, None)}, samples=3, big_id="b")
    assert sim.wh_saved_1k is None and sim.wh_reason == "cloud inference, not measurable locally"


def test_resume_after_crash(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    s = Store(db)
    s.write(rec("a", 0))
    s.write(rec("a", 1, error="timeout"))
    s.db.close()  # simulated crash: no clean shutdown

    s = Store(db)
    assert s.done() == {("bench", "m/Q4", "qa", "a", 0)}  # errored call is retried, finished one is skipped
    s.write(rec("a", 1))
    assert len(s.rows()) == 2 and all(r.error is None for r in s.rows())


def test_fixture_export_is_valid_and_honest(tmp_path: Path) -> None:
    from bench.export import main

    main(["--fixtures", "--out", str(tmp_path)])
    r = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert r["sample"] is True
    api = [x for x in r["results"] if x["config"].endswith("/API")]
    assert api and all(x["wh_q"] is None and x["energy_reason"] for x in api)  # never fabricated
    assert all(p["config"] not in {"gpt-4o/API", "claude-sonnet/API"} for p in r["pareto"] if p["axis"] == "energy")
    decisions = json.loads((tmp_path / "replay.json").read_text(encoding="utf-8"))["decisions"]
    assert all(sum(d["big"] == v["big"] for d in decisions) == 10 for v in r["router"]["variants"])
    sims = {v["big"]: v["sim"] for v in r["router"]["variants"]}
    assert sims["llama3.1-8b/Q8"]["wh_saved_1k"] is not None  # local big model ⇒ energy saving is measured
    assert sims["gpt-4o/API"]["wh_saved_1k"] is None and sims["gpt-4o/API"]["wh_reason"]


def test_confidence_needs_disagreement() -> None:
    assert agreement(["Rs 1800.", "rs 1800", "1800 rs"]) == 1.0  # wording differs, answer agrees
    assert agreement(["Delhi", "Delhi", "Mumbai"]) == 2 / 3
    assert agreement(["भाखड़ा नांगल", "टिहरी", "हीराकुंड"]) == 1 / 3  # Devanagari tokenises correctly


def test_conf_rows_never_reach_aggregates(tmp_path: Path) -> None:
    from bench.analysis.aggregate import items

    conf = rec("a", 0).model_copy(update={"kind": "conf", "temperature": 0.7, "correct": False})
    assert items([rec("a", 0), conf])[("m/Q4", "qa", "a")].acc == 1.0


def test_idle_subtraction() -> None:
    assert idle_subtracted_wh(gross_j=3600 * 2, seconds=3600, idle_w=1.0) == pytest.approx(1.0)  # 2 Wh − 1 W·h
    assert idle_subtracted_wh(gross_j=90, seconds=10, idle_w=10.0) == pytest.approx(-10 / 3600)  # below idle: kept, not clamped


def test_scorers() -> None:
    assert exact_match("So 6*4*5 = 120 cubic feet, $12.\nAnswer: 12", "12") == 1.0  # last number wins
    assert exact_match("Answer: 1,800", "1800") == 1.0
    assert exact_match("उत्तर: B", "B") == 1.0 and exact_match("C", "B") == 0.0
    assert exact_match("Sci/Tech.", "Sci/Tech") == 1.0 and exact_match("Business", "Sci/Tech") == 0.0
    assert token_f1("Lara; Australia; Geoff Marsh", "Lara; Australia; Geoff Marsh; Ian Healy") == pytest.approx(2 * 1 * (4 / 6) / (1 + 4 / 6))


def test_split_is_stable() -> None:
    assert split("qa-001") == split("qa-001")
    counts = {k: 0 for k in ("train", "val", "test")}
    for i in range(2000):
        counts[split(f"t-{i}")] += 1
    assert 0.55 < counts["train"] / 2000 < 0.65


def test_tau_holds_the_quality_floor() -> None:
    small, big = {"qa": cell(1.0, 0.1)}, {"qa": cell(100.0, 1.0)}
    val = [q(1.0, True)] * 8 + [q(1 / 3, True, small_ok=False)] * 2  # unsure ⇒ small is wrong
    tau = tune_tau(val, small, big, samples=3, big_id="b")
    assert 1 / 3 < tau <= 1.0  # escalates the unsure ones, keeps the confident ones small


def test_harness_plan_and_job_wiring(tmp_path: Path) -> None:
    import argparse

    from bench.core.schemas import Metrics, TaskItem, TaskSpec, load_configs, load_pricing
    from bench.harness import Job, plan, run_job
    from bench.measure.energy import Meter
    from bench.runners.base import Runner

    task = TaskSpec(id="math", name="Math", lang="en", file="x", scorer="exact_match", max_tokens=8, pass_score=1.0)
    its = [TaskItem(id=f"math-{i:03d}", prompt="2+2?", gold="4", display="2+2?") for i in range(5)]
    local = next(c for c in load_configs() if c.backend == "ollama")
    a = argparse.Namespace(phase="bench", limit=None, repeats=3, energy_items=2)
    jobs = plan([local], [task], {"math": its}, a)
    assert len(jobs) == 2 * 3 + 3 * 1  # first 2 items ×3 repeats, the rest once
    assert all(j.temperature == 0.0 for j in jobs)

    class Fake(Runner):
        def generate(self, prompt: str, max_tokens: int, temperature: float, seed: int) -> tuple[str, Metrics]:
            if prompt == "boom":
                raise TimeoutError("slow")
            return "Answer: 4", Metrics(ttft_ms=5, total_ms=20, input_tokens=3, output_tokens=4, tokens_per_sec=200)

    class S:  # the parts of harness.Session that run_job touches
        runner, metered, c = Fake(), False, local
        cost = staticmethod(lambda wh, m: None)

    meter = Meter(idle_w=0.0)
    ok = run_job(jobs[0], S(), meter, "fp")  # type: ignore[arg-type]
    assert ok.correct is True and ok.error is None and ok.energy_source == "unavailable" and ok.energy_wh is None
    bad = run_job(Job(local, task, its[0].model_copy(update={"prompt": "boom"}), 0, "bench", 0.0, 0), S(), meter, "fp")  # type: ignore[arg-type]
    assert bad.error == "TimeoutError: slow" and bad.correct is None  # stored, then retried on resume
    assert load_pricing()  # config files parse
