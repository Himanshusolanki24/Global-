"""The contracts. results.json and replay.json are built from these models and validated before
they are written, so frontend/src/lib/schema.ts mirrors this file and nothing else.

Field names on AggregateRow / Router match the existing frontend payload (README "Payload shape");
everything new is additive. Null means "not measured", and every null metric has a reason field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, computed_field

SCHEMA_VERSION = "1.0.0"
CLOUD_ENERGY_REASON = "cloud inference, not measurable locally"
NO_NVML_REASON = "NVML unavailable on this host"

BENCH_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BENCH_DIR / "config"
TASKS_DIR = BENCH_DIR / "data" / "tasks"

Lang = Literal["en", "hi"]
Quant = Literal["Q4", "Q8", "FP16", "API"]
Backend = Literal["ollama", "openai", "anthropic"]
EnergySource = Literal["nvml", "unavailable"]
CI = tuple[float, float]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── inputs ────────────────────────────────────────────────────────────────────


class ModelConfig(Contract):
    id: str  # "phi3.5-mini/Q4" — the key used everywhere downstream
    model: str
    name: str
    params_b: float | None  # None for APIs: undisclosed
    quant: Quant
    kind: Literal["small", "big"]
    backend: Backend
    tag: str  # ollama tag or API model id
    mem_gb: float | None = None  # filled from NVML at load time; None for APIs

    @computed_field  # type: ignore[prop-decorator]
    @property
    def estimated(self) -> bool:
        """True ⇒ energy is never measured for this config; the UI tags anything energy-related est."""
        return self.backend != "ollama"


class TaskSpec(Contract):
    id: str
    name: str
    lang: Lang
    file: str
    scorer: Literal["exact_match", "token_f1", "llm_judge"]
    max_tokens: int
    pass_score: float
    n_items: int | None = None  # filled at export


class TaskItem(Contract):
    id: str
    prompt: str  # exactly what every model receives
    gold: str | list[str]
    display: str  # short form for the router UI (the question, not the passage)


class ApiPrice(Contract):
    in_usd_per_m: float
    out_usd_per_m: float


class ExternalEstimate(Contract):
    config: str
    wh_q: float | None
    source: str | None
    note: str


class Pricing(Contract):
    tariff_inr_per_kwh: float
    grid_g_per_wh: float
    grid_source: str
    usd_inr: float
    api: dict[str, ApiPrice]
    external_energy_estimates: list[ExternalEstimate]


class Judge(Contract):
    backend: Backend
    tag: str


class RouterSpec(Contract):
    small: str
    bigs: list[str]  # each is simulated against the same queries; the first is the default
    samples: int
    conf_temperature: float  # > 0, or every sample is identical and conf is always 1


# ── per call ──────────────────────────────────────────────────────────────────


class Metrics(Contract):
    """What a runner measures for one call. Energy is measured around the call by measure/energy.py."""

    ttft_ms: float  # from the first streamed token, not after completion
    total_ms: float
    input_tokens: int | None
    output_tokens: int
    tokens_per_sec: float | None  # decode rate


class RunRecord(Contract):
    """One row per call in SQLite. (kind, config_id, task_id, item_id, repeat_index) is the resume key.

    kind="bench": temperature 0, feeds every aggregate.
    kind="conf":  the router's self-consistency samples at RouterSpec.conf_temperature; never aggregated.
    """

    run_id: str
    kind: Literal["bench", "conf"] = "bench"
    temperature: float = 0.0
    config_id: str
    task_id: str
    item_id: str
    repeat_index: int
    output: str | None
    score: float | None
    correct: bool | None
    ttft_ms: float | None
    total_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    tokens_per_sec: float | None
    energy_wh: float | None
    energy_source: EnergySource
    cost_inr: float | None
    host_fingerprint: str
    timestamp: str  # ISO 8601 with offset
    error: str | None = None
    judge_output: str | None = None  # raw llm_judge reply, kept for audit


class HostInfo(Contract):
    fingerprint: str
    gpu: str
    gpu_mem_gb: float
    driver: str
    cpu: str
    ram_gb: float
    os: str
    ollama_version: str | None
    idle_w: float  # mean GPU draw over the idle window, subtracted from every call
    idle_w_std: float
    idle_s: float


# ── aggregates (results.json) ─────────────────────────────────────────────────


class AggregateRow(Contract):
    config: str
    task: str
    lang: Lang
    n_items: int
    n_calls: int
    n_errors: int
    acc: float  # % over items; repeats are averaged per item first
    acc_ci: CI
    ttft_ms_mean: float | None
    ttft_ms_median: float | None
    ttft_p95_ms: float | None
    total_ms_mean: float
    total_ms_median: float
    total_ms_p95: float
    tok_s_mean: float | None
    wh_q: float | None
    wh_ci: CI | None
    mwh_tok: float | None
    energy_n: int  # calls with an NVML reading
    energy_reason: str | None  # set whenever wh_q is None
    inr_1k: float | None
    inr_ci: CI | None
    co2_g_1k: float | None
    ei: float | None  # Efficiency Index = acc ÷ Wh per 1k queries


class ParetoPoint(Contract):
    task: str
    lang: Lang
    axis: Literal["energy", "cost"]  # x = Wh/1k or ₹/1k; y = acc
    config: str
    x: float
    y: float
    on_frontier: bool


class CoQPair(Contract):
    """Read from the less accurate (lo) to the more accurate (hi) config. Significance is a paired
    bootstrap on per-item ΔAccuracy, not CI overlap."""

    task: str
    lang: Lang
    lo: str
    hi: str
    d_acc: float
    d_acc_ci: CI
    significant: bool
    d_inr_1k: float | None
    d_wh_1k: float | None
    inr_per_pt: float | None  # None when d_acc ≤ 0 or cost unknown
    wh_per_pt: float | None


class QuantRow(Contract):
    model: str
    lang: Lang
    quant: Quant
    n_items: int
    acc: float
    acc_ci: CI
    d_acc_vs_fp16: float | None  # None on the FP16 row itself
    d_acc_ci: CI | None
    significant: bool | None
    wh_q: float | None
    inr_1k: float | None


class RouterQuery(Contract):
    id: str
    item_id: str  # links back to the recorded calls in SQLite
    task: str
    lang: Lang
    text: str
    conf: float  # agreement across `samples` small-model answers: 1/3, 2/3 or 1
    pred_small: bool  # classifier: does the small tier suffice?
    small_ok: bool
    big_ok: bool  # for the default big model (Router.big)
    ok: dict[str, bool]  # every config's verdict on this item, so any big model can be simulated
    smallest_ok: str | None  # smallest config that answered correctly


class PolicyTotals(Contract):
    acc: float
    pct_small: float
    inr_1k: float | None
    wh_1k: float | None


class RouterSim(Contract):
    router: PolicyTotals
    always_small: PolicyTotals
    always_big: PolicyTotals
    quality_retained_pct: float
    inr_saved_1k: float | None
    wh_saved_1k: float | None
    wh_reason: str | None


class Classifier(Contract):
    encoder: str
    heldout_acc: float
    n_train: int
    n_val: int
    n_test: int


class RouterVariant(Contract):
    big: str
    tau: float  # tuned on the validation split for this big model
    sim: RouterSim  # on the test split


class Router(Contract):
    small: str
    big: str  # default variant
    samples: int
    conf_temperature: float
    tau: float  # default variant's τ
    classifier: Classifier
    queries: list[RouterQuery]  # held-out test split only
    variants: list[RouterVariant]


class Run(Contract):
    id: str
    hardware: str
    boundary: str
    grid_g_per_wh: float
    grid_source: str
    tariff_inr_per_kwh: float
    usd_inr: float
    n_per_cell: int
    repeats: int
    measured_at: str
    host: HostInfo


class Results(Contract):
    schema_version: str = SCHEMA_VERSION
    sample: bool
    run: Run
    tasks: list[str]
    task_meta: list[TaskSpec]
    configs: list[ModelConfig]
    results: list[AggregateRow]
    pareto: list[ParetoPoint]
    coq: list[CoQPair]
    quant: list[QuantRow]
    router: Router
    external_estimates: list[ExternalEstimate]


# ── replay.json ───────────────────────────────────────────────────────────────


class Outcome(Contract):
    model: str
    latency_ms: float
    energy_wh: float | None
    energy_reason: str | None
    cost_inr: float | None
    correct: bool


class RouteDecision(Contract):
    big: str  # which router variant this decision belongs to
    tau: float
    query_id: str
    task: str
    lang: Lang
    text: str
    predicted_tier: Literal["small", "big"]
    confidence: float
    escalated: bool
    routed: Outcome
    counterfactual: Outcome  # had the big model answered directly


class Replay(Contract):
    schema_version: str = SCHEMA_VERSION
    sample: bool
    run_id: str
    decisions: list[RouteDecision]  # 10 per router variant


# ── loaders ───────────────────────────────────────────────────────────────────


def _yaml(name: str) -> object:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def load_configs() -> list[ModelConfig]:
    raw = _yaml("models.yaml")
    out = [
        ModelConfig(id=f"{m['model']}/{q}", model=m["model"], name=m["name"], params_b=m["params_b"],
                    quant=q, kind="small", backend="ollama", tag=tag)
        for m in raw["small"] for q, tag in m["tags"].items()
    ]
    out += [ModelConfig(id=f"{b['model']}/API", model=b["model"], name=b["name"], params_b=None,
                        quant="API", kind="big", backend=b["backend"], tag=b["tag"]) for b in raw["big"]]
    return out


def load_router_spec() -> RouterSpec:
    return RouterSpec(**_yaml("models.yaml")["router"])


def load_tasks() -> list[TaskSpec]:
    return [TaskSpec(**t) for t in _yaml("tasks.yaml")["tasks"]]


def load_judge() -> Judge:
    return Judge(**_yaml("tasks.yaml")["judge"])


def load_items(task: TaskSpec, tasks_dir: Path = TASKS_DIR) -> list[TaskItem]:
    path = tasks_dir / task.file
    if not path.exists():
        raise SystemExit(f"{path} missing — run: python -m bench.data.build_tasks")
    return [TaskItem.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pricing() -> Pricing:
    return Pricing(**_yaml("pricing.yaml"))
