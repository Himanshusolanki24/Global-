# RightSize AI

**How small can the model be before quality breaks, and what does going bigger really cost?**

RightSize benchmarks 17 model configurations (5 small open models at FP16, Q8 and Q4, plus 2 frontier APIs) on 6 tasks in English and Hindi, on hardware you actually own: one laptop or one GPU. Every number is measured inside one declared power boundary and ships with its 95% confidence interval.

It is built for Indian college labs, startups and government teams who need to choose a model with evidence, not vibes.

> **Heads up:** the front end ships with **sample data, not results**. A striped banner stays on screen until a real run is loaded from the backend.

---

## What it measures

| Metric | Unit | Notes |
|---|---|---|
| Accuracy | % | 95% bootstrap CI, 2,000 resamples |
| Latency | TTFT p95, ms | time to first token |
| Energy | Wh/query, mWh/token | integrated across GPU+CPU+RAM |
| Cost | ₹ per 1,000 queries | electricity tariff + hardware amortisation, or API pricing |
| Memory | GB | local models only |
| Carbon | g CO₂e | CEA grid emission factor |

**Tasks:** classify · extract · summarise · Q&A · math · Hindi Q&A
**Models:** Qwen2.5 0.5B · Llama 3.2 1B · Gemma 2 2B · Phi-3.5 mini · Llama 3.1 8B (each at FP16 / Q8 / Q4) · GPT-4o · Claude Sonnet

### The formulas

```
E    = Σ(P − P_idle)·Δt ÷ N            energy per query, idle draw subtracted
EI   = Accuracy ÷ Wh per 1k queries     efficiency index
CoQ  = ΔCost ÷ ΔAccuracy                ₹ and Wh paid per extra accuracy point
Route: small if conf ≥ τ, else big      conf = agreement across 3 samples
```

Frontier API energy cannot be metered from outside, so it is **estimated** from provider disclosures. Everywhere it appears in the UI it is dotted, hatched and tagged `est.`: estimates never look like measurements.

---

## The interface: a calibration bench

The UI is designed as a measuring instrument, not a dashboard: graph paper, tick scales, a meter disc, a thermal receipt printer, a railway-style sorting rail.

| Screen | What you do there |
|---|---|
| **Bench** | Turn the **size dial** from 0.5B to frontier. The accuracy needle, energy meter disc, latency trace and ₹ counter all re-measure live. Set your accuracy floor and the dial marks where *quality breaks*. |
| **Frontier** | Pareto plot of accuracy against Wh per 1k queries (log scale). Threads show each model sliding FP16 → Q8 → Q4. Hover or tab for a crosshair readout; open *Read as a table* for the raw numbers. |
| **Receipt** | Pick two models and print a cost-of-quality receipt: ₹ and Wh per extra accuracy point, carbon, and a verdict. Tear it off as PNG, or save as PDF. |
| **Router** | Queries arrive as ticket stubs. Confident ones drop to the small model and unsure ones escalate to the big one. Drag τ and watch % routed small, ₹ saved and accuracy update. |
| **Method** | The 6-step measurement protocol, the measurement boundary drawn as a dashed box, and a comparison with prior work (HF AI Energy Score, RouteLLM, MILU). |

Switch the task and language (EN / हिं) from the header. In Hindi, headings are re-set in Devanagari rather than swapped as labels. **Lights off** switches to the dark bench.

---

## Quick start

Requires **Node 22+**.

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173
```

| Script | What it does |
|---|---|
| `npm run dev` | Start the dev server |
| `npm run build` | Type-check and build to `frontend/dist/` |
| `npm run preview` | Serve the production build |
| `npm run check` | Self-check for the Pareto, cost-of-quality and routing maths |
| `npm run mock` | Regenerate the sample data in `mock/results.json` |

### Connecting the backend

The front end reads one JSON payload. Point it at the FastAPI backend with an env var:

```bash
# frontend/.env
VITE_API_URL=http://localhost:8000
```

It then calls `GET {VITE_API_URL}/results`. If the variable is unset, or the backend is unreachable, the app falls back to the bundled sample data and keeps the sample banner visible.

**Payload shape** (`mock/gen.mjs` is the reference implementation):

```jsonc
{
  "sample": false,
  "run": { "id": "RUN-0412", "hardware": "RTX 3090 24 GB + Ryzen 7 5800X", "boundary": "GPU+CPU+RAM",
           "grid_g_per_wh": 0.716, "grid_source": "CEA CO₂ Baseline Database",
           "tariff_inr_per_kwh": 8.5, "n_per_cell": 500, "measured_at": "2026-09-18T14:22:00+05:30" },
  "tasks": ["classify", "extract", "summarise", "qa", "math", "hindi_qa"],
  "configs": [{ "id": "phi3.5-mini/Q4", "model": "phi3.5-mini", "name": "Phi-3.5 mini",
                "params_b": 3.8, "quant": "Q4", "kind": "small", "mem_gb": 2.3, "estimated": false }],
  "results": [{ "config": "phi3.5-mini/Q4", "task": "qa", "lang": "en",
                "acc": 66.6, "acc_ci": [62.3, 71.1], "ttft_p95_ms": 55,
                "wh_q": 0.0727, "wh_ci": [0.0676, 0.0778], "mwh_tok": 0.808,
                "inr_1k": 2.21, "co2_g_1k": 52.1 }],
  "router": { "small": "phi3.5-mini/Q4", "big": "gpt-4o/API", "samples": 3,
              "queries": [{ "id": "Q-001", "task": "classify", "text": "…", "conf": 0.667,
                            "small_ok": true, "big_ok": true }] }
}
```

Configurations must be ordered smallest to largest, because the size dial follows that order.

---

## Running the benchmark (Kaggle, nothing on your laptop)

All compute runs in one Kaggle notebook, `notebook/rightsize_bench.ipynb`. Its only outputs are `results.json` and `replay.json`.

1. **New notebook** on kaggle.com → File → Import → `notebook/rightsize_bench.ipynb` from this repo.
2. **Settings:** Accelerator **GPU T4 x2**, Internet **on**. (Two GPUs let Llama 3.1 8B FP16 sit fully in VRAM, and energy is read from both.)
3. **Secrets** (Add-ons → Secrets): `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN`. Accept the [MILU](https://huggingface.co/datasets/ai4bharat/MILU) terms first; it is gated.
4. **Save Version → Save & Run All.** The cells run in order:

| Cell | Does |
|---|---|
| 1 Ollama | installs Ollama natively, starts it with `OLLAMA_MAX_LOADED_MODELS=1` |
| 2 Code | clones this repo, `pip install -r requirements.txt`, picks up a previous DB if attached |
| 3 Tasks | builds the 6 × 200-item task bank (seeded) and checks every Ollama tag on the registry |
| 4 Host | GPU/CPU fingerprint + 60 s idle baseline |
| 5 Smoke | 3 items × 1 local + 1 API model; stops the notebook if any call errors |
| 6 Bench | the full run; every call is committed, so it resumes |
| 7 Conf | router confidence samples (small model, temperature 0.7, × 3) |
| 8 Router + export | trains the router, writes `results.json` and `replay.json` |

5. **A run takes longer than one 12-hour session.** The harness stops cleanly before the limit. When the version finishes, open a new version, **Add Input → Your Work → this notebook's output**, and Save & Run All again. Finished calls are skipped.
6. Download `results.json` and `replay.json` from the Output tab and commit them to `frontend/public/`.

Locally (or in Codespaces), `python -m bench.export --fixtures` writes **sample** files with the real shape, and `pytest` runs the checks.

### Deploying the site (Vercel)

Import the repo on vercel.com, set **Root Directory** to `frontend`, keep the Vite preset (`npm run build` → `dist`). Every commit of new `results.json` redeploys it. No backend, no environment variables.

---

## Project structure

```
frontend/
├── index.html              fonts: Bricolage Grotesque, IBM Plex Sans, JetBrains Mono, Tiro Devanagari Hindi, Mukta
├── mock/
│   ├── gen.mjs             sample data generator (seeded, reproducible)
│   └── results.json        SAMPLE data, not results
└── src/
    ├── tokens.css          design tokens: colour, type, spacing, radii, easing, durations
    ├── app.css             all component styles, dark theme, print, reduced motion
    ├── App.tsx             header, navigation, lazy-loaded screens, drawer transition
    ├── data.ts             data loading, lookup, Hindi headings, formatters
    ├── metrics.ts          pure maths: Pareto frontier, cost of quality, routing
    ├── metrics.check.ts    runnable self-check (npm run check)
    ├── ui.tsx              Odometer, CI Whisker, Est tag, headings, calibration loader
    └── screens/            Bench · Frontier · Receipt · Router · Method
```

**Stack:** React 18 · Vite 6 · TypeScript · framer-motion (needle springs, point re-flow) · d3-scale · html-to-image (receipt export). The stub physics and plotter-pen drawing are hand-rolled with `requestAnimationFrame` and CSS, with no extra libraries.

---

## Design principles

- **Motion shows measurement.** Nothing moves to decorate: the disc spins in proportion to Wh per query, counters roll like odometers, lines draw like a plotter.
- **No bare numbers.** Every accuracy and energy figure carries its 95% CI whisker.
- **Estimates look different.** API energy is always dotted, hatched and tagged `est.`.
- **Accessible by default.** The dial and sliders work fully from the keyboard (arrow keys step one config), readouts are announced via `aria-live`, every chart has a table view, and with `prefers-reduced-motion` everything goes straight to its end state and shows a *last measured* timestamp.
- **Two colours carry meaning:** signal `#D8FF3E` marks the chosen or live value, warn `#FF5A1F` marks where quality breaks or a query escalates.

---

## Before publishing

- [ ] Replace the sample data with a real run (the sample banner disappears once `"sample": false`).
- [ ] Verify the four landing figures and their cited sources (IEA 945 TWh; ~30× reasoning energy; 2.4× boundary gap; 74% MILU ceiling).
- [ ] Review the prior-work matrix on the Method screen, especially the "partly" marks.
- [ ] Replace the CI-overlap significance test in `metrics.ts` with a paired bootstrap on ΔAccuracy once the backend sends per-query results.
