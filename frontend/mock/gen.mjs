// SAMPLE data generator. These numbers are invented to exercise the UI; they are NOT results.
// Shape mirrors what the FastAPI backend should return from GET /results.  Run: npm run mock
import { writeFileSync } from 'node:fs';

let seed = 412;
const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;
const r2 = (x, d = 2) => +x.toFixed(d);

const models = [
  { id: 'qwen2.5-0.5b', name: 'Qwen2.5 0.5B', params: 0.5, q: 0.4, gb: 1.0 },
  { id: 'llama3.2-1b', name: 'Llama 3.2 1B', params: 1.2, q: 0.5, gb: 2.5 },
  { id: 'gemma2-2b', name: 'Gemma 2 2B', params: 2.6, q: 0.61, gb: 5.2 },
  { id: 'phi3.5-mini', name: 'Phi-3.5 mini', params: 3.8, q: 0.69, gb: 7.6 },
  { id: 'llama3.1-8b', name: 'Llama 3.1 8B', params: 8.0, q: 0.76, gb: 16.1 },
];
const quants = [
  { id: 'Q4', mem: 0.3, dq: -0.028, e: 0.55, lat: 0.7 },
  { id: 'Q8', mem: 0.53, dq: -0.006, e: 0.72, lat: 0.8 },
  { id: 'FP16', mem: 1, dq: 0, e: 1, lat: 1 },
];
const frontier = [
  { id: 'gpt-4o', name: 'GPT-4o', q: 0.87, wh: 0.9, ttft: 620, inr: 275 },
  { id: 'claude-sonnet', name: 'Claude Sonnet', q: 0.89, wh: 1.2, ttft: 780, inr: 380 },
];
const tasks = [
  { id: 'classify', off: 0.13, e: 0.4, tok: 8, prompt: 1 },
  { id: 'extract', off: 0.05, e: 0.8, tok: 60, prompt: 1.3 },
  { id: 'summarise', off: 0.03, e: 1.6, tok: 140, prompt: 1.8 },
  { id: 'qa', off: 0, e: 1, tok: 90, prompt: 1 },
  { id: 'math', off: -0.1, e: 1.8, tok: 180, prompt: 1 },
  { id: 'hindi_qa', off: -0.06, e: 1.2, tok: 100, prompt: 1.1 },
];
const N = 500, GRID = 0.716, TARIFF = 8.5;

const configs = [
  ...models.flatMap((m) => quants.map((q) => ({
    id: `${m.id}/${q.id}`, model: m.id, name: m.name, params_b: m.params, quant: q.id,
    kind: 'small', mem_gb: r2(m.gb * q.mem, 1), estimated: false,
  }))),
  ...frontier.map((f) => ({ id: `${f.id}/API`, model: f.id, name: f.name, params_b: null, quant: 'API', kind: 'big', mem_gb: null, estimated: true })),
];

const ci = (p) => { const h = 1.96 * Math.sqrt((p * (1 - p)) / N); return [r2((p - h * (0.9 + rnd() * 0.2)) * 100, 1), r2((p + h * (0.9 + rnd() * 0.2)) * 100, 1)]; };
const clamp = (x) => Math.min(0.97, Math.max(0.05, x));

const results = [];
for (const c of configs) for (const t of tasks) for (const lang of ['en', 'hi']) {
  const m = models.find((m) => m.id === c.model), f = frontier.find((f) => f.id === c.model);
  const q = quants.find((q) => q.id === c.quant);
  const base = (m ?? f).q;
  let acc = base + t.off - (t.id === 'math' ? (1 - base) * 0.35 : 0) - (lang === 'hi' ? (1 - base) * 0.3 : 0);
  if (t.id === 'hindi_qa' && lang === 'hi' && f) acc = Math.min(acc, 0.74);
  if (q) acc += q.dq * (1.3 - base) * 1.6;
  acc = clamp(acc + (rnd() - 0.5) * 0.03);
  const hiTok = lang === 'hi' ? 1.7 : 1;
  const tokens = Math.round(t.tok * hiTok);
  let wh, whCi, ttft, inr;
  if (m) {
    wh = (0.012 + 0.03 * m.params) * q.e * t.e * (lang === 'hi' ? 1.5 : 1) * (0.95 + rnd() * 0.1);
    whCi = [wh * 0.93, wh * 1.07];
    ttft = (28 + 14 * m.params) * q.lat * t.prompt * (lang === 'hi' ? 1.3 : 1) * (0.95 + rnd() * 0.1);
    inr = wh * TARIFF + 0.6 * m.params * t.e * q.lat; // electricity + GPU amortisation
  } else {
    wh = f.wh * t.e * (lang === 'hi' ? 1.5 : 1);
    whCi = [wh * 0.35, wh * 2.6]; // provider disclosures are loose
    ttft = f.ttft * t.prompt * (lang === 'hi' ? 1.15 : 1);
    inr = f.inr * t.e * hiTok * 0.8;
  }
  results.push({
    config: c.id, task: t.id, lang,
    acc: r2(acc * 100, 1), acc_ci: ci(acc),
    ttft_p95_ms: Math.round(ttft),
    wh_q: r2(wh, 4), wh_ci: whCi.map((x) => r2(x, 4)),
    mwh_tok: r2((wh * 1000) / tokens, 3),
    inr_1k: r2(inr, 2),
    co2_g_1k: r2(wh * 1000 * GRID, 1),
  });
}

// Router sample: small = phi3.5-mini/Q4, big = gpt-4o/API. conf = agreement of 3 samples → ⅓, ⅔ or 1.
const texts = [
  ['classify', 'Is this complaint about billing or delivery? "Order came 4 days late"'],
  ['extract', 'Pull the PAN and invoice total from: "PAN ABCDE1234F, total ₹14,560"'],
  ['qa', 'Which article of the Constitution abolishes untouchability?'],
  ['math', 'A tank fills in 6 h and drains in 9 h. Both open, how long to fill?'],
  ['hindi_qa', 'भारत का सबसे लंबा बाँध कौन-सा है?'],
  ['summarise', 'Summarise this 400-word RTI reply in two lines.'],
  ['classify', 'वर्गीकरण करें: "मेरा राशन कार्ड अब तक नहीं बना"'],
  ['qa', 'What is the repo rate set by RBI used for?'],
  ['math', 'Simple interest on ₹8,000 at 7.5% for 3 years?'],
  ['extract', 'Get the train number and date: "12951 Rajdhani on 14 Oct"'],
  ['hindi_qa', 'पंचायती राज किस संविधान संशोधन से आया?'],
  ['summarise', 'इस सरकारी परिपत्र का सार तीन पंक्तियों में लिखें।'],
];
const queries = Array.from({ length: 60 }, (_, i) => {
  const [task, text] = texts[i % texts.length];
  const u = rnd();
  const conf = u < 0.55 ? 1 : u < 0.82 ? 2 / 3 : 1 / 3;
  const pSmall = conf === 1 ? 0.9 : conf > 0.5 ? 0.6 : 0.3;
  return { id: `Q-${String(i + 1).padStart(3, '0')}`, task, text, conf: r2(conf, 3), small_ok: rnd() < pSmall, big_ok: rnd() < 0.88 };
});

const out = {
  sample: true,
  run: {
    id: 'RUN-0412', hardware: 'RTX 3090 24 GB + Ryzen 7 5800X', boundary: 'GPU+CPU+RAM',
    grid_g_per_wh: GRID, grid_source: 'CEA CO₂ Baseline Database', tariff_inr_per_kwh: TARIFF,
    n_per_cell: N, measured_at: '2026-09-18T14:22:00+05:30',
  },
  tasks: tasks.map((t) => t.id),
  configs, results,
  router: { small: 'phi3.5-mini/Q4', big: 'gpt-4o/API', samples: 3, queries },
};
writeFileSync(new URL('./results.json', import.meta.url), JSON.stringify(out));
console.log(`wrote ${configs.length} configs, ${results.length} cells, ${queries.length} router queries`);
