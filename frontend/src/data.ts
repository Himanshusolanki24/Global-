import { z } from 'zod';

// Mirrors bench/core/schemas.py (only the fields the UI reads). A mismatch fails loudly: no fake data.
const CI = z.tuple([z.number(), z.number()]);
const Schema = z.object({
  schema_version: z.string(),
  sample: z.boolean(),
  run: z.object({
    id: z.string(), hardware: z.string(), boundary: z.string(), grid_g_per_wh: z.number(), grid_source: z.string(),
    tariff_inr_per_kwh: z.number(), n_per_cell: z.number(), repeats: z.number(), measured_at: z.string(),
    host: z.object({ gpu: z.string(), idle_w: z.number(), idle_w_std: z.number(), idle_s: z.number() }),
  }),
  tasks: z.array(z.string()),
  task_meta: z.array(z.object({ id: z.string(), name: z.string(), lang: z.enum(['en', 'hi']), n_items: z.number().nullable() })),
  configs: z.array(z.object({
    id: z.string(), model: z.string(), name: z.string(), params_b: z.number().nullable(),
    quant: z.enum(['Q4', 'Q8', 'FP16', 'API']), kind: z.enum(['small', 'big']), mem_gb: z.number().nullable(), estimated: z.boolean(),
  })),
  // ponytail: energy fields are required, so an API config (wh_q null) fails validation; make them nullable
  // and render "not measurable" with energy_reason when a cloud model is added back.
  results: z.array(z.object({
    config: z.string(), task: z.string(), lang: z.enum(['en', 'hi']), n_items: z.number(),
    acc: z.number(), acc_ci: CI, ttft_p95_ms: z.number(), wh_q: z.number(), wh_ci: CI, mwh_tok: z.number(),
    inr_1k: z.number(), co2_g_1k: z.number(), energy_reason: z.string().nullable(),
  })),
  coq: z.array(z.object({ task: z.string(), lo: z.string(), hi: z.string(), d_acc: z.number(), d_acc_ci: CI, significant: z.boolean() })),
  router: z.object({
    small: z.string(), big: z.string(), samples: z.number(), tau: z.number(), conf_temperature: z.number(),
    classifier: z.object({ encoder: z.string(), heldout_acc: z.number(), n_test: z.number() }),
    queries: z.array(z.object({
      id: z.string(), task: z.string(), text: z.string(), conf: z.number(), pred_small: z.boolean(),
      small_ok: z.boolean(), big_ok: z.boolean(), ok: z.record(z.boolean()),
    })),
  }),
});

export type Bench = z.infer<typeof Schema>;
export type Config = Bench['configs'][number];
export type Result = Bench['results'][number];
export type Query = Bench['router']['queries'][number];
export type Lang = 'en' | 'hi';

/** results.json is produced by the Kaggle run (bench/export.py) and served as a static file. */
export async function loadBench(): Promise<Bench> {
  const r = await fetch(`${import.meta.env.BASE_URL}results.json`);
  if (!r.ok) throw new Error(`results.json not found (HTTP ${r.status}). Run bench/export.py and commit it to frontend/public/.`);
  const parsed = Schema.safeParse(await r.json());
  if (!parsed.success) throw new Error(`results.json does not match the schema: ${parsed.error.issues.slice(0, 3).map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`);
  return parsed.data;
}

const index = new WeakMap<Bench, Map<string, Result>>();
/** Each task is measured in one language; when the requested language has no cell, the task's own one is used. */
export function cell(b: Bench, config: string, task: string, lang: Lang): Result {
  let m = index.get(b);
  if (!m) {
    m = new Map();
    for (const r of b.results) m.set(`${r.config}|${r.task}|${r.lang}`, r).set(`${r.config}|${r.task}`, r);
    index.set(b, m);
  }
  return m.get(`${config}|${task}|${lang}`) ?? m.get(`${config}|${task}`)!;
}

export const shortName = (c: Config) => (c.kind === 'big' ? c.name : `${c.name} ${c.quant}`);

// Headings only. When lang = hi they are re-set in Devanagari, not swapped labels.
export const T = {
  bench: { en: 'Bench', hi: 'बेंच' },
  frontier: { en: 'Frontier', hi: 'सीमा रेखा' },
  receipt: { en: 'Receipt', hi: 'रसीद' },
  router: { en: 'Router', hi: 'राउटर' },
  method: { en: 'Method', hi: 'पद्धति' },
  dial: { en: 'Turn the dial. Watch what size buys.', hi: 'डायल घुमाइए, देखिए आकार क्या खरीदता है।' },
  frontierH: { en: 'Accuracy against energy, per 1,000 queries', hi: 'हर 1,000 सवालों पर सटीकता बनाम ऊर्जा' },
  receiptH: { en: 'What each extra accuracy point costs', hi: 'हर अतिरिक्त अंक की असली कीमत' },
  routerH: { en: 'Send it small unless the small model is unsure', hi: 'पहले छोटा मॉडल, शक हो तो बड़ा' },
  methodH: { en: 'How a number gets onto this bench', hi: 'यह संख्या बेंच तक कैसे पहुँचती है' },
  priorH: { en: 'What earlier work already measures', hi: 'पिछला काम क्या-क्या मापता है' },
} as const;
export const t = (k: keyof typeof T, lang: Lang) => T[k][lang];

export const TASKS: Record<string, string> = {
  classify: 'Classify', extract: 'Extract', summarise: 'Summarise', qa: 'Q&A', math: 'Math', hindi_qa: 'Hindi Q&A',
};

export const fmt = {
  pct: (x: number) => `${x.toFixed(1)}%`,
  inr: (x: number) => `₹${x < 10 ? x.toFixed(2) : x < 1000 ? x.toFixed(1) : Math.round(x).toLocaleString('en-IN')}`,
  wh: (x: number) => (x < 10 ? x.toFixed(2) : x < 100 ? x.toFixed(1) : Math.round(x).toLocaleString('en-IN')),
  when: (iso: string) =>
    new Date(iso).toLocaleString('en-IN', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata' }) + ' IST',
};
