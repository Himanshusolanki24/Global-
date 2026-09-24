import type { Cell } from './metrics.ts';

export type Bench = typeof import('../mock/results.json');
export type Config = Bench['configs'][number];
export type Result = Bench['results'][number] & Cell;
export type Query = Bench['router']['queries'][number];
export type Lang = 'en' | 'hi';

const API = import.meta.env.VITE_API_URL as string | undefined;

/** Real runs from FastAPI when VITE_API_URL is set; otherwise the bundled sample (flagged sample: true). */
export async function loadBench(): Promise<Bench> {
  const sample = async () => (await import('../mock/results.json')).default as Bench;
  if (!API) return sample();
  try {
    const r = await fetch(`${API}/results`);
    if (!r.ok) throw new Error(`GET /results → ${r.status}`);
    return await r.json();
  } catch (e) {
    console.warn('Backend unreachable, showing sample data.', e);
    return sample();
  }
}

const index = new WeakMap<Bench, Map<string, Result>>();
export function cell(b: Bench, config: string, task: string, lang: Lang): Result {
  let m = index.get(b);
  if (!m) index.set(b, (m = new Map(b.results.map((r) => [`${r.config}|${r.task}|${r.lang}`, r as Result]))));
  return m.get(`${config}|${task}|${lang}`)!;
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
