// Pure maths behind the screens. No imports, so `npm run check` can run it under plain Node.

export type Cell = { acc: number; acc_ci: number[]; inr_1k: number; wh_q: number; co2_g_1k: number };

/** Pareto frontier: lowest energy (x) for the highest accuracy (y). Returned in ascending x. */
export function pareto<T extends { x: number; y: number }>(pts: T[]): T[] {
  const sorted = [...pts].sort((a, b) => a.x - b.x || b.y - a.y);
  const out: T[] = [];
  let best = -Infinity;
  for (const p of sorted) if (p.y > best) (out.push(p), (best = p.y));
  return out;
}

/** CoQ = ΔCost ÷ ΔAccuracy, always read from the less accurate to the more accurate config. */
export function costOfQuality<T extends Cell>(a: T, b: T) {
  const [lo, hi] = a.acc <= b.acc ? [a, b] : [b, a];
  const dAcc = hi.acc - lo.acc;
  const dInr = hi.inr_1k - lo.inr_1k;
  const dWh = (hi.wh_q - lo.wh_q) * 1000; // Wh per 1k queries
  return {
    lo, hi, dAcc, dInr, dWh,
    dCo2: hi.co2_g_1k - lo.co2_g_1k,
    inrPerPt: dAcc > 0 ? dInr / dAcc : Infinity,
    whPerPt: dAcc > 0 ? dWh / dAcc : Infinity,
    // ponytail: CI overlap as the significance test; swap for a paired bootstrap on ΔAcc when per-query data ships
    significant: hi.acc_ci[0] > lo.acc_ci[1],
  };
}

export type Query = { conf: number; small_ok: boolean; big_ok: boolean };

/** Route small if conf ≥ τ else big. The small model always runs `samples` times to measure conf. */
export function route(qs: Query[], tau: number, small: Cell, big: Cell, samples: number) {
  let nSmall = 0, ok = 0, inr = 0, wh = 0;
  for (const q of qs) {
    const s = q.conf >= tau;
    nSmall += +s;
    ok += +(s ? q.small_ok : q.big_ok);
    inr += (small.inr_1k * samples + (s ? 0 : big.inr_1k)) / 1000;
    wh += small.wh_q * samples + (s ? 0 : big.wh_q);
  }
  const n = qs.length || 1;
  return {
    n: qs.length,
    pctSmall: (100 * nSmall) / n,
    acc: (100 * ok) / n,
    inrSaved1k: big.inr_1k - (inr / n) * 1000,
    whSaved1k: (big.wh_q - wh / n) * 1000,
  };
}
