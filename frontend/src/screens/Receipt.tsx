import { useRef, useState } from 'react';
import { toPng } from 'html-to-image';
import type { ScreenProps } from '../App';
import { cell, fmt, shortName, TASKS } from '../data';
import { costOfQuality } from '../metrics';
import { H } from '../ui';

type Line = [string, string?] | '--' | { verdict: string };

export default function Receipt({ bench, lang, task }: ScreenProps) {
  const [a, setA] = useState(bench.router.small);
  const [b, setB] = useState(bench.router.big);
  const [printed, setPrinted] = useState(1); // bump to re-feed the paper
  const [tearing, setTearing] = useState(false);
  const paper = useRef<HTMLDivElement>(null);

  const ca = bench.configs.find((c) => c.id === a)!, cb = bench.configs.find((c) => c.id === b)!;
  const ra = cell(bench, a, task, lang), rb = cell(bench, b, task, lang);
  const q = costOfQuality({ ...ra, c: ca }, { ...rb, c: cb });
  // Paired bootstrap on per-item ΔAccuracy from the bench; CI overlap only if this pair is not in results.json.
  const paired = bench.coq.find((p) => p.task === task && p.lo === q.lo.c.id && p.hi === q.hi.c.id);
  const significant = paired ? paired.significant : q.significant;
  const est = ca.estimated || cb.estimated;
  const e = est ? ' est.' : '';
  const sign = (x: number) => (x >= 0 ? '+' : '−');

  const verdict = !significant
    ? `No clear gain${paired ? ` (paired 95% CI ${paired.d_acc_ci[0]} to ${paired.d_acc_ci[1]} pts includes 0)` : ': the CIs overlap'}. Stay with ${shortName(q.lo.c)}.`
    : `Each extra point costs ${fmt.inr(q.inrPerPt)} per 1k queries. Go bigger only if a wrong answer costs you more.`;

  const lines: Line[] = [
    ['RightSize · cost of quality'],
    [`${bench.run.id} · ${bench.run.boundary}`],
    [fmt.when(bench.run.measured_at)],
    [`${TASKS[task]} · ${ra.lang === 'en' ? 'English' : 'Hindi'} · n=${ra.n_items}`],
    '--',
    [`A  ${shortName(q.lo.c)}`, `${fmt.pct(q.lo.acc)}`],
    ['   ₹ / 1k queries', fmt.inr(q.lo.inr_1k)],
    ['   Wh / 1k queries', `${fmt.wh(q.lo.wh_q * 1000)}${q.lo.c.estimated ? ' est.' : ''}`],
    [`B  ${shortName(q.hi.c)}`, `${fmt.pct(q.hi.acc)}`],
    ['   ₹ / 1k queries', fmt.inr(q.hi.inr_1k)],
    ['   Wh / 1k queries', `${fmt.wh(q.hi.wh_q * 1000)}${q.hi.c.estimated ? ' est.' : ''}`],
    '--',
    ['Extra accuracy', `${sign(q.dAcc)}${Math.abs(q.dAcc).toFixed(1)} pts`],
    ['Extra cost / 1k', `${sign(q.dInr)}${fmt.inr(Math.abs(q.dInr))}`],
    ['Extra energy / 1k', `${sign(q.dWh)}${fmt.wh(Math.abs(q.dWh))} Wh${e}`],
    ['₹ per extra point', Number.isFinite(q.inrPerPt) ? fmt.inr(q.inrPerPt) : '—'],
    ['Wh per extra point', Number.isFinite(q.whPerPt) ? `${fmt.wh(q.whPerPt)}${e}` : '—'],
    [`CO₂e / 1k @ ${bench.run.grid_g_per_wh} g/Wh`, `${sign(q.dCo2)}${fmt.wh(Math.abs(q.dCo2))} g${e}`],
    '--',
    ['CoQ = ΔCost ÷ ΔAccuracy'],
    { verdict },
    '--',
    [est ? 'est. = provider disclosure, not measured here' : 'All values measured on this bench'],
    [`Grid factor: ${bench.run.grid_source}`],
  ];

  const tear = async () => {
    setTearing(true);
    try {
      const url = await toPng(paper.current!, { pixelRatio: 2 });
      const link = Object.assign(document.createElement('a'), { href: url, download: `rightsize-receipt-${task}-${lang}.png` });
      link.click();
    } finally {
      setTimeout(() => setTearing(false), 480);
    }
  };

  const pick = (v: string, set: (s: string) => void, label: string) => (
    <label className="field">
      <span className="small graphite">{label}</span>
      <select value={v} onChange={(e) => set(e.target.value)}>
        {bench.configs.map((c) => <option key={c.id} value={c.id}>{shortName(c)}</option>)}
      </select>
    </label>
  );

  return (
    <div className="screen receipt-screen">
      <div className="screen-head">
        <H k="receiptH" lang={lang} className="h-section" />
        <p className="lede">Pick two configurations. The printer itemises what the more accurate one charges for every point it adds, in rupees and in watt-hours.</p>
        <div className="receipt-controls">
          {pick(a, setA, 'Model A')}
          {pick(b, setB, 'Model B')}
          <button className="btn" onClick={() => setPrinted((n) => n + 1)}>Print receipt</button>
        </div>
        <div className="receipt-actions">
          <button className="btn-quiet" onClick={tear}>Tear off as PNG</button>
          <button className="btn-quiet" onClick={() => print()}>Save as PDF</button>
        </div>
      </div>

      <div className="printer">
        <div className="printer-slot" aria-hidden />
        <div ref={paper} key={`${printed}${a}${b}${task}${lang}`} className={`thermal ${tearing ? 'torn' : ''}`} style={{ ['--n' as string]: lines.length }} aria-live="polite">
          {lines.map((l, k) => {
            const style = { animationDelay: `calc(var(--line-feed) * ${k})`, ['--jit' as string]: `${((k * 37) % 5) - 2}px` };
            if (l === '--') return <hr key={k} className="feed" style={style} />;
            if ('verdict' in l) return <p key={k} className="feed verdict" style={style}>{l.verdict}</p>;
            return (
              <p key={k} className={`feed ${k === 0 ? 'thermal-title' : ''}`} style={style}>
                <span>{l[0]}</span>{l[1] && <span>{l[1]}</span>}
              </p>
            );
          })}
        </div>
      </div>
    </div>
  );
}
