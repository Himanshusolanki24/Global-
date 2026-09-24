import { useMemo, useRef, useState, type PointerEvent } from 'react';
import { motion } from 'framer-motion';
import { scaleLinear, scaleLog } from 'd3-scale';
import type { ScreenProps } from '../App';
import { cell, fmt, shortName, TASKS } from '../data';
import { pareto } from '../metrics';
import { Est, H } from '../ui';

const W = 960, HT = 540, M = { l: 70, r: 28, t: 20, b: 56 };
const EASE = { duration: 0.48, ease: [0.22, 1, 0.36, 1] } as const;

export default function Frontier({ bench, lang, task }: ScreenProps) {
  // Axes are fixed across every task and language, so switching re-flows the points against the same ruler.
  const { xs, ys } = useMemo(() => {
    const lo = Math.min(...bench.results.map((r) => r.wh_ci[0])) * 1000;
    const hi = Math.max(...bench.results.map((r) => r.wh_ci[1])) * 1000;
    return {
      xs: scaleLog().domain([lo * 0.8, hi * 1.2]).range([M.l, W - M.r]),
      ys: scaleLinear().domain([0, 100]).range([HT - M.b, M.t]),
    };
  }, [bench]);

  const pts = bench.configs.map((c) => {
    const r = cell(bench, c.id, task, lang);
    return { c, r, x: r.wh_q * 1000, y: r.acc, px: xs(r.wh_q * 1000), py: ys(r.acc) };
  });
  const front = pareto(pts);
  const frontD = front.map((p, k) => (k ? `H${p.px} V${p.py}` : `M${p.px} ${p.py}`)).join(' ');
  const threads = Object.values(Object.groupBy(pts.filter((p) => p.c.kind === 'small'), (p) => p.c.model)).map((g) => g!);

  const svg = useRef<SVGSVGElement>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [hover, setHover] = useState<number | null>(null);

  const move = (e: PointerEvent<SVGSVGElement>) => {
    const ctm = svg.current!.getScreenCTM()!.inverse();
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm);
    if (p.x < M.l || p.x > W - M.r || p.y < M.t || p.y > HT - M.b) return (setCursor(null), setHover(null));
    let best = -1, bd = 28 ** 2;
    pts.forEach((q, k) => { const d = (q.px - p.x) ** 2 + (q.py - p.y) ** 2; if (d < bd) (bd = d), (best = k); });
    setHover(best < 0 ? null : best);
    setCursor(best < 0 ? { x: p.x, y: p.y } : { x: pts[best].px, y: pts[best].py });
  };

  const h = hover === null ? null : pts[hover];
  const xTicks = xs.ticks().filter((v) => [1, 2, 5].includes(+String(v)[0]) && /^[125]0*$/.test(String(v)));

  return (
    <div className="screen">
      <div className="screen-head">
        <H k="frontierH" lang={lang} className="h-section" />
        <p className="lede">
          Each point is one configuration on {TASKS[task]} in {lang === 'en' ? 'English' : 'Hindi'}. A thread joins one model as it is quantised, FP16 to Q8 to Q4. The stepped line is the frontier: nothing to its left is more accurate.
        </p>
      </div>

      <div className="readline mono" aria-live="polite">
        {h ? (
          <>
            <b>{shortName(h.c)}</b> acc {fmt.pct(h.r.acc)} [{h.r.acc_ci.join(', ')}] · {fmt.wh(h.x)} Wh/1k [{fmt.wh(h.r.wh_ci[0] * 1000)}, {fmt.wh(h.r.wh_ci[1] * 1000)}] · {fmt.inr(h.r.inr_1k)}/1k · TTFT p95 {h.r.ttft_p95_ms} ms {h.c.estimated && <Est />}
          </>
        ) : cursor ? (
          <>x {fmt.wh(xs.invert(cursor.x))} Wh/1k · y {ys.invert(cursor.y).toFixed(1)}%</>
        ) : (
          <span className="graphite">Move across the plot, or tab through the points, to read a configuration.</span>
        )}
      </div>

      <svg ref={svg} className="plot-frame" viewBox={`0 0 ${W} ${HT}`} onPointerMove={move} onPointerLeave={() => (setCursor(null), setHover(null))} role="img" aria-label="Pareto plot of accuracy against energy per 1,000 queries">
        <defs>
          <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" className="hatch-line" />
          </pattern>
        </defs>

        {ys.ticks(10).map((v) => (
          <g key={`y${v}`}>
            <line x1={M.l} x2={W - M.r} y1={ys(v)} y2={ys(v)} className="gridline" />
            <text x={M.l - 10} y={ys(v)} className="axis-num" textAnchor="end" dominantBaseline="middle">{v}</text>
          </g>
        ))}
        {xs.ticks().map((v) => <line key={`x${v}`} x1={xs(v)} x2={xs(v)} y1={M.t} y2={HT - M.b} className="gridline" />)}
        {xTicks.map((v) => <text key={`xl${v}`} x={xs(v)} y={HT - M.b + 20} className="axis-num" textAnchor="middle">{v.toLocaleString('en-IN')}</text>)}
        <text x={W - M.r} y={HT - 10} textAnchor="end" className="axis-title">Wh per 1,000 queries, log scale →</text>
        <text x={16} y={M.t} className="axis-title" transform={`rotate(-90 16 ${M.t})`} textAnchor="end">Accuracy, % ↑</text>

        {threads.map((g) => (
          <motion.path key={g[0].c.model} initial={false} animate={{ d: `M${g.map((p) => `${p.px} ${p.py}`).join(' L')}` }} transition={EASE} className="thread" />
        ))}

        <path key={`${task}${lang}`} d={frontD} pathLength={1} className="frontier plot plot-slow" />

        {pts.map((p, k) => (
          <motion.g
            key={p.c.id}
            initial={false}
            animate={{ x: p.px, y: p.py }}
            transition={EASE}
            tabIndex={0}
            role="img"
            aria-label={`${shortName(p.c)}: ${fmt.pct(p.r.acc)}, ${fmt.wh(p.x)} Wh per 1,000 queries${p.c.estimated ? ', estimated' : ''}`}
            onFocus={() => (setHover(k), setCursor({ x: p.px, y: p.py }))}
            onBlur={() => (setHover(null), setCursor(null))}
            className={`pt pt-${p.c.kind} ${p.c.estimated ? 'pt-est' : ''} ${hover === k ? 'pt-on' : ''}`}
          >
            <motion.line initial={false} animate={{ y1: ys(p.r.acc_ci[0]) - p.py, y2: ys(p.r.acc_ci[1]) - p.py }} transition={EASE} className="ci" />
            <motion.line initial={false} animate={{ x1: xs(p.r.wh_ci[0] * 1000) - p.px, x2: xs(p.r.wh_ci[1] * 1000) - p.px }} transition={EASE} className={`ci ${p.c.estimated ? 'ci-est' : ''}`} />
            {p.c.kind === 'big' ? <rect x={-7} y={-7} width={14} height={14} /> : <circle r={p.c.quant === 'FP16' ? 6 : p.c.quant === 'Q8' ? 5 : 4} />}
            {(p.c.quant === 'FP16' || p.c.kind === 'big') && (
              // the two API points sit close together, so the first one is labelled on its left
              <text x={p.c.id === bench.router.big ? -12 : 10} y={p.c.kind === 'big' && p.c.id !== bench.router.big ? -10 : 20} textAnchor={p.c.id === bench.router.big ? 'end' : 'start'} className="pt-label">{p.c.name}{p.c.estimated ? ' (est.)' : ''}</text>
            )}
          </motion.g>
        ))}

        {cursor && (
          <g className="crosshair" aria-hidden>
            <line x1={M.l} x2={W - M.r} y1={cursor.y} y2={cursor.y} />
            <line x1={cursor.x} x2={cursor.x} y1={M.t} y2={HT - M.b} />
            <text x={cursor.x + 4} y={HT - M.b - 6} className="cross-read">{fmt.wh(xs.invert(cursor.x))}</text>
            <text x={M.l + 4} y={cursor.y - 6} className="cross-read">{ys.invert(cursor.y).toFixed(1)}%</text>
          </g>
        )}
      </svg>

      <div className="legend small">
        <span><i className="key key-small" /> small, local, measured</span>
        <span><i className="key key-big" /> frontier API, energy estimated from provider disclosures</span>
        <span><i className="key key-ci" /> 95% bootstrap CI, both axes</span>
        <span><i className="key key-front" /> Pareto frontier</span>
      </div>

      <details className="table-view">
        <summary>Read as a table</summary>
        <table>
          <thead><tr><th>Configuration</th><th>Accuracy %</th><th>95% CI</th><th>Wh / 1k</th><th>95% CI</th><th>₹ / 1k</th><th>On frontier</th></tr></thead>
          <tbody>
            {pts.map((p) => (
              <tr key={p.c.id}>
                <td>{shortName(p.c)}</td><td>{p.r.acc}</td><td>{p.r.acc_ci.join('–')}</td>
                <td>{fmt.wh(p.x)}{p.c.estimated && ' est.'}</td><td>{fmt.wh(p.r.wh_ci[0] * 1000)}–{fmt.wh(p.r.wh_ci[1] * 1000)}</td>
                <td>{fmt.inr(p.r.inr_1k)}</td><td>{front.includes(p) ? 'yes' : ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
