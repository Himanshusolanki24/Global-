import { useEffect, useRef, useState } from 'react';
import type { ScreenProps } from '../App';
import { H } from '../ui';

/** The protocol as it actually ran, read from results.json where the run records it. */
const steps = ({ run }: ScreenProps['bench']) => [
  { h: 'Declare the boundary', f: `boundary = ${run.boundary}`, p: `NVML reads the energy counters of every GPU (${run.host.gpu}). CPU, RAM, monitor and network sit outside the box and are not claimed.` },
  { h: 'Measure idle', f: `P_idle = mean P over ${run.host.idle_s} s = ${run.host.idle_w.toFixed(1)} ± ${run.host.idle_w_std.toFixed(1)} W`, p: 'Every model is unloaded and the GPUs sit still, so their resting draw can be subtracted from every call.' },
  { h: 'Warm up, then run N queries', f: `N = ${run.n_per_cell} per task × config · temperature 0`, p: `One discarded warm-up call per model. Identical prompts and max_tokens for every model; a subset runs ${run.repeats}× for timing and energy spread.` },
  { h: 'Integrate energy', f: 'E = ΔE_counter − P_idle·Δt, per call', p: 'The GPU hardware energy counters are read before and after each call and summed across GPUs. Nothing is interpolated: a call without a reading stays unmeasured.' },
  { h: 'Score with a confidence interval', f: 'CI₉₅ = bootstrap, 2,000 resamples', p: 'Accuracy is averaged per item, then resampled over items. Differences between models use a paired bootstrap, not whisker overlap.' },
  { h: 'Compare', f: 'EI = Accuracy ÷ Wh per 1k\nCoQ = ΔCost ÷ ΔAccuracy', p: 'Efficiency index ranks configs; cost of quality prices each extra point in ₹ and Wh.' },
];

type Mark = 'yes' | 'part' | 'no';
const COLS = ['HF AI Energy Score', 'RouteLLM', 'MILU', 'RightSize'];
const ROWS: [string, Mark[]][] = [
  ['Measures energy per query', ['yes', 'no', 'no', 'yes']],
  ['Runs on hardware you own', ['no', 'part', 'yes', 'yes']],
  ['Indian languages', ['no', 'no', 'yes', 'yes']],
  ['Routes small ↔ big', ['no', 'yes', 'no', 'yes']],
  ['Prices quality in ₹', ['no', 'part', 'no', 'yes']],
  ['Reports 95% CIs', ['part', 'no', 'part', 'yes']],
  ['Quantisation as a variable', ['part', 'no', 'no', 'yes']],
];

export default function Method({ bench, lang }: ScreenProps) {
  const matrix = useRef<HTMLTableElement>(null);
  const [go, setGo] = useState(false);
  useEffect(() => {
    const el = matrix.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        setGo(true);
        io.disconnect();
      }
    }, { threshold: 0.3 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div className="screen notebook">
      <H k="methodH" lang={lang} className="h-section" />
      <div className="nb-grid">
        <ol className="nb-steps">
          {steps(bench).map((s) => (
            <li key={s.h}>
              <h3>{s.h}</h3>
              <p className="formula mono">{s.f}</p>
              <p>{s.p}</p>
            </li>
          ))}
        </ol>
        <figure className="boundary">
          <svg viewBox="0 0 320 300" role="img" aria-label={`Measurement boundary: ${bench.run.boundary}. Parts drawn dotted are not metered.`}>
            <rect x={20} y={20} width={280} height={190} className="bnd-box" />
            <text x={28} y={14} className="bnd-label">metered: {bench.run.boundary.split(',')[0]}</text>
            {[['GPU', 40, 50, 120, 70], ['CPU', 180, 50, 100, 70], ['RAM', 40, 140, 240, 44]].map(([n, x, y, w, h]) => {
              const inside = bench.run.boundary.includes(String(n)); // only what the run metered is drawn solid
              return (
                <g key={n}>
                  <rect x={+x} y={+y} width={+w} height={+h} className={inside ? 'bnd-part' : 'bnd-out'} />
                  <text x={+x + 10} y={+y + 22} className={inside ? 'bnd-part-label' : 'bnd-out-label'}>{n}</text>
                  {!inside && <text x={+x + 10} y={+y + 40} className="bnd-out-label" style={{ fontSize: 11 }}>not metered</text>}
                </g>
              );
            })}
            <rect x={40} y={236} width={110} height={40} className="bnd-out" />
            <text x={50} y={260} className="bnd-out-label">monitor</text>
            <rect x={170} y={236} width={110} height={40} className="bnd-out" />
            <text x={180} y={260} className="bnd-out-label">network</text>
          </svg>
          <figcaption className="small graphite">{bench.run.hardware}. Dotted parts are not metered and not claimed.</figcaption>
        </figure>
      </div>

      <H k="priorH" lang={lang} className="h-section" />
      <table ref={matrix} className={`matrix ${go ? 'go' : ''}`}>
        <thead><tr><th />{COLS.map((c) => <th key={c} scope="col">{c}</th>)}</tr></thead>
        <tbody>
          {ROWS.map(([row, marks], r) => (
            <tr key={row}>
              <th scope="row">{row}</th>
              {marks.map((m, c) => (
                <td key={c} className={c === 3 ? 'ours' : ''}>
                  <PenMark m={m} delay={c * 420 + r * 60} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="small graphite">Filled column by column. A slash means partly: for example RouteLLM prices in dollars per call, not rupees per accuracy point.</p>
    </div>
  );
}

function PenMark({ m, delay }: { m: Mark; delay: number }) {
  const label = { yes: 'yes', part: 'partly', no: 'no' }[m];
  const d = m === 'yes' ? 'M3 11 L9 17 L21 4' : m === 'part' ? 'M5 18 L19 6' : 'M8 12 L16 12';
  return (
    <svg viewBox="0 0 24 22" className={`pen pen-${m}`} role="img" aria-label={label}>
      <path d={d} pathLength={1} className="plot" style={{ animationDelay: `${delay}ms` }} />
    </svg>
  );
}
