import type { ScreenProps } from '../App';
import { H } from '../ui';

const STEPS = [
  { h: 'Declare the boundary', f: 'boundary = GPU + CPU + RAM', p: 'Everything inside the dashed box is metered. Anything outside, the monitor or the network, is not claimed.' },
  { h: 'Measure idle', f: 'P_idle = median P over 120 s, no load', p: 'The machine sits still so its resting draw can be subtracted from every run.' },
  { h: 'Warm up, then run N queries', f: 'N = 500 per task × language × config', p: 'Twenty discarded warm-up queries first, so caches and clocks settle before anything is counted.' },
  { h: 'Integrate energy', f: 'E = Σ(P − P_idle)·Δt ÷ N', p: 'Power is sampled every 100 ms across the boundary and integrated, then divided per query.' },
  { h: 'Score with a confidence interval', f: 'CI₉₅ = bootstrap, 2,000 resamples', p: 'No number leaves the bench without its whisker. Overlapping whiskers are reported as no difference.' },
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
  return (
    <div className="screen notebook">
      <H k="methodH" lang={lang} className="h-section" />
      <div className="nb-grid">
        <ol className="nb-steps">
          {STEPS.map((s) => (
            <li key={s.h}>
              <h3>{s.h}</h3>
              <p className="formula mono">{s.f}</p>
              <p>{s.p}</p>
            </li>
          ))}
        </ol>
        <figure className="boundary">
          <svg viewBox="0 0 320 300" role="img" aria-label="Measurement boundary: a dashed box around GPU, CPU and RAM; monitor and network sit outside">
            <rect x={20} y={20} width={280} height={190} className="bnd-box" />
            <text x={28} y={14} className="bnd-label">measurement boundary: {bench.run.boundary}</text>
            {[['GPU', 40, 50, 120, 70], ['CPU', 180, 50, 100, 70], ['RAM', 40, 140, 240, 44]].map(([n, x, y, w, h]) => (
              <g key={n}>
                <rect x={+x} y={+y} width={+w} height={+h} className="bnd-part" />
                <text x={+x + 10} y={+y + 22} className="bnd-part-label">{n}</text>
              </g>
            ))}
            <rect x={40} y={236} width={110} height={40} className="bnd-out" />
            <text x={50} y={260} className="bnd-out-label">monitor</text>
            <rect x={170} y={236} width={110} height={40} className="bnd-out" />
            <text x={180} y={260} className="bnd-out-label">network</text>
          </svg>
          <figcaption className="small graphite">{bench.run.hardware}. Outside the box is not counted and not claimed.</figcaption>
        </figure>
      </div>

      <H k="priorH" lang={lang} className="h-section" />
      <table className="matrix">
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
