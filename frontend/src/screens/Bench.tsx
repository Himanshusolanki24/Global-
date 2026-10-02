import { useEffect, useMemo, useRef, useState, type PointerEvent, type KeyboardEvent } from 'react';
import { motion, useReducedMotion, useSpring } from 'framer-motion';
import type { ScreenProps } from '../App';
import { cell, fmt, shortName, type Config, type Result } from '../data';
import { Cropped, Est, H, Odometer, Whisker } from '../ui';

const NEEDLE = { stiffness: 180, damping: 14 };

// Figures quoted on the landing bench. Verify sources before publishing; see README note.
type Readout = { pre: string; v: string; unit: string; label: string; src: string; min: number; max: number; log?: boolean; mark?: number; markLabel?: string };
const READOUTS: Readout[] = [
  { pre: '', v: '945', unit: 'TWh', label: 'Data-centre electricity demand projected for 2030, about double 2024', src: 'IEA, Energy and AI, 2025', min: 0, max: 1000, mark: 415, markLabel: '2024' },
  { pre: '~', v: '30', unit: '×', label: 'Energy of a reasoning-mode answer against a standard answer to the same prompt', src: 'Hugging Face AI Energy Score, 2025', min: 1, max: 100, log: true },
  { pre: '', v: '74', unit: '%', label: 'Best score any model reached on MILU, the Indic multi-domain benchmark', src: 'MILU, AI4Bharat, 2024', min: 0, max: 100, mark: 100, markLabel: 'perfect' },
];

export default function Bench({ bench, lang, task }: ScreenProps) {
  return (
    <>
      <Cropped className="hero">
        <p className="hero-line" lang="en" data-first={lang === 'en'}>Bigger costs something.</p>
        <p className="hero-line deva" lang="hi" data-first={lang === 'hi'}>बड़ा मॉडल, बड़ा बिल।</p>
      </Cropped>
      <Readouts bench={bench} />
      <SizeDial bench={bench} lang={lang} task={task} />
    </>
  );
}

/** The bench's own headline, computed from results.json: on Q&A, how much more energy the largest local
 *  model spends than the cheapest configuration that is at least as accurate. */
function measuredReadout(bench: ScreenProps['bench']): Readout {
  const big = bench.configs.at(-1)!, rb = cell(bench, big.id, 'qa', 'en');
  const best = bench.configs.map((c) => ({ c, r: cell(bench, c.id, 'qa', 'en') })).filter((x) => x.r.acc >= rb.acc).sort((a, b) => a.r.wh_q - b.r.wh_q)[0];
  const ratio = rb.wh_q / best.r.wh_q;
  return {
    pre: '', v: ratio.toFixed(1), unit: '×', min: 1, max: 10,
    label: `Energy ${shortName(big)} spends per Q&A answer against ${shortName(best.c)}, which scored ${fmt.pct(best.r.acc)} to its ${fmt.pct(rb.acc)}`,
    src: `RightSize, ${bench.run.id} (measured)`,
  };
}

function Readouts({ bench }: { bench: ScreenProps['bench'] }) {
  // Counters start at zero and roll to the quoted figure once, on arrival.
  const [on, setOn] = useState(false);
  useEffect(() => { const id = setTimeout(() => setOn(true), 500); return () => clearTimeout(id); }, []); // after the drawer settles
  const readouts = useMemo(() => [READOUTS[0], READOUTS[1], measuredReadout(bench), READOUTS[2]], [bench]);
  return (
    <div className="readouts">
      {readouts.map((r) => (
        <figure key={r.v} className="readout">
          <div className="readout-value">
            {r.pre}<Odometer value={on ? r.v : r.v.replace(/\d/g, '0')} />
            <span className="readout-unit">{r.unit}</span>
          </div>
          <Ruler {...r} value={+r.v} />
          <figcaption>
            <p>{r.label}</p>
            <p className="spec mono">{r.src}</p>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

function Ruler({ min, max, value, log, mark, markLabel }: { min: number; max: number; value: number; log?: boolean; mark?: number; markLabel?: string }) {
  const pos = (v: number) => (log ? Math.log(v / min) / Math.log(max / min) : (v - min) / (max - min)) * 100;
  const ticks = log ? [1, 2, 5, 10, 20, 50, 100] : Array.from({ length: 21 }, (_, i) => min + ((max - min) * i) / 20);
  return (
    <svg className="ruler" viewBox="0 0 100 14" preserveAspectRatio="none" aria-hidden>
      {ticks.map((v, i) => <line key={v} x1={pos(v)} x2={pos(v)} y1={0} y2={log || i % 5 === 0 ? 6 : 3} vectorEffect="non-scaling-stroke" />)}
      {mark !== undefined && <line className="ruler-ref" x1={pos(mark)} x2={pos(mark)} y1={0} y2={14} vectorEffect="non-scaling-stroke"><title>{markLabel}</title></line>}
      <rect className="ruler-mark" x={pos(value) - 0.6} width={1.2} y={0} height={14} />
    </svg>
  );
}

/* ---------------- size dial ---------------- */

const SWEEP = 270;
const angleOf = (i: number, n: number) => -SWEEP / 2 + (i * SWEEP) / (n - 1);
const polar = (r: number, deg: number) => [r * Math.sin((deg * Math.PI) / 180), -r * Math.cos((deg * Math.PI) / 180)] as const;

function SizeDial({ bench, lang, task }: ScreenProps) {
  const configs = bench.configs;
  const cells = useMemo(() => configs.map((c) => cell(bench, c.id, task, lang)), [bench, configs, task, lang]);
  const [i, setI] = useState(() => configs.findIndex((c) => c.id === bench.router.small));
  const [floor, setFloor] = useState(65);
  const pass = cells.map((c) => c.acc >= floor);
  const rightsize = pass.findIndex(Boolean);
  const c = configs[i], r = cells[i];
  const broke = !pass[i];

  return (
    <section className="dial-bench" aria-labelledby="dial-h">
      <div className="dial-head">
        <H k="dial" lang={lang} className="h-section" />
        <span className="mono small graphite" id="dial-h">
          {bench.run.id} · {c.quant} · {bench.run.hardware.split(' + ')[0]} · boundary: {c.estimated ? 'provider (est.)' : bench.run.boundary}
        </span>
      </div>

      <div className="dial-grid">
        <div className="dial-col">
          <Knob configs={configs} pass={pass} rightsize={rightsize} i={i} setI={setI} />
          <p className="dial-verdict">
            {rightsize === -1
              ? `No configuration reaches ${floor}% on this task.`
              : rightsize === 0
                ? `Even the smallest configuration clears ${floor}%.`
                : `Quality breaks below ${shortName(configs[rightsize])}. That is the right size for a ${floor}% floor.`}
          </p>
          <label className="field field-range">
            <span>Your accuracy floor <b className="mono">{floor}%</b></span>
            <input type="range" min={30} max={95} value={floor} onChange={(e) => setFloor(+e.target.value)} />
          </label>
        </div>

        <Gauge r={r} floor={floor} broke={broke} />
        <MeterDisc r={r} est={c.estimated} maxWh={Math.max(...cells.map((x) => x.wh_q))} />
        <Trace key={`${c.id}${task}${lang}`} r={r} id={c.id} yMax={Math.ceil((r.ttft_p95_ms * 1.6) / 50) * 50} />
        <Money r={r} c={c} />
      </div>

      <p className="sr-only" aria-live="polite">
        {shortName(c)}: accuracy {fmt.pct(r.acc)}, {broke ? 'below' : 'above'} your {floor}% floor. Latency p95 {r.ttft_p95_ms} ms. Energy {r.wh_q} Wh per query. Cost {fmt.inr(r.inr_1k)} per 1,000 queries.
      </p>
    </section>
  );
}

function Knob({ configs, pass, rightsize, i, setI }: { configs: Config[]; pass: boolean[]; rightsize: number; i: number; setI: (n: number) => void }) {
  const n = configs.length;
  const rot = useSpring(angleOf(i, n), NEEDLE);
  useEffect(() => rot.set(angleOf(i, n)), [i, n, rot]);
  const drag = useRef(false);

  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const b = e.currentTarget.getBoundingClientRect();
    const deg = (Math.atan2(e.clientX - (b.left + b.width / 2), -(e.clientY - (b.top + b.height / 2))) * 180) / Math.PI;
    const clamped = Math.max(-SWEEP / 2, Math.min(SWEEP / 2, deg));
    setI(Math.round(((clamped + SWEEP / 2) / SWEEP) * (n - 1)));
  };
  const key = (e: KeyboardEvent) => {
    const step = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 3, PageDown: -3 }[e.key];
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : step !== undefined ? i + step : null;
    if (next === null) return;
    e.preventDefault();
    setI(Math.max(0, Math.min(n - 1, next)));
  };

  // group labels sit at the middle tick of each model family
  const groups = configs.reduce<{ label: string; at: number[] }[]>((g, c, k) => {
    const label = c.params_b ? `${c.params_b}B` : 'API';
    const last = g[g.length - 1];
    last?.label === label ? last.at.push(k) : g.push({ label, at: [k] });
    return g;
  }, []);
  const teeth = Array.from({ length: 72 }, (_, k) => polar(k % 2 ? 96 : 102, k * 5).join(',')).join(' ');
  const breakDeg = rightsize > 0 ? (angleOf(rightsize - 1, n) + angleOf(rightsize, n)) / 2 : null;

  return (
    <svg
      className="knob"
      viewBox="-330 -236 660 440"
      role="slider"
      tabIndex={0}
      aria-label="Size dial"
      aria-valuemin={0}
      aria-valuemax={n - 1}
      aria-valuenow={i}
      aria-valuetext={shortName(configs[i])}
      onKeyDown={key}
      onPointerDown={(e) => { drag.current = true; e.currentTarget.setPointerCapture(e.pointerId); pick(e); }}
      onPointerMove={(e) => drag.current && pick(e)}
      onPointerUp={() => (drag.current = false)}
    >
      {configs.map((c, k) => {
        const a = angleOf(k, n);
        const len = c.quant === 'FP16' || c.quant === 'API' ? 22 : c.quant === 'Q8' ? 15 : 9;
        const [x1, y1] = polar(122, a), [x2, y2] = polar(122 + len, a);
        return <line key={c.id} x1={x1} y1={y1} x2={x2} y2={y2} className={`tick ${pass[k] ? '' : 'tick-fail'} ${k === i ? 'tick-on' : ''}`} />;
      })}
      {groups.map((g) => {
        const [x, y] = polar(166, angleOf(g.at[Math.floor(g.at.length / 2)], n));
        return <text key={g.label} x={x} y={y} className="knob-label" textAnchor="middle" dominantBaseline="middle">{g.label}</text>;
      })}
      {breakDeg !== null && (() => {
        const [x1, y1] = polar(104, breakDeg), [x2, y2] = polar(192, breakDeg);
        return (
          <g className="break">
            <line x1={x1} y1={y1} x2={x2} y2={y2} />
            <text x={x2 + (x2 < 0 ? -4 : 4)} y={y2 - 4} textAnchor={x2 < 0 ? 'end' : 'start'}>quality breaks here</text>
          </g>
        );
      })()}
      <motion.g style={{ rotate: rot }}>
        <polygon points={teeth} className="knob-body" />
        <circle r={78} className="knob-face" />
        <line x1={0} y1={-40} x2={0} y2={-92} className="knob-pointer" />
      </motion.g>
      <text className="knob-readout" textAnchor="middle" y={-4}>{configs[i].params_b ? `${configs[i].params_b}B` : configs[i].name}</text>
      <text className="knob-readout-sub" textAnchor="middle" y={16}>{configs[i].quant}</text>
      <text className="knob-ends" x={polar(150, -SWEEP / 2)[0] - 8} y={polar(150, -SWEEP / 2)[1] + 44} textAnchor="middle">smallest</text>
      <text className="knob-ends" x={polar(150, SWEEP / 2)[0] + 8} y={polar(150, SWEEP / 2)[1] + 44} textAnchor="middle">frontier</text>
    </svg>
  );
}

function Gauge({ r, floor, broke }: { r: Result; floor: number; broke: boolean }) {
  const deg = (p: number) => -90 + (p / 100) * 180;
  const rot = useSpring(deg(r.acc), NEEDLE);
  useEffect(() => rot.set(deg(r.acc)), [r.acc, rot]);
  const arc = (a: number, b: number, rad: number) => {
    const [x1, y1] = polar(rad, deg(a)), [x2, y2] = polar(rad, deg(b));
    return `M${x1} ${y1} A${rad} ${rad} 0 0 1 ${x2} ${y2}`;
  };
  const [fx1, fy1] = polar(84, deg(floor)), [fx2, fy2] = polar(124, deg(floor));
  return (
    <div className={`inst inst-gauge ${broke ? 'broke' : ''}`} key={broke ? 'b' : 'ok'}>
      <p className="inst-label">Accuracy, 95% CI</p>
      <svg viewBox="-130 -118 260 136" aria-hidden>
        {Array.from({ length: 51 }, (_, k) => {
          const [x1, y1] = polar(100, deg(k * 2)), [x2, y2] = polar(k % 5 ? 94 : 88, deg(k * 2));
          return <line key={k} x1={x1} y1={y1} x2={x2} y2={y2} className="gauge-tick" />;
        })}
        {[0, 50, 100].map((p) => { const [x, y] = polar(112, deg(p)); return <text key={p} x={x} y={y} className="gauge-num" textAnchor="middle">{p}</text>; })}
        <path d={arc(r.acc_ci[0], r.acc_ci[1], 104)} className="gauge-ci" />
        <line x1={fx1} y1={fy1} x2={fx2} y2={fy2} className="gauge-floor" />
        <motion.g style={{ rotate: rot }}>
          <circle r={90} fill="none" />
          <line x1={0} y1={12} x2={0} y2={-90} className="gauge-needle" />
          <circle r={5} className="gauge-hub" />
        </motion.g>
      </svg>
      <div className="inst-read">
        <Odometer value={r.acc.toFixed(1).padStart(4, '0')} className="big-read" /><span className="unit">%</span>
        <Whisker v={r.acc} lo={r.acc_ci[0]} hi={r.acc_ci[1]} min={0} max={100} />
        <span className="mono small graphite">[{r.acc_ci[0]}, {r.acc_ci[1]}]</span>
      </div>
      {broke && <p className="broke-note">Below your {floor}% floor. Quality breaks here.</p>}
    </div>
  );
}

function MeterDisc({ r, est, maxWh }: { r: Result; est: boolean; maxWh: number }) {
  const disc = useRef<SVGGElement>(null);
  const reduced = useReducedMotion();
  const speed = useRef(0);
  speed.current = r.wh_q * 400; // °/s per Wh/query: the disc turns faster as each answer costs more

  useEffect(() => {
    if (reduced) return;
    let a = 0, last = performance.now(), id = 0;
    const tick = (now: number) => {
      a = (a + ((now - last) / 1000) * speed.current) % 360;
      last = now;
      if (disc.current) disc.current.style.transform = `rotate(${a}deg)`;
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [reduced]);

  return (
    <div className="inst inst-disc">
      <p className="inst-label">Energy per query {est && <Est />}</p>
      <svg viewBox="-80 -80 160 160" aria-hidden className={est ? 'disc-est' : ''}>
        <g ref={disc} className="disc">
          <circle r={70} className="disc-plate" />
          {Array.from({ length: 60 }, (_, k) => { const [x1, y1] = polar(70, k * 6), [x2, y2] = polar(k % 5 ? 66 : 62, k * 6); return <line key={k} x1={x1} y1={y1} x2={x2} y2={y2} className="disc-grad" />; })}
          <path d={`M0 0 L${polar(70, -9).join(' ')} A70 70 0 0 1 ${polar(70, 9).join(' ')} Z`} className="disc-mark" />
        </g>
        <circle r={8} className="disc-spindle" />
      </svg>
      <div className="inst-read">
        <Odometer value={r.wh_q.toFixed(3)} className="big-read" /><span className="unit">Wh</span>
        <Whisker v={r.wh_q} lo={r.wh_ci[0]} hi={r.wh_ci[1]} min={0} max={Math.max(maxWh, r.wh_ci[1])} est={est} />
        <span className="mono small graphite">{r.mwh_tok} mWh/token</span>
      </div>
      {reduced && <p className="mono small graphite">Disc stopped (reduced motion). Speed would be {Math.round(speed.current)}°/s.</p>}
    </div>
  );
}

// Deterministic per-query TTFT samples shaped around the measured p95. A trace, not a claim.
function samples(id: string, p95: number) {
  let s = [...id].reduce((a, ch) => a * 31 + ch.charCodeAt(0), 7) >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  return Array.from({ length: 96 }, () => { const u = rnd(); return p95 * (u > 0.95 ? 1 + rnd() * 0.25 : 0.55 + u * 0.4); });
}

function Trace({ r, id, yMax }: { r: Result; id: string; yMax: number }) {
  const W = 600, Hh = 150;
  const pts = samples(id, r.ttft_p95_ms);
  const y = (ms: number) => Hh - (ms / yMax) * Hh;
  const d = `M0 ${y(pts[0]).toFixed(1)} ` + pts.map((v, k) => `H${(k * W) / pts.length} V${y(v).toFixed(1)}`).join(' ') + ` H${W}`;
  return (
    <div className="inst inst-trace">
      <p className="inst-label">Time to first token, 96 queries</p>
      <svg viewBox={`0 0 ${W} ${Hh}`} preserveAspectRatio="none" aria-hidden>
        {Array.from({ length: 11 }, (_, k) => <line key={`v${k}`} x1={(k * W) / 10} x2={(k * W) / 10} y1={0} y2={Hh} className="graticule" vectorEffect="non-scaling-stroke" />)}
        {Array.from({ length: 5 }, (_, k) => <line key={`h${k}`} x1={0} x2={W} y1={(k * Hh) / 4} y2={(k * Hh) / 4} className="graticule" vectorEffect="non-scaling-stroke" />)}
        <line x1={0} x2={W} y1={y(r.ttft_p95_ms)} y2={y(r.ttft_p95_ms)} className="p95" vectorEffect="non-scaling-stroke" />
        <path d={d} pathLength={1} className="trace plot" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="inst-read">
        <Odometer value={String(r.ttft_p95_ms).padStart(4, '0')} className="big-read" /><span className="unit">ms p95</span>
        <span className="mono small graphite">{yMax / 4} ms/div, auto-ranged</span>
      </div>
    </div>
  );
}

function Money({ r, c }: { r: Result; c: Config }) {
  const ei = r.acc / (r.wh_q * 1000);
  return (
    <div className="inst inst-money">
      <p className="inst-label">Cost per 1,000 queries</p>
      <div className="money-read">
        <span className="rupee">₹</span>
        <Odometer value={r.inr_1k.toFixed(2).padStart(7, '0')} className="money-odo" />
      </div>
      <dl className="spec-strip mono">
        <div><dt>memory</dt><dd>{c.mem_gb === null ? 'n/a (API)' : `${c.mem_gb} GB`}</dd></div>
        <div><dt>carbon</dt><dd>{fmt.wh(r.co2_g_1k)} g CO₂e/1k {c.estimated && <Est />}</dd></div>
        <div><dt>EI = Acc ÷ Wh/1k</dt><dd>{ei.toFixed(3)}</dd></div>
      </dl>
    </div>
  );
}
