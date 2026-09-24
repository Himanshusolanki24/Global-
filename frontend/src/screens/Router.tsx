import { useEffect, useMemo, useRef, useState } from 'react';
import { useReducedMotion } from 'framer-motion';
import type { ScreenProps } from '../App';
import { cell, fmt, shortName, TASKS, type Query } from '../data';
import { route } from '../metrics';
import { H, Odometer } from '../ui';

const STUB_W = 150, SPEED = 280, G = 2200, GATE_HOLD = 0.35, TRAY_KEEP = 9;
type Flight = { q: Query; key: number; x: number; y: number; vx: number; vy: number; phase: 'rail' | 'gate' | 'drop'; t: number; small: boolean };
type Done = { q: Query; key: number; small: boolean };

const confLabel = (c: number) => (c > 0.9 ? '3/3' : c > 0.5 ? '2/3' : '1/3');

export default function Router({ bench, lang, task }: ScreenProps) {
  const { queries, samples } = bench.router;
  const small = cell(bench, bench.router.small, task, lang), big = cell(bench, bench.router.big, task, lang);
  const smallC = bench.configs.find((c) => c.id === bench.router.small)!, bigC = bench.configs.find((c) => c.id === bench.router.big)!;
  const reduced = useReducedMotion();

  const [tau, setTau] = useState(0.6);
  const [paused, setPaused] = useState(false);
  const [done, setDone] = useState<Done[]>([]);
  const [flights, setFlights] = useState<Flight[]>([]);
  const [atGate, setAtGate] = useState<Query | null>(null);

  const tauRef = useRef(tau);
  tauRef.current = tau;
  const rail = useRef<HTMLDivElement>(null);
  const els = useRef(new Map<number, HTMLDivElement>());
  const fl = useRef<Flight[]>([]);
  const seq = useRef(0);

  // τ changed: start a fresh tally. With motion off the whole sample is routed at once.
  useEffect(() => {
    fl.current = [];
    setFlights([]);
    setDone(reduced ? queries.map((q, k) => ({ q, key: k, small: q.conf >= tau })) : []);
  }, [tau, reduced, queries]);

  useEffect(() => {
    if (reduced || paused) return;
    let last = performance.now(), spawnIn = 0, id = 0;
    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const box = rail.current!;
      const w = box.clientWidth, h = box.clientHeight;
      const railY = h * 0.5 - 23, gateX = w * 0.55 - STUB_W / 2, smallY = h - 60, bigY = 14;

      spawnIn -= dt;
      if (spawnIn <= 0 && fl.current.length < 3) {
        const q = queries[seq.current % queries.length];
        fl.current.push({ q, key: seq.current++, x: -STUB_W, y: railY, vx: SPEED, vy: 0, phase: 'rail', t: 0, small: true });
        spawnIn = 0.9;
        setFlights([...fl.current]);
      }

      const landed: Flight[] = [];
      for (const f of fl.current) {
        if (f.phase === 'rail') {
          f.x += f.vx * dt;
          const ahead = fl.current.find((o) => o !== f && o.x > f.x && o.phase !== 'drop');
          if (ahead && ahead.x - f.x < STUB_W + 12) f.x = ahead.x - STUB_W - 12; // queue behind the stub at the gate
          if (f.x >= gateX) (f.x = gateX), (f.phase = 'gate'), (f.t = 0), setAtGate(f.q);
        } else if (f.phase === 'gate') {
          f.t += dt;
          if (f.t >= GATE_HOLD) {
            f.small = f.q.conf >= tauRef.current;
            f.phase = 'drop';
            f.vx = 90;
            f.vy = f.small ? 0 : -120;
            const el = els.current.get(f.key);
            if (el && !f.small) {
              el.classList.add('esc', 'flash');
              requestAnimationFrame(() => el.classList.remove('flash')); // one frame of --warn
            }
          }
        } else {
          f.vy += (f.small ? G : -G) * dt; // small falls, escalation is pulled up to the big tray
          f.x += f.vx * dt;
          f.y += f.vy * dt;
          if ((f.small && f.y >= smallY) || (!f.small && f.y <= bigY)) landed.push(f);
        }
        const el = els.current.get(f.key);
        if (el) el.style.transform = `translate3d(${f.x}px, ${f.y}px, 0)`;
      }
      if (landed.length) {
        fl.current = fl.current.filter((f) => !landed.includes(f));
        setFlights([...fl.current]);
        setDone((d) => [...d, ...landed.map((f) => ({ q: f.q, key: f.key, small: f.small }))]);
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [reduced, paused, queries]);

  const live = route(done.map((d) => d.q), tau, small, big, samples);
  const full = useMemo(() => route(queries, tau, small, big, samples), [queries, tau, small, big, samples]);
  const allBig = (100 * queries.filter((q) => q.big_ok).length) / queries.length;
  const trays = { small: done.filter((d) => d.small).slice(-TRAY_KEEP), big: done.filter((d) => !d.small).slice(-TRAY_KEEP) };

  return (
    <div className="screen">
      <div className="screen-head">
        <H k="routerH" lang={lang} className="h-section" />
        <p className="formula mono">route = small if conf ≥ τ, else big · conf = agreement of {samples} samples from {shortName(smallC)}</p>
      </div>

      <div className="router-controls">
        <label className="field field-range">
          <span>Threshold τ <b className="mono">{tau.toFixed(2)}</b></span>
          <input type="range" min={0.3} max={1} step={0.01} value={tau} list="conf-levels" onChange={(e) => setTau(+e.target.value)} />
          <datalist id="conf-levels"><option value={1 / 3} /><option value={2 / 3} /><option value={1} /></datalist>
          <span className="small graphite">conf can only be 1/3, 2/3 or 3/3, so the result moves in three steps.</span>
        </label>
        {!reduced && <button className="btn-quiet" aria-pressed={paused} onClick={() => setPaused(!paused)}>{paused ? 'Resume feed' : 'Pause feed'}</button>}
      </div>

      <div className="sorter">
        <div className="tray tray-big">
          <span className="tray-label">{shortName(bigC)} · escalated</span>
          <div className="tray-stack">{trays.big.map((d) => <Stub key={d.key} q={d.q} esc />)}</div>
        </div>

        <div className="rail" ref={rail} aria-hidden>
          <div className="rail-track" />
          <div className="gate">
            <div className="gate-meter">
              <span className="gate-tau" style={{ left: `${tau * 100}%` }} />
              {atGate && <span className={`gate-conf ${atGate.conf >= tau ? '' : 'gate-conf-low'}`} style={{ left: `${atGate.conf * 100}%` }} />}
            </div>
            <span className="mono small">{atGate ? `${atGate.id} conf ${confLabel(atGate.conf)} ${atGate.conf >= tau ? '≥' : '<'} τ` : 'gate idle'}</span>
          </div>
          {flights.map((f) => (
            <div key={f.key} ref={(el) => void (el ? els.current.set(f.key, el) : els.current.delete(f.key))} className="stub-fly" style={{ transform: `translate3d(${f.x}px, ${f.y}px, 0)` }}>
              <Stub q={f.q} />
            </div>
          ))}
        </div>

        <div className="tray tray-small">
          <span className="tray-label">{shortName(smallC)} · kept local</span>
          <div className="tray-stack">{trays.small.map((d) => <Stub key={d.key} q={d.q} />)}</div>
        </div>
      </div>

      <div className="tally" aria-live="polite">
        <div><span className="inst-label">Routed small</span><Odometer value={live.pctSmall.toFixed(0).padStart(3, '0')} className="big-read" /><span className="unit">%</span></div>
        <div><span className="inst-label">₹ saved per 1k vs always-big</span><span className="big-read">₹</span><Odometer value={Math.max(0, live.n ? live.inrSaved1k : 0).toFixed(1).padStart(5, '0')} className="big-read" /></div>
        <div><span className="inst-label">Wh saved per 1k (est.)</span><Odometer value={Math.max(0, live.n ? live.whSaved1k : 0).toFixed(0).padStart(4, '0')} className="big-read" /></div>
        <div><span className="inst-label">Accuracy, routed</span><Odometer value={(live.n ? live.acc : 0).toFixed(1).padStart(4, '0')} className="big-read" /><span className="unit">%</span></div>
      </div>
      <p className="small graphite tally-note">
        Live tally over {live.n} stubs. Full sample of {queries.length} at τ {tau.toFixed(2)}: {full.pctSmall.toFixed(0)}% small, accuracy {fmt.pct(full.acc)} against {fmt.pct(allBig)} always-big, {fmt.inr(full.inrSaved1k)} saved per 1k. Priced at {TASKS[task]} rates; escalations also pay for the {samples} small samples.
      </p>
    </div>
  );
}

function Stub({ q, esc = false }: { q: Query; esc?: boolean }) {
  return (
    <div className={`stub ${esc ? 'esc' : ''}`} lang={/[ऀ-ॿ]/.test(q.text) ? 'hi' : 'en'}>
      <span className="stub-id mono">{q.id} · {confLabel(q.conf)}</span>
      <span className="stub-text">{q.text}</span>
    </div>
  );
}
