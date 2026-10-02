import { lazy, Suspense, useEffect, useState } from 'react';
import { AnimatePresence, motion, MotionConfig } from 'framer-motion';
import { loadBench, t, TASKS, fmt, type Bench, type Lang } from './data';
import { Calibrating } from './ui';

const screens = {
  bench: lazy(() => import('./screens/Bench')),
  frontier: lazy(() => import('./screens/Frontier')),
  receipt: lazy(() => import('./screens/Receipt')),
  router: lazy(() => import('./screens/Router')),
  method: lazy(() => import('./screens/Method')),
};
type Screen = keyof typeof screens;
const readHash = (): Screen => (location.hash.slice(1) in screens ? (location.hash.slice(1) as Screen) : 'bench');

export type ScreenProps = { bench: Bench; lang: Lang; task: string };

export default function App() {
  const [bench, setBench] = useState<Bench | null>(null);
  const [screen, setScreen] = useState<Screen>(readHash);
  const [lang, setLang] = useState<Lang>('en');
  const [task, setTask] = useState('qa');
  const [dark, setDark] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => void loadBench().then(setBench, (e: Error) => setError(e.message)), []);
  useEffect(() => {
    const on = () => setScreen(readHash());
    addEventListener('hashchange', on);
    return () => removeEventListener('hashchange', on);
  }, []);
  useEffect(() => void (document.documentElement.dataset.theme = dark ? 'dark' : 'light'), [dark]);

  const Current = screens[screen];
  return (
    <MotionConfig reducedMotion="user">
      {bench?.sample && (
        <p className="tape" role="note">
          <span>Sample data, not results. Every number here is invented until a real run's results.json is loaded.</span>
        </p>
      )}
      <header className="bar">
        <div className="bar-id">
          <span className="wordmark">RightSize</span>
          {bench && (
            <span className="mono small graphite">
              {bench.run.id} · {bench.run.hardware} · boundary: {bench.run.boundary} · n={bench.run.n_per_cell}/cell · last measured {fmt.when(bench.run.measured_at)}
            </span>
          )}
        </div>
        <nav className="bar-nav" aria-label="Screens">
          {(Object.keys(screens) as Screen[]).map((s) => (
            <a key={s} href={`#${s}`} lang={lang} aria-current={s === screen ? 'page' : undefined}>{t(s, lang)}</a>
          ))}
        </nav>
        <div className="bar-controls">
          <label className="field">
            <span className="small graphite">Task</span>
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              {Object.entries(TASKS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          <div className="seg" role="radiogroup" aria-label="Language">
            {(['en', 'hi'] as const).map((l) => (
              <button key={l} role="radio" aria-checked={lang === l} onClick={() => setLang(l)} lang={l}>{l === 'en' ? 'EN' : 'हिं'}</button>
            ))}
          </div>
          <button className="btn-quiet" aria-pressed={dark} onClick={() => setDark(!dark)}>{dark ? 'Lights on' : 'Lights off'}</button>
        </div>
      </header>
      <main>
        {error ? <p className="tape" role="alert"><span>Could not load results: {error}</span></p> : !bench ? <Calibrating label="Loading bench" /> : (
          <AnimatePresence mode="wait" initial={false}>
            <motion.section
              key={screen}
              className="drawer"
              initial={{ y: 40, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -16, opacity: 0 }}
              transition={{ duration: 0.48, ease: [0.22, 1, 0.36, 1] }}
            >
              <div className="scan" aria-hidden />
              <Suspense fallback={<Calibrating />}>
                <Current bench={bench} lang={lang} task={task} />
              </Suspense>
            </motion.section>
          </AnimatePresence>
        )}
      </main>
    </MotionConfig>
  );
}
