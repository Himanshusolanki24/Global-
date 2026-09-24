import type { ReactNode } from 'react';
import { t, T, type Lang } from './data';

/** Mechanical counter: each digit is a drum that rolls; drums settle right-to-left, 30ms apart. */
export function Odometer({ value, className = '' }: { value: string; className?: string }) {
  const chars = [...value];
  return (
    <span className={`odo ${className}`}>
      <span className="sr-only">{value}</span>
      <span aria-hidden>
        {chars.map((c, i) => {
          const fromRight = chars.length - 1 - i;
          if (!/\d/.test(c)) return <span key={`c${fromRight}`}>{c}</span>;
          return (
            <span key={fromRight} className="odo-d">
              <span className="odo-col" style={{ transform: `translateY(${-1.2 * +c}em)`, transitionDelay: `calc(var(--digit-stagger) * ${fromRight})` }}>
                {'0123456789'.split('').map((d) => <span key={d}>{d}</span>)}
              </span>
            </span>
          );
        })}
      </span>
    </span>
  );
}

/** 95% CI as a whisker on a fixed scale. Estimates get a dotted, hatched body. */
export function Whisker({ v, lo, hi, min, max, est = false, w = 120 }: { v: number; lo: number; hi: number; min: number; max: number; est?: boolean; w?: number }) {
  const x = (n: number) => ((Math.min(max, Math.max(min, n)) - min) / (max - min)) * w;
  return (
    <svg width={w} height={12} className="whisker" role="img" aria-label={`95% CI ${lo} to ${hi}`}>
      <line x1={0} x2={w} y1={6} y2={6} className="whisker-scale" />
      <rect x={x(lo)} y={3} width={Math.max(1, x(hi) - x(lo))} height={6} className={est ? 'whisker-est' : 'whisker-body'} />
      <line x1={x(lo)} x2={x(lo)} y1={1} y2={11} className="whisker-cap" />
      <line x1={x(hi)} x2={x(hi)} y1={1} y2={11} className="whisker-cap" />
      <line x1={x(v)} x2={x(v)} y1={0} y2={12} className="whisker-v" />
    </svg>
  );
}

export const Est = () => <span className="est" title="Estimated from provider disclosures, not measured on this bench">est.</span>;

/** Heading that re-sets in Devanagari when the bench language is Hindi. */
export function H({ k, lang, as: Tag = 'h2', className = '' }: { k: keyof typeof T; lang: Lang; as?: 'h1' | 'h2' | 'h3'; className?: string }) {
  return <Tag lang={lang} className={`heading ${className}`}>{t(k, lang)}</Tag>;
}

/** Loading state: a tick scale warming up, left to right. Never a grey shimmer bar. */
export function Calibrating({ label = 'Calibrating' }: { label?: string }) {
  return (
    <div className="calibrating" role="status">
      <div className="calibrating-ticks" aria-hidden>
        {Array.from({ length: 41 }, (_, i) => <span key={i} style={{ animationDelay: `${i * 24}ms`, height: i % 10 === 0 ? 18 : i % 5 === 0 ? 12 : 7 }} />)}
      </div>
      <span className="mono small">{label}…</span>
    </div>
  );
}

/** Four crop marks just outside a block, like a print proof. */
export const Cropped = ({ children, className = '' }: { children: ReactNode; className?: string }) => (
  <div className={`cropped ${className}`}>
    <i className="crop tl" /><i className="crop tr" /><i className="crop bl" /><i className="crop br" />
    {children}
  </div>
);
