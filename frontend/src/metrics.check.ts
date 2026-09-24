// Run: npm run check. Fails loudly if the frontier, CoQ or routing maths breaks.
import assert from 'node:assert/strict';
import { pareto, costOfQuality, route } from './metrics.ts';

const f = pareto([{ x: 1, y: 50 }, { x: 2, y: 40 }, { x: 3, y: 70 }, { x: 0.5, y: 30 }, { x: 3, y: 60 }]);
assert.deepEqual(f.map((p) => p.x), [0.5, 1, 3]);

const small = { acc: 60, acc_ci: [56, 64], inr_1k: 2, wh_q: 0.1, co2_g_1k: 71.6 };
const big = { acc: 80, acc_ci: [76, 84], inr_1k: 202, wh_q: 1.1, co2_g_1k: 787.6 };
const c = costOfQuality(big, small); // order must not matter
assert.equal(c.dAcc, 20);
assert.equal(c.inrPerPt, 10);
assert.equal(c.whPerPt, 50);
assert.ok(c.significant);
assert.ok(!costOfQuality(small, { ...small, acc: 62, acc_ci: [58, 66] }).significant);

const qs = [{ conf: 1, small_ok: true, big_ok: true }, { conf: 1 / 3, small_ok: false, big_ok: true }];
const r = route(qs, 0.5, small, big, 3);
assert.equal(r.pctSmall, 50);
assert.equal(r.acc, 100);
assert.equal(+r.inrSaved1k.toFixed(6), 202 - (6 + 6 + 202) / 2); // 3 small samples always paid
assert.equal(route(qs, 0, small, big, 3).pctSmall, 100);
console.log('metrics ok');
