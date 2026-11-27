'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const sim = require('../src/sim.js');

const data = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data.json'), 'utf8'));

// 1. Simulator determinism and invariants
const t1 = sim.makeTrace({ n: 64, rate: 4, pLong: 0.25, seed: 7 });
const t2 = sim.makeTrace({ n: 64, rate: 4, pLong: 0.25, seed: 7 });
assert.deepStrictEqual(t1, t2, 'makeTrace must be deterministic for fixed seed');

const s = sim.simStatic(t1, 8);
const c = sim.simContinuous(t1, 8);
assert(c.goodput > s.goodput * 1.2, `expected continuous (${c.goodput}) to beat static (${s.goodput})`);
assert(c.latP50 < s.latP50, 'expected continuous median latency below static');
for (const r of [s, c]) {
  assert(r.makespan > 0 && Number.isFinite(r.goodput));
  for (const it of r.items) {
    assert(it.start >= it.arrival - 1e-9, 'start before arrival');
    assert(it.decodeStart >= it.start - 1e-9, 'decode before prefill');
    assert(it.done >= it.decodeStart - 1e-9, 'done before decode');
    assert(it.hold >= it.done - 1e-9, 'hold before done');
  }
}

// 2. Prefill budget & KV memory cap invariants (Medium #9)
const burst = sim.makeTrace({ n: 64, rate: 8, pLong: 0.25, seed: 3 });
const cUnlim = sim.simContinuous(burst, 16);
const cPf1 = sim.simContinuous(burst, 16, { maxPrefills: 1 });
assert(cPf1.maxItlMs <= cUnlim.maxItlMs, `prefill budget=1 maxItlMs (${cPf1.maxItlMs}) should not exceed unlimited (${cUnlim.maxItlMs})`);

const cContig = sim.simContinuous(burst, 16, { kvSlots: 4, paged: false });
const cPaged = sim.simContinuous(burst, 16, { kvSlots: 4, paged: true });
assert(cContig.kvBlockedSteps > 0, '4-slot contiguous KV cap should block admissions at B=16');
assert(cPaged.goodput > cContig.goodput, `paged KV cap (${cPaged.goodput}) should deliver higher goodput than contiguous (${cContig.goodput})`);

// 3. Calibration sanity on closed-loop runs
for (const B of [4, 8, 16, 32]) {
  const tr = sim.makeTrace({ n: 64, closed: true, seed: 1 });
  const sm = sim.simStatic(tr, B);
  const cm = sim.simContinuous(tr, B);
  assert(cm.goodput / sm.goodput > 1.5, `B=${B}: expected continuous speedup > 1.5x`);
}

// 4. data.json completeness checks (including Stages 12, 13, 14)
for (const key of [
  'roofline', 'kv_crossover', 'batch_scaling', 'closed', 'load', 'prefill_budget',
  'mem', 'kv_breakdown', 'block_size', 'block_size_ref', 'vllm_load',
  's12_modes', 's13_http', 's13_ablation', 's14_http', 's14_ablation',
  'latency', 'capacity', 'slo',
]) {
  assert(data[key] != null, `missing data.json key: ${key}`);
}
assert.strictEqual(data.roofline.context_len, 192);
assert(data.s13_http.find((r) => r.size === '1.5B' && r.rate === 8).tpot_p50_ms < 70);
assert(data.s14_http.find((r) => r.size === '1.5B' && r.rate === 8).tpot_p50_ms < 50);
assert(data.s12_modes.find((r) => r.size === '1.5B' && r.mode === 'graph').tpot_vs_eager > 2.0);

// 5. Built HTML sanity checks
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert(!html.includes('/*__DATA__*/') && !html.includes('/*__SIM__*/'), 'unreplaced build placeholders');
assert(!html.includes('chizkidd/minivllm'), 'stale chizkidd/minivllm repo reference in built HTML');

console.log('inference-playground tests: OK');
