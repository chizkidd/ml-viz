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

// 2. Calibration sanity on closed-loop runs (ratio should match measured direction)
for (const B of [4, 8, 16, 32]) {
  const tr = sim.makeTrace({ n: 64, closed: true, seed: 1 });
  const sm = sim.simStatic(tr, B);
  const cm = sim.simContinuous(tr, B);
  assert(cm.goodput / sm.goodput > 1.5, `B=${B}: expected continuous speedup > 1.5x`);
}

// 3. data.json completeness checks
for (const key of ['roofline', 'kv_crossover', 'batch_scaling', 'closed', 'load', 'mem', 'kv_breakdown', 'block_size', 'block_size_ref', 'vllm_load', 'latency', 'capacity', 'slo']) {
  assert(data[key] != null, `missing data.json key: ${key}`);
}
assert.strictEqual(data.roofline.context_len, 192, 'Stage 3 roofline context_len should be 192');
assert(data.kv_crossover.points.length >= 8, 'expected >= 8 KV crossover points');
assert(data.kv_crossover.crossover_probe > 400 && data.kv_crossover.crossover_probe < 450);
assert(data.load.length === 14, `expected 14 Stage 4 open-loop rows, got ${data.load.length}`);
assert(data.kv_breakdown.length === 8, `expected 8 Stage 5 breakdown rows, got ${data.kv_breakdown.length}`);
assert(data.block_size.every((r) => r.step_vs_contiguous_x > 1.05), 'expected gather overhead on block_size rows');

// 4. Built HTML sanity checks
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert(!html.includes('/*__DATA__*/') && !html.includes('/*__SIM__*/'), 'unreplaced build placeholders');
assert(!html.includes('chizkidd/minivllm'), 'stale chizkidd/minivllm repo reference in built HTML');

console.log('inference-playground tests: OK');
