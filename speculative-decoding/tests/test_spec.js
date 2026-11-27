'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const sim = require('../src/sim.js');

const rootDir = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(rootDir, 'data.json'), 'utf8'));
const html = fs.readFileSync(path.join(rootDir, 'index.html'), 'utf8');

// 1. Exact geometric acceptance & speedup law
for (let k = 1; k <= 12; k++) {
  assert.strictEqual(sim.expectedTokens(1, k), k + 1, `alpha=1 must yield k+1 for k=${k}`);
  assert.strictEqual(sim.expectedTokens(0, k), 1, `alpha=0 must yield 1 for k=${k}`);
}
assert.strictEqual(sim.expectedTokens(0.85, 0), 1, 'k=0 must yield 1');

for (const alpha of [0.2, 0.5, 0.75, 0.9, 0.99]) {
  for (const k of [1, 2, 4, 6, 8]) {
    let direct = 0;
    for (let i = 0; i <= k; i++) direct += Math.pow(alpha, i);
    const closed = sim.expectedTokens(alpha, k);
    assert(Math.abs(closed - direct) < 1e-11, `closed (${closed}) vs direct (${direct}) for alpha=${alpha}, k=${k}`);
    for (const c of [0, 0.27, 0.84]) {
      for (const v of [1.0, 1.09, 1.94]) {
        const expectedSpd = closed / (k * c + v);
        const gotSpd = sim.specSpeedup(alpha, k, c, v);
        assert(Math.abs(gotSpd - expectedSpd) < 1e-12, 'specSpeedup mismatch');
      }
    }
  }
}

// Measured regime checks via speedup law
assert(sim.specSpeedup(0.885, 4, 0.835, 1.096) < 1.0, 'Stage 7 eager 0.5B draft on 1.5B must be < 1.0x');
assert(sim.specSpeedup(0.90, 4, 0.275, 1.09) > 1.5, 'Stage 7 CUDA-graph 0.5B draft on 3B must win > 1.5x');
const fp16Spd = sim.specSpeedup(0.987, 4, 0, 1.013);
const int8Spd = sim.specSpeedup(0.987, 4, 0, 1.943);
assert(int8Spd < fp16Spd * 0.55, 'High int8 verify cost v=1.943 must nearly halve speculation speedup');

// optimalK and breakEvenAlpha
const opt = sim.optimalK(0.8, 0.25, 1.05, 12);
assert(opt.k >= 1 && opt.k <= 12 && opt.speedup > 1.2, 'optimalK must find profitable k');
for (let kk = 1; kk <= 12; kk++) {
  assert(opt.speedup >= sim.specSpeedup(0.8, kk, 0.25, 1.05) - 1e-12, 'optimalK must maximize speedup');
}
const aMin = sim.breakEvenAlpha(4, 0.27, 1.09);
assert(aMin !== null && aMin > 0 && aMin < 1, 'breakEvenAlpha must exist for c=0.27, v=1.09');
assert(Math.abs(sim.specSpeedup(aMin, 4, 0.27, 1.09) - 1.0) < 1e-6, 'breakEvenAlpha must hit 1.0x');
assert.strictEqual(sim.breakEvenAlpha(4, 1.0, 2.0), null, 'impossible break-even must return null');

// 2. Trace simulator determinism, parity, and rollback invariants
const tA = sim.simulateTrace({ workload: 'code', proposer: 'ngram', k: 4, seed: 42 });
const tB = sim.simulateTrace({ workload: 'code', proposer: 'ngram', k: 4, seed: 42 });
assert.deepStrictEqual(tA, tB, 'simulateTrace must be deterministic for fixed seed');

for (const workload of ['copy', 'code', 'open']) {
  for (const proposer of ['ngram', 'draft_0p5b', 'int8_self']) {
    for (const k of [1, 4, 8]) {
      const tr = sim.simulateTrace({ workload, proposer, k, seed: 11, maxTokens: 20 });
      assert.strictEqual(tr.exactParity, true, `parity failed for ${workload}/${proposer}/k=${k}`);
      assert.deepStrictEqual(tr.emitted, tr.greedyReference, 'emitted tokens must equal greedy reference');
      assert(tr.totalRounds >= 1 && tr.totalEmitted >= 12);

      let sumAcc = 0, sumRej = 0, sumDraft = 0, sumEmit = 0;
      for (const r of tr.rounds) {
        assert.strictEqual(r.acceptedCount + r.rejectedCount, r.k, 'accepted + rejected must equal round k');
        assert.strictEqual(r.emittedThisRound.length, r.acceptedCount + 1, 'each round emits accepted + 1 bonus/correction');
        assert.strictEqual(r.kvPeak, r.kvBefore + r.k, 'kvPeak must be kvBefore + k');
        assert.strictEqual(r.kvAfter, r.kvBefore + r.acceptedCount + 1, 'kvAfter rollback pointer must equal kvBefore + accepted + 1');
        // Prefix invariant: accepted tokens first, rejected suffix after
        for (let i = 0; i < r.k; i++) {
          const want = i < r.acceptedCount ? 'accepted' : 'rejected';
          assert.strictEqual(r.statuses[i], want, 'statuses must be accepted prefix then rejected suffix');
        }
        sumAcc += r.acceptedCount;
        sumRej += r.rejectedCount;
        sumDraft += r.k;
        sumEmit += r.emittedThisRound.length;
      }
      assert.strictEqual(sumAcc, tr.totalAccepted);
      assert.strictEqual(sumRej, tr.totalRejected);
      assert.strictEqual(sumDraft, tr.totalDrafted);
      assert.strictEqual(sumEmit, tr.totalEmitted);
      assert.strictEqual(tr.totalBonus, tr.totalRounds);
    }
  }
}

// Perfect alpha=1 check: int8_self on copy has alpha=1.0 -> zero rollbacks
const perfect = sim.simulateTrace({ workload: 'copy', proposer: 'int8_self', k: 4, seed: 3, maxTokens: 20 });
assert.strictEqual(perfect.totalRejected, 0, 'alpha=1.0 must have zero rollbacks');
assert(perfect.rounds.every((r) => r.bonusKind === 'bonus'), 'alpha=1.0 rounds must all earn bonus');

// 3. data.json completeness & real Tesla T4 data integrity checks
assert.strictEqual(data.gpu, 'Tesla T4');
for (const key of ['stage6', 'stage7', 'stage8', 'stage9', 'stage10', 'stage11']) {
  assert(data[key] != null, `missing data.json key: ${key}`);
}
assert(data.stage6.verify_cost.length === 24, 'expected 24 stage6 verify_cost rows');
assert(data.stage6.speculative.length === 12, 'expected 12 stage6 speculative rows');
assert(data.stage6.oracle.length === 3, 'expected 3 stage6 oracle rows');
assert(data.stage6.projection.length === 36, 'expected 36 stage6 projection rows');

const c15Eager = data.stage7.cost.find((r) => r.target === '1p5b' && r.mode === 'eager');
const c3Graph = data.stage7.cost.find((r) => r.target === '3b' && r.mode === 'graph');
assert(c15Eager.c > 0.8 && c15Eager.c < 0.9, `expected 1.5B eager c ~0.84, got ${c15Eager.c}`);
assert(c3Graph.c > 0.25 && c3Graph.c < 0.3, `expected 3B graph c ~0.27, got ${c3Graph.c}`);

// Stage 8 PyTorch int8 slowdown vs Stage 10 fused Triton int8 win
for (const sz of ['0.5B', '1.5B', '3B']) {
  const fp8 = data.stage8.speed.find((r) => r.model === sz && r.variant === 'fp16' && r.mode === 'eager');
  const pt8 = data.stage8.speed.find((r) => r.model === sz && r.variant === 'int8' && r.mode === 'eager');
  const tr10 = data.stage10.plain.find((r) => r.model === sz && r.variant === 'int8' && r.mode === 'graph');
  assert(pt8.median_ms > fp8.median_ms, `Stage 8 PyTorch int8 must be slower than fp16 on ${sz}`);
  assert(tr10.step_ms < fp8.median_ms, `Stage 10 fused Triton int8+graph must beat fp16 eager on ${sz}`);
}

// Stage 9 int8 self-draft high acceptance
const selfAcc = data.stage9.acceptance.filter((r) => r.proposer === 'int8 self-draft');
assert(selfAcc.length === 6 && selfAcc.every((r) => r.alpha > 0.96), 'int8 self-draft alpha must exceed 0.96');

// Stage 11 block_cost and non-composition
const vFp3b = data.stage11.verdict.v_fp16['3B'];
const vNew3b = data.stage11.verdict.v_new['3B'];
assert(vFp3b < 1.05, `expected 3B fp16 5-token v < 1.05, got ${vFp3b}`);
assert(vNew3b > 1.85 && vNew3b < 2.05, `expected 3B int8 Stage 11 5-token v ~1.94, got ${vNew3b}`);

const mult3bCopy4 = data.stage11.multiply.find((r) => r.target === '3B' && r.workload === 'copy' && r.k === 4);
assert(mult3bCopy4.both_over_product > 0.5 && mult3bCopy4.both_over_product < 0.6, 'expected 3B copy k=4 survival ~53%');

const best3bCopy = data.stage11.best_stack.find((r) => r.target === '3B' && r.workload === 'copy');
const best3bCode = data.stage11.best_stack.find((r) => r.target === '3B' && r.workload === 'code');
const best3bOpen = data.stage11.best_stack.find((r) => r.target === '3B' && r.workload === 'open');
assert(best3bCopy.best_vs_eager > 6.0, `expected 3B copy > 6.0x, got ${best3bCopy.best_vs_eager}`);
assert(best3bCode.best_vs_eager > 2.0, `expected 3B code > 2.0x, got ${best3bCode.best_vs_eager}`);
assert(best3bOpen.best_vs_eager > 1.85, `expected 3B open > 1.85x, got ${best3bOpen.best_vs_eager}`);

// 4. Built index.html structural & JS syntax checks
for (const token of ['/*__DATA__*/', '/*__SIM__*/', '<!--__THEME__-->', '<!--__UI__-->']) {
  assert(!html.includes(token), `unreplaced placeholder in index.html: ${token}`);
}
assert(html.includes('href="../shared/theme.css"'), 'index.html must link ../shared/theme.css');
assert(html.includes('src="../shared/ui.js"'), 'index.html must link ../shared/ui.js');
assert(html.includes('property="og:title"'), 'index.html must contain OpenGraph meta tags');
assert(html.includes("slugs: { p1: 'trace', p2: 'law', p3: 'compose' }"), 'index.html must configure tab slugs');

// Extract inline <script> block and verify valid JS syntax and initTabs placement after declarations
const scripts = html.match(/<script>([\s\S]*?)<\/script>/g) || [];
assert(scripts.length >= 1, 'expected inline script in index.html');
const mainScript = scripts[scripts.length - 1].replace(/^<script>|<\/script>$/g, '');
new Function(mainScript); // throws SyntaxError if invalid JS
const idxRendered = mainScript.indexOf('const rendered = {};');
const idxInitTabs = mainScript.indexOf('window.mlViz.initTabs(');
assert(idxRendered > 0 && idxInitTabs > idxRendered, 'initTabs must be called after state declarations at end of IIFE');

console.log('speculative-decoding tests: OK');
