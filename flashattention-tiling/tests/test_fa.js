'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const match = html.match(/\/\* =+[\s\S]*?window\.__fa = \{[\s\S]*?\};/);
assert(match, 'could not locate pure math core in flashattention-tiling/index.html');

// Extract rng + gauss helpers right above the math core and run in isolated VM
const helpers = html.match(/function rng\(seed\)[\s\S]*?const gauss = [^;]+;/);
assert(helpers, 'could not locate rng/gauss helpers');

const sandbox = { window: {} };
vm.runInNewContext(`${helpers[0]}\n${match[0]}`, sandbox);
const fa = sandbox.window.__fa;
assert(fa && typeof fa.makeProblem === 'function', 'window.__fa not exposed');

// 1. Tiled FlashAttention matches naive reference to floating-point precision
for (const [N, Br, Bc, causal] of [
  [8, 2, 2, false],
  [8, 4, 2, true],
  [12, 4, 4, false],
  [12, 4, 4, true],
  [16, 4, 8, true],
  [16, 8, 4, false],
]) {
  const p = fa.makeProblem(N, Br, Bc, causal, 42);
  assert(p.maxErr < 1e-12, `FlashAttention mismatch for N=${N},Br=${Br},Bc=${Bc},causal=${causal}: ${p.maxErr}`);
  if (causal) {
    assert(p.tiles.some((t) => t.skipped), `expected skipped tiles for causal N=${N}`);
  }
}

// 2. Online softmax survives extreme magnitude where plain exp(x) overflows
const rp = fa.rowProblem(4, 1000);
const tw = fa.threeWays(rp.xs, rp.vs);
assert.strictEqual(tw.expInf, true, 'plain exp(x) should overflow at magnitude x1000');
assert(!Number.isFinite(tw.naive), 'plain softmax should be NaN/Inf at magnitude x1000');
const os = fa.onlineStates(rp.xs, rp.vs, 3);
const last = os[os.length - 1];
const onlineVal = last.o / last.l;
assert(Math.abs(onlineVal - tw.safe) < 1e-12, `online softmax (${onlineVal}) != safe softmax (${tw.safe})`);

console.log('flashattention-tiling tests: OK');
