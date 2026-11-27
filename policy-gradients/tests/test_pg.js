'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const S = require('../src/sim.js');

// 1. Exact Short Corridor closed-form vs 3x3 linear solver & Sutton & Barto Fig. 13.1 landmarks
const testPs = [0.05, 0.2, 0.35, 0.5, S.P_STAR, 0.75, 0.95];
for (const p of testPs) {
  const ex = S.corridorValueExact(p);
  const lin = S.corridorValueLinear(p, 1.0);
  assert(Math.abs(ex.v0 - lin.v0) < 1e-11, `v0 mismatch at p=${p}: ${ex.v0} vs ${lin.v0}`);
  assert(Math.abs(ex.v1 - lin.v1) < 1e-11, `v1 mismatch at p=${p}: ${ex.v1} vs ${lin.v1}`);
  assert(Math.abs(ex.v2 - lin.v2) < 1e-11, `v2 mismatch at p=${p}: ${ex.v2} vs ${lin.v2}`);
}

assert(Math.abs(S.corridorValueExact(0.5).v0 - (-12)) < 1e-12, 'v0(0.5) must equal -12');
assert(Math.abs(S.corridorValueExact(0.05).v0 - (-82.10526315789474)) < 1e-9, 'v0(0.05) should be ~-82.105');
assert(Math.abs(S.corridorValueExact(0.95).v0 - (-44.21052631578947)) < 1e-9, 'v0(0.95) should be ~-44.211');

// 2. Algebraic optimum p* = 2 - sqrt(2) and v* = -(6 + 4*sqrt(2))
assert(Math.abs(S.P_STAR - (2 - Math.sqrt(2))) < 1e-15, 'P_STAR must be 2 - sqrt(2)');
assert(Math.abs(S.P_STAR - 0.5857864376) < 1e-9, 'P_STAR should be ~0.5858');
assert(Math.abs(S.V_STAR - (-(6 + 4 * Math.sqrt(2)))) < 1e-12, 'V_STAR must be -(6 + 4*sqrt(2))');
assert(Math.abs(S.corridorValueExact(S.P_STAR).v0 - S.V_STAR) < 1e-12, 'corridorValueExact(P_STAR) must equal V_STAR');
assert(Math.abs(S.corridorDv0Dp(S.P_STAR)) < 1e-12, 'dv0/dp at P_STAR must be 0');
assert(S.corridorDv0Dp(0.35) > 0, 'dv0/dp must be positive below P_STAR');
assert(S.corridorDv0Dp(0.80) < 0, 'dv0/dp must be negative above P_STAR');
for (let p = 0.05; p <= 0.95; p += 0.01) {
  assert(S.corridorValueExact(p).v0 <= S.V_STAR + 1e-12, `p=${p} beat V_STAR`);
}

// 3. Exact analytical value vs Monte Carlo average over 20,000 episodes
function mcMean(p, episodes, seed) {
  const rng = S.mulberry32(seed);
  let sum = 0, sq = 0;
  for (let i = 0; i < episodes; i++) {
    const ep = S.corridorEpisode(rng, p, 2000);
    sum += ep.totalReturn;
    sq += ep.totalReturn * ep.totalReturn;
  }
  const mean = sum / episodes;
  const variance = sq / episodes - mean * mean;
  return { mean, se: Math.sqrt(variance / episodes) };
}

for (const p of [0.5, S.P_STAR]) {
  const exact = S.corridorValueExact(p).v0;
  const mc = mcMean(p, 20000, 12345);
  const diff = Math.abs(mc.mean - exact);
  assert(diff < 3.5 * mc.se && diff < 0.35, `20k MC mismatch at p=${p}: exact=${exact.toFixed(3)}, mc=${mc.mean.toFixed(3)}, se=${mc.se.toFixed(3)}`);
}

// 4. Baseline unbiasedness & variance reduction at fixed policy p = 0.35
const bv = S.baselineVarianceCheck({ p: 0.35, episodes: 12000, seed: 77 });
const b0 = bv.results[0], bV0 = bv.results[1];
assert(Math.abs(b0.mean - bv.trueGrad) < 3.5 * b0.se, `b=0 gradient biased: ${b0.mean} vs ${bv.trueGrad}`);
assert(Math.abs(bV0.mean - bv.trueGrad) < 3.5 * bV0.se, `b=v0 gradient biased: ${bV0.mean} vs ${bv.trueGrad}`);
assert(bV0.variance < b0.variance * 0.55, `Expected baseline to cut fixed-policy variance sharply: ${bV0.variance} vs ${b0.variance}`);

// 5. REINFORCE with baseline converges close to V_STAR with much lower gradient variance
const rfBase = S.runReinforce({
  alphaTheta: Math.pow(2, -9),
  alphaW: Math.pow(2, -6),
  useBaseline: true,
  episodes: 1000,
  runs: 30,
  seed: 42
});
const rfNoSame = S.runReinforce({
  alphaTheta: Math.pow(2, -9),
  useBaseline: false,
  episodes: 1000,
  runs: 30,
  seed: 42
});
const rfNoSmall = S.runReinforce({
  alphaTheta: Math.pow(2, -12),
  useBaseline: false,
  episodes: 1000,
  runs: 30,
  seed: 42
});
assert(rfBase.finalReturn > S.V_STAR - 1.2 && rfBase.finalReturn < S.V_STAR + 1.2, `REINFORCE+baseline did not converge near V_STAR: ${rfBase.finalReturn}`);
assert(Math.abs(rfBase.finalP - S.P_STAR) < 0.08, `REINFORCE+baseline finalP far from P_STAR: ${rfBase.finalP}`);
assert(rfBase.meanGradVar < rfNoSame.meanGradVar * 0.1, `Expected >=10x lower grad variance at same alpha: ${rfBase.meanGradVar} vs ${rfNoSame.meanGradVar}`);
assert(rfBase.meanGradVar < rfNoSmall.meanGradVar * 0.25, `Expected much lower grad variance than no-baseline even at small alpha: ${rfBase.meanGradVar} vs ${rfNoSmall.meanGradVar}`);

// 6. One-step Actor-Critic: tabular (unbiased, ultra-low variance), aliased (bootstrapping bias), and 5-state corridor
const acTab = S.runActorCritic({
  env: 'corridor',
  criticMode: 'tabular',
  alphaTheta: 0.015,
  alphaW: 0.10,
  episodes: 500,
  runs: 30,
  seed: 42
});
assert(acTab.finalReturn > S.V_STAR - 1.5, `Actor-Critic (tabular critic) failed to converge near V_STAR: ${acTab.finalReturn}`);
assert(acTab.meanGradVar < rfBase.meanGradVar * 0.05, `Expected Actor-Critic grad variance << REINFORCE+baseline: ${acTab.meanGradVar} vs ${rfBase.meanGradVar}`);

const acAli = S.runActorCritic({
  env: 'corridor',
  criticMode: 'aliased',
  alphaTheta: 0.015,
  alphaW: 0.10,
  episodes: 500,
  runs: 20,
  seed: 42
});
assert(acAli.finalReturn < -28, `Expected aliased Actor-Critic to show bootstrapping bias: ${acAli.finalReturn}`);

const ac5 = S.runActorCritic({ env: 'line5', alphaTheta: 0.12, alphaW: 0.25, episodes: 250, runs: 25, seed: 42 });
const rf5 = S.runReinforceLine5({ alphaTheta: 0.08, alphaW: 0.20, useBaseline: true, episodes: 250, runs: 25, seed: 42 });
assert(ac5.finalReturn > -5.5, `Actor-Critic on 5-state corridor should approach -5: ${ac5.finalReturn}`);
assert(ac5.meanGradVar < rf5.meanGradVar * 0.05, `Actor-Critic on line5 should have far lower grad variance than REINFORCE`);

const tr = S.traceEpisodeCompare({ env: 'corridor', criticMode: 'tabular', seed: 19, maxSteps: 30 });
assert(tr.acSteps.length > 0 && tr.acSteps.length === tr.mcSteps.length, 'traceEpisodeCompare step count mismatch');

// 7. Check built index.html integrity, accessibility, and script syntax
const htmlPath = path.join(__dirname, '..', 'index.html');
assert(fs.existsSync(htmlPath), 'policy-gradients/index.html does not exist');
const html = fs.readFileSync(htmlPath, 'utf8');

assert(!html.includes('/*SIM*/'), 'unreplaced /*SIM*/ placeholder in index.html');
assert(!html.includes('<!--__THEME__-->'), 'unreplaced <!--__THEME__--> placeholder in index.html');
assert(!html.includes('<!--__UI__-->'), 'unreplaced <!--__UI__--> placeholder in index.html');
assert(html.includes('href="../shared/theme.css"'), 'missing ../shared/theme.css link');
assert(html.includes('src="../shared/ui.js"'), 'missing ../shared/ui.js script');
assert(html.includes('https://chizkidd.github.io/RL-Sutton-Barto-notes/chapters/ch13-policy-gradient-methods.html'), 'missing Ch. 13 notes link');
assert(html.includes('property="og:title"') && html.includes('property="og:description"') && html.includes('property="og:url"'), 'missing OpenGraph tags');
assert(html.includes("p1: 'corridor'") && html.includes("p2: 'baseline'") && html.includes("p3: 'actor-critic'"), 'missing expected tab slugs');
assert(html.includes('window.mlViz.initTheme(') && html.includes('window.mlViz.initTabs('), 'missing mlViz initTheme / initTabs');

// Verify initTabs is near the end of the IIFE (after window.__pg)
assert(html.indexOf('window.mlViz.initTabs(') > html.indexOf('window.__pg'), 'initTabs must be called at the END of the IIFE');

// Verify every <input type="range" id="..."> has a matching <label for="...">
const rangeIds = [...html.matchAll(/<input\s+[^>]*type="range"[^>]*id="([^"]+)"/g)].map((m) => m[1]);
assert(rangeIds.length >= 4, `expected at least 4 range inputs, found ${rangeIds.length}`);
for (const id of rangeIds) {
  assert(html.includes(`for="${id}"`), `missing <label for="${id}">`);
}

// Verify every <svg class="chart" has role="img" and aria-label, and <title> tooltips exist
const svgs = [...html.matchAll(/<svg\s+class="chart"[^>]*>/g)].map((m) => m[0]);
assert(svgs.length >= 6, `expected at least 6 svg.chart elements, found ${svgs.length}`);
for (const tag of svgs) {
  assert(tag.includes('role="img"') && tag.includes('aria-label="'), `svg.chart missing role/aria-label: ${tag}`);
}
assert(html.includes('<title>'), 'missing <title> tooltips inside svg charts');

// Verify all inline <script> blocks compile without syntax errors
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
assert(scripts.length === 2, `expected 2 inline script blocks (sim + ui), found ${scripts.length}`);
scripts.forEach((code, idx) => {
  new vm.Script(code, { filename: `inline-script-${idx}.js` });
});

console.log('policy-gradients tests: OK');
console.log(`  Short Corridor p* = ${S.P_STAR.toFixed(6)}, v* = ${S.V_STAR.toFixed(4)}`);
console.log(`  REINFORCE+base final = ${rfBase.finalReturn.toFixed(2)} (gradVar ${rfBase.meanGradVar.toFixed(1)}) vs no-base same-alpha gradVar ${rfNoSame.meanGradVar.toFixed(1)}`);
console.log(`  Actor-Critic (tabular) final = ${acTab.finalReturn.toFixed(2)} (gradVar ${acTab.meanGradVar.toFixed(1)}) vs aliased final = ${acAli.finalReturn.toFixed(2)}`);
