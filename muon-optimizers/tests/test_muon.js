'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const M = require('../src/sim.js');

/* 1. Exact singular values & Jacobi SVD invariants */
{
  const mk = M.makeSkewedMatrix(4, 200, 42);
  const { U, S, V } = M.svd(mk.G);

  // Singular values match constructed spectrum [1, ..., 1/200]
  assert.strictEqual(S.length, 4);
  for (let i = 0; i < 4; i++) {
    assert(Math.abs(S[i] - mk.sigmas[i]) < 1e-9, `sv[${i}] mismatch: ${S[i]} vs ${mk.sigmas[i]}`);
  }

  // Exact Frobenius identity: ||G||_F^2 === sum(s_i^2)
  const f2 = Math.pow(M.frobNorm(mk.G), 2);
  const sumS2 = S.reduce((acc, s) => acc + s * s, 0);
  assert(Math.abs(f2 - sumS2) < 1e-10, `Frobenius identity failed: ${f2} vs ${sumS2}`);

  // Exact reconstruction G = U * diag(S) * V^T
  const Smat = M.zeros(4, 4);
  for (let i = 0; i < 4; i++) Smat[i][i] = S[i];
  const recon = M.matmul(M.matmul(U, Smat), M.transpose(V));
  const diff = M.frobNorm(M.matAdd(mk.G, recon, 1, -1));
  assert(diff < 1e-10, `SVD reconstruction error too large: ${diff}`);

  // U and V are orthonormal
  assert(M.orthError(U) < 1e-10, `U not orthogonal: ${M.orthError(U)}`);
  assert(M.orthError(V) < 1e-10, `V not orthogonal: ${M.orthError(V)}`);

  // Rectangular matrix singular values also satisfy ||G||_F^2 = sum s_i^2
  const rect = [[3, -1], [0.5, 2], [-2, 1.5], [1, 0.25]];
  const svRect = M.singularValues(rect);
  const f2Rect = Math.pow(M.frobNorm(rect), 2);
  const sumRect = svRect.reduce((acc, s) => acc + s * s, 0);
  assert(Math.abs(f2Rect - sumRect) < 1e-10, `rectangular Frobenius identity failed: ${f2Rect} vs ${sumRect}`);
}

/* 2. Newton-Schulz orthogonalization convergence */
{
  const mk = M.makeSkewedMatrix(4, 200, 42);
  const nsKeller = M.newtonSchulzSteps(mk.G, 5, M.NS_PRESETS.keller);
  const nsCubic  = M.newtonSchulzSteps(mk.G, 5, M.NS_PRESETS.cubic);

  const h0 = nsKeller.history[0];
  const h5 = nsKeller.history[5];

  // Orthogonality error drops sharply
  assert(h5.orthErr < h0.orthErr * 0.45, `expected orthErr to drop sharply: ${h0.orthErr} -> ${h5.orthErr}`);
  // Effective condition number collapses from 200 to < 2
  assert(h0.cond > 199 && h5.cond < 2.0, `expected cond collapse: ${h0.cond} -> ${h5.cond}`);
  // All singular values cluster near 1 (within [0.65, 1.25])
  for (const s of h5.sv) {
    assert(s >= 0.65 && s <= 1.25, `singular value ${s} not clustered near 1 after 5 Keller NS steps`);
  }
  // Keller quintic lifts smallest singular value far more aggressively than classic cubic
  assert(h5.sv[3] > nsCubic.history[5].sv[3] * 10, `expected Keller quintic to beat cubic on rare direction`);

  // Smooth quintic (2, -1.5, 0.5) converges tightly toward exact 1 on moderate cond=10
  const mkMod = M.makeSkewedMatrix(4, 10, 7);
  const nsBlog = M.newtonSchulzSteps(mkMod.G, 5, M.NS_PRESETS.blog);
  assert(nsBlog.history[5].orthErr < 0.05, `expected smooth quintic orthErr < 0.05 on cond=10, got ${nsBlog.history[5].orthErr}`);

  // Odd polynomial action matches exact scalar composition on every singular value
  for (let i = 0; i < 4; i++) {
    let s = h0.sv[i];
    for (let k = 0; k < 5; k++) s = M.evalNSPoly(s, nsKeller.coeffs);
    assert(Math.abs(s - h5.sv[i]) < 1e-6 || h5.sv.some((v) => Math.abs(v - s) < 1e-6),
      `polynomial transfer did not match matrix SVD for sigma_${i + 1}`);
  }
}

/* 3. AdamW vs Muon on ill-conditioned matrix regression */
{
  const res = M.runLandscape({ fn: 'matrix', cond: 80, lr: 0.10, beta: 0.90, steps: 100 });
  const { sgd, momentum, adamw, muon } = res.optimizers;

  // Muon converges dramatically faster than plain SGD and rotated AdamW
  assert(muon.finalLoss < sgd.finalLoss * 0.01, `expected Muon (${muon.finalLoss}) << SGD (${sgd.finalLoss})`);
  assert(muon.finalLoss < adamw.finalLoss * 0.05, `expected Muon (${muon.finalLoss}) << rotated AdamW (${adamw.finalLoss})`);
  assert(muon.finalLoss < 1e-4, `expected Muon finalLoss < 1e-4, got ${muon.finalLoss}`);
  assert(muon.stepsTo1e3 !== null && muon.stepsTo1e3 < 80, `expected Muon to hit 1e-3 fast, got ${muon.stepsTo1e3}`);
  assert(sgd.stepsTo1e3 === null || muon.stepsTo1e3 < sgd.stepsTo1e3, 'expected Muon faster than SGD to 1e-3');

  // Muon's update matrix is nearly isotropic (low update condition number)
  assert(muon.geomUpdateCond < 1.5, `expected Muon update cond < 1.5, got ${muon.geomUpdateCond}`);
  assert(adamw.geomUpdateCond > muon.geomUpdateCond * 3, `expected rotated AdamW update cond (${adamw.geomUpdateCond}) >> Muon (${muon.geomUpdateCond})`);
  assert(Number.isFinite(momentum.finalLoss), 'momentum loss must be finite');
}

/* 4. MuonClip (QK-Clip) per-head logit capping & untouched healthy heads */
{
  for (const tau of [40, 100, 140]) {
    const off1 = M.simQKClip({ heads: 8, steps: 100, tau, clipOn: false, seed: 2026 });
    const off2 = M.simQKClip({ heads: 8, steps: 100, tau, clipOn: false, seed: 2026 });
    assert.deepStrictEqual(off1.globalMaxLogit, off2.globalMaxLogit, 'simQKClip must be deterministic');

    const on = M.simQKClip({ heads: 8, steps: 100, tau, clipOn: true, seed: 2026 });

    // Unclipped explodes past tau
    const peakOff = Math.max(...off1.globalMaxLogit);
    assert(peakOff > tau * 1.8, `expected unclipped peak (${peakOff}) well above tau=${tau}`);

    // Clipped caps EVERY head at EVERY step <= tau
    for (let h = 0; h < 8; h++) {
      for (let t = 0; t <= 100; t++) {
        assert(on.smaxSeries[h][t] <= tau + 1e-8,
          `tau=${tau} head ${h} step ${t} exceeded tau: ${on.smaxSeries[h][t]}`);
      }
    }

    // For tau=100, healthy heads (never exceeding tau) are completely untouched: gamma === 1
    if (tau === 100) {
      let untouchedCount = 0;
      let clippedCount = 0;
      for (let h = 0; h < 8; h++) {
        const hs = on.headSummary[h];
        if (!hs.runaway) {
          untouchedCount++;
          assert.strictEqual(hs.clipCount, 0, `healthy head ${h} should not be clipped at tau=100`);
          assert.strictEqual(hs.minGamma, 1, `healthy head ${h} minGamma should be === 1`);
          assert.strictEqual(hs.cumGamma, 1, `healthy head ${h} cumGamma should be === 1`);
          for (let t = 0; t <= 100; t++) {
            assert.strictEqual(on.gammaSeries[h][t], 1, `healthy head ${h} gamma at step ${t} must be === 1`);
            assert(Math.abs(on.smaxSeries[h][t] - off1.smaxSeries[h][t]) < 1e-10,
              `healthy head ${h} trajectory must be identical with/without QK-Clip`);
          }
        } else {
          clippedCount++;
          assert(hs.clipCount > 0 && hs.cumGamma < 0.5, `runaway head ${h} should be clipped`);
        }
      }
      assert(untouchedCount === 5 && clippedCount === 3, `expected 5 healthy and 3 runaway heads`);

      // Softmax entropy preserved vs collapse
      const entOn = on.runawayEntropy[100];
      const entOff = off1.runawayEntropy[100];
      assert(entOn > entOff * 1.8, `expected MuonClip entropy (${entOn}) >> unclipped (${entOff})`);
    }
  }
}

/* 5. Built index.html sanity checks */
{
  const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
  assert(!html.includes('/*SIM*/'), 'unreplaced /*SIM*/ placeholder');
  assert(!html.includes('<!--__THEME__-->'), 'unreplaced <!--__THEME__--> placeholder');
  assert(!html.includes('<!--__UI__-->'), 'unreplaced <!--__UI__--> placeholder');
  assert(html.includes('../shared/theme.css') && html.includes('../shared/ui.js'), 'missing shared asset links');
  assert(html.includes('computed live'), 'missing computed live badge');
  assert(html.includes("p1: 'ns', p2: 'trajectories', p3: 'muonclip'"), 'missing expected tab slugs');
  assert(html.includes('https://chizkidd.github.io/2026/04/04/muon-muonclip/'), 'missing blog post link');
  assert(html.includes('og:title') && html.includes('og:description'), 'missing OpenGraph meta tags');
}

console.log('muon-optimizers tests: ALL PASSED');
