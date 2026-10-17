/* TD(lambda) and deadly-triad simulations. Pure functions, no DOM.
   Works in a browser (window.TDSIM) and in node (module.exports). */
(function (root) {
  'use strict';

  /* ---------- seeded RNG (mulberry32) ---------- */
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------- 19-state random walk (Sutton & Barto, ch. 7 and 12) ----------
     states 0..18, start in the middle (9), both ends terminal,
     reward -1 off the left end, +1 off the right end, 0 otherwise, gamma = 1. */
  var N = 19, START = 9;
  var TRUE_V = [];
  for (var i = 0; i < N; i++) TRUE_V.push((i - START) / 10);

  /* one episode as arrays: states visited, and the terminal reward.
     policy: 'random' | 'right' | 'left' */
  function walkEpisode(rng, policy) {
    var s = START, states = [s], term = 0;
    for (;;) {
      var right = policy === 'right' ? true : policy === 'left' ? false : rng() < 0.5;
      s += right ? 1 : -1;
      if (s < 0) { term = -1; break; }
      if (s >= N) { term = 1; break; }
      states.push(s);
    }
    return { states: states, term: term };
  }

  /* One online TD(lambda) update (gamma = 1). Mutates w and e.
     s2 === -1 means "terminal next state"; r is the reward on that step.
     Returns delta. kind: 'acc' (accumulating) or 'rep' (replacing). */
  function tdStep(w, e, s, s2, r, p) {
    var v2 = s2 < 0 ? 0 : w[s2];
    var delta = r + v2 - w[s];
    var gl = p.lambda; // gamma = 1
    for (var i = 0; i < N; i++) e[i] *= gl;
    if (p.kind === 'rep') e[s] = 1; else e[s] += 1;
    for (var j = 0; j < N; j++) w[j] += p.alpha * delta * e[j];
    return delta;
  }

  /* run a whole episode through tdStep; returns number of steps */
  function runEpisode(w, ep, p) {
    var e = new Float64Array(N), n = ep.states.length;
    for (var k = 0; k < n; k++) {
      var last = k === n - 1;
      tdStep(w, e, ep.states[k], last ? -1 : ep.states[k + 1], last ? ep.term : 0, p);
    }
    return n;
  }

  function rms(w) {
    var s = 0;
    for (var i = 0; i < N; i++) { var d = w[i] - TRUE_V[i]; s += d * d; }
    return Math.sqrt(s / N);
  }

  /* lambda sweep: for each (lambda, alpha) the RMS error of the value estimates at
     the end of each episode, averaged over `episodes` episodes and `runs` runs.
     Every (lambda, alpha) sees the same random episodes (common random numbers).
     Diverged runs are capped at `cap`. */
  function sweepPrepare(runs, episodes, seed) {
    var rng = mulberry32(seed), all = [];
    for (var r = 0; r < runs; r++) {
      var eps = [];
      for (var k = 0; k < episodes; k++) eps.push(walkEpisode(rng, 'random'));
      all.push(eps);
    }
    return all;
  }
  function sweepCell(data, lambda, alpha, kind, cap) {
    var tot = 0, cnt = 0, p = { lambda: lambda, alpha: alpha, kind: kind };
    for (var r = 0; r < data.length; r++) {
      var w = new Float64Array(N);
      for (var k = 0; k < data[r].length; k++) {
        runEpisode(w, data[r][k], p);
        var x = rms(w);
        tot += (isFinite(x) ? Math.min(x, cap) : cap); cnt++;
      }
    }
    return tot / cnt;
  }

  /* ---------- the two-state "w, 2w" example (S&B 11.2, first example) ---------- */
  /* v(s1) = w, v(s2) = 2w; s1 -> s2 with reward 0, only that transition is updated. */
  function twoStateGrowth(alpha, gamma) { return 1 + alpha * (2 * gamma - 1); }
  function twoStateRun(w0, alpha, gamma, steps) {
    var w = w0, out = [w];
    for (var k = 0; k < steps; k++) { w = w + alpha * (gamma * 2 * w - w) * 1; out.push(w); }
    return out;
  }

  /* ---------- Baird's counterexample (S&B 11.2) ----------
     7 states. 'dashed' goes to a uniformly random state among 1..6, 'solid' goes to state 7.
     Target policy: solid always. Behaviour policy: dashed 6/7, solid 1/7. Rewards 0, gamma 0.99.
     Linear features: v(s_i) = 2 w_i + w_8 (i = 1..6), v(s_7) = w_7 + 2 w_8.
     Tabular features: v(s_i) = w_i (7 weights). Initial w = (1,1,1,1,1,1,10,1) (linear) or all ones (tabular). */
  var GAMMA = 0.99, CAP = 1e12;

  function bairdFeatures(tab) {
    var X = [], i, row;
    if (tab) {
      for (i = 0; i < 7; i++) { row = new Array(7).fill(0); row[i] = 1; X.push(row); }
      return X;
    }
    for (i = 0; i < 6; i++) { row = new Array(8).fill(0); row[i] = 2; row[7] = 1; X.push(row); }
    row = new Array(8).fill(0); row[6] = 1; row[7] = 2; X.push(row);
    return X;
  }
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }

  /* cfg: { dist: 'off' | 'on' | 'uniform', tab: bool, boot: bool, algo: 'td' | 'tdc',
            alpha, beta, steps, seed }
     Returns { W: [step][weight] including step 0, V: [step] max |v(s)|, diverged: bool } */
  function bairdRun(cfg) {
    var X = bairdFeatures(cfg.tab), d = X[0].length;
    var w = new Array(d).fill(1); if (!cfg.tab) w[6] = 10;
    var vv = new Array(d).fill(0);               // TDC secondary weights
    var rng = mulberry32(cfg.seed >>> 0);
    var W = [w.slice()], V = [maxAbsV(X, w)], blown = false;
    var s = Math.floor(rng() * 7);
    var alpha = cfg.alpha, beta = cfg.beta == null ? 5 * cfg.alpha : cfg.beta;

    for (var t = 0; t < cfg.steps; t++) {
      if (!blown) {
        if (cfg.dist === 'uniform') {
          /* expected (DP-style) update, uniform over the 7 states, target policy = solid -> state 7 */
          var x7 = X[6], v7 = dot(x7, w);
          var dw = new Array(d).fill(0), dv = new Array(d).fill(0);
          for (var i = 0; i < 7; i++) {
            var xi = X[i], vi = dot(xi, w), tgt = cfg.boot ? GAMMA * v7 : 0, delta = tgt - vi;
            for (var k = 0; k < d; k++) {
              if (cfg.algo === 'tdc') {
                dw[k] += (delta * xi[k] - (cfg.boot ? GAMMA * dot(xi, vv) * x7[k] : 0)) / 7;
                dv[k] += (delta - dot(xi, vv)) * xi[k] / 7;
              } else dw[k] += delta * xi[k] / 7;
            }
          }
          for (var m = 0; m < d; m++) { w[m] += alpha * dw[m]; if (cfg.algo === 'tdc') vv[m] += beta * dv[m]; }
        } else {
          var solid = rng() < 1 / 7;
          var s2 = solid ? 6 : Math.floor(rng() * 6);
          var rho = cfg.dist === 'off' ? (solid ? 7 : 0) : 1;
          var xs = X[s], x2 = X[s2];
          var delta2 = (cfg.boot ? GAMMA * dot(x2, w) : 0) - dot(xs, w);
          if (cfg.algo === 'tdc') {
            var xv = dot(xs, vv);
            for (var q = 0; q < d; q++) {
              var wq = w[q];
              w[q] = wq + alpha * rho * (delta2 * xs[q] - (cfg.boot ? GAMMA * x2[q] * xv : 0));
              vv[q] += beta * rho * (delta2 - xv) * xs[q];
            }
          } else {
            for (var u = 0; u < d; u++) w[u] += alpha * rho * delta2 * xs[u];
          }
          s = s2;
        }
        for (var z = 0; z < d; z++) if (!isFinite(w[z]) || Math.abs(w[z]) > CAP) { blown = true; }
        if (blown) { for (var y = 0; y < d; y++) w[y] = Math.max(-CAP, Math.min(CAP, isFinite(w[y]) ? w[y] : CAP)); }
      }
      W.push(w.slice()); V.push(maxAbsV(X, w));
    }
    return { W: W, V: V, diverged: blown || V[V.length - 1] > 1e3 };
  }
  function maxAbsV(X, w) { var m = 0; for (var i = 0; i < X.length; i++) m = Math.max(m, Math.abs(dot(X[i], w))); return m; }

  var api = {
    mulberry32: mulberry32, N: N, START: START, TRUE_V: TRUE_V,
    walkEpisode: walkEpisode, tdStep: tdStep, runEpisode: runEpisode, rms: rms,
    sweepPrepare: sweepPrepare, sweepCell: sweepCell,
    twoStateGrowth: twoStateGrowth, twoStateRun: twoStateRun,
    bairdRun: bairdRun, bairdFeatures: bairdFeatures, GAMMA: GAMMA, CAP: CAP
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.TDSIM = api;
})(typeof window !== 'undefined' ? window : this);
