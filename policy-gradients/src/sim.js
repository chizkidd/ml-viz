/* Policy Gradient simulations (Sutton & Barto, ch. 13). Pure functions, no DOM.
   Works in a browser (window.PGSIM) and in node (module.exports). */
(function (root) {
  'use strict';

  /* ---------- seeded RNG (mulberry32, identical to td-lambda) ---------- */
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------- Short Corridor with Switched Actions (Sutton & Barto Ex. 13.1) ----------
     States: 0, 1, 2 (non-terminal), 3 (terminal goal).
     Reward: -1 per step, gamma = 1.
     Actions: 0 = left, 1 = right.
     State 0: right -> 1, left -> 0 (clamped).
     State 1 (SWITCHED): right -> 0, left -> 2.
     State 2: right -> 3 (terminal), left -> 1.
     All states share identical action features: x(s, left) = [1, 0], x(s, right) = [0, 1]. */

  var CORRIDOR_TERMINAL = 3;
  var P_STAR = 2 - Math.sqrt(2);                  // ~0.5857864376
  var V_STAR = -(6 + 4 * Math.sqrt(2));           // ~-11.6568542495

  /* Exact closed-form state values for Short Corridor under p = pi(right | s):
     From Bellman equations (gamma = 1):
       v0 = -1 + (1 - p) v0 + p v1          => v0 = v1 - 1/p
       v2 = -1 + (1 - p) v1 + p * 0         => v2 = -1 + (1 - p) v1
       v1 = -1 + p v0 + (1 - p) v2          => p(1 - p) v1 = -(3 - p)
     Yields:
       v0(p) = -2(2 - p) / (p(1 - p))
       v1(p) = -(3 - p) / (p(1 - p))
       v2(p) = -3 / p */
  function corridorValueExact(p) {
    if (p <= 0 || p >= 1) {
      return { v0: -Infinity, v1: -Infinity, v2: -Infinity };
    }
    var q = 1 - p;
    var denom = p * q;
    return {
      v0: -2 * (2 - p) / denom,
      v1: -(3 - p) / denom,
      v2: -3 / p
    };
  }

  /* Exact 3x3 linear system solver for (I - gamma * P(p)) v = -1.
     Independent numerical cross-check of corridorValueExact(p) and works for any gamma <= 1. */
  function solve3x3(A, b) {
    var M = [
      [A[0][0], A[0][1], A[0][2], b[0]],
      [A[1][0], A[1][1], A[1][2], b[1]],
      [A[2][0], A[2][1], A[2][2], b[2]]
    ];
    for (var col = 0; col < 3; col++) {
      var piv = col;
      for (var r = col + 1; r < 3; r++) {
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      }
      if (piv !== col) {
        var tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
      }
      var diag = M[col][col];
      if (Math.abs(diag) < 1e-14) return [NaN, NaN, NaN];
      for (var j = col; j < 4; j++) M[col][j] /= diag;
      for (var row = 0; row < 3; row++) {
        if (row === col) continue;
        var f = M[row][col];
        for (var k = col; k < 4; k++) M[row][k] -= f * M[col][k];
      }
    }
    return [M[0][3], M[1][3], M[2][3]];
  }

  function corridorValueLinear(p, gamma) {
    var g = gamma == null ? 1 : gamma;
    var q = 1 - p;
    var A = [
      [1 - g * q, -g * p, 0],
      [-g * p, 1, -g * q],
      [0, -g * q, 1]
    ];
    var v = solve3x3(A, [-1, -1, -1]);
    return { v0: v[0], v1: v[1], v2: v[2] };
  }

  /* Derivative d v0 / d p and d v0 / d theta_R (for 2-weight softmax, d p / d theta_R = p(1 - p),
     while total directional derivative w.r.t. logit diff (theta_R - theta_L) is p(1-p) dv0/dp). */
  function corridorDv0Dp(p) {
    var q = 1 - p;
    var denom = p * q;
    return 2 * (p * p - 4 * p + 2) / (denom * denom);
  }

  function corridorStep(s, actionRight) {
    var next;
    if (s === 0) next = actionRight ? 1 : 0;
    else if (s === 1) next = actionRight ? 0 : 2; // switched!
    else if (s === 2) next = actionRight ? 3 : 1;
    else next = 3;
    return { nextState: next, reward: -1, done: next === CORRIDOR_TERMINAL };
  }

  function corridorEpisode(rng, p, maxSteps) {
    var cap = maxSteps == null ? 1000 : maxSteps;
    var s = 0, states = [], actions = [], rewards = [], total = 0, truncated = false;
    for (var t = 0; t < cap; t++) {
      var a = rng() < p ? 1 : 0;
      var tr = corridorStep(s, a === 1);
      states.push(s);
      actions.push(a);
      rewards.push(tr.reward);
      total += tr.reward;
      s = tr.nextState;
      if (tr.done) break;
      if (t === cap - 1) truncated = true;
    }
    return {
      states: states,
      actions: actions,
      rewards: rewards,
      totalReturn: total,
      steps: states.length,
      truncated: truncated
    };
  }

  /* ---------- 5-state non-switched corridor (distinct state features) ----------
     States 0..4, terminal 5. Reward -1 per step, gamma = 1 (or < 1).
     Action 1 (right) -> s + 1, action 0 (left) -> max(0, s - 1).
     Optimal policy always goes right: v*(0) = -5. */
  var LINE5_N = 5;
  var LINE5_TERMINAL = 5;

  function line5Step(s, actionRight) {
    var next = actionRight ? s + 1 : Math.max(0, s - 1);
    return { nextState: next, reward: -1, done: next === LINE5_TERMINAL };
  }

  /* ---------- Numerically stable 2-action softmax helpers ---------- */
  function softmaxProbRight(thetaL, thetaR) {
    var diff = thetaR - thetaL;
    if (diff > 30) return 1 - 1e-13;
    if (diff < -30) return 1e-13;
    return 1 / (1 + Math.exp(-diff));
  }

  function clampTheta(theta, epsFloor) {
    if (!epsFloor || epsFloor <= 0) return;
    var maxDiff = Math.log((1 - epsFloor) / epsFloor);
    var diff = theta[1] - theta[0];
    if (diff > maxDiff) {
      var mid1 = 0.5 * (theta[0] + theta[1]);
      theta[0] = mid1 - 0.5 * maxDiff;
      theta[1] = mid1 + 0.5 * maxDiff;
    } else if (diff < -maxDiff) {
      var mid2 = 0.5 * (theta[0] + theta[1]);
      theta[0] = mid2 + 0.5 * maxDiff;
      theta[1] = mid2 - 0.5 * maxDiff;
    }
    // Keep theta centered to avoid drift in absolute magnitude
    var avg = 0.5 * (theta[0] + theta[1]);
    theta[0] -= avg;
    theta[1] -= avg;
  }

  function initThetaFromP(pInit) {
    var p = Math.max(1e-4, Math.min(1 - 1e-4, pInit == null ? 0.05 : pInit));
    var halfLogit = 0.5 * Math.log(p / (1 - p));
    return [-halfLogit, halfLogit];
  }

  /* Helper to compute quantile from sorted numeric array */
  function quantileSorted(arr, q) {
    var n = arr.length;
    if (n === 0) return 0;
    var pos = (n - 1) * q;
    var lo = Math.floor(pos), hi = Math.ceil(pos);
    if (lo === hi) return arr[lo];
    return arr[lo] + (pos - lo) * (arr[hi] - arr[lo]);
  }

  function summarizeMatrix(mat, episodes) {
    var mean = new Array(episodes);
    var std = new Array(episodes);
    var q25 = new Array(episodes);
    var q50 = new Array(episodes);
    var q75 = new Array(episodes);
    var runs = mat.length;
    var col = new Array(runs);
    for (var e = 0; e < episodes; e++) {
      var sum = 0;
      for (var r = 0; r < runs; r++) {
        var v = mat[r][e];
        col[r] = v;
        sum += v;
      }
      var m = sum / runs;
      var sq = 0;
      for (var r2 = 0; r2 < runs; r2++) {
        var d = col[r2] - m;
        sq += d * d;
      }
      col.sort(function (a, b) { return a - b; });
      mean[e] = m;
      std[e] = Math.sqrt(sq / runs);
      q25[e] = quantileSorted(col, 0.25);
      q50[e] = quantileSorted(col, 0.50);
      q75[e] = quantileSorted(col, 0.75);
    }
    return { mean: mean, std: std, q25: q25, q50: q50, q75: q75 };
  }

  /* ---------- REINFORCE (with and without learned baseline) on Short Corridor ----------
     Follows Sutton & Barto Section 13.3 & 13.4 (Figs. 13.1 and 13.2).
     Parameterization: theta = [thetaL, thetaR], x(left) = [1, 0], x(right) = [0, 1].
     grad_theta log pi(a) = [ (a==0 ? 1 : 0) - (1-p), (a==1 ? 1 : 0) - p ] = [ -(a - p), +(a - p) ].
     Learned state-value baseline: v_hat(s) = w (single scalar, since states are aliased). */
  function runReinforce(cfg) {
    var alphaTheta = cfg.alphaTheta == null ? Math.pow(2, -12) : cfg.alphaTheta;
    var alphaW = cfg.alphaW == null ? Math.pow(2, -6) : cfg.alphaW;
    var useBaseline = Boolean(cfg.useBaseline);
    var episodes = cfg.episodes == null ? 1000 : cfg.episodes;
    var runs = cfg.runs == null ? 30 : cfg.runs;
    var seed = cfg.seed == null ? 42 : (cfg.seed >>> 0);
    var pInit = cfg.pInit == null ? 0.05 : cfg.pInit;
    var epsFloor = cfg.epsFloor == null ? 0.05 : cfg.epsFloor;
    var maxSteps = cfg.maxSteps == null ? 1000 : cfg.maxSteps;
    var gamma = cfg.gamma == null ? 1.0 : cfg.gamma;

    var returnsByRun = new Array(runs);
    var pByRun = new Array(runs);
    var wByRun = new Array(runs);
    var gradByRun = new Array(runs);
    var stepGradAbsByRun = new Array(runs);

    for (var r = 0; r < runs; r++) {
      var rng = mulberry32((seed + Math.imul(r + 1, 0x9E3779B1)) >>> 0);
      var theta = initThetaFromP(pInit);
      clampTheta(theta, epsFloor);
      var w = 0;
      var retArr = new Float64Array(episodes);
      var pArr = new Float64Array(episodes);
      var wArr = new Float64Array(episodes);
      var gArr = new Float64Array(episodes);
      var gAbsArr = new Float64Array(episodes);

      for (var ep = 0; ep < episodes; ep++) {
        var p = softmaxProbRight(theta[0], theta[1]);
        var traj = corridorEpisode(rng, p, maxSteps);
        var T = traj.steps;
        retArr[ep] = traj.totalReturn;

        // Precompute returns G_t from t = 0..T-1
        var G = new Float64Array(T);
        var runningG = 0;
        for (var k = T - 1; k >= 0; k--) {
          runningG = traj.rewards[k] + gamma * runningG;
          G[k] = runningG;
        }

        // REINFORCE episode updates (t = 0..T-1)
        var epGradR = 0;
        var epAbsGrad = 0;
        var gammaPow = 1;
        for (var t = 0; t < T; t++) {
          var pCur = softmaxProbRight(theta[0], theta[1]);
          var a = traj.actions[t];
          var scoreR = (a === 1 ? 1 : 0) - pCur;
          var delta = useBaseline ? (G[t] - w) : G[t];
          var gR = gammaPow * delta * scoreR;
          epGradR += gR;
          epAbsGrad += Math.abs(gR);

          if (useBaseline) {
            w += alphaW * delta;
          }
          theta[0] -= alphaTheta * gR;
          theta[1] += alphaTheta * gR;
          clampTheta(theta, epsFloor);
          gammaPow *= gamma;
        }

        pArr[ep] = softmaxProbRight(theta[0], theta[1]);
        wArr[ep] = w;
        gArr[ep] = epGradR;
        gAbsArr[ep] = epAbsGrad / Math.max(1, T);
      }

      returnsByRun[r] = retArr;
      pByRun[r] = pArr;
      wByRun[r] = wArr;
      gradByRun[r] = gArr;
      stepGradAbsByRun[r] = gAbsArr;
    }

    var retStats = summarizeMatrix(returnsByRun, episodes);
    var pStats = summarizeMatrix(pByRun, episodes);
    var wStats = summarizeMatrix(wByRun, episodes);
    var gStats = summarizeMatrix(gradByRun, episodes);
    var gAbsStats = summarizeMatrix(stepGradAbsByRun, episodes);

    // Per-episode variance of the episode gradient estimate across runs
    var gradVar = new Array(episodes);
    var totVar = 0;
    for (var e = 0; e < episodes; e++) {
      var v = gStats.std[e] * gStats.std[e];
      gradVar[e] = v;
      totVar += v;
    }

    var tailStart = Math.max(0, Math.floor(episodes * 0.9));
    var tailSum = 0, tailPSum = 0, tailCnt = episodes - tailStart;
    for (var te = tailStart; te < episodes; te++) {
      tailSum += retStats.mean[te];
      tailPSum += pStats.mean[te];
    }

    return {
      episodes: episodes,
      runs: runs,
      useBaseline: useBaseline,
      alphaTheta: alphaTheta,
      alphaW: alphaW,
      meanReturn: retStats.mean,
      stdReturn: retStats.std,
      q25Return: retStats.q25,
      q50Return: retStats.q50,
      q75Return: retStats.q75,
      meanP: pStats.mean,
      meanBaseline: wStats.mean,
      gradMean: gStats.mean,
      gradStd: gStats.std,
      gradVar: gradVar,
      meanStepGradAbs: gAbsStats.mean,
      meanGradVar: totVar / episodes,
      finalReturn: tailSum / Math.max(1, tailCnt),
      finalP: tailPSum / Math.max(1, tailCnt)
    };
  }

  /* ---------- Fixed-policy baseline unbiasedness & variance experiment ----------
     Evaluates g_hat(b) = sum_{t=0}^{T-1} (G_t - b) * (I(A_t=R) - p) on independent
     Short Corridor episodes at fixed p, comparing b = 0 vs b = v0(p) vs optimal b*. */
  function baselineVarianceCheck(cfg) {
    var p = cfg.p == null ? 0.35 : cfg.p;
    var episodes = cfg.episodes == null ? 4000 : cfg.episodes;
    var seed = cfg.seed == null ? 101 : (cfg.seed >>> 0);
    var maxSteps = cfg.maxSteps == null ? 1000 : cfg.maxSteps;
    var rng = mulberry32(seed);
    var exact = corridorValueExact(p);
    var baselines = cfg.baselines || [0, exact.v0, exact.v1];

    var samples = baselines.map(function () { return new Float64Array(episodes); });
    for (var ep = 0; ep < episodes; ep++) {
      var traj = corridorEpisode(rng, p, maxSteps);
      var T = traj.steps;
      for (var bi = 0; bi < baselines.length; bi++) {
        var b = baselines[bi];
        var g = 0;
        for (var t = 0; t < T; t++) {
          var Gt = -(T - t);
          var scoreR = (traj.actions[t] === 1 ? 1 : 0) - p;
          g += (Gt - b) * scoreR;
        }
        samples[bi][ep] = g;
      }
    }

    var results = baselines.map(function (b, bi) {
      var arr = samples[bi], sum = 0;
      for (var i = 0; i < episodes; i++) sum += arr[i];
      var mean = sum / episodes;
      var sq = 0;
      for (var j = 0; j < episodes; j++) {
        var d = arr[j] - mean;
        sq += d * d;
      }
      var variance = sq / (episodes - 1);
      return {
        baseline: b,
        mean: mean,
        variance: variance,
        std: Math.sqrt(variance),
        se: Math.sqrt(variance / episodes)
      };
    });

    // Analytical score-function gradient for single logit theta_R:
    // Because d/d(theta_R) log pi(a) = I(a=R) - p = d/d(diff) log pi(a),
    // E[sum_t G_t (I(A_t=R) - p)] = d v0 / d(diff) = p(1 - p) * d v0 / d p
    var trueGrad = p * (1 - p) * corridorDv0Dp(p);
    return { p: p, exact: exact, trueGrad: trueGrad, results: results };
  }

  /* ---------- One-Step Actor-Critic (Sutton & Barto Section 13.5) ----------
     Supports:
       env: 'corridor' (Short Corridor, state-aliased actor; criticMode 'tabular' [3 state values]
                        or 'aliased' [1 shared weight, demonstrating bootstrapping bias])
       env: 'line5'    (5-state corridor with distinct state features for both actor & critic) */
  function runActorCritic(cfg) {
    var env = cfg.env || 'corridor';
    var criticMode = cfg.criticMode || 'tabular';
    var alphaTheta = cfg.alphaTheta == null ? (env === 'line5' ? 0.15 : 0.015) : cfg.alphaTheta;
    var alphaW = cfg.alphaW == null ? (env === 'line5' ? 0.25 : 0.1) : cfg.alphaW;
    var episodes = cfg.episodes == null ? 500 : cfg.episodes;
    var runs = cfg.runs == null ? 30 : cfg.runs;
    var seed = cfg.seed == null ? 42 : (cfg.seed >>> 0);
    var pInit = cfg.pInit == null ? (env === 'line5' ? 0.3 : 0.05) : cfg.pInit;
    var epsFloor = cfg.epsFloor == null ? (env === 'line5' ? 0.01 : 0.05) : cfg.epsFloor;
    var maxSteps = cfg.maxSteps == null ? 1000 : cfg.maxSteps;
    var gamma = cfg.gamma == null ? 1.0 : cfg.gamma;

    var returnsByRun = new Array(runs);
    var pByRun = new Array(runs);
    var tdErrSqByRun = new Array(runs);
    var gradByRun = new Array(runs);

    for (var r = 0; r < runs; r++) {
      var rng = mulberry32((seed + Math.imul(r + 1, 0x9E3779B1)) >>> 0);
      var nStates = env === 'line5' ? LINE5_N : 3;
      var actorPerState = env === 'line5';
      var thetas = [];
      for (var i = 0; i < (actorPerState ? nStates : 1); i++) {
        var th = initThetaFromP(pInit);
        clampTheta(th, epsFloor);
        thetas.push(th);
      }
      var criticDim = (env === 'corridor' && criticMode === 'aliased') ? 1 : nStates;
      var w = new Float64Array(criticDim);

      var retArr = new Float64Array(episodes);
      var pArr = new Float64Array(episodes);
      var tdSqArr = new Float64Array(episodes);
      var gArr = new Float64Array(episodes);

      for (var ep = 0; ep < episodes; ep++) {
        var s = 0, I = 1, totalR = 0, epGrad = 0, tdSq = 0, steps = 0;
        for (var t = 0; t < maxSteps; t++) {
          var thS = actorPerState ? thetas[s] : thetas[0];
          var pCur = softmaxProbRight(thS[0], thS[1]);
          var a = rng() < pCur ? 1 : 0;
          var tr = env === 'line5' ? line5Step(s, a === 1) : corridorStep(s, a === 1);
          totalR += tr.reward;
          steps++;

          var cIdx = criticDim === 1 ? 0 : s;
          var vS = w[cIdx];
          var vNext = tr.done ? 0 : w[criticDim === 1 ? 0 : tr.nextState];
          var delta = tr.reward + gamma * vNext - vS;
          tdSq += delta * delta;

          // Critic update
          w[cIdx] += alphaW * delta;

          // Actor update (online, immediately at step t!)
          var scoreR = (a === 1 ? 1 : 0) - pCur;
          var gR = I * delta * scoreR;
          epGrad += gR;
          thS[0] -= alphaTheta * gR;
          thS[1] += alphaTheta * gR;
          clampTheta(thS, epsFloor);

          I *= gamma;
          s = tr.nextState;
          if (tr.done) break;
        }
        retArr[ep] = totalR;
        if (actorPerState) {
          var pAvg = 0;
          for (var si = 0; si < nStates; si++) pAvg += softmaxProbRight(thetas[si][0], thetas[si][1]);
          pArr[ep] = pAvg / nStates;
        } else {
          pArr[ep] = softmaxProbRight(thetas[0][0], thetas[0][1]);
        }
        tdSqArr[ep] = tdSq / Math.max(1, steps);
        gArr[ep] = epGrad;
      }

      returnsByRun[r] = retArr;
      pByRun[r] = pArr;
      tdErrSqByRun[r] = tdSqArr;
      gradByRun[r] = gArr;
    }

    var retStats = summarizeMatrix(returnsByRun, episodes);
    var pStats = summarizeMatrix(pByRun, episodes);
    var tdStats = summarizeMatrix(tdErrSqByRun, episodes);
    var gStats = summarizeMatrix(gradByRun, episodes);

    var gradVar = new Array(episodes);
    var totVar = 0;
    for (var e = 0; e < episodes; e++) {
      var v = gStats.std[e] * gStats.std[e];
      gradVar[e] = v;
      totVar += v;
    }

    var tailStart = Math.max(0, Math.floor(episodes * 0.9));
    var tailSum = 0, tailPSum = 0, tailCnt = episodes - tailStart;
    for (var te = tailStart; te < episodes; te++) {
      tailSum += retStats.mean[te];
      tailPSum += pStats.mean[te];
    }

    return {
      env: env,
      criticMode: criticMode,
      episodes: episodes,
      runs: runs,
      alphaTheta: alphaTheta,
      alphaW: alphaW,
      meanReturn: retStats.mean,
      stdReturn: retStats.std,
      q25Return: retStats.q25,
      q50Return: retStats.q50,
      q75Return: retStats.q75,
      meanP: pStats.mean,
      meanTdSq: tdStats.mean,
      gradStd: gStats.std,
      gradVar: gradVar,
      meanGradVar: totVar / episodes,
      finalReturn: tailSum / Math.max(1, tailCnt),
      finalP: tailPSum / Math.max(1, tailCnt)
    };
  }

  /* ---------- REINFORCE on 5-state corridor (for fair Tab 3 side-by-side comparison) ---------- */
  function runReinforceLine5(cfg) {
    var alphaTheta = cfg.alphaTheta == null ? 0.05 : cfg.alphaTheta;
    var alphaW = cfg.alphaW == null ? 0.15 : cfg.alphaW;
    var useBaseline = cfg.useBaseline == null ? true : Boolean(cfg.useBaseline);
    var episodes = cfg.episodes == null ? 500 : cfg.episodes;
    var runs = cfg.runs == null ? 30 : cfg.runs;
    var seed = cfg.seed == null ? 42 : (cfg.seed >>> 0);
    var pInit = cfg.pInit == null ? 0.3 : cfg.pInit;
    var epsFloor = cfg.epsFloor == null ? 0.01 : cfg.epsFloor;
    var maxSteps = cfg.maxSteps == null ? 1000 : cfg.maxSteps;
    var gamma = cfg.gamma == null ? 1.0 : cfg.gamma;

    var returnsByRun = new Array(runs);
    var pByRun = new Array(runs);
    var gradByRun = new Array(runs);

    for (var r = 0; r < runs; r++) {
      var rng = mulberry32((seed + Math.imul(r + 1, 0x9E3779B1)) >>> 0);
      var thetas = [];
      for (var i = 0; i < LINE5_N; i++) {
        var th = initThetaFromP(pInit);
        clampTheta(th, epsFloor);
        thetas.push(th);
      }
      var w = new Float64Array(LINE5_N);
      var retArr = new Float64Array(episodes);
      var pArr = new Float64Array(episodes);
      var gArr = new Float64Array(episodes);

      for (var ep = 0; ep < episodes; ep++) {
        var s = 0, states = [], actions = [], rewards = [], totalR = 0;
        for (var t = 0; t < maxSteps; t++) {
          var pCur = softmaxProbRight(thetas[s][0], thetas[s][1]);
          var a = rng() < pCur ? 1 : 0;
          var tr = line5Step(s, a === 1);
          states.push(s); actions.push(a); rewards.push(tr.reward);
          totalR += tr.reward;
          s = tr.nextState;
          if (tr.done) break;
        }
        var T = states.length;
        var G = new Float64Array(T);
        var runG = 0;
        for (var k = T - 1; k >= 0; k--) { runG = rewards[k] + gamma * runG; G[k] = runG; }

        var epGrad = 0, gPow = 1;
        for (var u = 0; u < T; u++) {
          var su = states[u], au = actions[u];
          var thU = thetas[su];
          var pU = softmaxProbRight(thU[0], thU[1]);
          var scoreR = (au === 1 ? 1 : 0) - pU;
          var delta = useBaseline ? (G[u] - w[su]) : G[u];
          var gR = gPow * delta * scoreR;
          epGrad += gR;
          if (useBaseline) w[su] += alphaW * delta;
          thU[0] -= alphaTheta * gR;
          thU[1] += alphaTheta * gR;
          clampTheta(thU, epsFloor);
          gPow *= gamma;
        }
        retArr[ep] = totalR;
        var pAvg = 0;
        for (var si = 0; si < LINE5_N; si++) pAvg += softmaxProbRight(thetas[si][0], thetas[si][1]);
        pArr[ep] = pAvg / LINE5_N;
        gArr[ep] = epGrad;
      }
      returnsByRun[r] = retArr;
      pByRun[r] = pArr;
      gradByRun[r] = gArr;
    }

    var retStats = summarizeMatrix(returnsByRun, episodes);
    var pStats = summarizeMatrix(pByRun, episodes);
    var gStats = summarizeMatrix(gradByRun, episodes);
    var gradVar = new Array(episodes), totVar = 0;
    for (var e = 0; e < episodes; e++) { gradVar[e] = gStats.std[e] * gStats.std[e]; totVar += gradVar[e]; }
    var tailStart = Math.max(0, Math.floor(episodes * 0.9));
    var tailSum = 0, tailPSum = 0, tailCnt = episodes - tailStart;
    for (var te = tailStart; te < episodes; te++) { tailSum += retStats.mean[te]; tailPSum += pStats.mean[te]; }
    return {
      env: 'line5',
      episodes: episodes,
      runs: runs,
      useBaseline: useBaseline,
      alphaTheta: alphaTheta,
      alphaW: alphaW,
      meanReturn: retStats.mean,
      stdReturn: retStats.std,
      q25Return: retStats.q25,
      q50Return: retStats.q50,
      q75Return: retStats.q75,
      meanP: pStats.mean,
      gradStd: gStats.std,
      gradVar: gradVar,
      meanGradVar: totVar / episodes,
      finalReturn: tailSum / Math.max(1, tailCnt),
      finalP: tailPSum / Math.max(1, tailCnt)
    };
  }

  /* ---------- Single-episode interactive trace comparing Actor-Critic vs REINFORCE ----------
     Generates one episode trajectory and returns step-by-step records for both:
     - Actor-Critic (updates online at every step t using TD error delta_t = R + gamma*v(S') - v(S))
     - REINFORCE with baseline (waits until terminal state, then computes G_t and updates retroactively)
     Both start from the same initial state values vInit and policy pInit on the same episode actions
     (or Actor-Critic updates online as it walks). */
  function traceEpisodeCompare(cfg) {
    var env = cfg.env || 'corridor';
    var criticMode = cfg.criticMode || 'tabular';
    var pInit = cfg.pInit == null ? (env === 'line5' ? 0.45 : 0.45) : cfg.pInit;
    var alphaTheta = cfg.alphaTheta == null ? 0.05 : cfg.alphaTheta;
    var alphaW = cfg.alphaW == null ? 0.2 : cfg.alphaW;
    var gamma = cfg.gamma == null ? 1.0 : cfg.gamma;
    var epsFloor = cfg.epsFloor == null ? 0.05 : cfg.epsFloor;
    var seed = cfg.seed == null ? 7 : (cfg.seed >>> 0);
    var maxSteps = cfg.maxSteps == null ? 40 : cfg.maxSteps;
    var rng = mulberry32(seed);

    var nStates = env === 'line5' ? LINE5_N : 3;
    var actorPerState = env === 'line5';
    var criticDim = (env === 'corridor' && criticMode === 'aliased') ? 1 : nStates;

    // Seed initial critic slightly warm (or zero if cfg.zeroInit) so TD deltas are illuminating
    var wInit = new Float64Array(criticDim);
    if (!cfg.zeroInit) {
      if (env === 'corridor') {
        if (criticDim === 1) wInit[0] = -10;
        else { wInit[0] = -11; wInit[1] = -9; wInit[2] = -5; }
      } else {
        for (var ii = 0; ii < nStates; ii++) wInit[ii] = -(nStates - ii) * 1.4;
      }
    }

    var wAC = new Float64Array(wInit);
    var thAC = [];
    for (var i = 0; i < (actorPerState ? nStates : 1); i++) {
      var th = initThetaFromP(pInit);
      clampTheta(th, epsFloor);
      thAC.push(th);
    }

    var acSteps = [];
    var s = 0, I = 1, totalR = 0;
    for (var t = 0; t < maxSteps; t++) {
      var thS = actorPerState ? thAC[s] : thAC[0];
      var pBefore = softmaxProbRight(thS[0], thS[1]);
      var a = rng() < pBefore ? 1 : 0;
      var tr = env === 'line5' ? line5Step(s, a === 1) : corridorStep(s, a === 1);
      totalR += tr.reward;

      var cIdx = criticDim === 1 ? 0 : s;
      var vS = wAC[cIdx];
      var vNext = tr.done ? 0 : wAC[criticDim === 1 ? 0 : tr.nextState];
      var delta = tr.reward + gamma * vNext - vS;
      var dw = alphaW * delta;
      wAC[cIdx] += dw;

      var scoreR = (a === 1 ? 1 : 0) - pBefore;
      var dThetaR = alphaTheta * I * delta * scoreR;
      thS[0] -= dThetaR;
      thS[1] += dThetaR;
      clampTheta(thS, epsFloor);
      var pAfter = softmaxProbRight(thS[0], thS[1]);

      acSteps.push({
        t: t,
        s: s,
        a: a,
        r: tr.reward,
        nextState: tr.nextState,
        done: tr.done,
        vS: vS,
        vNext: vNext,
        delta: delta,
        dw: dw,
        scoreR: scoreR,
        dThetaR: dThetaR,
        pBefore: pBefore,
        pAfter: pAfter,
        wSnap: Array.from(wAC)
      });

      I *= gamma;
      s = tr.nextState;
      if (tr.done) break;
    }

    // Now compute what REINFORCE (with baseline) would compute on the EXACT same episode
    // once it finishes (retroactive Monte Carlo returns G_t):
    var T = acSteps.length;
    var G = new Float64Array(T);
    var runG = 0;
    for (var k = T - 1; k >= 0; k--) {
      runG = acSteps[k].r + gamma * runG;
      G[k] = runG;
    }
    var wMC = new Float64Array(wInit);
    var thMC = [];
    for (var m = 0; m < (actorPerState ? nStates : 1); m++) {
      var tm = initThetaFromP(pInit);
      clampTheta(tm, epsFloor);
      thMC.push(tm);
    }
    var mcSteps = [];
    var gPow = 1;
    for (var u = 0; u < T; u++) {
      var st = acSteps[u];
      var thM = actorPerState ? thMC[st.s] : thMC[0];
      var pB = softmaxProbRight(thM[0], thM[1]);
      var cIdxM = criticDim === 1 ? 0 : st.s;
      var bVal = wMC[cIdxM];
      var deltaM = G[u] - bVal;
      var dwM = alphaW * deltaM;
      wMC[cIdxM] += dwM;
      var scM = (st.a === 1 ? 1 : 0) - pB;
      var dThM = alphaTheta * gPow * deltaM * scM;
      thM[0] -= dThM;
      thM[1] += dThM;
      clampTheta(thM, epsFloor);
      mcSteps.push({
        t: u,
        s: st.s,
        a: st.a,
        G: G[u],
        baseline: bVal,
        delta: deltaM,
        dw: dwM,
        scoreR: scM,
        dThetaR: dThM,
        pBefore: pB,
        pAfter: softmaxProbRight(thM[0], thM[1])
      });
      gPow *= gamma;
    }

    return {
      env: env,
      criticMode: criticMode,
      nStates: nStates,
      totalReturn: totalR,
      steps: T,
      done: acSteps[T - 1].done,
      acSteps: acSteps,
      mcSteps: mcSteps
    };
  }

  var api = {
    mulberry32: mulberry32,
    CORRIDOR_TERMINAL: CORRIDOR_TERMINAL,
    P_STAR: P_STAR,
    V_STAR: V_STAR,
    LINE5_N: LINE5_N,
    LINE5_TERMINAL: LINE5_TERMINAL,
    corridorValueExact: corridorValueExact,
    corridorValueLinear: corridorValueLinear,
    corridorDv0Dp: corridorDv0Dp,
    corridorStep: corridorStep,
    corridorEpisode: corridorEpisode,
    line5Step: line5Step,
    softmaxProbRight: softmaxProbRight,
    initThetaFromP: initThetaFromP,
    runReinforce: runReinforce,
    runReinforceLine5: runReinforceLine5,
    baselineVarianceCheck: baselineVarianceCheck,
    runActorCritic: runActorCritic,
    traceEpisodeCompare: traceEpisodeCompare
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.PGSIM = api;
})(typeof window !== 'undefined' ? window : this);
