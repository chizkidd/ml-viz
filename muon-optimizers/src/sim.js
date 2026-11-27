/* Muon, Newton-Schulz orthogonalization, AdamW vs Muon matrix optimization,
   and MuonClip (QK-Clip) multi-head attention simulations.
   Pure functions, no DOM. Works in a browser (window.MUONSIM) and in Node (module.exports). */
(function (root) {
  'use strict';

  /* ---------- seeded PRNG (mulberry32 + Box-Muller Gaussian) ---------- */
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeRandn(rng) {
    var spare = null;
    return function () {
      if (spare !== null) {
        var v = spare;
        spare = null;
        return v;
      }
      var u = 0, w = 0;
      while (u <= 1e-12) u = rng();
      w = rng();
      var r = Math.sqrt(-2 * Math.log(u));
      var theta = 2 * Math.PI * w;
      spare = r * Math.sin(theta);
      return r * Math.cos(theta);
    };
  }

  /* ---------- small dense matrix utilities (2D arrays A[row][col]) ---------- */
  function zeros(m, n) {
    var A = new Array(m);
    for (var i = 0; i < m; i++) {
      var row = new Array(n);
      for (var j = 0; j < n; j++) row[j] = 0;
      A[i] = row;
    }
    return A;
  }

  function eye(n) {
    var I = zeros(n, n);
    for (var i = 0; i < n; i++) I[i][i] = 1;
    return I;
  }

  function cloneMat(A) {
    var m = A.length, out = new Array(m);
    for (var i = 0; i < m; i++) out[i] = A[i].slice();
    return out;
  }

  function transpose(A) {
    var m = A.length, n = A[0].length;
    var T = zeros(n, m);
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) T[j][i] = A[i][j];
    }
    return T;
  }

  function matmul(A, B) {
    var m = A.length, k = A[0].length, n = B[0].length;
    var C = zeros(m, n);
    for (var i = 0; i < m; i++) {
      var ai = A[i], ci = C[i];
      for (var p = 0; p < k; p++) {
        var aip = ai[p];
        if (aip === 0) continue;
        var bp = B[p];
        for (var j = 0; j < n; j++) ci[j] += aip * bp[j];
      }
    }
    return C;
  }

  function matAdd(A, B, alpha, beta) {
    if (alpha === undefined) alpha = 1;
    if (beta === undefined) beta = 1;
    var m = A.length, n = A[0].length;
    var C = zeros(m, n);
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) C[i][j] = alpha * A[i][j] + beta * B[i][j];
    }
    return C;
  }

  function matScale(A, s) {
    var m = A.length, n = A[0].length;
    var C = zeros(m, n);
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) C[i][j] = A[i][j] * s;
    }
    return C;
  }

  function frobNorm(A) {
    var s = 0, m = A.length, n = A[0].length;
    for (var i = 0; i < m; i++) {
      for (var j = 0; j < n; j++) {
        var v = A[i][j];
        s += v * v;
      }
    }
    return Math.sqrt(s);
  }

  function gram(A) {
    return matmul(transpose(A), A);
  }

  function orthError(X) {
    var GtG = gram(X);
    var n = GtG.length;
    var s = 0;
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        var d = GtG[i][j] - (i === j ? 1 : 0);
        s += d * d;
      }
    }
    return Math.sqrt(s);
  }

  /* ---------- Exact symmetric Jacobi eigendecomposition ---------- */
  /* Given real symmetric n x n matrix S, returns { values, vectors } sorted descending,
     such that S = V * diag(values) * V^T with V orthogonal. */
  function jacobiEigenSym(S, maxIter, tol) {
    var n = S.length;
    maxIter = maxIter || (120 * n * n);
    tol = tol || 1e-13;
    var A = cloneMat(S);
    var V = eye(n);

    for (var iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal entry |A[p][q]|
      var p = 0, q = 1, maxOff = 0;
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          var v = Math.abs(A[i][j]);
          if (v > maxOff) { maxOff = v; p = i; q = j; }
        }
      }
      if (maxOff <= tol) break;

      var app = A[p][p], aqq = A[q][q], apq = A[p][q];
      var tau = (aqq - app) / (2 * apq);
      var t = tau >= 0 ? 1 / (tau + Math.sqrt(1 + tau * tau)) : -1 / (-tau + Math.sqrt(1 + tau * tau));
      var c = 1 / Math.sqrt(1 + t * t);
      var s = t * c;

      // Update A = J^T A J
      A[p][p] = app - t * apq;
      A[q][q] = aqq + t * apq;
      A[p][q] = 0;
      A[q][p] = 0;
      for (var k = 0; k < n; k++) {
        if (k === p || k === q) continue;
        var akp = A[k][p], akq = A[k][q];
        var nkp = c * akp - s * akq;
        var nkq = s * akp + c * akq;
        A[k][p] = nkp; A[p][k] = nkp;
        A[k][q] = nkq; A[q][k] = nkq;
      }
      // Accumulate eigenvectors V = V J
      for (var r = 0; r < n; r++) {
        var vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
      }
    }

    var idx = [];
    for (var m = 0; m < n; m++) idx.push(m);
    idx.sort(function (a, b) { return A[b][b] - A[a][a]; });

    var values = new Array(n);
    var vectors = zeros(n, n);
    for (var col = 0; col < n; col++) {
      var src = idx[col];
      values[col] = A[src][src];
      for (var row = 0; row < n; row++) vectors[row][col] = V[row][src];
    }
    return { values: values, vectors: vectors };
  }

  /* Exact singular values of m x n matrix G via Jacobi on G^T G (or G G^T if m < n). */
  function singularValues(G) {
    var m = G.length, n = G[0].length;
    var sym = m >= n ? matmul(transpose(G), G) : matmul(G, transpose(G));
    var eig = jacobiEigenSym(sym);
    return eig.values.map(function (lam) { return Math.sqrt(Math.max(0, lam)); });
  }

  /* Full SVD for square/tall matrices (m >= n): G = U * diag(S) * V^T */
  function svd(G) {
    var m = G.length, n = G[0].length;
    var eig = jacobiEigenSym(matmul(transpose(G), G));
    var S = eig.values.map(function (lam) { return Math.sqrt(Math.max(0, lam)); });
    var V = eig.vectors;
    var GV = matmul(G, V);
    var U = zeros(m, n);
    for (var j = 0; j < n; j++) {
      if (S[j] > 1e-12) {
        for (var i = 0; i < m; i++) U[i][j] = GV[i][j] / S[j];
      } else {
        // Fill nullspace column orthonormal to previous columns via Gram-Schmidt
        var vec = new Array(m);
        for (var r = 0; r < m; r++) vec[r] = (r === j ? 1 : 0);
        for (var prev = 0; prev < j; prev++) {
          var dot = 0;
          for (var r2 = 0; r2 < m; r2++) dot += vec[r2] * U[r2][prev];
          for (var r3 = 0; r3 < m; r3++) vec[r3] -= dot * U[r3][prev];
        }
        var nrm = 0;
        for (var r4 = 0; r4 < m; r4++) nrm += vec[r4] * vec[r4];
        nrm = Math.sqrt(nrm);
        if (nrm > 1e-12) {
          for (var r5 = 0; r5 < m; r5++) U[r5][j] = vec[r5] / nrm;
        }
      }
    }
    return { U: U, S: S, V: V };
  }

  /* Random n x n orthogonal matrix via modified Gram-Schmidt (2 passes for precision). */
  function randomOrthogonal(n, rng) {
    var randn = makeRandn(rng);
    var Q = zeros(n, n);
    for (var j = 0; j < n; j++) {
      var v = new Array(n);
      for (var i = 0; i < n; i++) v[i] = randn();
      for (var pass = 0; pass < 2; pass++) {
        for (var k = 0; k < j; k++) {
          var dot = 0;
          for (var i2 = 0; i2 < n; i2++) dot += v[i2] * Q[i2][k];
          for (var i3 = 0; i3 < n; i3++) v[i3] -= dot * Q[i3][k];
        }
      }
      var nrm = 0;
      for (var i4 = 0; i4 < n; i4++) nrm += v[i4] * v[i4];
      nrm = Math.sqrt(nrm) || 1;
      for (var i5 = 0; i5 < n; i5++) Q[i5][j] = v[i5] / nrm;
    }
    return Q;
  }

  /* Build a random n x n matrix with geometrically spaced singular values [1 .. 1/cond]. */
  function makeSkewedMatrix(n, cond, seed) {
    var rng = mulberry32(seed == null ? 42 : seed);
    var U = randomOrthogonal(n, rng);
    var V = randomOrthogonal(n, rng);
    var sigmas = new Array(n);
    var S = zeros(n, n);
    for (var i = 0; i < n; i++) {
      var t = n > 1 ? i / (n - 1) : 0;
      sigmas[i] = Math.pow(cond, -t);
      S[i][i] = sigmas[i];
    }
    return {
      G: matmul(matmul(U, S), transpose(V)),
      sigmas: sigmas,
      U: U,
      V: V
    };
  }

  /* ---------- 1. Newton-Schulz Orthogonalization ---------- */
  var NS_PRESETS = {
    keller: { id: 'keller', label: 'Keller / Muon quintic', a: 3.4445, b: -4.7750, c: 2.0315 },
    blog:   { id: 'blog',   label: 'Quintic (2, -1.5, 0.5)', a: 2.0,    b: -1.5,    c: 0.5 },
    cubic:  { id: 'cubic',  label: 'Classic cubic (1.5s - 0.5s^3)', a: 1.5, b: -0.5, c: 0.0 }
  };

  function evalNSPoly(s, coeffs) {
    var s2 = s * s;
    return coeffs.a * s + coeffs.b * s * s2 + coeffs.c * s * s2 * s2;
  }

  function polyCurve(k, coeffs, numPts) {
    numPts = numPts || 101;
    var pts = new Array(numPts);
    for (var i = 0; i < numPts; i++) {
      var s = i / (numPts - 1);
      var y = s;
      for (var it = 0; it < k; it++) y = evalNSPoly(y, coeffs);
      pts[i] = { s: s, y: y };
    }
    return pts;
  }

  function newtonSchulzStep(X, coeffs) {
    var XXt = matmul(X, transpose(X));
    var B = matScale(XXt, coeffs.b);
    if (coeffs.c !== 0) {
      var XXt2 = matmul(XXt, XXt);
      B = matAdd(B, XXt2, 1, coeffs.c);
    }
    return matAdd(matScale(X, coeffs.a), matmul(B, X), 1, 1);
  }

  /* Fast orthogonalizer returning only final X (used inside Muon optimizer). */
  function newtonSchulz5(G, steps, coeffs) {
    steps = steps == null ? 5 : steps;
    coeffs = coeffs || NS_PRESETS.keller;
    var nrm = frobNorm(G);
    if (nrm < 1e-12) return cloneMat(G);
    var X = matScale(G, 1 / (nrm + 1e-7));
    for (var k = 0; k < steps; k++) X = newtonSchulzStep(X, coeffs);
    return X;
  }

  function newtonSchulzSteps(G, steps, coeffs) {
    steps = steps == null ? 5 : steps;
    coeffs = coeffs || { a: 3.4445, b: -4.7750, c: 2.0315 };
    var rawSV = singularValues(G);
    var rawNorm = frobNorm(G);
    var rawCond = rawSV[0] / Math.max(1e-12, rawSV[rawSV.length - 1]);

    // Step 0: Frobenius normalization X_0 = G / (||G||_F + 1e-7)
    var X = matScale(G, 1 / (rawNorm + 1e-7));
    var history = [];

    function record(iter, M) {
      var sv = singularValues(M);
      var GtG = gram(M);
      var err = orthError(M);
      var cnd = sv[0] / Math.max(1e-12, sv[sv.length - 1]);
      history.push({
        iter: iter,
        X: cloneMat(M),
        sv: sv,
        gram: GtG,
        orthErr: err,
        cond: cnd,
        frob: frobNorm(M),
        curve: polyCurve(iter, coeffs, 101)
      });
    }

    record(0, X);
    for (var k = 1; k <= steps; k++) {
      X = newtonSchulzStep(X, coeffs);
      record(k, X);
    }

    return {
      coeffs: { a: coeffs.a, b: coeffs.b, c: coeffs.c },
      rawSV: rawSV,
      rawNorm: rawNorm,
      rawCond: rawCond,
      steps: steps,
      history: history
    };
  }

  /* ---------- 2. AdamW vs Muon on Ill-Conditioned Matrix Regression ---------- */
  /* Objective: L(W) = 0.5 * ||A (W - W*)||_F^2 = 0.5 * ||A W - Y||_F^2
     where A = R(theta) * diag(1, 1/sqrt(cond)) * R(phi)^T so A^T A has exact eigenvalues
     1 and 1/cond, rotated by angle phi relative to the coordinate axes!
     Also tracks the exact 2D eigen-coordinate projection of column 0 of (W - W*) so we
     can plot trajectories over exact elliptical loss contours. */
  function rot2(rad) {
    var c = Math.cos(rad), s = Math.sin(rad);
    return [[c, -s], [s, c]];
  }

  function runLandscape(opts) {
    opts = opts || {};
    var fn = opts.fn || 'matrix';       // 'matrix' (rotated 2x2) | 'aligned' (axis-aligned 2x2) | 'bowl2d'
    var cond = Math.max(1, opts.cond == null ? 50 : opts.cond);
    var lr = opts.lr == null ? 0.10 : opts.lr;
    var beta = opts.beta == null ? 0.90 : opts.beta;
    var steps = opts.steps == null ? 100 : opts.steps;
    var schedule = opts.schedule || 'cosine'; // 'cosine' | 'constant'
    var angleDeg = fn === 'aligned' ? 0 : (opts.angleDeg == null ? 36 : opts.angleDeg);
    var phi = angleDeg * Math.PI / 180;

    // Build A so H = A^T A = R(phi) diag(1, 1/cond) R(phi)^T
    var Rphi = rot2(phi);
    var Rtheta = rot2(fn === 'aligned' ? 0 : 0.42);
    var D = [[1, 0], [0, 1 / Math.sqrt(cond)]];
    var A = matmul(matmul(Rtheta, D), transpose(Rphi));
    var H = matmul(transpose(A), A); // exact Hessian on columns of W

    // Target W* and starting W0 (starting offset aligned so both strong and weak directions have error)
    var Wstar = [[0.4, -0.3], [0.2, 0.5]];
    var delta0 = fn === 'bowl2d'
      ? matmul(Rphi, [[1.4, 0], [1.4, 0]])
      : matmul(Rphi, [[1.3, -0.7], [1.4, 1.2]]);
    var W0 = matAdd(Wstar, delta0, 1, 1);
    var Y = matmul(A, Wstar);

    function lossOf(W) {
      var R = matAdd(matmul(A, W), Y, 1, -1);
      var f = frobNorm(R);
      return 0.5 * f * f;
    }

    function gradOf(W) {
      var R = matAdd(matmul(A, W), Y, 1, -1);
      return matmul(transpose(A), R);
    }

    // Project (W - W*) column 0 into eigenbasis of H for 2D contour plot
    // eigen coord z = Rphi^T * (W - W*)[:,0] -> contour energy 0.5*(z0^2 + z1^2 / cond)
    function proj2D(W) {
      var dx = W[0][0] - Wstar[0][0];
      var dy = W[1][0] - Wstar[1][0];
      var c = Math.cos(phi), s = Math.sin(phi);
      return [c * dx + s * dy, -s * dx + c * dy];
    }

    function lrAt(t) {
      if (schedule === 'constant') return lr;
      // Cosine decay to 1% of peak lr so unit-norm updates (AdamW & Muon) settle cleanly
      var frac = t / Math.max(1, steps);
      return lr * (0.01 + 0.99 * 0.5 * (1 + Math.cos(Math.PI * frac)));
    }

    function runOne(name) {
      var W = cloneMat(W0);
      var M = zeros(2, 2);
      var V = zeros(2, 2);
      var beta2 = 0.999, eps = 1e-8;
      var losses = [lossOf(W)];
      var traj = [proj2D(W)];
      var updateSVs = [];
      var updateConds = [];
      var hit1e3 = null;
      if (losses[0] <= 1e-3) hit1e3 = 0;

      for (var t = 1; t <= steps; t++) {
        var G = gradOf(W);
        var stepLr = lrAt(t - 1);
        var U = zeros(2, 2);

        if (name === 'sgd') {
          U = G;
        } else if (name === 'momentum') {
          // Heavy-ball / EMA-scaled momentum: M <- beta*M + (1-beta)*G keeps step scale comparable,
          // or Nesterov-style with standard scale. Let's use classical momentum with Nesterov lookahead
          // M <- beta*M + G; U = G + beta*M scaled? Actually classical M <- beta*M + G with stepLr:
          // at lr <= 0.35, 0.35 < 2(1+beta) = 3.8 is always stable!
          M = matAdd(M, G, beta, 1);
          U = M;
        } else if (name === 'adamw') {
          M = matAdd(M, G, beta, 1 - beta);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 2; j++) {
              V[i][j] = beta2 * V[i][j] + (1 - beta2) * G[i][j] * G[i][j];
            }
          }
          var bc1 = 1 - Math.pow(beta, t);
          var bc2 = 1 - Math.pow(beta2, t);
          for (var i2 = 0; i2 < 2; i2++) {
            for (var j2 = 0; j2 < 2; j2++) {
              var mh = M[i2][j2] / bc1;
              var vh = V[i2][j2] / bc2;
              U[i2][j2] = mh / (Math.sqrt(vh) + eps);
            }
          }
        } else if (name === 'muon') {
          // Muon: Nesterov momentum + 5-step quintic Newton-Schulz orthogonalization
          M = matAdd(M, G, beta, 1);
          var Bmat = matAdd(G, M, 1, beta); // Nesterov lookahead (Keller Jordan Muon default)
          U = newtonSchulz5(Bmat, 5, NS_PRESETS.keller);
        }

        // Actual matrix displacement dW = -stepLr * U
        var dW = matScale(U, -stepLr);
        if (fn === 'bowl2d') {
          // Zero out column 1 in pure 2D bowl mode
          dW[0][1] = 0; dW[1][1] = 0;
        }
        W = matAdd(W, dW, 1, 1);

        var L = lossOf(W);
        losses.push(L);
        traj.push(proj2D(W));
        if (hit1e3 === null && L <= 1e-3) hit1e3 = t;

        var usv = singularValues(U);
        updateSVs.push(usv);
        updateConds.push(usv[0] / Math.max(1e-12, usv[usv.length - 1]));
      }

      // Compute median / geometric mean of update condition number across first half of steps
      var sampleN = Math.max(1, Math.floor(steps * 0.6));
      var sumLogCond = 0, sv0Sum = 0, sv1Sum = 0;
      for (var k = 0; k < sampleN; k++) {
        sumLogCond += Math.log(Math.min(1e6, updateConds[k]));
        var s0 = updateSVs[k][0], s1 = updateSVs[k][1];
        var mx = Math.max(s0, 1e-12);
        sv0Sum += s0 / mx;
        sv1Sum += s1 / mx;
      }

      return {
        name: name,
        losses: losses,
        finalLoss: losses[losses.length - 1],
        stepsTo1e3: hit1e3,
        traj: traj,
        updateSVs: updateSVs,
        updateConds: updateConds,
        avgNormSV: [sv0Sum / sampleN, sv1Sum / sampleN],
        geomUpdateCond: Math.exp(sumLogCond / sampleN),
        finalW: W
      };
    }

    var optimizers = {
      sgd: runOne('sgd'),
      momentum: runOne('momentum'),
      adamw: runOne('adamw'),
      muon: runOne('muon')
    };

    return {
      fn: fn,
      cond: cond,
      lr: lr,
      beta: beta,
      steps: steps,
      schedule: schedule,
      angleDeg: angleDeg,
      A: A,
      H: H,
      W0: W0,
      Wstar: Wstar,
      initLoss: lossOf(W0),
      optimizers: optimizers
    };
  }

  /* ---------- 3. MuonClip (QK-Clip) Multi-Head Attention Simulation ---------- */
  /* Simulates H attention heads with real d x d query/key projection matrices W_q^h, W_k^h
     over T normalized token embeddings X (T x d).
     Under Muon orthogonalized updates, outlier/runaway heads align W_q^h and W_k^h along
     dominant feature pairs, causing S_max^h = max_{i,j} (q_i^h . k_j^h) / sqrt(d) to grow
     quadratically and collapse attention softmax entropy toward 0.
     MuonClip (QK-Clip) monitors S_max^h and rescales ONLY offending heads:
       if S_max^h > tau: gamma_h = tau / S_max^h
                         W_q^h <- sqrt(gamma_h) W_q^h
                         W_k^h <- sqrt(gamma_h) W_k^h
     leaving healthy heads (S_max^h <= tau) completely untouched (gamma_h = 1). */
  function simQKClip(opts) {
    opts = opts || {};
    var heads = opts.heads == null ? 8 : opts.heads;
    var steps = opts.steps == null ? 100 : opts.steps;
    var tau = opts.tau == null ? 100 : opts.tau;
    var clipOn = opts.clipOn == null ? true : Boolean(opts.clipOn);
    var seed = opts.seed == null ? 2026 : opts.seed;

    var d = 8;   // head dimension
    var T = 10;  // sequence length
    var invSqrtD = 1 / Math.sqrt(d);
    var rng = mulberry32(seed);
    var randn = makeRandn(rng);

    // Normalized token embeddings X (T x d), unit-norm rows scaled mildly so initial logits ~ O(1..5)
    var X = zeros(T, d);
    for (var ti = 0; ti < T; ti++) {
      var nrm2 = 0;
      for (var j = 0; j < d; j++) {
        var v = randn();
        X[ti][j] = v;
        nrm2 += v * v;
      }
      var sc = Math.sqrt(d / (nrm2 || 1));
      for (var j2 = 0; j2 < d; j2++) X[ti][j2] *= sc;
    }

    // Choose runaway/outlier heads (3 of 8 heads act like attention-sink / high-alignment heads)
    var headProfiles = [];
    for (var h = 0; h < heads; h++) {
      var runaway = (h === 1 || h === 3 || h === 6) || (heads > 8 && h % 3 === 1);
      var driftStrength = runaway
        ? (h === 1 ? 1.15 : h === 3 ? 0.96 : 0.84)
        : (0.28 + 0.18 * rng());
      // Preferred token pair (tokQ, tokK) for runaway alignment pressure
      var tokQ = Math.floor(rng() * T);
      var tokK = Math.floor(rng() * T);
      // Random unit target direction u_h in R^d
      var u = new Array(d), un2 = 0;
      for (var k = 0; k < d; k++) { u[k] = randn(); un2 += u[k] * u[k]; }
      un2 = Math.sqrt(un2) || 1;
      for (var k2 = 0; k2 < d; k2++) u[k2] /= un2;

      // Initialize W_q^h, W_k^h so initial S_max is ~5..12 and initial attention entropy is ~0.85..0.95
      var Wq = zeros(d, d), Wk = zeros(d, d);
      for (var r = 0; r < d; r++) {
        for (var c = 0; c < d; c++) {
          Wq[r][c] = randn() * 0.44;
          Wk[r][c] = randn() * 0.44;
        }
      }
      headProfiles.push({
        head: h,
        runaway: runaway,
        driftStrength: driftStrength,
        tokQ: tokQ,
        tokK: tokK,
        u: u,
        Wq: Wq,
        Wk: Wk,
        Mq: zeros(d, d),
        Mk: zeros(d, d),
        cumGamma: 1
      });
    }

    function computeLogitStats(Wq, Wk) {
      var Q = matmul(X, Wq); // T x d
      var K = matmul(X, Wk); // T x d
      var maxLogit = -Infinity;
      var entropySum = 0;
      var logT = Math.log(T);
      // Scale factor mapping small d=8 dot products to realistic LLM logit magnitudes (equivalent to d_model norm)
      var logitGain = 1.6;
      // Softmax temperature matches realistic per-row logit spread in multi-token sequences
      var temp = 8.0;

      for (var i = 0; i < T; i++) {
        var row = new Array(T);
        var rowMax = -Infinity;
        for (var j = 0; j < T; j++) {
          var dot = 0;
          for (var k = 0; k < d; k++) dot += Q[i][k] * K[j][k];
          var s = dot * invSqrtD * logitGain;
          row[j] = s;
          if (s > rowMax) rowMax = s;
          if (s > maxLogit) maxLogit = s;
        }
        var sumExp = 0;
        var exps = new Array(T);
        for (var j2 = 0; j2 < T; j2++) {
          var e = Math.exp(Math.max(-60, (row[j2] - rowMax) / temp));
          exps[j2] = e;
          sumExp += e;
        }
        var Hrow = 0;
        for (var j3 = 0; j3 < T; j3++) {
          var p = exps[j3] / sumExp;
          if (p > 1e-15) Hrow -= p * Math.log(p);
        }
        entropySum += Hrow / logT;
      }
      return { maxLogit: maxLogit, entropy: entropySum / T };
    }

    var smaxSeries = new Array(heads);
    var rawSmaxSeries = new Array(heads);
    var gammaSeries = new Array(heads);
    var cumGammaSeries = new Array(heads);
    var entropySeries = new Array(heads);
    var clipCounts = new Array(heads);
    for (var h0 = 0; h0 < heads; h0++) {
      smaxSeries[h0] = [];
      rawSmaxSeries[h0] = [];
      gammaSeries[h0] = [];
      cumGammaSeries[h0] = [];
      entropySeries[h0] = [];
      clipCounts[h0] = 0;
    }

    var meanEntropy = [];
    var runawayEntropy = [];
    var globalMaxLogit = [];
    var stepClippedHeads = [];
    var beta = 0.85;

    // Record step 0
    var max0 = -Infinity, ent0 = 0, runEnt0 = 0, runCnt = 0;
    for (var h1 = 0; h1 < heads; h1++) {
      var st0 = computeLogitStats(headProfiles[h1].Wq, headProfiles[h1].Wk);
      smaxSeries[h1].push(st0.maxLogit);
      rawSmaxSeries[h1].push(st0.maxLogit);
      gammaSeries[h1].push(1);
      cumGammaSeries[h1].push(1);
      entropySeries[h1].push(st0.entropy);
      if (st0.maxLogit > max0) max0 = st0.maxLogit;
      ent0 += st0.entropy;
      if (headProfiles[h1].runaway) { runEnt0 += st0.entropy; runCnt++; }
    }
    globalMaxLogit.push(max0);
    meanEntropy.push(ent0 / heads);
    runawayEntropy.push(runCnt ? runEnt0 / runCnt : ent0 / heads);
    stepClippedHeads.push(0);

    for (var t = 1; t <= steps; t++) {
      var stepMax = -Infinity;
      var stepEnt = 0;
      var stepRunEnt = 0;
      var clippedThisStep = 0;

      for (var hh = 0; hh < heads; hh++) {
        var hp = headProfiles[hh];
        var Gq = zeros(d, d), Gk = zeros(d, d);
        var xq = X[hp.tokQ], xk = X[hp.tokK];
        var alignWeight = hp.runaway
          ? hp.driftStrength * (0.9 + 0.25 * Math.sin(t * 0.08 + hh))
          : hp.driftStrength * Math.cos(t * 0.42 + hh * 1.4);

        for (var r = 0; r < d; r++) {
          for (var c = 0; c < d; c++) {
            Gq[r][c] = alignWeight * xq[r] * hp.u[c] + 0.28 * randn();
            Gk[r][c] = alignWeight * xk[r] * hp.u[c] + 0.28 * randn();
          }
        }

        // Momentum + Newton-Schulz orthogonalized Muon update + weight decay
        hp.Mq = matAdd(hp.Mq, Gq, beta, 1);
        hp.Mk = matAdd(hp.Mk, Gk, beta, 1);
        var Oq = newtonSchulz5(hp.Mq, 3, NS_PRESETS.keller);
        var Ok = newtonSchulz5(hp.Mk, 3, NS_PRESETS.keller);

        var muLr = hp.runaway ? 0.20 : 0.16;
        var wd = hp.runaway ? 0.008 : 0.014;
        hp.Wq = matAdd(hp.Wq, Oq, 1 - wd, muLr);
        hp.Wk = matAdd(hp.Wk, Ok, 1 - wd, muLr);

        var pre = computeLogitStats(hp.Wq, hp.Wk);
        var gamma = 1;
        var post = pre;

        if (pre.maxLogit > tau) {
          gamma = tau / pre.maxLogit;
          if (clipOn) {
            var rt = Math.sqrt(gamma);
            hp.Wq = matScale(hp.Wq, rt);
            hp.Wk = matScale(hp.Wk, rt);
            hp.cumGamma *= gamma;
            post = computeLogitStats(hp.Wq, hp.Wk);
            clipCounts[hh]++;
            clippedThisStep++;
          }
        }

        rawSmaxSeries[hh].push(pre.maxLogit);
        smaxSeries[hh].push(post.maxLogit);
        gammaSeries[hh].push(clipOn ? gamma : 1);
        cumGammaSeries[hh].push(hp.cumGamma);
        entropySeries[hh].push(post.entropy);

        if (post.maxLogit > stepMax) stepMax = post.maxLogit;
        stepEnt += post.entropy;
        if (hp.runaway) stepRunEnt += post.entropy;
      }

      globalMaxLogit.push(stepMax);
      meanEntropy.push(stepEnt / heads);
      runawayEntropy.push(runCnt ? stepRunEnt / runCnt : stepEnt / heads);
      stepClippedHeads.push(clippedThisStep);
    }

    var headSummary = [];
    for (var hs = 0; hs < heads; hs++) {
      var arr = smaxSeries[hs];
      var gArr = gammaSeries[hs];
      var eArr = entropySeries[hs];
      var peak = -Infinity, minGamma = 1;
      for (var idx = 0; idx < arr.length; idx++) {
        if (arr[idx] > peak) peak = arr[idx];
        if (gArr[idx] < minGamma) minGamma = gArr[idx];
      }
      headSummary.push({
        head: hs,
        runaway: headProfiles[hs].runaway,
        finalSmax: arr[arr.length - 1],
        peakSmax: peak,
        finalGamma: gArr[gArr.length - 1],
        minGamma: minGamma,
        cumGamma: headProfiles[hs].cumGamma,
        clipCount: clipCounts[hs],
        finalEntropy: eArr[eArr.length - 1]
      });
    }

    return {
      heads: heads,
      steps: steps,
      tau: tau,
      clipOn: clipOn,
      seed: seed,
      smaxSeries: smaxSeries,
      rawSmaxSeries: rawSmaxSeries,
      gammaSeries: gammaSeries,
      cumGammaSeries: cumGammaSeries,
      entropySeries: entropySeries,
      meanEntropy: meanEntropy,
      runawayEntropy: runawayEntropy,
      globalMaxLogit: globalMaxLogit,
      stepClippedHeads: stepClippedHeads,
      headSummary: headSummary
    };
  }

  var api = {
    mulberry32: mulberry32,
    zeros: zeros,
    eye: eye,
    cloneMat: cloneMat,
    transpose: transpose,
    matmul: matmul,
    matAdd: matAdd,
    matScale: matScale,
    frobNorm: frobNorm,
    gram: gram,
    orthError: orthError,
    jacobiEigenSym: jacobiEigenSym,
    singularValues: singularValues,
    svd: svd,
    randomOrthogonal: randomOrthogonal,
    makeSkewedMatrix: makeSkewedMatrix,
    NS_PRESETS: NS_PRESETS,
    evalNSPoly: evalNSPoly,
    polyCurve: polyCurve,
    newtonSchulz5: newtonSchulz5,
    newtonSchulzSteps: newtonSchulzSteps,
    runLandscape: runLandscape,
    simQKClip: simQKClip
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.MUONSIM = api;
})(typeof window !== 'undefined' ? window : globalThis);
