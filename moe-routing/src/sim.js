/* Mixture of Experts (MoE) routing, load-balancing, and serving roofline core.
   Pure functions, no DOM. Works in a browser (window.MOESIM) and in Node (module.exports). */
(function (root) {
  'use strict';

  /* ---------- seeded RNG (mulberry32) & Box-Muller normal ---------- */
  function mulberry32(a) {
    a = a >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeNormal(rng) {
    var spare = null;
    return function () {
      if (spare !== null) {
        var out = spare;
        spare = null;
        return out;
      }
      var u = 0, v = 0;
      while (u <= 1e-12) u = rng();
      while (v <= 1e-12) v = rng();
      var mag = Math.sqrt(-2.0 * Math.log(u));
      spare = mag * Math.sin(2.0 * Math.PI * v);
      return mag * Math.cos(2.0 * Math.PI * v);
    };
  }

  /* ---------- numerically stable softmax + log-normalizer Z(x) ---------- */
  function softmaxWithZ(logits, temp) {
    var invT = 1 / Math.max(0.05, temp || 1);
    var n = logits.length;
    var scaled = new Array(n);
    var maxV = -Infinity;
    for (var i = 0; i < n; i++) {
      var v = logits[i] * invT;
      scaled[i] = v;
      if (v > maxV) maxV = v;
    }
    var exps = new Array(n);
    var sum = 0;
    for (var j = 0; j < n; j++) {
      var e = Math.exp(scaled[j] - maxV);
      exps[j] = e;
      sum += e;
    }
    var probs = new Array(n);
    for (var k = 0; k < n; k++) probs[k] = exps[k] / sum;
    var z = maxV + Math.log(sum);
    return { probs: probs, z: z };
  }

  /* =====================================================================
     1. TOP-K ROUTING, CAPACITY FACTOR & DROPPED TOKENS
     ===================================================================== */
  function routeBatch(opts) {
    opts = opts || {};
    var nTokens = Math.max(1, Math.round(opts.nTokens || 32));
    var nExperts = Math.max(2, Math.round(opts.nExperts || 8));
    var topK = Math.min(nExperts, Math.max(1, Math.round(opts.topK || 2)));
    var temp = Math.max(0.05, opts.temp != null ? opts.temp : 1.0);
    var capacityFactor = Math.max(0.1, opts.capacityFactor != null ? opts.capacityFactor : 1.25);
    var skew = opts.skew != null ? opts.skew : 0.0;
    var seed = opts.seed != null ? opts.seed : 1;
    var dropless = Boolean(opts.dropless);
    var biases = opts.biases && opts.biases.length === nExperts ? opts.biases : null;

    var rng = mulberry32((seed * 2654435761) ^ (nExperts * 97) ^ (nTokens * 31));
    var randn = makeNormal(rng);

    // Build zero-mean skew profile favoring lower-index experts when skew > 0
    var skewProfile = new Array(nExperts);
    var decay = Math.max(1.0, nExperts * 0.25);
    var meanExp = 0;
    for (var i = 0; i < nExperts; i++) {
      skewProfile[i] = Math.exp(-i / decay);
      meanExp += skewProfile[i];
    }
    meanExp /= nExperts;
    for (var i2 = 0; i2 < nExperts; i2++) {
      skewProfile[i2] = skew * (skewProfile[i2] - meanExp);
    }

    var capacity = Math.max(1, Math.ceil(capacityFactor * nTokens * topK / nExperts));
    var demand = new Array(nExperts).fill(0);
    var assigned = new Array(nExperts).fill(0);
    var dropped = new Array(nExperts).fill(0);
    var probSum = new Array(nExperts).fill(0);
    var expertSlots = new Array(nExperts);
    for (var e = 0; e < nExperts; e++) expertSlots[e] = [];

    var tokens = new Array(nTokens);
    var totalSlots = nTokens * topK;
    var droppedSlots = 0;
    var tokensWithDrop = 0;
    var fullyDroppedTokens = 0;
    var zSqSum = 0;

    for (var t = 0; t < nTokens; t++) {
      var rowLogits;
      if (opts.logits && opts.logits[t]) {
        rowLogits = opts.logits[t].slice(0, nExperts);
      } else {
        rowLogits = new Array(nExperts);
        for (var j = 0; j < nExperts; j++) {
          rowLogits[j] = randn() * 0.85 + skewProfile[j];
        }
      }

      var sm = softmaxWithZ(rowLogits, temp);
      var probs = sm.probs;
      zSqSum += sm.z * sm.z;
      for (var p = 0; p < nExperts; p++) probSum[p] += probs[p];

      // DeepSeek-V3 aux-loss-free trick: additive bias b_i shifts the selection
      // score h'(x)_i = h(x)_i/T + b_i for top-k routing ONLY, NEVER changing
      // the unbiased softmax probabilities p_i or final gating weights.
      var selScores = new Array(nExperts);
      var order = new Array(nExperts);
      for (var s = 0; s < nExperts; s++) {
        selScores[s] = (rowLogits[s] / temp) + (biases ? biases[s] : 0);
        order[s] = s;
      }
      // Deterministic round-robin tie-breaker when scores are identical
      order.sort(function (a, b) {
        var diff = selScores[b] - selScores[a];
        if (Math.abs(diff) > 1e-12) return diff;
        var ra = (((a - t) % nExperts) + nExperts) % nExperts;
        var rb = (((b - t) % nExperts) + nExperts) % nExperts;
        return ra - rb;
      });

      var selected = order.slice(0, topK);
      // Gating weights use ONLY the original unbiased probabilities p_i
      var gateSum = 0;
      for (var r = 0; r < topK; r++) gateSum += probs[selected[r]];
      var gates = new Array(topK);
      for (var r2 = 0; r2 < topK; r2++) {
        gates[r2] = gateSum > 0 ? probs[selected[r2]] / gateSum : 1 / topK;
      }

      var assignments = new Array(topK);
      var tokDropped = 0;
      for (var r3 = 0; r3 < topK; r3++) {
        var expIdx = selected[r3];
        var slotIdx = demand[expIdx];
        demand[expIdx] += 1;
        var isDropped = !dropless && assigned[expIdx] >= capacity;
        if (isDropped) {
          dropped[expIdx] += 1;
          droppedSlots += 1;
          tokDropped += 1;
        } else {
          assigned[expIdx] += 1;
        }
        var rec = {
          token: t,
          rank: r3,
          expert: expIdx,
          prob: probs[expIdx],
          selScore: selScores[expIdx],
          gate: gates[r3],
          dropped: isDropped,
          slot: slotIdx
        };
        assignments[r3] = rec;
        expertSlots[expIdx].push(rec);
      }

      if (tokDropped > 0) tokensWithDrop += 1;
      if (tokDropped === topK) fullyDroppedTokens += 1;

      tokens[t] = {
        id: t,
        logits: rowLogits,
        probs: probs,
        selScores: selScores,
        selected: selected,
        gates: gates,
        assignments: assignments,
        droppedCount: tokDropped,
        z: sm.z
      };
    }

    // Switch load-balancing loss: L_aux = N * sum_i (f_i * P_i)
    // f_i = fraction of routed token-expert assignments going to expert i (sums to 1)
    // P_i = mean router probability assigned to expert i (sums to 1)
    // Perfect uniform balance (f_i = P_i = 1/N) yields L_aux = 1.0 for every topK.
    var f = new Array(nExperts);
    var P = new Array(nExperts);
    var lAux = 0;
    var maxDemand = 0;
    var minDemand = Infinity;
    for (var m = 0; m < nExperts; m++) {
      f[m] = demand[m] / totalSlots;
      P[m] = probSum[m] / nTokens;
      lAux += f[m] * P[m];
      if (demand[m] > maxDemand) maxDemand = demand[m];
      if (demand[m] < minDemand) minDemand = demand[m];
    }
    lAux *= nExperts;

    var meanDemand = totalSlots / nExperts;
    var imbalanceRatio = meanDemand > 0 ? maxDemand / meanDemand : 1.0;
    var assignedSlots = totalSlots - droppedSlots;
    var paddedSlots = dropless ? assignedSlots : nExperts * capacity;
    var paddingWaste = paddedSlots > 0 ? Math.max(0, paddedSlots - assignedSlots) / paddedSlots : 0;

    return {
      nTokens: nTokens,
      nExperts: nExperts,
      topK: topK,
      temp: temp,
      capacityFactor: capacityFactor,
      dropless: dropless,
      capacity: capacity,
      totalSlots: totalSlots,
      assignedSlots: assignedSlots,
      droppedSlots: droppedSlots,
      dropRate: totalSlots > 0 ? droppedSlots / totalSlots : 0,
      tokensWithDrop: tokensWithDrop,
      fullyDroppedTokens: fullyDroppedTokens,
      paddedSlots: paddedSlots,
      paddingWaste: paddingWaste,
      demand: demand,
      assigned: assigned,
      dropped: dropped,
      f: f,
      P: P,
      lAux: lAux,
      zLoss: zSqSum / nTokens,
      meanDemand: meanDemand,
      maxDemand: maxDemand,
      minDemand: minDemand,
      imbalanceRatio: imbalanceRatio,
      expertSlots: expertSlots,
      tokens: tokens
    };
  }

  /* =====================================================================
     2. ROUTER COLLAPSE & LOAD BALANCING DYNAMICS
     Modes:
       'none' : no load balancing -> rich-get-richer router collapse
       'aux'  : Switch auxiliary load-balancing loss (alphaAux * L_aux)
       'bias' : DeepSeek-V3 aux-loss-free dynamic bias (b_i += gamma * sign(1/N - f_i))
     ===================================================================== */
  function simTrainingBalance(opts) {
    opts = opts || {};
    var nExperts = Math.max(2, Math.round(opts.nExperts || 8));
    var topK = Math.min(nExperts, Math.max(1, Math.round(opts.topK || 2)));
    var mode = opts.mode || 'none';
    var steps = Math.max(10, Math.round(opts.steps || 80));
    var alphaAux = opts.alphaAux != null ? Math.max(0, opts.alphaAux) : 0.08;
    var biasGamma = opts.biasGamma != null ? Math.max(0, opts.biasGamma) : 0.06;
    var seed = opts.seed != null ? opts.seed : 42;

    var batchTokens = Math.max(64, nExperts * 16);
    var targetShare = 1 / nExperts;
    var rng = mulberry32((seed * 1597334677) ^ (nExperts * 131) ^ (topK * 17));
    var randn = makeNormal(rng);

    // Router global logits w_i, expert competence q_i, and DeepSeek dynamic bias b_i
    var w = new Array(nExperts);
    var q = new Array(nExperts);
    var b = new Array(nExperts);
    for (var i = 0; i < nExperts; i++) {
      // Realistic random init with mild early-winner symmetry breaking
      w[i] = randn() * 0.16 + (i < topK ? 0.28 : 0);
      q[i] = 0.22 + (i < topK ? 0.08 : 0);
      b[i] = 0;
    }

    // Semantic specialization scale: how strongly router prefers the token's
    // matching specialist expert. Auxiliary loss directly fights this sharpness.
    var semWeight = 1.0;
    var lrW = 0.36;
    var lrSem = 0.14;
    var history = new Array(steps);

    for (var step = 0; step < steps; step++) {
      var counts = new Array(nExperts).fill(0);
      var pSum = new Array(nExperts).fill(0);
      var dLAux_dw = new Array(nExperts).fill(0);
      var dLAux_dSem = 0;
      var matchedSpecialist = 0;
      var gateQualitySum = 0;

      var batchProbs = new Array(batchTokens);
      var batchCluster = new Array(batchTokens);

      for (var t = 0; t < batchTokens; t++) {
        var cluster = Math.floor(rng() * nExperts);
        batchCluster[t] = cluster;
        var logits = new Array(nExperts);
        for (var e = 0; e < nExperts; e++) {
          var affinity = e === cluster ? 1.0 : ((e + 1) % nExperts === cluster ? 0.30 : 0.0);
          logits[e] = w[e] + semWeight * affinity + randn() * 0.28;
        }
        var probs = softmaxWithZ(logits, 1.0).probs;
        batchProbs[t] = probs;
        for (var p = 0; p < nExperts; p++) pSum[p] += probs[p];

        // Top-k selection: DeepSeek-V3 adds b_i to routing logits h'(x)_i = h(x)_i + b_i
        // while keeping original unbiased probs for gating weights.
        var order = new Array(nExperts);
        for (var s = 0; s < nExperts; s++) order[s] = s;
        order.sort(function (x, y) {
          var sx = logits[x] + (mode === 'bias' ? b[x] : 0);
          var sy = logits[y] + (mode === 'bias' ? b[y] : 0);
          return sy - sx;
        });
        var sel = order.slice(0, topK);

        var gSum = 0;
        var tokScore = 0;
        var hitSpec = false;
        for (var r = 0; r < topK; r++) {
          var idx = sel[r];
          counts[idx] += 1;
          gSum += probs[idx];
          if (idx === cluster) hitSpec = true;
        }
        if (hitSpec) matchedSpecialist += 1;
        for (var r2 = 0; r2 < topK; r2++) {
          var idx2 = sel[r2];
          var g = gSum > 0 ? probs[idx2] / gSum : 1 / topK;
          var matchBonus = idx2 === cluster ? 1.0 : ((idx2 + 1) % nExperts === cluster ? 0.45 : 0.15);
          tokScore += g * (0.45 * q[idx2] + 0.55 * matchBonus);
        }
        gateQualitySum += tokScore;
      }

      var totalSlots = batchTokens * topK;
      var f = new Array(nExperts);
      var P = new Array(nExperts);
      var lAux = 0;
      var maxF = 0;
      var deadCount = 0;
      var entropy = 0;

      for (var k = 0; k < nExperts; k++) {
        f[k] = counts[k] / totalSlots;
        P[k] = pSum[k] / batchTokens;
        lAux += f[k] * P[k];
        if (f[k] > maxF) maxF = f[k];
        if (f[k] < 0.2 * targetShare) deadCount += 1;
        if (f[k] > 1e-12) entropy -= f[k] * Math.log(f[k]);
      }
      lAux *= nExperts;
      var normEntropy = entropy / Math.log(nExperts);
      var imbalance = maxF / targetShare;

      // Exact gradient of L_aux = N * sum_j f_j P_j wrt w_i and semWeight
      for (var t2 = 0; t2 < batchTokens; t2++) {
        var pr = batchProbs[t2];
        var cl = batchCluster[t2];
        var dotFP = 0;
        for (var j = 0; j < nExperts; j++) dotFP += f[j] * pr[j];
        var meanAff = pr[cl] + 0.30 * pr[(cl - 1 + nExperts) % nExperts];
        for (var j2 = 0; j2 < nExperts; j2++) {
          var gW = (nExperts / batchTokens) * pr[j2] * (f[j2] - dotFP);
          dLAux_dw[j2] += gW;
          var aff = j2 === cl ? 1.0 : ((j2 + 1) % nExperts === cl ? 0.30 : 0.0);
          dLAux_dSem += (nExperts / batchTokens) * f[j2] * pr[j2] * (aff - meanAff);
        }
      }

      // Rich-get-richer LM feedback: experts receiving more tokens train faster,
      // increasing competence q_i, which reinforces the LM router preference.
      var meanQ = 0;
      for (var iq = 0; iq < nExperts; iq++) {
        q[iq] += 0.14 * (1 - q[iq]) * Math.min(1, f[iq] * nExperts * 0.65);
        meanQ += q[iq];
      }
      meanQ /= nExperts;

      // Scale factor so dynamics remain consistent across N = 4, 8, 16
      var nScale = nExperts / 8;
      var lmNormSq = 0;
      var auxNormSq = 0;
      var gradLM = new Array(nExperts);
      var gradAuxVec = new Array(nExperts);
      for (var iw = 0; iw < nExperts; iw++) {
        var gLM = -(0.85 * (q[iw] - meanQ) + 0.38 * nScale * (f[iw] - targetShare)) + 0.02 * w[iw];
        gradLM[iw] = gLM;
        lmNormSq += gLM * gLM;
        // Scale aux gradient by N so alphaAux has invariant meaning across N=4,8,16
        var gAux = alphaAux * 46.0 * nScale * dLAux_dw[iw];
        gradAuxVec[iw] = gAux;
        if (mode === 'aux') auxNormSq += gAux * gAux;
      }

      var gradInterference = 0;
      if (mode === 'aux' && alphaAux > 0) {
        var ratio = Math.sqrt(auxNormSq) / (Math.sqrt(lmNormSq) + 0.12);
        var semPush = Math.abs(alphaAux * 6.0 * dLAux_dSem) + alphaAux * 2.1;
        gradInterference = Math.min(1.0, ratio * 0.32 + semPush * 0.48);
      }

      // Apply gradient updates to router weights w_i
      for (var iu = 0; iu < nExperts; iu++) {
        if (mode === 'aux') {
          w[iu] -= lrW * (gradLM[iu] + gradAuxVec[iu]);
        } else {
          // Both 'none' and 'bias' apply ONLY the clean LM gradient to w_i!
          w[iu] -= lrW * gradLM[iu];
        }
        w[iu] = Math.max(-4.0, Math.min(4.0, w[iu]));
      }

      // Zero-center w so shift-invariance does not drift logits
      var meanW = 0;
      for (var mc = 0; mc < nExperts; mc++) meanW += w[mc];
      meanW /= nExperts;
      for (var mc2 = 0; mc2 < nExperts; mc2++) w[mc2] -= meanW;

      // Update semantic sharpness: LM wants strong specialization (~1.75);
      // Switch aux loss fights it proportionally to alphaAux (gradient interference).
      var targetSem = mode === 'aux' ? Math.max(0.25, 1.75 - alphaAux * 5.2) : 1.75;
      if (mode === 'none') {
        targetSem = 1.75 * (0.40 + 0.60 * normEntropy);
      }
      semWeight += lrSem * (targetSem - semWeight);

      // DeepSeek-V3 auxiliary-loss-free dynamic bias update:
      // b_i <- b_i + gamma * sign(target - load_i), zero-centered
      if (mode === 'bias' && biasGamma > 0) {
        var meanB = 0;
        for (var ib = 0; ib < nExperts; ib++) {
          var err = targetShare - f[ib];
          var sgn = Math.abs(err) < 0.004 ? 0 : (err > 0 ? 1 : -1);
          b[ib] += biasGamma * sgn;
          b[ib] = Math.max(-3.0, Math.min(3.0, b[ib]));
          meanB += b[ib];
        }
        meanB /= nExperts;
        for (var ib2 = 0; ib2 < nExperts; ib2++) b[ib2] -= meanB;
      }

      // Proxy LM loss (lower is better): rewards active specialist capacity & clean gating
      var meanGateQ = gateQualitySum / batchTokens;
      var auxPenalty = mode === 'aux' ? 0.24 * gradInterference : 0;
      var lmLossProxy = 2.65 - 1.18 * meanGateQ + 0.30 * (1 - normEntropy) + auxPenalty;

      history[step] = {
        step: step + 1,
        shares: f,
        probs: P,
        biases: b.slice(),
        lAux: lAux,
        imbalance: imbalance,
        deadExperts: deadCount,
        normEntropy: normEntropy,
        specialistHitRate: matchedSpecialist / batchTokens,
        gradInterference: gradInterference,
        lmLossProxy: lmLossProxy
      };
    }

    var last = history[steps - 1];
    return {
      mode: mode,
      nExperts: nExperts,
      topK: topK,
      steps: steps,
      alphaAux: alphaAux,
      biasGamma: biasGamma,
      history: history,
      final: last
    };
  }

  /* =====================================================================
     3. ACTIVE VS TOTAL PARAMS SERVING ROOFLINE (PUBLISHED SPECS + FORMULA)
     ===================================================================== */
  var MODELS = {
    'mixtral-8x7b': {
      key: 'mixtral-8x7b',
      name: 'Mixtral-8x7B',
      family: 'MoE (coarse)',
      isMoE: true,
      totalParamsB: 46.7,
      activeParamsB: 12.9,
      nExperts: 8,
      topK: 2,
      sharedExperts: 0,
      layers: 32,
      kvBytesPerTokFP16: 131072, // 2 * 32 * 8 * 128 * 2 B = 128 KiB
      note: '8 experts, top-2; classic coarse sparse MoE'
    },
    'qwen3-30b-a3b': {
      key: 'qwen3-30b-a3b',
      name: 'Qwen3-30B-A3B',
      family: 'MoE (fine-grained)',
      isMoE: true,
      totalParamsB: 30.5,
      activeParamsB: 3.3,
      nExperts: 128,
      topK: 8,
      sharedExperts: 0,
      layers: 48,
      kvBytesPerTokFP16: 98304, // 2 * 48 * 4 * 128 * 2 B = 96 KiB
      note: '128 fine-grained experts, top-8 (9.2x total/active ratio)'
    },
    'deepseek-v3': {
      key: 'deepseek-v3',
      name: 'DeepSeek-V3',
      family: 'MoE (fine-grained + shared)',
      isMoE: true,
      totalParamsB: 671.0,
      activeParamsB: 37.0,
      nExperts: 256,
      topK: 8,
      sharedExperts: 1,
      layers: 61,
      kvBytesPerTokFP16: 70272, // MLA compressed latent: 61 * (512 + 64) * 2 B ≈ 68.6 KiB
      note: '256 routed experts (top-8) + 1 shared expert, aux-loss-free bias + MLA'
    },
    'llama-3.1-8b': {
      key: 'llama-3.1-8b',
      name: 'Llama-3.1-8B',
      family: 'Dense',
      isMoE: false,
      totalParamsB: 8.0,
      activeParamsB: 8.0,
      nExperts: 1,
      topK: 1,
      sharedExperts: 0,
      layers: 32,
      kvBytesPerTokFP16: 131072, // 2 * 32 * 8 * 128 * 2 B = 128 KiB
      note: 'Dense 8B baseline (100% params active every token)'
    },
    'llama-3.1-70b': {
      key: 'llama-3.1-70b',
      name: 'Llama-3.1-70B',
      family: 'Dense',
      isMoE: false,
      totalParamsB: 70.6,
      activeParamsB: 70.6,
      nExperts: 1,
      topK: 1,
      sharedExperts: 0,
      layers: 80,
      kvBytesPerTokFP16: 327680, // 2 * 80 * 8 * 128 * 2 B = 320 KiB
      note: 'Dense 70.6B baseline (100% params active every token)'
    }
  };

  var GPUS = {
    'a100-80': {
      key: 'a100-80',
      name: '1x A100 80GB SXM',
      vramGB: 80,
      bwGBs: 2039,
      tflopsBF16: 312
    },
    'h100-80': {
      key: 'h100-80',
      name: '1x H100 80GB SXM',
      vramGB: 80,
      bwGBs: 3350,
      tflopsBF16: 989
    },
    'h100-8x': {
      key: 'h100-8x',
      name: '8x H100 80GB SXM',
      vramGB: 640,
      bwGBs: 26800,
      tflopsBF16: 7912
    },
    'h200-8x': {
      key: 'h200-8x',
      name: '8x H200 141GB SXM',
      vramGB: 1128,
      bwGBs: 38400,
      tflopsBF16: 7912
    }
  };

  /* Coupon-collector expected distinct experts hit by a batch of B independent tokens:
     E[distinct] = N * (1 - (1 - k/N)^B)
     At B = 1: equals k. As B -> inf: approaches N. */
  function expectedDistinctExperts(nExperts, topK, batchSize) {
    var N = Math.max(1, nExperts);
    var k = Math.min(N, Math.max(1, topK));
    var B = Math.max(1, batchSize);
    if (k >= N) return N;
    return N * (1 - Math.pow(1 - k / N, B));
  }

  function moeRoofline(opts) {
    opts = opts || {};
    var model = MODELS[opts.modelKey || 'qwen3-30b-a3b'] || MODELS['qwen3-30b-a3b'];
    var gpu = GPUS[opts.gpuKey || 'h100-80'] || GPUS['h100-80'];
    var weightBytes = opts.weightBytes != null ? Math.max(0.25, opts.weightBytes) : 2;
    var batchSize = Math.max(1, Math.round(opts.batchSize || 1));
    var ctx = Math.max(1, Math.round(opts.ctx || 2048));

    var expExperts = expectedDistinctExperts(model.nExperts, model.topK, batchSize);
    var expertFrac = expExperts / model.nExperts;

    // Split model params into always-active (attention, embeddings, shared experts)
    // and per-routed-expert weights so B=1 reads activeParamsB and B->inf reads totalParamsB.
    var perExpertB = model.nExperts > model.topK
      ? (model.totalParamsB - model.activeParamsB) / (model.nExperts - model.topK)
      : 0;
    var alwaysActiveB = model.activeParamsB - model.topK * perExpertB;
    var stepParamsReadB = alwaysActiveB + expExperts * perExpertB;

    var weightBytesStep = stepParamsReadB * 1e9 * weightBytes;
    var kvBytesStep = batchSize * ctx * model.kvBytesPerTokFP16; // FP16 KV cache
    var totalBytesStep = weightBytesStep + kvBytesStep;

    var weightBytesPerTok = weightBytesStep / batchSize;
    var kvBytesPerTok = kvBytesStep / batchSize;
    var totalBytesPerTok = totalBytesStep / batchSize;

    // VRAM must ALWAYS hold ALL total parameters + batch KV cache
    var weightVramGB = (model.totalParamsB * 1e9 * weightBytes) / 1e9;
    var kvVramGB = kvBytesStep / 1e9;
    var totalVramGB = weightVramGB + kvVramGB;
    var fitsVram = totalVramGB <= gpu.vramGB;

    // Roofline decode latency ceiling (idealized: ignores all-to-all EP comm overhead)
    var flopsStep = 2 * model.activeParamsB * 1e9 * batchSize;
    var tMemMs = (totalBytesStep / (gpu.bwGBs * 1e9)) * 1000;
    var tCompMs = (flopsStep / (gpu.tflopsBF16 * 1e12)) * 1000;
    var tStepMs = Math.max(tMemMs, tCompMs);
    var tokPerSecPerUser = 1000 / tStepMs;
    var totalTokPerSec = batchSize * tokPerSecPerUser;
    var regime = tMemMs >= tCompMs ? 'memory-bound' : 'compute-bound';

    var batchTo90Pct = model.nExperts > model.topK
      ? Math.ceil(Math.log(0.1) / Math.log(1 - model.topK / model.nExperts))
      : 1;

    return {
      model: model,
      gpu: gpu,
      weightBytes: weightBytes,
      batchSize: batchSize,
      ctx: ctx,
      expectedDistinctExperts: expExperts,
      expertHitFraction: expertFrac,
      stepParamsReadB: stepParamsReadB,
      weightReadFraction: stepParamsReadB / model.totalParamsB,
      weightBytesStepGB: weightBytesStep / 1e9,
      kvBytesStepGB: kvBytesStep / 1e9,
      totalBytesStepGB: totalBytesStep / 1e9,
      weightBytesPerTokGB: weightBytesPerTok / 1e9,
      kvBytesPerTokGB: kvBytesPerTok / 1e9,
      totalBytesPerTokGB: totalBytesPerTok / 1e9,
      weightVramGB: weightVramGB,
      kvVramGB: kvVramGB,
      totalVramGB: totalVramGB,
      vramUtilPct: (totalVramGB / gpu.vramGB) * 100,
      fitsVram: fitsVram,
      tMemMs: tMemMs,
      tCompMs: tCompMs,
      tStepMs: tStepMs,
      tokPerSecPerUser: tokPerSecPerUser,
      totalTokPerSec: totalTokPerSec,
      regime: regime,
      batchTo90Pct: batchTo90Pct
    };
  }

  function moeRooflineCurve(opts) {
    opts = opts || {};
    var batches = opts.batches || [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256];
    return batches.map(function (B) {
      return moeRoofline({
        modelKey: opts.modelKey,
        gpuKey: opts.gpuKey,
        weightBytes: opts.weightBytes,
        batchSize: B,
        ctx: opts.ctx
      });
    });
  }

  var api = {
    mulberry32: mulberry32,
    softmaxWithZ: softmaxWithZ,
    routeBatch: routeBatch,
    simTrainingBalance: simTrainingBalance,
    expectedDistinctExperts: expectedDistinctExperts,
    moeRoofline: moeRoofline,
    moeRooflineCurve: moeRooflineCurve,
    MODELS: MODELS,
    GPUS: GPUS
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MOESIM = api;
})(typeof window !== 'undefined' ? window : globalThis);
