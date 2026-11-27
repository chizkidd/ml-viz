/* Pure speculative-decoding math and seeded draft-verify-rollback trace simulator.
   No DOM access. UMD export: window.SPECSIM (browser) and module.exports (Node). */
(function (root) {
  'use strict';

  function mulberry32(seed) {
    var a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Exact geometric expected tokens emitted per verify round.
   * For k drafted tokens and per-token acceptance probability alpha:
   *   E[tokens] = sum_{i=0}^{k} alpha^i = (1 - alpha^(k+1)) / (1 - alpha)
   * Handles alpha === 1 -> k + 1, alpha <= 0 -> 1, k <= 0 -> 1.
   */
  function expectedTokens(alpha, k) {
    if (k <= 0) return 1;
    if (alpha <= 0) return 1;
    if (alpha >= 1 - 1e-12) return k + 1;
    return (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
  }

  /**
   * Exact speculative decoding speedup relative to a single plain target step.
   *   S(alpha, k, c, v) = E[tokens](alpha, k) / (k * c + v)
   * where c = draft_ms / target_ms (0 for free n-gram lookup)
   * and   v = verify_ms(k+1) / target_ms(1) (~1.01 in fp16, ~1.83-1.94 in int8).
   */
  function specSpeedup(alpha, k, c, v) {
    var draftCost = c == null ? 0 : c;
    var verifyCost = v == null ? 1 : v;
    var denom = k * draftCost + verifyCost;
    if (denom <= 0) return Infinity;
    return expectedTokens(alpha, k) / denom;
  }

  /**
   * Find the integer lookahead k in [1..maxK] maximizing specSpeedup(alpha, k, c, v).
   */
  function optimalK(alpha, c, v, maxK) {
    var limit = maxK || 12;
    var best = { k: 1, speedup: specSpeedup(alpha, 1, c, v), expected: expectedTokens(alpha, 1) };
    for (var k = 2; k <= limit; k++) {
      var s = specSpeedup(alpha, k, c, v);
      if (s > best.speedup) {
        best = { k: k, speedup: s, expected: expectedTokens(alpha, k) };
      }
    }
    return best;
  }

  /**
   * Minimum acceptance rate alpha in [0, 1] required to reach speedup >= 1.0x.
   * Returns null if even alpha = 1.0 cannot break even.
   */
  function breakEvenAlpha(k, c, v) {
    if (specSpeedup(0, k, c, v) >= 1) return 0;
    if (specSpeedup(1, k, c, v) < 1) return null;
    var lo = 0;
    var hi = 1;
    for (var i = 0; i < 40; i++) {
      var mid = 0.5 * (lo + hi);
      if (specSpeedup(mid, k, c, v) >= 1) hi = mid;
      else lo = mid;
    }
    return hi;
  }

  // Realistic workloads mirroring mini-vllm Stage 6 / 7 / 9 / 11 prompt suites.
  var WORKLOADS = {
    copy: {
      id: 'copy',
      label: 'Copy / rewrite',
      prompt: 'Rewrite the changelog entry with version v2.4: "Fixed paged KV block release when a speculative stop token lands inside an accepted prefix and updated CUDA graph buckets."',
      tokens: [
        'Version', 'v2.4', ':', 'Fixed', 'paged', 'KV', 'block', 'release',
        'when', 'a', 'speculative', 'stop', 'token', 'lands', 'inside', 'an',
        'accepted', 'prefix', ',', 'and', 'updated', 'the', 'CUDA', 'graph',
        'bucket', 'table', 'for', 'static', 'decode', 'steps', 'with', 'zero',
        'padding', 'leaks', 'in', 'batch', 'mode', '.'
      ],
      distractors: [
        'Release', 'v2.3', 'Updated', 'cache', 'slot', 'free', 'during', 'draft',
        'check', 'falls', 'within', 'verified', 'span', 'buffer', 'layout', 'dynamic',
        'kernel', 'memory', 'bounds', 'stream'
      ]
    },
    code: {
      id: 'code',
      label: 'Code completion',
      prompt: 'def verify_draft(target, cache, drafts):\n    # Run one forward pass over k+1 tokens and roll back rejected tail',
      tokens: [
        'logits', '=', 'target', '.', 'forward', '(', 'drafts', ',',
        'cache', ')', '\n', 'preds', '=', 'logits', '.', 'argmax',
        '(', 'dim', '=', '-1', ')', '\n', 'matched', '=',
        '0', '\n', 'for', 'i', ',', 'tok', 'in', 'enumerate',
        '(', 'drafts', '[', '1', ':', ']', ')', ':'
      ],
      distractors: [
        'out', 'model', 'step', 'tokens', 'kv', 'probs', 'sample', 'axis',
        'count', 'idx', 'val', 'range', 'len', 'return', 'cache_len', 'break',
        'bonus', 'draft', 'tail', 'slice'
      ]
    },
    open: {
      id: 'open',
      label: 'Open prose',
      prompt: 'Explain in plain terms why reading quantized int8 weights once per verify pass can still slow down speculative decoding on a small GPU.',
      tokens: [
        'Checking', 'five', 'draft', 'tokens', 'together', 'shares', 'every', 'weight',
        'read', 'across', 'the', 'whole', 'block', ',', 'so', 'fp16',
        'barely', 'slows', 'down', '.', 'With', 'int8', ',', 'however',
        ',', 'each', 'extra', 'row', 'adds', 'dequant', 'math', 'that',
        'saturates', 'a', 'power', '-', 'capped', 'T4', '.'
      ],
      distractors: [
        'Verifying', 'several', 'guessed', 'words', 'once', 'reuses', 'all', 'matrix',
        'loads', 'over', 'one', 'step', 'while', 'float', 'stays', 'fast',
        'under', 'eight', 'bit', 'compute', 'limits', 'bandwidth', 'cache', 'latency'
      ]
    }
  };

  // Calibrated to Stage 9 / Stage 11 Tesla T4 Qwen2.5-3B measured rates.
  var PROPOSERS = {
    ngram: {
      id: 'ngram',
      label: 'N-gram lookup',
      shortLabel: 'N-gram',
      c: 0.0,
      v: 1.013,
      alphaByWorkload: { copy: 0.987, code: 0.644, open: 0.207 },
      note: 'Free prompt-lookup n-gram (c = 0). Huge win on copy/code; misses on open prose.'
    },
    draft_0p5b: {
      id: 'draft_0p5b',
      label: '0.5B draft (CUDA graph)',
      shortLabel: '0.5B draft',
      c: 0.261,
      v: 1.013,
      alphaByWorkload: { copy: 0.983, code: 0.865, open: 0.711 },
      note: 'Qwen2.5-0.5B CUDA-graphed draft for 3B target (c = 0.26). High acceptance even on open text.'
    },
    int8_self: {
      id: 'int8_self',
      label: 'Int8 self-draft (graph)',
      shortLabel: 'Int8 self',
      c: 0.562,
      v: 1.013,
      alphaByWorkload: { copy: 1.0, code: 0.982, open: 0.969 },
      note: 'Same 3B weights in int8 drafting for fp16 (alpha ~0.97-1.00), but c = 0.56 is too heavy to win.'
    }
  };

  function pickDistractor(rng, distractors, avoid) {
    for (var tries = 0; tries < 8; tries++) {
      var cand = distractors[Math.floor(rng() * distractors.length)];
      if (cand !== avoid) return cand;
    }
    return avoid + '*';
  }

  /**
   * Simulate a complete speculative decoding trace with deterministic RNG.
   * Guarantees exact greedy parity: emitted === greedyReference token-for-token.
   */
  function simulateTrace(opts) {
    opts = opts || {};
    var workloadId = WORKLOADS[opts.workload] ? opts.workload : 'code';
    var proposerId = PROPOSERS[opts.proposer] ? opts.proposer : 'ngram';
    var k = Math.max(1, Math.min(8, Math.round(opts.k == null ? 4 : opts.k)));
    var seed = opts.seed == null ? 7 : opts.seed;
    var maxTokens = Math.max(8, Math.min(30, opts.maxTokens == null ? 22 : opts.maxTokens));

    var wl = WORKLOADS[workloadId];
    var prop = PROPOSERS[proposerId];
    var alpha = opts.alphaOverride != null ? opts.alphaOverride : prop.alphaByWorkload[workloadId];
    var c = opts.cOverride != null ? opts.cOverride : prop.c;
    var v = opts.vOverride != null ? opts.vOverride : prop.v;

    var rng = mulberry32((seed * 2654435761) ^ (k * 97) ^ workloadId.charCodeAt(0) ^ (proposerId.charCodeAt(0) << 4));
    var target = wl.tokens;
    var promptLen = 32; // illustrative prompt KV length before generation begins

    var rounds = [];
    var emitted = [];
    var totalDrafted = 0;
    var totalAccepted = 0;
    var totalRejected = 0;
    var totalBonus = 0;
    var pos = 0;

    while (pos < maxTokens && pos < target.length - 1) {
      var remainingBeforeLast = target.length - 1 - pos;
      var kEff = Math.min(k, remainingBeforeLast);
      var drafts = [];
      var targetWindow = [];
      var statuses = [];
      var firstReject = kEff;

      for (var j = 0; j < kEff; j++) {
        var trueTok = target[pos + j];
        targetWindow.push(trueTok);
        if (firstReject === kEff) {
          var hit = alpha >= 1 - 1e-12 ? true : (alpha <= 0 ? false : rng() < alpha);
          if (hit) {
            drafts.push(trueTok);
            statuses.push('accepted');
          } else {
            firstReject = j;
            drafts.push(pickDistractor(rng, wl.distractors, trueTok));
            statuses.push('rejected');
          }
        } else {
          // Suffix after first mismatch is always rolled back regardless of token value.
          drafts.push(pickDistractor(rng, wl.distractors, trueTok));
          statuses.push('rejected');
        }
      }

      var acceptedCount = firstReject;
      var rejectedCount = kEff - acceptedCount;
      var bonusToken = target[pos + acceptedCount];
      var bonusKind = acceptedCount === kEff ? 'bonus' : 'correction';
      var emittedThisRound = [];
      for (var a = 0; a < acceptedCount; a++) {
        emittedThisRound.push({ token: drafts[a], kind: 'accepted', round: rounds.length + 1 });
        emitted.push(drafts[a]);
      }
      emittedThisRound.push({ token: bonusToken, kind: bonusKind, round: rounds.length + 1 });
      emitted.push(bonusToken);

      var kvBefore = promptLen + pos;
      var kvPeak = kvBefore + kEff;
      var kvAfter = kvBefore + acceptedCount + 1;
      var costSteps = kEff * c + v;

      rounds.push({
        roundIndex: rounds.length + 1,
        startPos: pos,
        k: kEff,
        drafts: drafts,
        targetWindow: targetWindow,
        statuses: statuses,
        acceptedCount: acceptedCount,
        rejectedCount: rejectedCount,
        bonusToken: bonusToken,
        bonusKind: bonusKind,
        emittedThisRound: emittedThisRound,
        kvBefore: kvBefore,
        kvPeak: kvPeak,
        kvAfter: kvAfter,
        costSteps: costSteps
      });

      totalDrafted += kEff;
      totalAccepted += acceptedCount;
      totalRejected += rejectedCount;
      totalBonus += 1;
      pos += acceptedCount + 1;
    }

    var greedyReference = target.slice(0, emitted.length);
    var exactParity = emitted.length === greedyReference.length &&
      emitted.every(function (tok, idx) { return tok === greedyReference[idx]; });
    var totalRounds = rounds.length;
    var specCostSteps = rounds.reduce(function (acc, r) { return acc + r.costSteps; }, 0);
    var plainSteps = emitted.length;

    return {
      workload: workloadId,
      workloadLabel: wl.label,
      prompt: wl.prompt,
      proposer: proposerId,
      proposerLabel: prop.label,
      proposerNote: prop.note,
      k: k,
      seed: seed,
      alpha: alpha,
      c: c,
      v: v,
      rounds: rounds,
      emitted: emitted,
      greedyReference: greedyReference,
      exactParity: exactParity,
      totalRounds: totalRounds,
      totalDrafted: totalDrafted,
      totalAccepted: totalAccepted,
      totalRejected: totalRejected,
      totalBonus: totalBonus,
      totalEmitted: emitted.length,
      empiricalAlpha: totalDrafted > 0 ? totalAccepted / totalDrafted : 0,
      tokensPerRound: totalRounds > 0 ? emitted.length / totalRounds : 1,
      plainSteps: plainSteps,
      specCostSteps: specCostSteps,
      effectiveSpeedup: specCostSteps > 0 ? plainSteps / specCostSteps : 1,
      theoreticalTokensPerRound: expectedTokens(alpha, k),
      theoreticalSpeedup: specSpeedup(alpha, k, c, v)
    };
  }

  var SPECSIM = {
    mulberry32: mulberry32,
    expectedTokens: expectedTokens,
    specSpeedup: specSpeedup,
    optimalK: optimalK,
    breakEvenAlpha: breakEvenAlpha,
    WORKLOADS: WORKLOADS,
    PROPOSERS: PROPOSERS,
    simulateTrace: simulateTrace
  };

  if (typeof root !== 'undefined') root.SPECSIM = SPECSIM;
  if (typeof module !== 'undefined' && module.exports) module.exports = SPECSIM;
})(typeof window !== 'undefined' ? window : globalThis);
