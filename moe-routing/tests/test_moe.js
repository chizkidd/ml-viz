'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const moe = require('../src/sim.js');

// 1. Uniform routing yields Switch L_aux === 1.0 and 0 dropped tokens at capacityFactor = 1.0
const zeros = Array.from({ length: 16 }, () => new Array(8).fill(0));
const uni = moe.routeBatch({ nTokens: 16, nExperts: 8, topK: 2, temp: 1.0, capacityFactor: 1.0, logits: zeros });
assert(Math.abs(uni.lAux - 1.0) < 1e-9, `expected L_aux = 1.0 under uniform routing, got ${uni.lAux}`);
assert.strictEqual(uni.droppedSlots, 0, 'expected 0 dropped slots under uniform routing at C=1.0');
assert(Math.abs(uni.imbalanceRatio - 1.0) < 1e-9);

// 2. Tight capacity / high skew increases dropped tokens, while dropless drops 0
const skewed = moe.routeBatch({ nTokens: 64, nExperts: 8, topK: 2, temp: 0.4, capacityFactor: 0.75, skew: 3.0, seed: 5 });
const generous = moe.routeBatch({ nTokens: 64, nExperts: 8, topK: 2, temp: 0.4, capacityFactor: 2.0, skew: 3.0, seed: 5 });
const dropless = moe.routeBatch({ nTokens: 64, nExperts: 8, topK: 2, temp: 0.4, capacityFactor: 0.75, skew: 3.0, seed: 5, dropless: true });
assert(skewed.droppedSlots > 0, 'expected dropped slots under high skew + tight capacity');
assert(skewed.dropRate > generous.dropRate, 'tighter capacity must drop more tokens');
assert.strictEqual(dropless.droppedSlots, 0, 'dropless MoE must never drop tokens');

// 3. Training load-balance dynamics: 'none' collapses while 'aux' and 'bias' stay balanced
const trNone = moe.simTrainingBalance({ nExperts: 8, topK: 2, mode: 'none', steps: 80, seed: 1 });
const trAux = moe.simTrainingBalance({ nExperts: 8, topK: 2, mode: 'aux', steps: 80, alphaAux: 0.1, seed: 1 });
const trBias = moe.simTrainingBalance({ nExperts: 8, topK: 2, mode: 'bias', steps: 80, biasGamma: 0.08, seed: 1 });
const lastImb = (tr) => tr.final.imbalance;
assert(lastImb(trNone) > lastImb(trBias) * 1.3, `expected 'none' (${lastImb(trNone)}) to be far more imbalanced than 'bias' (${lastImb(trBias)})`);
assert(lastImb(trNone) > lastImb(trAux) * 1.3, `expected 'none' (${lastImb(trNone)}) to be far more imbalanced than 'aux' (${lastImb(trAux)})`);
assert(trNone.final.deadExperts > trBias.final.deadExperts, 'expected dead experts under no balancing');
assert.strictEqual(trBias.final.gradInterference, 0, 'aux-loss-free bias must have zero auxiliary gradient interference');

// 4. Coupon-collector active experts: equals k at B=1, approaches N at large B
assert(Math.abs(moe.expectedDistinctExperts(8, 2, 1) - 2) < 1e-9, 'at B=1 expected distinct experts must equal k');
assert(moe.expectedDistinctExperts(8, 2, 64) > 7.99, 'at B=64 almost all 8 experts should be active');

// 5. Roofline invariant: step time === max(tMemMs, tCompMs) and VRAM holds total params
const rf1 = moe.moeRoofline({ modelKey: 'mixtral-8x7b', gpuKey: 'h100-80', weightBytes: 2, batchSize: 1, ctx: 2048 });
const rf64 = moe.moeRoofline({ modelKey: 'mixtral-8x7b', gpuKey: 'h100-80', weightBytes: 2, batchSize: 64, ctx: 2048 });
assert(Math.abs(rf1.tStepMs - Math.max(rf1.tMemMs, rf1.tCompMs)) < 1e-9);
assert(rf64.stepParamsReadB > rf1.stepParamsReadB, 'larger batch must read more expert weights per step');
assert(rf1.weightVramGB > 40, 'VRAM must hold all total parameters even at B=1');

// 6. Built HTML checks
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert(!html.includes('/*SIM*/') && !html.includes('<!--__THEME__-->') && !html.includes('<!--__UI__-->'), 'unreplaced build placeholders in moe-routing/index.html');

console.log('moe-routing tests: OK');
