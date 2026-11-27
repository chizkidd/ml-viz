'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const payload = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data.json'), 'utf8'));
const rows = payload.rows;

// 1. Row count, 1-based IDs, audio files, and alignment ambiguity
assert.strictEqual(rows.length, 21, 'expected 21 clip rows');
rows.forEach((r, idx) => {
  assert.strictEqual(r.id, idx + 1, `expected 1-based id ${idx + 1}, got ${r.id}`);
  assert.strictEqual(r.row_idx, idx);
  assert.strictEqual(r.clip_id, String(idx + 1).padStart(2, '0'));
  assert(typeof r.group === 'string' && r.group.length > 0);
  assert(fs.existsSync(path.join(__dirname, '..', r.audio)), `missing audio file ${r.audio}`);
});
assert.strictEqual(rows.find((r) => r.clip_id === '19').n_opt, 480, 'clip 19 should have 480 optimal alignments');
for (const cid of ['06', '07', '08', '09']) {
  assert.strictEqual(rows.find((r) => r.clip_id === cid).n_opt, 1, `clip ${cid} optimal alignment should be unique`);
}

// 2. Pinned canonical_results.csv CER / WER aggregates
const near = (a, b, eps = 1e-3, msg = '') => assert(Math.abs(a - b) < eps, `${msg}: ${a} != ${b}`);
const pooled = (sub, dk, nk) => sub.reduce((s, r) => s + r[dk], 0) / sub.reduce((s, r) => s + r[nk], 0);
const mean = (sub, dk, nk) => sub.reduce((s, r) => s + r[dk] / r[nk], 0) / sub.length;

near(pooled(rows, 'cer_strict_d', 'cer_strict_n'), 0.287, 1e-3, 'overall strict CER pooled');
near(mean(rows, 'cer_strict_d', 'cer_strict_n'), 0.305, 1e-3, 'overall strict CER mean');
near(pooled(rows, 'cer_base_d', 'cer_base_n'), 0.180, 1e-3, 'overall base_letters CER pooled');
near(pooled(rows, 'wer_strict_d', 'wer_strict_n'), 0.753, 1e-3, 'overall strict WER pooled');
near(mean(rows, 'cer_legacy_d', 'cer_legacy_n'), 0.365, 1e-3, 'overall legacy CER mean');

const pilot = rows.filter((r) => r.group === 'igbo_tonal_pilot');
assert.strictEqual(pilot.length, 4);
near(pooled(pilot, 'cer_strict_d', 'cer_strict_n'), 0.416, 1e-3, 'pilot strict CER pooled');
near(pooled(pilot, 'cer_tone_d', 'cer_tone_n'), 0.197, 1e-3, 'pilot tone_stripped CER pooled');
near(pooled(pilot, 'cer_base_d', 'cer_base_n'), 0.121, 1e-3, 'pilot base_letters CER pooled');

// 3. Verify built HTML + pure alignment & exact DP resampling math from template
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert(!html.includes('/*__ROWS__*/'), 'unreplaced /*__ROWS__*/ placeholder');
assert(!html.includes('Hallucination rate'), 'stale Hallucination rate wording in HTML');
assert(!html.includes('marks that were not spoken'), 'withdrawn C18 wording in HTML');
assert(!html.includes('.hl-tone .m-dot, .hl-tone .m-both { color: inherit'), 'dual-mark CSS filter bug returned');

// Extract exactResampleDist & alignRowDetailed and verify canonical 0/32 + support_size=169
const scriptMatch = html.match(/<script>\nconst ROWS =[\s\S]*?<\/script>/);
assert(scriptMatch, 'could not find inline script');
const fnChunk = scriptMatch[0]
  .replace(/<script>|<\/script>/g, '')
  .replace(/^seg\('sideGrpSeg'[\s\S]*?^buildClipList\(\);/m, '')
  .replace(/\ndrawClip\(\);[\s\S]*$/, '\nwindow.__ig.totFam = totFam; window.__ig.MET = MET;\n})();');

const sandbox = {
  window: {},
  document: { getElementById: () => ({ innerHTML: '', querySelectorAll: () => [], dataset: {} }) },
};
vm.runInNewContext(fnChunk, sandbox);
const ig = sandbox.window.__ig;

// Primary headline: igbo_tonal_pilot tone-only base-letter-aware = 0/32 (28 missing, 4 base_substituted, 0 base_deleted, 3 inserted)
const pilotRows = ig.ROWS.filter((r) => r.group === 'igbo_tonal_pilot');
const sumAl = pilotRows.reduce((acc, r) => {
  const a = ig.alignRowDetailed(r, 'tone', 'aware');
  for (const k of Object.keys(acc)) acc[k] += a[k];
  return acc;
}, { correct: 0, wrong: 0, missing: 0, base_substituted: 0, base_deleted: 0, inserted: 0, expected: 0, produced: 0 });
assert.deepStrictEqual(sumAl, { correct: 0, wrong: 0, missing: 28, base_substituted: 4, base_deleted: 0, inserted: 3, expected: 32, produced: 3 }, `pilot tone alignment mismatch: ${JSON.stringify(sumAl)}`);

// Exact DP resample distribution on 6 historical tonal clips: support_size === 169
const histTonal = ig.ROWS.filter((r) => r.cat === 'tonal_diacritics');
const dist = ig.exactResampleDist(histTonal.map((r) => Math.max(0, r.e - r.p)), histTonal.map((r) => r.e));
assert.strictEqual(dist.supportSize, 169, `expected support_size 169 on 6 historical tonal clips, got ${dist.supportSize}`);
const looPilot = ig.looRange(pilotRows, (sub) => { const t = ig.totFam(sub, 'all'); return t.D / t.E; });
near(looPilot[0], 0.75, 1e-3, 'pilot LOO min');
near(looPilot[1], 0.889, 1e-3, 'pilot LOO max');

console.log('igbo-tone-loss tests: OK (0/32 headline + exact DP support=169 + LOO + 21 audio clips verified)');
