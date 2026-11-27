'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');

const payload = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data.json'), 'utf8'));
const rows = payload.rows;

// 1. Row count and 1-based ID integrity
assert.strictEqual(rows.length, 21, 'expected 21 clip rows');
rows.forEach((r, idx) => {
  assert.strictEqual(r.id, idx + 1, `expected 1-based id ${idx + 1}, got ${r.id}`);
  assert.strictEqual(r.row_idx, idx, `expected 0-based row_idx ${idx}, got ${r.row_idx}`);
  assert.strictEqual(r.clip_id, String(idx + 1).padStart(2, '0'));
  assert(typeof r.group === 'string' && r.group.length > 0, `missing analysis_group on clip ${r.clip_id}`);
});

// 2. Mark-counting rule reproduces expected & produced counts on 21/21 rows
const COUNTED = new Set([0x300, 0x301, 0x304, 0x323]);
function countMarks(s) {
  const out = [];
  for (const ch of s.normalize('NFD')) {
    if (/\p{M}/u.test(ch) && out.length) out[out.length - 1] += ch;
    else out.push(ch);
  }
  return out.filter((cl) => /\p{L}/u.test(cl[0]) && [...cl.slice(1)].some((c) => COUNTED.has(c.codePointAt(0)))).length;
}
for (const r of rows) {
  assert.strictEqual(countMarks(r.gt), r.e, `clip ${r.clip_id}: expected count mismatch`);
  assert.strictEqual(countMarks(r.out), r.p, `clip ${r.clip_id}: produced count mismatch`);
}

// 3. Pinned canonical_results.csv CER / WER aggregates
const near = (a, b, eps = 1e-3, msg = '') => assert(Math.abs(a - b) < eps, `${msg}: ${a} != ${b}`);
const pooled = (sub, dk, nk) => sub.reduce((s, r) => s + r[dk], 0) / sub.reduce((s, r) => s + r[nk], 0);
const mean = (sub, dk, nk) => sub.reduce((s, r) => s + r[dk] / r[nk], 0) / sub.length;

near(pooled(rows, 'cer_strict_d', 'cer_strict_n'), 0.287, 1e-3, 'overall strict CER pooled');
near(mean(rows, 'cer_strict_d', 'cer_strict_n'), 0.305, 1e-3, 'overall strict CER mean');
near(pooled(rows, 'cer_base_d', 'cer_base_n'), 0.180, 1e-3, 'overall base_letters CER pooled');
near(pooled(rows, 'wer_strict_d', 'wer_strict_n'), 0.753, 1e-3, 'overall strict WER pooled');
near(mean(rows, 'cer_legacy_d', 'cer_legacy_n'), 0.365, 1e-3, 'overall legacy CER mean');

const pilot = rows.filter((r) => r.group === 'igbo_tonal_pilot');
assert.strictEqual(pilot.length, 4, 'igbo_tonal_pilot must have 4 clips');
near(pooled(pilot, 'cer_strict_d', 'cer_strict_n'), 0.416, 1e-3, 'pilot strict CER pooled');
near(pooled(pilot, 'cer_tone_d', 'cer_tone_n'), 0.197, 1e-3, 'pilot tone_stripped CER pooled');
near(pooled(pilot, 'cer_base_d', 'cer_base_n'), 0.121, 1e-3, 'pilot base_letters CER pooled');

// 4. Built HTML policy & regression checks
const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
assert(!html.includes('/*__ROWS__*/'), 'unreplaced /*__ROWS__*/ placeholder');
assert(!html.includes('Hallucination rate'), 'stale Hallucination rate wording in HTML');
assert(!html.includes('marks that were not spoken'), 'withdrawn C18 wording in HTML');
assert(!html.includes('.hl-tone .m-dot, .hl-tone .m-both { color: inherit'), 'dual-mark CSS filter bug returned');

console.log('igbo-tone-loss tests: OK');
