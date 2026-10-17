// Pure simulation code. No DOM. Inlined into index.html by build.py and also unit-checked in node.
// Constants are calibrated to the T4 measurements in chizkidd/minivllm (stage 3 and stage 4).

const CAL = {
  // Fitted by grid search (see CALIBRATION in the page) so that simulated goodput for static and continuous batching
  // matches the stage 4 closed-loop and open-loop runs (RMS log error about 9%). The stage 3 microbenchmark
  // (128-token prompts) has a flatter step time, 24 ms at B=1 to 25 ms at B=64; the mixed workload has longer
  // contexts and more scheduler work, so the fitted per-sequence cost is higher.
  stepBase: 24, stepPerSeq: 0.3,       // decode step (ms) = base + perSeq * active sequences
  prefillPerTok: 0.12, prefillFloor: 30, // prefill (ms) = max(floor, perTok * padded prompt tokens)
};
const stepMs = (n) => CAL.stepBase + CAL.stepPerSeq * n;
const prefillMs = (tokens) => Math.max(CAL.prefillFloor, CAL.prefillPerTok * tokens);

function rng(seed) { // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// Same "mixed" workload the engine benchmarks use: prompts from 32..512, 25% long outputs by default.
function makeTrace({ n = 64, rate = 4, pLong = 0.25, seed = 1, closed = false } = {}) {
  const r = rng(seed), prompts = [32, 64, 128, 256, 512];
  let t = 0; const reqs = [];
  for (let i = 0; i < n; i++) {
    if (!closed) t += -Math.log(1 - r()) / rate; // Poisson arrivals, seconds
    const isLong = r() < pLong;
    const out = isLong ? 128 + Math.floor(r() * 129) : 16 + Math.floor(r() * 33);
    reqs.push({ id: i, arrival: closed ? 0 : t, prompt: prompts[Math.floor(r() * prompts.length)], out, isLong });
  }
  return reqs;
}

// Static batching: take up to B queued requests, prefill them padded to the longest prompt, then decode until the
// longest output in the batch is done. Slots of finished requests sit idle until the whole batch ends.
function simStatic(reqs, B) {
  const q = reqs.map((r) => ({ ...r })), items = []; let t = 0, i = 0;
  while (i < q.length) {
    if (q[i].arrival > t) t = q[i].arrival;
    const batch = []; while (i < q.length && batch.length < B && q[i].arrival <= t) batch.push(q[i++]);
    const maxP = Math.max(...batch.map((r) => r.prompt)), maxO = Math.max(...batch.map((r) => r.out));
    const pf = prefillMs(batch.length * maxP) / 1000, st = stepMs(batch.length) / 1000;
    const start = t, decodeStart = t + pf, end = decodeStart + maxO * st;
    batch.forEach((r, k) => items.push({ id: r.id, slot: k, arrival: r.arrival, start, decodeStart,
      done: decodeStart + r.out * st, hold: end, out: r.out, isLong: r.isLong }));
    t = end;
  }
  return summarize(items, B, t);
}

// Continuous batching: every iteration, admit queued requests into free slots, then one decode step for all active.
function simContinuous(reqs, B) {
  const q = reqs.map((r) => ({ ...r })), items = [], slots = new Array(B).fill(null); let t = 0, i = 0, active = 0;
  while (i < q.length || active > 0) {
    if (active === 0 && i < q.length && q[i].arrival > t) t = q[i].arrival;
    let admitTok = 0; const admitted = [];
    for (let s = 0; s < B && i < q.length && q[i].arrival <= t; s++) {
      if (slots[s]) continue;
      const r = q[i++]; admitTok += r.prompt;
      slots[s] = { r, left: r.out, start: t, decodeStart: 0 }; active++; admitted.push(slots[s]);
    }
    if (admitted.length) { const pf = prefillMs(admitTok) / 1000; t += pf; admitted.forEach((a) => { a.decodeStart = t; }); }
    t += stepMs(active) / 1000;
    for (let s = 0; s < B; s++) { const a = slots[s]; if (!a) continue;
      if (--a.left === 0) { items.push({ id: a.r.id, slot: s, arrival: a.r.arrival, start: a.start, decodeStart: a.decodeStart,
        done: t, hold: t, out: a.r.out, isLong: a.r.isLong }); slots[s] = null; active--; } }
  }
  return summarize(items, B, t);
}

function pct(arr, p) { const a = [...arr].sort((x, y) => x - y); return a.length ? a[Math.min(a.length - 1, Math.floor(p * a.length))] : 0; }
function summarize(items, B, makespan) {
  items.sort((a, b) => a.id - b.id);
  const toks = items.reduce((s, x) => s + x.out, 0);
  const lat = items.map((x) => x.done - x.arrival), ttft = items.map((x) => x.decodeStart - x.arrival);
  const heldLane = items.reduce((s, x) => s + (x.hold - x.start), 0), usefulLane = items.reduce((s, x) => s + (x.done - x.start), 0);
  return { items, B, makespan, goodput: toks / makespan, latP50: pct(lat, .5), latP99: pct(lat, .99), ttftP50: pct(ttft, .5),
    laneUse: usefulLane / (B * makespan), laneHeld: heldLane / (B * makespan) };
}

if (typeof module !== 'undefined') module.exports = { CAL, stepMs, prefillMs, rng, makeTrace, simStatic, simContinuous };
