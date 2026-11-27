# ml-viz

Interactive explainers for ML, RL, optimization, architectures, speech, and LLM inference, built from real experiments. Static HTML, CSS and JavaScript, no framework and no tracking.

| Folder | What it is | Status |
|---|---|---|
| `inference-playground/` | Continuous batching (prefill budgets, KV admission caps, closed + open loop), paged KV cache + measured pool breakdown, decode roofline + KV crossover + A100/H100 predictions, and Stage 12 &rarr; 13 &rarr; 14 vs vLLM with Graph-in-Batch paradox &amp; ablations. Real Tesla T4 numbers from [mini-vllm](https://github.com/chizkidd/mini-vllm). | ready |
| `speculative-decoding/` | Token-by-token draft/verify/rollback, the exact speedup law `E[tokens]/(k·c + v)`, and real T4 measurements from [mini-vllm](https://github.com/chizkidd/mini-vllm) Stages 6–11 showing why int8 and speculation fail to multiply. | ready |
| `flashattention-tiling/` | Tile-by-tile FlashAttention with online softmax, verified against naive attention in the browser. Single file. | ready |
| `moe-routing/` | Sparse top-k routing &amp; capacity overflow (dropped tokens), router collapse vs Switch aux loss vs DeepSeek-V3 aux-loss-free bias, and active vs total serving memory roofline. | ready |
| `muon-optimizers/` | Step-by-step quintic Newton-Schulz orthogonalization with exact singular values, SGD/Momentum/AdamW/Muon comparison on ill-conditioned matrices, and MuonClip per-head QK-logit clipping. | ready |
| `igbo-tone-loss/` | Tone and dot-below mark probe on 21 single-speaker clips with inline audio, acoustic/export-chain audit metadata, language-based `analysis_group`s + Historical toggle, 8-class base-letter-aware alignment (`0/32` headline + alignment ambiguity), Levenshtein CER/WER profiles, and exact DP resample distributions + LOO ranges. Data from [omniASR-igbo-blindspots](https://huggingface.co/datasets/Chiz/omniASR-igbo-blindspots) / [igbo-asr-tonal-evaluation](https://github.com/chizkidd/igbo-asr-tonal-evaluation). | ready |
| `td-lambda/` | TD(lambda) on the 19-state random walk (step through eligibility traces, sweep lambda and alpha), the w/2w example, and Baird's counterexample with toggles for function approximation, bootstrapping, off-policy data and TDC. Simulated live with fixed seeds. | ready |
| `policy-gradients/` | Sutton &amp; Barto Ch. 13 sequel to `td-lambda`: Short Corridor with Switched Actions (exact analytical value curve vs &epsilon;-greedy), REINFORCE vs learned baseline variance reduction, and online Actor-Critic vs Monte Carlo. | ready |

## Publish with GitHub Pages
1. Create a **public** repo named `ml-viz`, push this folder to `main`.
2. Settings, Pages, Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The site appears at `https://<user>.github.io/ml-viz/`.

Every page loads `shared/theme.css` and `shared/ui.js` (persisted dark/light theme, accessible ARIA tabs with `#hash` deep-linking, reduced-motion check, and delegated hover/touch/focus SVG tooltips) through relative links, so keep the folder layout as is.

## Build & verify everything

```bash
sh scripts/check.sh                # rebuilds all pages and runs all 8 test suites
sh scripts/check.sh --check-clean  # also fails if generated index.html drifted (run in CI)
```

Requires `python3`, `node`, and `numpy` (for the `td-lambda` Node-vs-NumPy cross-check).

## Extracting measured data
- **`inference-playground/`**: `python3 inference-playground/extract_data.py /path/to/mini-vllm/results > inference-playground/data.json && python3 inference-playground/build.py`
- **`speculative-decoding/`**: `python3 speculative-decoding/extract_data.py /path/to/mini-vllm/results > speculative-decoding/data.json && python3 speculative-decoding/build.py`
- **`igbo-tone-loss/`**: `python3 igbo-tone-loss/extract_data.py /path/to/igbo-asr-tonal-evaluation-research > igbo-tone-loss/data.json && python3 igbo-tone-loss/build.py`

Every builder also supports `--inline` (`python3 <folder>/build.py --inline`) to write a single standalone `index.inline.html` with `shared/theme.css` and `shared/ui.js` inlined.

## Adding a visualization
Create a folder, link `../shared/theme.css` and `../shared/ui.js`, reuse the tokens (`--c-a`, `--c-b`, badges, tabs, cards), add a card to the root `index.html`, and wire its build/test into `scripts/check.sh`.
