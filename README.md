# ml-viz

Interactive explainers for ML, RL and LLM inference, built from real experiments. Static HTML, CSS and JavaScript, no framework and no tracking.

| Folder | What it is | Status |
|---|---|---|
| `inference-playground/` | Continuous batching (closed + open loop), paged KV cache + measured pool breakdown, decode roofline + KV crossover, engine vs vLLM. Real Tesla T4 numbers from [mini-vllm](https://github.com/chizkidd/mini-vllm). | ready |
| `flashattention-tiling/` | Tile-by-tile FlashAttention with online softmax, verified against naive attention in the browser. Single file. | ready |
| `igbo-tone-loss/` | Tone and dot-below mark probe on 21 single-speaker clips: clip explorer, scoreboard, mark types, alignment estimate with confusion matrix, Levenshtein CER/WER profiles, and method checks. Data from [omniASR-igbo-blindspots](https://huggingface.co/datasets/Chiz/omniASR-igbo-blindspots) / [igbo-asr-tonal-evaluation](https://github.com/chizkidd/igbo-asr-tonal-evaluation). | ready |
| `td-lambda/` | TD(lambda) on the 19-state random walk (step through eligibility traces, sweep lambda and alpha), the w/2w example, and Baird's counterexample with toggles for function approximation, bootstrapping, off-policy data and TDC. Simulated live with fixed seeds. | ready |

## Publish with GitHub Pages
1. Create a **public** repo named `ml-viz`, push this folder to `main`.
2. Settings, Pages, Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The site appears at `https://<user>.github.io/ml-viz/`.

Every page loads `shared/theme.css` and `shared/ui.js` (persisted dark/light theme, accessible ARIA tabs with `#hash` deep-linking, reduced-motion check) through relative links, so keep the folder layout as is.

## Build & verify everything

```bash
sh scripts/check.sh                # rebuilds all pages and runs all 4 test suites
sh scripts/check.sh --check-clean  # also fails if generated index.html drifted (run in CI)
```

Requires `python3`, `node`, and `numpy` (for the `td-lambda` Node-vs-NumPy cross-check).

## Inference playground data
Everything labelled *measured* comes from `inference-playground/data.json`, extracted from `mini-vllm`'s `results/*.json`:

```bash
python3 inference-playground/extract_data.py /path/to/mini-vllm/results > inference-playground/data.json
python3 inference-playground/build.py          # rewrites index.html
```

Everything labelled *simulated* comes from `inference-playground/src/sim.js`. Its four constants were fitted to the stage 4 runs; the calibration table on the first tab shows how close it gets. To check it in node: `node inference-playground/tests/test_sim.js`.

`python3 <folder>/build.py --inline` also writes `index.inline.html` (with `shared/theme.css` and `shared/ui.js` inlined) for sharing as a single standalone file.

## Adding a visualization
Create a folder, link `../shared/theme.css` and `../shared/ui.js`, reuse the tokens (`--c-a`, `--c-b`, badges, tabs, cards), and add a card to the root `index.html`.

## Igbo explorer data
`igbo-tone-loss/data.json` is extracted directly from the audited evaluation repo (`data/metadata.csv` + `data/analysis_groups.csv`, verified against `results/canonical_results.csv`):

```bash
python3 igbo-tone-loss/extract_data.py /path/to/igbo-asr-tonal-evaluation-research > igbo-tone-loss/data.json
python3 igbo-tone-loss/build.py
node igbo-tone-loss/tests/test_metrics.js
```

Mark counts reproduce the dataset's expected and produced counts for all 21 clips (count each letter carrying a grave, acute, macron or dot-below), and Levenshtein CER/WER profiles follow `docs/ANALYSIS_SPEC.md`.

## TD(lambda) page
`td-lambda/src/sim.js` holds all the simulation code (seeded RNG, TD(lambda) with accumulating or replacing traces, the lambda sweep, Baird's counterexample with semi-gradient TD(0) and TDC). `python3 td-lambda/build.py` inlines it into `td-lambda/template.html` and writes `index.html`.

`sh td-lambda/tests/run.sh` re-checks the JavaScript against an independent numpy implementation that uses the same random stream (needs node, python3 and numpy). Sweep values agree to about 1e-17 and the Baird runs agree exactly.
