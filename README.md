# ml-viz

Interactive explainers for ML, RL and LLM inference, built from real experiments. Static HTML, CSS and JavaScript, no framework and no tracking.

| Folder | What it is | Status |
|---|---|---|
| `inference-playground/` | Continuous batching, paged KV cache, decode roofline, engine vs vLLM. Real Tesla T4 numbers from [minivllm](https://github.com/chizkidd/minivllm). | ready |
| `flashattention-tiling/` | Tile-by-tile FlashAttention with online softmax, verified against naive attention in the browser. Single file. | ready |
| `igbo-tone-loss/` | Tone-mark loss in ASR on 21 Igbo clips: clip explorer, scoreboard with bootstrap intervals, mark types, a position-by-position alignment estimate with a confusion matrix, and method checks (resampling unit, CER vs tone loss, limitations). Data from [omniASR-igbo-blindspots](https://huggingface.co/datasets/Chiz/omniASR-igbo-blindspots). | ready |
| `td-lambda/` | TD(lambda) on the 19-state random walk (step through eligibility traces, sweep lambda and alpha), the w/2w example, and Baird's counterexample with toggles for function approximation, bootstrapping, off-policy data and TDC. Simulated live with fixed seeds. | ready |

## Publish with GitHub Pages
1. Create a **public** repo named `ml-viz`, push this folder to `main`.
2. Settings, Pages, Source: *Deploy from a branch*, branch `main`, folder `/ (root)`.
3. The site appears at `https://<user>.github.io/ml-viz/`.

Every page loads `shared/theme.css` through a relative link, so keep the folder layout as is.

## Inference playground data
Everything labelled *measured* comes from `inference-playground/data.json`, extracted from minivllm's `results/*.json`:

```bash
python3 inference-playground/extract_data.py /path/to/minivllm/results > inference-playground/data.json
python3 inference-playground/build.py          # rewrites index.html
```

Everything labelled *simulated* comes from `inference-playground/src/sim.js`. Its four constants were fitted to the stage 4 runs; the calibration table on the first tab shows how close it gets. To re-check it in node: `node -e "require('./inference-playground/src/sim.js')"`.

`python3 inference-playground/build.py --inline` also writes `index.inline.html`, the same page with the theme inlined, handy for sharing as a single file.

## Adding a visualization
Create a folder, link `../shared/theme.css`, reuse the tokens (`--c-a`, `--c-b`, badges, tabs, cards), and add a card to the root `index.html`.

## Igbo explorer data
`igbo-tone-loss/rows.py` holds the 21 dataset rows (text only, no audio). `python3 igbo-tone-loss/build.py` rebuilds `index.html` from `template.html`. Mark counts use a rule that reproduces the dataset's expected and produced counts for all 21 rows: count each letter carrying a grave, acute, macron or dot below.

## TD(lambda) page
`td-lambda/src/sim.js` holds all the simulation code (seeded RNG, TD(lambda) with accumulating or replacing traces, the lambda sweep, Baird's counterexample with semi-gradient TD(0) and TDC). `python3 td-lambda/build.py` inlines it into `td-lambda/template.html` and writes `index.html`.

`sh td-lambda/tests/run.sh` re-checks the JavaScript against an independent numpy implementation that uses the same random stream (needs node, python3 and numpy). Sweep values agree to about 1e-17 and the Baird runs agree exactly.
