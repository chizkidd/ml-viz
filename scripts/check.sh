#!/bin/sh
# Rebuild all template-backed explainers, optionally check for drift, and run all test suites.
# Usage:
#   sh scripts/check.sh                 # rebuild + run all tests
#   sh scripts/check.sh --check-clean   # rebuild + fail if generated index.html drifted + run all tests
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> Rebuilding pages"
python3 inference-playground/build.py
python3 igbo-tone-loss/build.py
python3 td-lambda/build.py

if [ "${1:-}" = "--check-clean" ]; then
  echo "==> Checking for generated HTML drift"
  git diff --exit-code -- inference-playground/index.html igbo-tone-loss/index.html td-lambda/index.html
fi

echo "==> Running inference-playground tests"
node inference-playground/tests/test_sim.js

echo "==> Running igbo-tone-loss tests"
node igbo-tone-loss/tests/test_metrics.js

echo "==> Running flashattention-tiling tests"
node flashattention-tiling/tests/test_fa.js

echo "==> Running td-lambda tests (Node vs NumPy)"
if python3 -c "import numpy" 2>/dev/null; then
  sh td-lambda/tests/run.sh
else
  echo "ERROR: numpy is required for td-lambda/tests/run.sh (pip install numpy)" >&2
  exit 1
fi

echo "ALL CHECKS PASSED"
