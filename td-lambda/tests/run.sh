#!/bin/sh
# Cross-check src/sim.js against an independent numpy implementation that uses the same random stream.
set -e
cd "$(dirname "$0")"
python3 ref.py > /dev/null 2>&1
python3 ref_baird.py > /dev/null 2>&1
node check_sweep.js ./ref_sweep.json
node check_baird.js ./ref_baird.json
rm -f ref_sweep.json ref_baird.json
