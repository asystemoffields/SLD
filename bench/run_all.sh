#!/usr/bin/env bash
# Reproduce the SLD results (CPU). Trains the teacher/draft on first run (cached
# after), then runs the frontier, the ablations, the summary tables and figure.
#
# Usage:  bash bench/run_all.sh
# Needs:  the jumprec substrate on the path. Set JUMPREC to its checkout.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
JUMPREC="${JUMPREC:-$(cd "$HERE/../SMOKE" 2>/dev/null && pwd || echo "")}"
PY="${PY:-python}"
export PYTHONPATH="$JUMPREC:$HERE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"

if [ -z "$JUMPREC" ] || [ ! -d "$JUMPREC/jumprec" ]; then
  echo "Set JUMPREC to the jumprec checkout (git clone github.com/asystemoffields/jumprec)"; exit 1
fi
echo "jumprec substrate: $JUMPREC"

echo "== frontier (depth / horizon / length-gen / controls / wall-clock) =="
"$PY" bench/experiment.py --tag main

echo "== draft-quality ablation (speedup tracks acceptance, always lossless) =="
"$PY" bench/draft_quality.py

echo "== convergent-loop ablation (SLD vs a fair early-exit) =="
"$PY" bench/convergent.py

echo "== summary tables + figure =="
"$PY" bench/summarize.py main
"$PY" bench/plot.py main

echo
echo "Done. See results/*.json, results/frontier_main.png, and RESULTS.md."
echo "Optional (slower / downloads): bench/incontext.py ; bench/parcae_cpu.py ; bench/parcae_sld.py"
