#!/bin/bash
#SBATCH --job-name=exp-plot
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Renders the experiment figures from the per-trace result directories.
# Run after 05_aggregate so each <trace>/summaries.csv is in place.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$SCRIPT_DIR"

OUT="$EXP_OUT/plots"
mkdir -p "$OUT"

# The plot script discovers per-trace dirs under --runs-root. Stage
# Experiment A and B output into one combined view so a single
# invocation produces all figures.
COMBINED="$EXP_OUT/_plot_combined"
rm -rf "$COMBINED"; mkdir -p "$COMBINED"
if [ -d "$EXP_OUT/runs" ]; then
    for d in "$EXP_OUT"/runs/*; do
        [ -d "$d" ] && ln -s "$d" "$COMBINED/$(basename "$d")"
    done
fi
if [ -d "$EXP_OUT/runs_b" ]; then
    for d in "$EXP_OUT"/runs_b/*; do
        [ -d "$d" ] && ln -s "$d" "$COMBINED/$(basename "$d")"
    done
fi

python harness/plot.py \
    --runs-root "$COMBINED" \
    --out "$OUT"
echo "[plot] figures under $OUT"
