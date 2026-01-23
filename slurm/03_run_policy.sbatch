#!/bin/bash
#SBATCH --job-name=exp-A
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --constraint=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# Experiment A — runs ONE policy on ONE interactive trace on a single
# 4xA40 node. The controller manages all vLLM lifecycle on that node:
# warm prewarm, cold load, switch in place. Pass policy and trace via
# --export, e.g.
#
#   sbatch --export=ALL,POLICY=hybrid_break_even,TRACE=interactive_balanced_mix \
#          slurm/03_run_policy.sbatch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$SCRIPT_DIR"

: "${POLICY:?must set POLICY=...}"
: "${TRACE:?must set TRACE=...}"

OUT_DIR="$EXP_OUT/runs/$TRACE"
LOG_DIR="$EXP_LOGS/$TRACE/$POLICY"
mkdir -p "$OUT_DIR" "$LOG_DIR"
bash slurm/lib/env_snapshot.sh "$LOG_DIR"

python harness/controller.py \
    --config "$EXP_CONFIG" \
    --primitives "$EXP_PRIMITIVES" \
    --policy "$POLICY" \
    --trace "$EXP_TRACES/$TRACE.csv" \
    --trace-name "$TRACE" \
    --pool-dir "$EXP_POOLS" \
    --out-dir "$OUT_DIR" \
    --log-dir "$LOG_DIR"

echo "[A:$TRACE/$POLICY] done"
