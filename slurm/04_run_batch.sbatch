#!/bin/bash
#SBATCH --job-name=exp-B
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --constraint=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
# Experiment B — replays one offline batch trace at a fixed concurrency
# against either the warm hot path or the cold baseline.
#
#   sbatch --export=ALL,TRACE=offline_batch_c4_primary,CONC=4,MODE=warm \
#          slurm/04_run_batch.sbatch
#
# CONC and MODE are independent: CONC controls the client-side dispatch
# semaphore, MODE controls whether the server is kept resident or
# reloaded between waves.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$SCRIPT_DIR"

: "${TRACE:?must set TRACE=...}"
: "${CONC:?must set CONC=...}"
: "${MODE:?must set MODE=warm|cold}"

OUT_DIR="$EXP_OUT/runs_b/$TRACE"
LOG_DIR="$EXP_LOGS/$TRACE/${MODE}_c${CONC}"
mkdir -p "$OUT_DIR" "$LOG_DIR"
bash slurm/lib/env_snapshot.sh "$LOG_DIR"

python harness/run_batch.py \
    --config "$EXP_CONFIG" \
    --trace "$EXP_TRACES/$TRACE.csv" \
    --trace-name "$TRACE" \
    --pool-dir "$EXP_POOLS" \
    --concurrency "$CONC" \
    --mode "$MODE" \
    --out "$OUT_DIR/${MODE}_c${CONC}__details.csv" \
    --log-dir "$LOG_DIR"

# Experiment B has no policy-level switches.csv; emit an empty one so
# aggregate.py and the plot pipeline see a uniform schema across A and B.
SW_PATH="$OUT_DIR/${MODE}_c${CONC}__switches.csv"
if [ ! -f "$SW_PATH" ]; then
    echo "policy,trace_name,at_time,from_model,to_model,cost,reason" > "$SW_PATH"
fi

echo "[B:$TRACE/$MODE/c=$CONC] done"
