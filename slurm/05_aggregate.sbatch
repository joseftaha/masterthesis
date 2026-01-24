#!/bin/bash
#SBATCH --job-name=exp-aggregate
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Compiles per-trace summaries.csv files from the per-policy details.
# Run after all 03_run_policy and 04_run_batch jobs for a given trace
# have completed. CPU-only.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$SCRIPT_DIR"

INTERACTIVE=(interactive_sparse_tail interactive_bursty_tail
             interactive_popularity_shift interactive_balanced_mix)
BATCH=(offline_batch_c1_primary offline_batch_c4_primary offline_batch_c8_primary)

for t in "${INTERACTIVE[@]}"; do
    if [ -d "$EXP_OUT/runs/$t" ]; then
        python harness/aggregate.py --runs-dir "$EXP_OUT/runs/$t" --trace "$t"
    fi
done
for t in "${BATCH[@]}"; do
    if [ -d "$EXP_OUT/runs_b/$t" ]; then
        python harness/aggregate.py --runs-dir "$EXP_OUT/runs_b/$t" --trace "$t"
    fi
done

echo "[aggregate] all summaries written"
