#!/bin/bash
#SBATCH --job-name=exp-primitives
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:4
#SBATCH --constraint=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
# Measures warm/cold TTFT and end-to-end latency for both models plus
# both directed switch costs at the fixed reference prompt
# (512 input tokens, 128 output tokens, concurrency 1).
# Output: $EXP_PRIMITIVES (JSON consumed by the policy controller).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
[ -n "$EXP_PARTITION" ] && scontrol show partition "$EXP_PARTITION" >/dev/null
cd "$SCRIPT_DIR"

mkdir -p "$EXP_LOGS/primitives"
bash slurm/lib/env_snapshot.sh "$EXP_LOGS/primitives"
python harness/measure_primitives.py \
    --config "$EXP_CONFIG" \
    --out "$EXP_PRIMITIVES" \
    --log-dir "$EXP_LOGS/primitives"
echo "[primitives] wrote $EXP_PRIMITIVES"
