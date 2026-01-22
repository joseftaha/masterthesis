#!/bin/bash
#SBATCH --job-name=exp-setup
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# This job runs ONCE on a login or CPU partition (no GPU). It builds
# the prompt pools and the seven CSV traces. Pure preparation - it never
# launches vLLM.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$SCRIPT_DIR"

python datasets/build_prompts.py --seed 20240501 \
                                 --n_sharegpt 4000 --n_cnn 4000

python traces/build_traces.py --seed 20240501 --out traces

echo "[setup] datasets + traces ready under $SCRIPT_DIR"
