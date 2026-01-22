#!/bin/bash
#SBATCH --job-name=exp-download
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Pre-fetches both Llama-3 model checkpoints into HF_HOME (local NVMe).
# Run this once on a login node or a CPU partition with internet egress.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/env.sh"
cd "$SCRIPT_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "[download] WARNING: HF_TOKEN not set; gated Llama-3 weights will fail."
fi
huggingface-cli login --token "${HF_TOKEN:-}" --add-to-git-credential || true

python harness/download_models.py --config "$EXP_CONFIG"
echo "[download] cache populated at $HF_HOME"
