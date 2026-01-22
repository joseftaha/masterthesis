#!/bin/bash
# Capture an environment snapshot for one experiment run. Writes a single
# file ``env_snapshot.txt`` into $1 (default: cwd) so each result
# directory carries a self-contained reproducibility receipt.
#
#   bash slurm/lib/env_snapshot.sh /path/to/run/dir
set -euo pipefail
OUT_DIR="${1:-$PWD}"
mkdir -p "$OUT_DIR"
SNAP="$OUT_DIR/env_snapshot.txt"

{
    echo "# env_snapshot for $OUT_DIR"
    echo "# generated: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "## host"
    echo "hostname     = $(hostname)"
    echo "user         = ${USER:-unknown}"
    echo "slurm_jobid  = ${SLURM_JOB_ID:-}"
    echo "slurm_node   = ${SLURMD_NODENAME:-}"
    echo "slurm_part   = ${SLURM_JOB_PARTITION:-}"
    echo
    echo "## git"
    if git -C "${EXP_ROOT:-.}" rev-parse --show-toplevel >/dev/null 2>&1; then
        echo "repo         = $(git -C "${EXP_ROOT:-.}" rev-parse --show-toplevel)"
        echo "git_sha      = $(git -C "${EXP_ROOT:-.}" rev-parse HEAD)"
        echo "git_branch   = $(git -C "${EXP_ROOT:-.}" rev-parse --abbrev-ref HEAD)"
        echo "git_dirty    = $(git -C "${EXP_ROOT:-.}" status --porcelain | wc -l) files"
    else
        echo "git_sha      = (not a git checkout)"
    fi
    echo
    echo "## software"
    echo "python       = $(python --version 2>&1)"
    echo "pip_freeze:"
    pip freeze 2>/dev/null | grep -E '^(vllm|torch|transformers|datasets|aiohttp|huggingface_hub|tokenizers|numpy|pandas|matplotlib)=' | sed 's/^/  /' || true
    echo "vllm_version = $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'n/a')"
    echo
    echo "## cuda / driver"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader | sed 's/^/  /'
    fi
    if command -v nvcc >/dev/null 2>&1; then
        echo "nvcc:"
        nvcc --version | sed 's/^/  /'
    fi
    echo
    echo "## env vars"
    env | grep -E '^(EXP_|HF_|VLLM_|CUDA_|CUBLAS_|PYTHONHASHSEED|TOKENIZERS_)' | sort | sed 's/HF_TOKEN=.*/HF_TOKEN=<redacted>/'
    echo
    echo "## datasets"
    if [ -f "$EXP_POOLS/dataset_revisions.json" ]; then
        cat "$EXP_POOLS/dataset_revisions.json" | sed 's/^/  /'
    fi
    echo
    echo "## models"
    python -c "import json,sys; c=json.load(open('${EXP_CONFIG:-/dev/null}')); m=c['models']; print(f\"  primary={m['primary']['hf_repo']}@{m['primary']['revision']} quant={m['primary']['quantization']}\"); print(f\"  secondary={m['secondary'].get('quant_repo_override') or m['secondary']['hf_repo']}@{m['secondary']['revision']} quant={m['secondary']['quantization']}\")" 2>/dev/null || true
} > "$SNAP"

echo "[env_snapshot] wrote $SNAP"
