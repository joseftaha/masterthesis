#!/bin/bash
# Site-specific environment. Sourced by every sbatch script. Edit the
# top of this file once when porting to a new Slurm cluster.

# --------------------------------------------------------------------
# 1. Cluster identity (only thing you must change per site)
# --------------------------------------------------------------------
export EXP_ROOT="${EXP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Source the site-specific cluster.env first, if it exists. Variables
# set there win over the defaults below. cluster.env.template documents
# every supported variable; copy it to cluster.env and edit once.
if [ -f "$EXP_ROOT/cluster.env" ]; then
    # shellcheck disable=SC1091
    source "$EXP_ROOT/cluster.env"
fi

export EXP_PARTITION="${EXP_PARTITION:-gpu}"
export EXP_ACCOUNT="${EXP_ACCOUNT:-}"
export EXP_QOS="${EXP_QOS:-}"
export EXP_NODE_CONSTRAINT="${EXP_NODE_CONSTRAINT:-a40}"

# Per-job wall clocks (override via env if your scheduler is stricter).
export EXP_TIME_WARM="${EXP_TIME_WARM:-06:00:00}"
export EXP_TIME_COLD="${EXP_TIME_COLD:-06:00:00}"
export EXP_TIME_BATCH="${EXP_TIME_BATCH:-04:00:00}"
export EXP_TIME_MEASURE="${EXP_TIME_MEASURE:-01:00:00}"

# --------------------------------------------------------------------
# 2. Filesystem layout
# --------------------------------------------------------------------
export EXP_OUT="${EXP_OUT:-$EXP_ROOT/../runs_real}"
export EXP_LOGS="${EXP_LOGS:-$EXP_OUT/logs}"
export EXP_PRIMITIVES="${EXP_PRIMITIVES:-$EXP_OUT/primitives.json}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf-cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$EXP_OUT" "$EXP_LOGS" "$HF_HOME"

# --------------------------------------------------------------------
# 3. Software stack
# --------------------------------------------------------------------
# Either an EasyBuild/Lmod module name OR a path to a pre-built venv.
# At least one of these must work on the compute node.
EXP_MODULE_LOAD="${EXP_MODULE_LOAD:-}"
EXP_VENV="${EXP_VENV:-$EXP_ROOT/.venv-real}"

if [ -n "$EXP_MODULE_LOAD" ]; then
    module purge
    # shellcheck disable=SC2086
    module load $EXP_MODULE_LOAD
fi
if [ -d "$EXP_VENV" ]; then
    # shellcheck disable=SC1091
    source "$EXP_VENV/bin/activate"
fi

# Make `vllm` and `python` deterministic.
export PYTHONHASHSEED=0
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --------------------------------------------------------------------
# 4. Pinned config + traces
# --------------------------------------------------------------------
export EXP_CONFIG="$EXP_ROOT/config/experiment.json"
export EXP_TRACES="$EXP_ROOT/traces"
export EXP_POOLS="$EXP_ROOT/datasets/cache"
