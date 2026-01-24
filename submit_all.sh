#!/bin/bash
# End-to-end orchestrator. Submits every job needed to reproduce the
# results in chapter 8 with the correct dependency chain.
#
#   bash submit_all.sh [--no-setup] [--policies-only] [--batch-only]
#
# The chain is:
#
#   00_setup        -> CPU only, builds prompt pools + traces
#   01_download     -> CPU only, populates HF_HOME
#   02_primitives   -> 4xA40,    measures warm/cold/switch primitives
#   03_run_policy   -> 4xA40,    one job per (policy x interactive trace) = 20
#   04_run_batch    -> 4xA40,    one job per (mode x concurrency x batch trace)
#                                = 2 modes x 3 concurrencies x 3 traces = 18
#   05_aggregate    -> CPU,      compiles summaries.csv per trace
#   06_plot         -> CPU,      regenerates the chapter figures
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source env.sh

DO_SETUP=1; DO_A=1; DO_B=1
for a in "$@"; do
    case "$a" in
        --no-setup)      DO_SETUP=0 ;;
        --policies-only) DO_B=0 ;;
        --batch-only)    DO_A=0 ;;
        *) echo "unknown arg: $a"; exit 1 ;;
    esac
done

submit() {
    # echo+submit, return jobid on stdout
    local id
    id=$(sbatch --parsable "$@" | tr -d '[:space:]')
    echo "$id"
}

DEP=""
if [ "$DO_SETUP" -eq 1 ]; then
    SETUP_ID=$(submit slurm/00_setup.sbatch)
    DL_ID=$(submit --dependency="afterok:$SETUP_ID" slurm/01_download_models.sbatch)
    PRIM_ID=$(submit --dependency="afterok:$DL_ID"     slurm/02_measure_primitives.sbatch)
    DEP="afterok:$PRIM_ID"
    echo "setup=$SETUP_ID download=$DL_ID primitives=$PRIM_ID"
else
    [ -f "$EXP_PRIMITIVES" ] || {
        echo "ERROR: --no-setup but $EXP_PRIMITIVES is missing"; exit 1; }
fi

A_IDS=()
if [ "$DO_A" -eq 1 ]; then
    POLICIES=(all_cold static_hot_8B static_hot_70B
              reactive_switch_on_miss hybrid_break_even)
    TRACES_A=(interactive_sparse_tail interactive_bursty_tail
              interactive_popularity_shift interactive_balanced_mix)
    for trace in "${TRACES_A[@]}"; do
        for policy in "${POLICIES[@]}"; do
            id=$(submit ${DEP:+--dependency=$DEP} \
                       --export=ALL,POLICY="$policy",TRACE="$trace" \
                       --job-name="A-${trace}-${policy}" \
                       slurm/03_run_policy.sbatch)
            A_IDS+=("$id")
            echo "  A $trace $policy -> $id"
        done
    done
fi

B_IDS=()
if [ "$DO_B" -eq 1 ]; then
    TRACES_B=(offline_batch_c1_primary offline_batch_c4_primary offline_batch_c8_primary)
    for trace in "${TRACES_B[@]}"; do
        for mode in warm cold; do
            for c in 1 4 8; do
                id=$(submit ${DEP:+--dependency=$DEP} \
                           --export=ALL,TRACE="$trace",CONC="$c",MODE="$mode" \
                           --job-name="B-${trace}-${mode}-c${c}" \
                           slurm/04_run_batch.sbatch)
                B_IDS+=("$id")
                echo "  B $trace $mode c=$c -> $id"
            done
        done
    done
fi

ALL_IDS=("${A_IDS[@]}" "${B_IDS[@]}")
if [ "${#ALL_IDS[@]}" -gt 0 ]; then
    AGG_DEP="afterany:$(IFS=:; echo "${ALL_IDS[*]}")"
    AGG_ID=$(submit --dependency="$AGG_DEP" slurm/05_aggregate.sbatch)
    PLOT_ID=$(submit --dependency="afterok:$AGG_ID" slurm/06_plot.sbatch)
    echo "aggregate=$AGG_ID plot=$PLOT_ID"
fi
