#!/bin/bash
# Combined launcher for expanded LOO + nonstationary experiments
# 320 + 40 = 360 total runs, throttled to 6 concurrent jobs.
# Estimated runtime: ~4-6 hours on 8-core machine.

set -e
cd "$(dirname "$0")/.."

PYTHON="/Users/aravraval/miniconda3/envs/dbtl/bin/python"
RUNNER="experiments/run_single_experiment.py"
MAX_PARALLEL=6
SEEDS="10 11 12 13 14 15 16 17 18 19"

APPROACHES="ours cbtl_adapted flat_model without_mood_learning"
EXPANDED_SPLITS="ultra asian indian mediterranean moroccan ethiopian thai spanish"

TOTAL=0
DONE=0
FAILED=0
RUNNING=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_one() {
    local env="$1" approach="$2" seed="$3" outdir="$4"
    mkdir -p "${outdir}"
    $PYTHON $RUNNER \
        env="${env}" approach="${approach}" seed="${seed}" \
        "hydra.sweep.dir=${outdir}" \
        "hydra.run.dir=${outdir}/${seed}" \
        > "${outdir}/${seed}_stdout.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        log "  OK: env=${env} approach=${approach} seed=${seed}"
    else
        log "  FAIL (rc=$rc): env=${env} approach=${approach} seed=${seed}"
    fi
    return $rc
}

wait_for_slot() {
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 1
    done
}

# --- Expanded LOO (320 runs) ---
log "=== Starting Expanded LOO experiments (8 splits x 4 approaches x 10 seeds = 320 runs) ==="
for split in $EXPANDED_SPLITS; do
    for approach in $APPROACHES; do
        for seed in $SEEDS; do
            wait_for_slot
            outdir="logs/runs/expanded_loo_${split}_${approach}"
            run_one "spices_expanded_loo_${split}" "${approach}" "${seed}" "${outdir}" &
            TOTAL=$((TOTAL + 1))
        done
    done
done

# --- Nonstationary (40 runs) ---
log "=== Starting Nonstationary experiments (4 approaches x 10 seeds = 40 runs) ==="
for approach in $APPROACHES; do
    for seed in $SEEDS; do
        wait_for_slot
        outdir="logs/runs/nonstationary_${approach}"
        run_one "spices_nonstationary" "${approach}" "${seed}" "${outdir}" &
        TOTAL=$((TOTAL + 1))
    done
done

log "All ${TOTAL} jobs submitted. Waiting for completion..."
wait

# --- Count results ---
DONE=0
FAILED=0
for split in $EXPANDED_SPLITS; do
    for approach in $APPROACHES; do
        for seed in $SEEDS; do
            outdir="logs/runs/expanded_loo_${split}_${approach}"
            if [ -f "${outdir}/${seed}/eval_results.csv" ] || [ -f "${outdir}/eval_results.csv" ]; then
                DONE=$((DONE + 1))
            else
                FAILED=$((FAILED + 1))
            fi
        done
    done
done
for approach in $APPROACHES; do
    for seed in $SEEDS; do
        outdir="logs/runs/nonstationary_${approach}"
        if [ -f "${outdir}/${seed}/eval_results.csv" ] || [ -f "${outdir}/eval_results.csv" ]; then
            DONE=$((DONE + 1))
        else
            FAILED=$((FAILED + 1))
        fi
    done
done

log "=== COMPLETE: ${DONE}/${TOTAL} succeeded, ${FAILED} failed ==="
