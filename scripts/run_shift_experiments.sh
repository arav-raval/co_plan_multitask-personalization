#!/bin/bash
# 4 shift variants x 4 approaches x 10 seeds = 160 runs
# Estimated: ~2 hours with 6 concurrent jobs.
set -e
cd "$(dirname "$0")/.."

PYTHON="/Users/aravraval/miniconda3/envs/dbtl/bin/python"
RUNNER="experiments/run_single_experiment.py"
MAX_PARALLEL=6
SEEDS="10 11 12 13 14 15 16 17 18 19"
APPROACHES="ours cbtl_adapted flat_model without_mood_learning"
SHIFT_ENVS="spices_shift_soft spices_shift_medium spices_shift_strong spices_shift_random"

TOTAL=0

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_one() {
    local env="$1" approach="$2" seed="$3"
    local outdir="logs/runs/${env}_${approach}"
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

log "=== Starting shift experiments (4 variants x 4 approaches x 10 seeds = 160 runs) ==="

for env in $SHIFT_ENVS; do
    for approach in $APPROACHES; do
        for seed in $SEEDS; do
            wait_for_slot
            run_one "$env" "$approach" "$seed" &
            TOTAL=$((TOTAL + 1))
        done
    done
done

log "All ${TOTAL} jobs submitted. Waiting for completion..."
wait

# Count results
DONE=0
FAILED=0
for env in $SHIFT_ENVS; do
    for approach in $APPROACHES; do
        for seed in $SEEDS; do
            outdir="logs/runs/${env}_${approach}"
            if [ -f "${outdir}/${seed}/eval_results.csv" ]; then
                DONE=$((DONE + 1))
            else
                FAILED=$((FAILED + 1))
            fi
        done
    done
done

log "=== COMPLETE: ${DONE}/${TOTAL} succeeded, ${FAILED} failed ==="
