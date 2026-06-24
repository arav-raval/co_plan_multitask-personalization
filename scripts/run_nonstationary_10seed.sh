#!/bin/bash
# Full 10-seed nonstationary experiments: 4 approaches x 10 seeds = 40 runs
# Much smaller than expanded LOO, should finish in ~30-60 min with 6 cores.

set -e
cd "$(dirname "$0")/.."

PYTHON="/Users/aravraval/miniconda3/envs/dbtl/bin/python"
RUNNER="experiments/run_single_experiment.py"

APPROACHES=(ours cbtl_adapted flat_model without_mood_learning)
SEEDS=$(seq 10 19)

MAX_PARALLEL=6

run_one() {
    local approach="$1" seed="$2"
    local outdir="logs/runs/nonstationary_${approach}"

    $PYTHON $RUNNER \
        env=spices_nonstationary approach="${approach}" seed="${seed}" \
        "hydra.sweep.dir=${outdir}" \
        "hydra.run.dir=${outdir}/${seed}" \
        2>&1 | tail -5

    echo "[DONE] approach=${approach} seed=${seed}"
}

export -f run_one
export PYTHON RUNNER

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with ${MAX_PARALLEL} jobs"
    for approach in "${APPROACHES[@]}"; do
        for seed in $SEEDS; do
            echo "$approach $seed"
        done
    done | parallel --colsep ' ' -j ${MAX_PARALLEL} run_one {1} {2}
else
    echo "GNU parallel not found. Using background jobs with throttle."
    running=0
    for approach in "${APPROACHES[@]}"; do
        for seed in $SEEDS; do
            run_one "$approach" "$seed" &
            running=$((running + 1))
            if [ $running -ge $MAX_PARALLEL ]; then
                wait -n
                running=$((running - 1))
            fi
        done
    done
    wait
fi

echo "All nonstationary experiments complete!"
