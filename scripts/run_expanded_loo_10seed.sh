#!/bin/bash
# Full 10-seed expanded LOO experiments: 8 splits x 4 approaches x 10 seeds = 320 runs
# Uses GNU parallel (or background jobs) to maximize CPU utilization.
# Each run takes ~2-5 min on spices, so total ~3-5 hours with 8 cores.

set -e
cd "$(dirname "$0")/.."

PYTHON="/Users/aravraval/miniconda3/envs/dbtl/bin/python"
RUNNER="experiments/run_single_experiment.py"

SPLITS=(ultra asian indian mediterranean moroccan ethiopian thai spanish)
APPROACHES=(ours cbtl_adapted flat_model without_mood_learning)
SEEDS=$(seq 10 19)

MAX_PARALLEL=6  # Leave 2 cores free for system

run_one() {
    local split="$1" approach="$2" seed="$3"
    local env="spices_expanded_loo_${split}"
    local outdir="logs/runs/expanded_loo_${split}_${approach}"

    $PYTHON $RUNNER \
        env="${env}" approach="${approach}" seed="${seed}" \
        "hydra.sweep.dir=${outdir}" \
        "hydra.run.dir=${outdir}/${seed}" \
        2>&1 | tail -5

    echo "[DONE] split=${split} approach=${approach} seed=${seed}"
}

export -f run_one
export PYTHON RUNNER

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel with ${MAX_PARALLEL} jobs"
    for split in "${SPLITS[@]}"; do
        for approach in "${APPROACHES[@]}"; do
            for seed in $SEEDS; do
                echo "$split $approach $seed"
            done
        done
    done | parallel --colsep ' ' -j ${MAX_PARALLEL} run_one {1} {2} {3}
else
    echo "GNU parallel not found. Using background jobs with throttle."
    running=0
    for split in "${SPLITS[@]}"; do
        for approach in "${APPROACHES[@]}"; do
            for seed in $SEEDS; do
                run_one "$split" "$approach" "$seed" &
                running=$((running + 1))
                if [ $running -ge $MAX_PARALLEL ]; then
                    wait -n
                    running=$((running - 1))
                fi
            done
        done
    done
    wait
fi

echo "All expanded LOO experiments complete!"
