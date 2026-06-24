# Experiment Reproduction Guide

## Quick Reference

**Python**: `.venv/bin/python`
**Single-experiment runner**: `experiments/run_single_experiment.py`
**Transfer experiment runner**: `experiments/run_transfer_experiment.py`
**Layout LOO runner**: `experiments/run_overcooked_layout_loo.py`
**Output base**: `logs/runs/`

### Run command template
```bash
.venv/bin/python experiments/run_single_experiment.py \
    env=<env_config> \
    approach=<approach_config> \
    seed=<seed> \
    "hydra.run.dir=logs/runs/<dir_name>/seed_<seed>"
```

---

## Approach Configs

| Config name | Description |
|-------------|-------------|
| `ours` | DBTL: full HBM (μ → θ → φ) + vector ψ + max-entropy exploration |
| `scalar_psi` | DBTL with scalar ψ (single shared session offset) |
| `without_mood_learning` | DBTL with ψ disabled (no-mood ablation) |
| `flat_model` | Flat HBM: independent posteriors per (human, context, subtask), no hierarchy |
| `cbtl_adapted` | CBTL: context-agnostic Beta-Bernoulli, pooled across humans |
| `exploit_only` | Pure exploitation, no exploration bonus |
| `no_learning` | Random assignment lower bound |

---

## Thesis Experiments (Chapter 5)

### §5.3a — SpiceEnv multi-recipe joint training (hero result)

**What it shows**: DBTL's hierarchy beats Flat by +0.186 and CBTL by +0.059
on neutral eval when the robot trains on 4 recipes with hidden per-recipe
preference conflicts. Flat cannot improve past ~0.47 without the upper
μ → θ levels.

**Env config**: `spices` (4-recipe pool: Ultra, Asian, Indian, Mediterranean)
**Seeds**: 10–19 (10 seeds)
**Log dirs**: `logs/runs/spices_c14_{ours,nomood,flat,cbtl}/`
**Approaches**: `ours`, `without_mood_learning`, `flat_model`, `cbtl_adapted`

```bash
.venv/bin/python experiments/run_single_experiment.py \
    env=spices approach=ours seed=10 \
    "hydra.run.dir=logs/runs/spices_c14_ours/0"
```

---

### §5.3b — SpiceEnv cross-recipe LOO (diverse pool, OOD cautionary tale)

**What it shows**: On a globally-diverse 4-recipe pool (~51% spice overlap),
DBTL wins on final satisfaction (+0.052 over Flat) but step-0 warm-start
is OOD-sensitive. An honest demonstration of where hierarchy over-commits.

**Env configs**: `spices_loo_ultra`, `spices_loo_asian`, `spices_loo_indian`
(leave-Mediterranean is the default `spices_cross_recipe`)
**Seeds**: 0–9 (10 seeds per split)
**Log dirs**: `logs/runs/spices_loo_{asian,indian,ultra}_{ours,nomood,flat,cbtl}/`
**Approaches**: `ours`, `without_mood_learning`, `flat_model`, `cbtl_adapted`

```bash
# Example: leave-asian-out, DBTL
.venv/bin/python experiments/run_single_experiment.py \
    env=spices_loo_asian approach=ours seed=0 \
    "hydra.run.dir=logs/runs/spices_loo_asian_ours/0"
```

---

### §5.3c — SpiceEnv Middle-Eastern tight-cluster LOO (cleanest LOO positive)

**What it shows**: On a tight 4-recipe ME cluster (~77% spice overlap),
DBTL's step-0 warm-start is +0.115 above CBTL and Flat across all 4 splits.
Flat does not improve from step-0 to final (0.337 → 0.337). CBTL catches
up by final but the warm-start advantage is clear.

**Env configs**: `spices_me_feast`, `spices_me_lebanese`, `spices_me_kebab`, `spices_me_doner`
**Seeds**: 0–9 (10 seeds per split × 4 splits × 3 methods = 120 runs)
**Log dir**: `logs/runs/spices_me_loo/`
**Approaches**: `ours`, `flat_model`, `cbtl_adapted`

```bash
# Example: leave-feast-out, DBTL
.venv/bin/python experiments/run_single_experiment.py \
    env=spices_me_feast approach=ours seed=0 \
    "hydra.run.dir=logs/runs/spices_me_loo/feast_ours_seed0"
```

---

### §5.2 — Overcooked cross-layout LOO (null result)

**What it shows**: At 10 seeds, DBTL does not beat CBTL on cross-layout
warm-start transfer in Overcooked. The three canonical layouts share
near-identical subtask spaces and differ primarily in feasibility rather
than preference variation — a weak test bed for the hierarchy.

**Runner**: `experiments/run_overcooked_layout_loo.py`
**Log dir**: `logs/runs/oc_layout_loo_10seed/`
**Seeds**: 0–9
**Approaches**: `ours`, `cbtl_adapted`

```bash
# Example: hold out AsymmetricAdvantages, train on CrampedRoom + CoordinationRing
.venv/bin/python experiments/run_overcooked_layout_loo.py \
    --approach ours --seed 0 \
    --phase1-steps 3000 --phase2-steps 4000 \
    --train-layouts "CrampedRoom,CoordinationRing" \
    --held-out AsymmetricAdvantages \
    --output-dir logs/runs/oc_layout_loo_10seed/AsymmetricAdvantages_ours_seed0
```

---

### §5.4 — Multi-human population transfer (mixed result)

**What it shows**: Across-human transfer via μ is positive on SpiceEnv
(+0.040 step-0 gap) but near-zero on Overcooked (−0.010). CBTL's pooled
posterior gives a stronger warm-start on Overcooked (+0.093). We diagnose
phi/ψ coordinate-ascent attribution as the likely cause.

**Runner**: `experiments/run_transfer_experiment.py`
**Log dirs**: `logs/runs/spices_transfer_pooled/`, `logs/runs/oc_transfer_5h_pooled/`
**Seeds**: 0–9
**Approaches**: `ours`, `without_mood_learning`, `flat_model`, `cbtl_adapted`

```bash
# SpiceEnv 3-human transfer
.venv/bin/python experiments/run_transfer_experiment.py \
    --env spices --approach ours --seed 0 \
    --output-dir logs/runs/spices_transfer_pooled/ours_seed0

# Overcooked 5-human transfer
.venv/bin/python experiments/run_transfer_experiment.py \
    --env overcooked --approach ours --seed 0 \
    --output-dir logs/runs/oc_transfer_5h_pooled/ours_seed0
```

---

### §5.5 — Session-level ψ ablation (reproducible null)

**What it shows**: Vector ψ does not beat scalar ψ (or ψ-free) on the
ExtremeAsymmetric session profile, even with train-neutral-dominant /
eval-mood-heavy separation. All three methods within ±0.009 of each other.

**Env config**: `overcooked_psi_ablation`
(train: `prob_neutral_session=0.8`, eval: `prob_neutral_session=0.5`,
zero-sum `ExtremeAsymmetric` weights)
**Seeds**: 0–9
**Log dir**: `logs/runs/psi_extreme/`
**Approaches**: `ours` (vector ψ), `scalar_psi`, `without_mood_learning`

```bash
.venv/bin/python experiments/run_single_experiment.py \
    env=overcooked_psi_ablation approach=ours seed=0 \
    "hydra.run.dir=logs/runs/psi_extreme/ours_seed0"
```

---

### §5.7 — Overcooked single-context saturation (precondition)

**What it shows**: On single-layout Overcooked, all methods saturate within
±0.003 of each other. Confirms that hierarchy does not help with only one
context to pool over.

**Env configs**: `overcooked` (CrampedRoom), `overcooked_strong` (StrongCook), `overcooked_tomato` (TomatoCook)
**Seeds**: 0–9
**Log dirs**: `logs/runs/oc_main_*`, `logs/runs/oc_strong_*`, `logs/runs/oc_tomato_*`
**Approaches**: `ours`, `without_mood_learning`, `flat_model`, `cbtl_adapted`, `exploit_only`, `no_learning`

```bash
.venv/bin/python experiments/run_single_experiment.py \
    env=overcooked approach=ours seed=0 \
    "hydra.run.dir=logs/runs/oc_main_ours/0"
```

---

## Key Parameters

| Parameter | Typical Value | Meaning |
|-----------|--------------|---------|
| `max_environment_steps` | 5000–6000 | Training budget per phase |
| `eval_frequency` | 250–300 | Steps between eval checkpoints |
| `max_eval_episode_length` | 40 (spices), 200 (overcooked) | Max steps per eval episode |
| `num_eval_trials` | 15–50 | Episodes per eval pass |
| `dt` | 0.01 | Time scaling for x-axis in plots |

## Key Metrics in eval_results.csv

| Column | Meaning |
|--------|---------|
| `neutral_eval_mean_user_satisfaction_per_step` | **Primary metric** — per-step satisfaction under forced-neutral sessions (isolates learned φ) |
| `natural_eval_mean_user_satisfaction_per_step` | Per-step satisfaction under sampled sessions (tests φ + ψ handling) |
| `neutral_eval_mean_prediction_accuracy` | Fraction of correct actor-assignment predictions (neutral eval) |
| `neutral_eval_phi_mae` | Mean absolute error of learned P(human) vs ground truth |

For transfer experiments (`phase2_train_results.csv`):

| Column | Meaning |
|--------|---------|
| `warm_satisfaction` | Per-step satisfaction from warm-start model (carried Phase 1 θ/μ) |
| `cold_satisfaction` | Per-step satisfaction from cold-start model (fresh prior) |
| `warm_prediction_accuracy` | Prediction accuracy from warm-start model |

---

## Visualization

```bash
# Thesis figures (all experiments)
.venv/bin/python scripts/plot_thesis_figures.py

# Claims bar charts
.venv/bin/python scripts/plot_claims.py

# SpiceEnv multirun analysis
.venv/bin/python scripts/visualize_spices_experiment_multirun.py

# Transfer experiment analysis
.venv/bin/python scripts/visualize_transfer_experiments.py
```

---

## Known Issues

1. **Vector vs scalar ψ is a reproducible null** — tested on two profiles
   (AsymmetricFatigue and ExtremeAsymmetric) with two train/eval session
   distributions. The signal-to-noise ratio for per-subtask ψ inference
   appears below the detection floor at n=10. Stage-4 ARD prior is the
   principled next step.

2. **Phi/ψ coordinate-ascent attribution** — on Overcooked multi-human
   transfer, DBTL-with-ψ underperforms DBTL-without-ψ. The joint VI
   optimization of phi and psi on limited per-human data can land in
   mixed-attribution local minima. Candidate fixes: phi warmup, σ_mood
   annealing, or Stage-4 ARD.

3. **Cross-layout LOO is a weak test bed** — the three canonical Overcooked
   layouts differ primarily in feasibility constraints rather than
   preference variation, giving the hierarchy little per-context signal
   to specialize on.