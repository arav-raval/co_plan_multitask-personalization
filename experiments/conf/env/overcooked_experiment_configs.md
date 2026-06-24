# Overcooked Experiment Configurations

## Active Hydra configs

| Config | Layout | Hidden human | Purpose |
|---|---|---|---|
| `overcooked.yaml` | CrampedRoom | RealisticCook | Main single-context baseline |
| `overcooked_psi_ablation.yaml` | CrampedRoom | RealisticCook + ExtremeAsymmetric session | ψ scalar vs vector ablation (train 80% neutral, eval 50/50) |

## Thesis experiments

### Single-context saturation (CrampedRoom)

**Config:** `overcooked.yaml`
**Log dirs:** `logs/runs/oc_main_{ours,nomood,flat,cbtl,exploit,nolearn,scalarpsi}/`
**Seeds:** 0–9, **Approaches:** ours, without_mood_learning, flat_model, cbtl_adapted, exploit_only, no_learning, scalar_psi

### Cross-layout LOO (null result)


**Runner:** `experiments/run_overcooked_layout_loo.py` (CLI args, no yaml)
**Log dir:** `logs/runs/oc_layout_loo_10seed/`
**Seeds:** 0–9, **Approaches:** ours, cbtl_adapted
**Held-out splits:** AsymmetricAdvantages, CoordinationRing

```bash
.venv/bin/python experiments/run_overcooked_layout_loo.py \
    --approach ours --seed 0 \
    --phase1-steps 3000 --phase2-steps 4000 \
    --train-layouts "CrampedRoom,CoordinationRing" \
    --held-out AsymmetricAdvantages \
    --output-dir logs/runs/oc_layout_loo_10seed/AsymmetricAdvantages_ours_seed0
```

---

### Multi-human population transfer

**Runner:** `experiments/run_transfer_experiment.py`
**Log dir:** `logs/runs/oc_transfer_5h_pooled/`
**Seeds:** 0–9, **Approaches:** ours, without_mood_learning, flat_model, cbtl_adapted

```bash
.venv/bin/python experiments/run_transfer_experiment.py \
    --env overcooked --approach ours --seed 0 \
    --output-dir logs/runs/oc_transfer_5h_pooled/ours_seed0
```

### Session-level ψ ablation (null result)

**Config:** `overcooked_psi_ablation.yaml`
**Log dir:** `logs/runs/psi_extreme/`
**Seeds:** 0–9, **Approaches:** ours (vector ψ), scalar_psi, without_mood_learning

## Common parameters

| Parameter | Value |
|---|---|
| `max_environment_steps` | 5000–6000 |
| `eval_frequency` | 250–300 |
| `max_eval_episode_length` | 200 |
| `num_eval_trials` | 15 |
| `hidden_hbm_config_name` | RealisticCook (default) |
