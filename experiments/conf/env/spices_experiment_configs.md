# SpiceEnv Experiment Configurations

## Active Hydra configs

| Config | Recipe pool | Purpose |
|---|---|---|
| `spices.yaml` | 4-recipe joint (Ultra, Asian, Indian, Mediterranean) | Main multi-recipe training |
| `spices_cross_recipe.yaml` | Leave-Mediterranean-out LOO | Cross-recipe LOO split |

## Thesis experiments

### Multi-recipe joint training

**Config:** `spices.yaml`
**Log dirs:** `logs/runs/spices_c14_{ours,nomood,flat,cbtl}/`
**Seeds:** 10–19, **Approaches:** ours, without_mood_learning, flat_model, cbtl_adapted

```bash
.venv/bin/python experiments/run_single_experiment.py \
    env=spices approach=ours seed=10
```

### Cross-recipe LOO, diverse pool

**Configs:** `spices_loo_asian`, `spices_loo_indian`, `spices_loo_ultra`
**Builders:** `build_spice_scene_spec_loo_train_{ultra,asian,indian}` / `_eval_{...}` in `spices_experiment.py`
**Log dirs:** `logs/runs/spices_loo_{asian,indian,ultra}_{ours,nomood,flat,cbtl}/`
**Seeds:** 0–9, **Splits:** leave-asian, leave-indian, leave-ultra


### Cross-recipe LOO, tight pool

**Configs:** `spices_me_feast`, `spices_me_lebanese`, `spices_me_kebab`, `spices_me_doner`
**Builders:** `build_spice_scene_spec_me_train_{feast,lebanese,kebab,doner}` / `_eval_{...}` in `spices_experiment.py`
**Pool:** MiddleEasternFeast, LebaneseKafta, TurkishKebab, TurkishDoner (~77% spice overlap)
**Log dir:** `logs/runs/spices_me_loo/`
**Seeds:** 0–9, **Approaches:** ours, flat_model, cbtl_adapted

### Multi-human population transfe

**Runner:** `experiments/run_transfer_experiment.py`
**Log dir:** `logs/runs/spices_transfer_pooled/`
**Seeds:** 0–9, **Approaches:** ours, without_mood_learning, flat_model, cbtl_adapted

## Common parameters

| Parameter | Value |
|---|---|
| `max_environment_steps` | 5000–6000 |
| `eval_frequency` | 250–300 |
| `max_eval_episode_length` | 40–50 |
| `num_eval_trials` | 50 |
| `hidden_hbm_config_name` | SpiceSpecificHumanRecipeConflict |
