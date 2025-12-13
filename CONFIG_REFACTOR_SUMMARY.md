# Configuration Refactoring Summary

## Overview
All hyperparameters and configuration values have been centralized into a single configuration file: `src/multitask_personalization/envs/spices/spices_config.py`

## Configuration Structure

The configuration is organized into four main dataclasses:

### 1. `HBMConfig` - Hierarchical Bayesian Model Parameters
- **Variances**: `mu0`, `sigma0`, `sigma_h`, `sigma_r`, `sigma_obs`
- **EMA Parameters**: `ema_alpha`, `ema_early_threshold`, `ema_early_alpha`, `ema_medium_threshold`, `ema_medium_alpha`
- **Single Recipe Updates**: `single_recipe_prior_var_multiplier`, `single_recipe_obs_var_multiplier`, `single_recipe_theta_prior_weight`
- **Learning Rates**: `base_learning_rate`, `max_learning_rate`
- **Preference Mismatch**: `preference_mismatch_threshold`, `preference_mismatch_penalty`

### 2. `MoodConfig` - Mood Inference Parameters
- **Prior**: `mood_prior` (tuple of 3 floats: [all_self, neutral, none_self])
- **Smoothing**: `mood_smoothing_alpha`, `mood_prior_weight`
- **Mood Bias**: `mood_bias_multiplier` (relative to base_satisfaction_bias)
- **Satisfaction Likelihood**: `satisfaction_sigma`, `satisfaction_loglik_min`, `satisfaction_loglik_max`
- **De-mooding**: `demood_confidence_threshold`, `demood_scale_factor`
- **Non-neutral Weights**: `non_neutral_pref_weight_match`, `non_neutral_pref_weight_mismatch`

### 3. `UpdateConfig` - Update Thresholds and Filtering
- **Thresholds**: `neutral_confidence_threshold`, `effective_threshold_min`
- **Confidence Weighting**: `confidence_weight_min`
- **Legacy**: `legacy_learning_threshold`

### 4. `SatisfactionConfig` - Satisfaction Computation
- **Base Bias**: `base_satisfaction_bias`
- **Beta Distribution**: `satisfaction_beta_kappa`

## Main Config Class: `SpicesConfig`

Combines all sub-configs and provides helper methods:
- `mood_prior_array`: Returns mood prior as tuple
- `mood_bias_strength`: Computes mood bias strength
- `get_mood_bias()`: Returns mood bias dictionary

## Usage

### Default Configuration
```python
from multitask_personalization.envs.spices.spices_config import DEFAULT_CONFIG

# Use default config
hbm = HierarchicalPreferenceModel(..., config=DEFAULT_CONFIG)
```

### Custom Configuration
```python
from multitask_personalization.envs.spices.spices_config import SpicesConfig, HBMConfig, MoodConfig

# Create custom config
custom_config = SpicesConfig(
    hbm=HBMConfig(ema_alpha=0.5, base_learning_rate=0.2),
    mood=MoodConfig(mood_prior=(0.05, 0.90, 0.05))
)

# Use custom config
hbm = HierarchicalPreferenceModel(..., config=custom_config)
```

## Files Modified

1. **`spices_config.py`** (NEW): Centralized configuration file
2. **`spices_hbm.py`**: Updated to use config for all hyperparameters
3. **`spices_csp.py`**: Updated to use config and pass it to HBM
4. **`spices_env.py`**: Updated to use config for satisfaction Beta kappa

## Benefits

1. **Single Source of Truth**: All hyperparameters in one place
2. **Easy Tuning**: Change values in one file to affect entire system
3. **Type Safety**: Dataclasses provide type hints and validation
4. **Documentation**: Each parameter has inline documentation
5. **Flexibility**: Can create custom configs for different experiments

## Migration Notes

- Old hardcoded values have been replaced with config references
- `MOOD_BIAS` constant removed (now generated from config)
- All magic numbers moved to config file
- Backward compatible: defaults match previous values

