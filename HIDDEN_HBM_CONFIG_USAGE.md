# Using Hidden HBM Configurations

## Overview

All hidden HBM parameters (theta means, variance parameters) are now configurable through `hidden_hbm_configs.py`, similar to how recipes are defined in `recipe.py`.

## Quick Start

### 1. Using a Predefined Configuration

In `test_spices_csp.py`, set the `hidden_hbm_config_name` parameter:

```python
PARAMETERS = {
    ...
    "use_hidden_hbm": True,
    "hidden_hbm_config_name": "SpiceSpecificHuman",  # Choose from available configs
    ...
}
```

### 2. Available Configurations

Run to see all available configs:
```python
from tests.envs.hidden_hbm_configs import list_hidden_hbm_configs
print(list_hidden_hbm_configs())
```

Current configs:
- `AlternatingHuman`: Alternating pattern (1.5, -1.5, 1.5, -1.5...)
- `HumanPrefersFirstHalf`: First half prefers human (2.0), second half prefers robot (-2.0)
- `HumanPrefersSecondHalf`: Opposite pattern
- `SpiceSpecificHuman`: Specific preferences for common spices
- `ConsistentHuman`: Similar to SpiceSpecificHuman but with smaller variance (sigma_r=0.5)
- `VariableHuman`: Similar to SpiceSpecificHuman but with larger variance (sigma_r=2.0)
- `StrongPreferencesHuman`: Very strong preferences (theta = ±3.0)
- `WeakPreferencesHuman`: Subtle preferences (theta = ±0.5)

### 3. Multiple Humans

For multiple humans, specify a list of config names:

```python
PARAMETERS = {
    ...
    "num_humans": 3,
    "hidden_hbm_config_names": [
        "HumanPrefersFirstHalf",
        "HumanPrefersSecondHalf", 
        "SpiceSpecificHuman"
    ],
    ...
}
```

## Creating Your Own Configuration

### Step 1: Add to `hidden_hbm_configs.py`

```python
MY_CUSTOM_HUMAN = HiddenHBMConfig(
    name="MyCustomHuman",
    theta_mean={
        "salt": 2.5,      # Strong human preference
        "pepper": -1.0,   # Moderate robot preference
        "garlic": 0.0,    # Neutral
        "cumin": -2.0,    # Strong robot preference
        # Add more spices as needed
    },
    sigma_r=1.2,          # Recipe-level variance
    sigma_h=0.8,          # Human-level variance
    sigma0=1.0,           # Global variance
    sigma_obs=1.0,        # Observation noise
    base_satisfaction_bias=3.0,
)
```

### Step 2: Add to Lookup Dictionary

```python
ALL_HIDDEN_HBM_CONFIGS = {
    ...
    "MyCustomHuman": MY_CUSTOM_HUMAN,
}
```

### Step 3: Use It

```python
PARAMETERS = {
    ...
    "hidden_hbm_config_name": "MyCustomHuman",
    ...
}
```

## Understanding Parameters

### Theta Mean (theta_mean)

- **Positive values**: Human prefers to add the spice themselves
- **Negative values**: Human prefers robot to add the spice
- **Magnitude**: Strength of preference
  - `|theta| < 0.5`: Very weak preference
  - `0.5 <= |theta| < 1.5`: Moderate preference
  - `1.5 <= |theta| < 2.5`: Strong preference
  - `|theta| >= 2.5`: Very strong preference

### Variance Parameters

- **sigma_r** (Recipe-level): Controls variation between recipes
  - `0.5`: Very consistent preferences across recipes
  - `1.0`: Moderate variation (default)
  - `2.0`: High variation between recipes

- **sigma_h** (Human-level): Controls uncertainty in human preferences
  - Typically `0.5` to `1.5`

- **sigma0** (Global): Population-level variance
  - Only matters for multi-human experiments

- **sigma_obs** (Observation): Noise in satisfaction feedback
  - Typically `1.0`

## Dynamic Patterns

Some configs use dynamic patterns that adapt to the spices in a recipe:

- `AlternatingHuman`: Automatically creates alternating pattern based on spice order
- `HumanPrefersFirstHalf`: Splits spices in half automatically
- `HumanPrefersSecondHalf`: Opposite split

These work with any recipe without modification.

## Examples

### Example 1: Single Human with Specific Preferences

```python
PARAMETERS = {
    "use_hidden_hbm": True,
    "hidden_hbm_config_name": "SpiceSpecificHuman",
    "num_seeds": 5,
}
```

### Example 2: Multiple Humans with Different Configs

```python
PARAMETERS = {
    "use_hidden_hbm": True,
    "num_humans": 3,
    "hidden_hbm_config_names": [
        "HumanPrefersFirstHalf",
        "HumanPrefersSecondHalf",
        "ConsistentHuman"
    ],
    "num_seeds": 5,
}
```

### Example 3: Custom Configuration

1. Edit `hidden_hbm_configs.py` to add your config
2. Use it:
```python
PARAMETERS = {
    "hidden_hbm_config_name": "MyCustomHuman",
}
```

## Benefits

1. **Easy to modify**: Just edit `hidden_hbm_configs.py` - no code changes needed
2. **Reusable**: Same config works with any recipe
3. **Clear documentation**: All parameters in one place
4. **Consistent**: Same pattern as `recipe.py` for familiarity
5. **Flexible**: Can specify exact values or use dynamic patterns

## File Structure

```
tests/envs/
├── recipe.py              # Recipe definitions (existing)
├── hidden_hbm_configs.py  # Hidden HBM configurations (new)
└── test_spices_csp.py     # Test code (updated to use configs)
```

## Tips

1. **Start with predefined configs** to understand the system
2. **Use `SpiceSpecificHuman`** as a template for custom configs
3. **Test with different sigma_r values** to see effect on learning
4. **Use multiple humans** to show importance of personalization
5. **Check convergence** - learned theta should match hidden theta

