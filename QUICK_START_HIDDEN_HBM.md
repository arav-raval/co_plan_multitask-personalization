# Quick Start: Using Hidden HBM Configurations

## Basic Usage

### Step 1: Set Parameters in `test_spices_csp.py`

Open `tests/envs/test_spices_csp.py` and find the `PARAMETERS` dictionary (around line 20):

```python
PARAMETERS = {
    "num_episodes": 100,
    "num_epochs": 5, 
    "profile": "ChefAExpanded",
    "recipe_name": "UltraComplexFeast",
    "env_seed": 123,
    "csp_seed": 369,
    "logging_level": logging.INFO,
    "train_frac": 0.7,
    "verbose": True,
    "use_hbm": True,
    "num_seeds": 5,
    "use_hidden_hbm": True,  # ← Enable hidden HBM
    "num_humans": 1,
    "hidden_hbm_config_name": "AlternatingHuman",  # ← Choose your config here
}
```

### Step 2: Choose a Configuration

Available configurations (see `hidden_hbm_configs.py` for details):

- **`"AlternatingHuman"`** - Simple pattern: 1.5, -1.5, 1.5, -1.5...
- **`"SpiceSpecificHuman"`** - Specific preferences for common spices
- **`"HumanPrefersFirstHalf"`** - First half prefers human, second half robot
- **`"HumanPrefersSecondHalf"`** - Opposite pattern
- **`"ConsistentHuman"`** - Low variance (more consistent)
- **`"VariableHuman"` - High variance (more variation)
- **`"StrongPreferencesHuman"`** - Very strong preferences
- **`"WeakPreferencesHuman"`** - Subtle preferences

### Step 3: Run the Test

```bash
pytest tests/envs/test_spices_csp.py::test_spices_csp_single_recipe -v
```

## Examples

### Example 1: Single Human with Alternating Preferences

```python
PARAMETERS = {
    "num_episodes": 100,
    "recipe_name": "UltraComplexFeast",
    "use_hidden_hbm": True,
    "hidden_hbm_config_name": "AlternatingHuman",  # Uses alternating pattern
    "num_seeds": 5,  # Run 5 seeds for error bars
}
```

### Example 2: Single Human with Specific Spice Preferences

```python
PARAMETERS = {
    "num_episodes": 100,
    "recipe_name": "UltraComplexFeast",
    "use_hidden_hbm": True,
    "hidden_hbm_config_name": "SpiceSpecificHuman",  # Has specific preferences for salt, pepper, etc.
    "num_seeds": 5,
}
```

### Example 3: Multiple Humans (Comparison)

```python
PARAMETERS = {
    "num_episodes": 100,
    "recipe_name": "UltraComplexFeast",
    "use_hidden_hbm": True,
    "num_humans": 3,
    "hidden_hbm_config_names": [  # List of configs for each human
        "HumanPrefersFirstHalf",
        "HumanPrefersSecondHalf",
        "SpiceSpecificHuman"
    ],
    "num_seeds": 5,
}
```

Then run:
```bash
pytest tests/envs/test_spices_csp.py::test_spices_csp_multiple_humans -v
```

## Creating Your Own Configuration

### Step 1: Edit `hidden_hbm_configs.py`

Add a new configuration:

```python
MY_HUMAN = HiddenHBMConfig(
    name="MyHuman",
    theta_mean={
        "salt": 2.0,      # Strong human preference for salt
        "pepper": -1.5,   # Moderate robot preference for pepper
        "garlic": 1.0,    # Moderate human preference for garlic
        "cumin": -0.5,    # Slight robot preference for cumin
        # Add more spices as needed
    },
    sigma_r=1.0,   # Recipe-level variance
    sigma_h=1.0,   # Human-level variance
)
```

### Step 2: Add to Lookup Dictionary

```python
ALL_HIDDEN_HBM_CONFIGS = {
    ...
    "MyHuman": MY_HUMAN,  # Add this line
}
```

### Step 3: Use It

```python
PARAMETERS = {
    ...
    "hidden_hbm_config_name": "MyHuman",
    ...
}
```

## Understanding Theta Values

- **Positive theta** (e.g., `2.0`): Human prefers to add the spice themselves
- **Negative theta** (e.g., `-1.5`): Human prefers robot to add the spice
- **Magnitude**: 
  - `|theta| < 0.5`: Very weak preference
  - `0.5 <= |theta| < 1.5`: Moderate preference
  - `1.5 <= |theta| < 2.5`: Strong preference
  - `|theta| >= 2.5`: Very strong preference

## Understanding Variance Parameters

- **`sigma_r`**: Controls how much preferences vary between recipes
  - `0.5` = Very consistent (preferences similar across recipes)
  - `1.0` = Moderate variation (default)
  - `2.0` = High variation (preferences can differ significantly)

- **`sigma_h`**: Controls uncertainty in human-level preferences
  - Typically `0.5` to `1.5`

## What You'll See

When you run the test:

1. **Satisfaction plot**: Shows learning over episodes (with error bars if `num_seeds > 1`)
2. **HBM evolution plots**: Shows how learned theta converges to hidden theta
   - μ (global preferences)
   - θ (human-level preferences) ← **This should converge to your hidden config**
   - φ (recipe-specific preferences)
3. **Convergence**: The learned `theta_mean` values should approach the hidden HBM's `theta_mean` values over time

## Quick Reference

| Parameter | What It Does | Example |
|-----------|--------------|---------|
| `use_hidden_hbm` | Enable/disable hidden HBM | `True` |
| `hidden_hbm_config_name` | Single human config name | `"SpiceSpecificHuman"` |
| `hidden_hbm_config_names` | List for multiple humans | `["Human1", "Human2"]` |
| `num_seeds` | Number of seeds for error bars | `5` |
| `num_humans` | Number of humans to test | `3` |

## Troubleshooting

**Q: Config not found?**
- Check available configs: `python -c "from tests.envs.hidden_hbm_configs import list_hidden_hbm_configs; print(list_hidden_hbm_configs())"`
- Make sure the name matches exactly (case-sensitive)

**Q: Spice not in config?**
- Configs with dynamic patterns (like `AlternatingHuman`) work with any recipe
- For configs with explicit spices, missing spices default to `theta = 0.0`

**Q: Want to see what a config does?**
- Check `hidden_hbm_configs.py` - each config shows its theta_mean values
- Run the test and check the logs - it will show the true preferences

## Next Steps

1. **Start simple**: Use `"AlternatingHuman"` to see how it works
2. **Try different configs**: See how different preference patterns affect learning
3. **Create custom config**: Add your own human with specific preferences
4. **Compare humans**: Use multiple humans to show personalization importance

