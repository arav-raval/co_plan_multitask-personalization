# Hidden HBM, Multiple Seeds, and Multiple Humans - Usage Guide

This document explains the new features added to support:
1. Hidden HBM with fixed parameters for generating preferences
2. Multiple seeds for error bars
3. Multiple humans (different hidden HBMs) to show importance of personalization

## Changes Made

### 1. Hidden HBM Support

**File: `src/multitask_personalization/envs/spices/spices_env.py`**

- Modified `SpiceHiddenSpec` to include an optional `hidden_hbm` field
- Added `__sample_preferences_from_hbm()` method that samples preferences from the hidden HBM's theta distribution for each recipe
- When a hidden HBM is provided, preferences are sampled from `phi ~ N(theta_s, sigma_r^2)` rather than being randomly assigned

### 2. Test Infrastructure

**File: `tests/envs/test_spices_csp.py`**

Added several helper functions:

- `_create_hidden_hbm()`: Creates a hidden HBM with fixed theta parameters for each spice
- `_create_multiple_humans()`: Creates multiple hidden HBMs representing different humans with varied preferences
- `_aggregate_metrics_across_seeds()`: Aggregates metrics across multiple seed runs, computing mean and std
- `_run_single_recipe_experiment()`: Core function that runs a single experiment and returns metrics
- `visualize_with_error_bars()`: Visualizes metrics with error bars across seeds

## Usage

### Running with Multiple Seeds

Set the `num_seeds` parameter in `PARAMETERS`:

```python
PARAMETERS = {
    ...
    "num_seeds": 5,  # Run with 5 different seeds
    ...
}
```

The test will run the experiment 5 times with different seeds and aggregate results with error bars.

### Running with Hidden HBM

Set the `use_hidden_hbm` parameter:

```python
PARAMETERS = {
    ...
    "use_hidden_hbm": True,  # Use hidden HBM instead of random preferences
    ...
}
```

This will create a hidden HBM with fixed theta parameters and sample preferences from it for each recipe.

### Running with Multiple Humans

Use the new test function:

```python
@pytest.mark.multiple_humans
def test_spices_csp_multiple_humans(...):
    ...
```

Set the `num_humans` parameter:

```python
PARAMETERS = {
    ...
    "num_humans": 3,  # Test with 3 different humans
    ...
}
```

Each human will have different fixed theta parameters, and the test will compare how well the learned HBM matches each human's true preferences.

### Combined Usage

You can combine all features:

```python
PARAMETERS = {
    "num_seeds": 5,          # 5 seeds per human
    "num_humans": 3,         # 3 different humans
    "use_hidden_hbm": True,  # Use hidden HBM
    ...
}
```

This will:
1. Create 3 different humans (hidden HBMs)
2. For each human, run 5 experiments with different seeds
3. Aggregate metrics across seeds with error bars
4. Compare learning performance across different humans

## Understanding the Results

### Hidden HBM Convergence

When using a hidden HBM:
- The hidden HBM has fixed `theta_mean` parameters (human-level preferences)
- For each recipe, preferences are sampled from `phi ~ N(theta, sigma_r^2)`
- The learned HBM should converge to match the hidden HBM's theta parameters over time
- Visualizations show how the learned `theta` converges to the true `theta`

### Multiple Seeds Error Bars

Error bars show the variability across different random seeds:
- Mean line: average across all seeds
- Error bars: standard deviation across seeds
- This shows the robustness of the learning algorithm

### Multiple Humans Comparison

Comparing across humans shows:
- How well the method personalizes to different preference profiles
- The importance of learning individual preferences vs. using global preferences
- Whether the learned preferences match the true preferences for each human

## Example: Creating Custom Hidden HBMs

You can create custom hidden HBMs with specific preferences:

```python
from tests.envs.test_spices_csp import _create_hidden_hbm

# Create a hidden HBM where some spices strongly prefer human, others robot
theta_params = {
    "salt": 2.0,      # Strongly prefer human
    "pepper": -2.0,   # Strongly prefer robot
    "garlic": 1.0,    # Moderately prefer human
    # ... etc
}

hidden_hbm = _create_hidden_hbm(
    spices=["salt", "pepper", "garlic", ...],
    recipes=["Recipe1", "Recipe2", ...],
    theta_params=theta_params,
    sigma_r=1.0  # Variance for recipe-specific preferences
)

# Use in environment
env = _make_env(seed=123, name="Recipe1", hidden_hbm=hidden_hbm)
```

## Next Steps

To extend these features:

1. **Update `test_spices_csp_multiple_recipes`**: Add support for hidden HBM and multiple seeds in the multi-recipe setting
2. **Enhanced visualizations**: Add more detailed error bars for HBM parameters (phi, theta, mu)
3. **Convergence metrics**: Add metrics to measure how well learned preferences match hidden preferences
4. **Different human profiles**: Create more sophisticated human preference patterns

