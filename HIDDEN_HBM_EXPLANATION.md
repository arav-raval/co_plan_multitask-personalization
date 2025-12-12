# Hidden HBM: In-Depth Explanation

## Overview

The hidden HBM is a fixed generative model that represents the "true" human preferences. It's used to **generate** preferences for each recipe, which the learning algorithm then tries to **infer** over time. The goal is to show that the learned HBM converges to match the hidden HBM.

## Hierarchical Structure

The HBM has three levels:

```
Level 1 (Global):     μ_s  ~ N(μ0, σ0²)           [Global preference for spice s]
                          ↓
Level 2 (Human):      θ_s  ~ N(μ_s, σ_h²)         [This human's preference for spice s]
                          ↓
Level 3 (Recipe):     φ_{r,s} ~ N(θ_s, σ_r²)      [Preference for spice s in recipe r]
```

## Fixed Parameters in Hidden HBM

When we create a hidden HBM, we fix parameters at **Level 2 (θ_s)**:

### What Gets Fixed

1. **`theta_mean[spice]`**: The human-level preference mean for each spice
   - This is the **true preference** we want the algorithm to learn
   - Example: `theta_mean["salt"] = 2.0` means this human strongly prefers to add salt themselves
   - Example: `theta_mean["pepper"] = -1.5` means this human moderately prefers robot to add pepper

2. **`theta_var[spice]`**: Variance at the human level (computed as `sigma_h² + sigma_r²`)
   - This represents uncertainty in the human-level preferences

### What Does NOT Get Fixed

1. **`mu_mean[spice]`**: Global preferences (Level 1)
   - Initially set to `mu0` (default 0.0)
   - These represent global preferences across all humans
   - For a single hidden HBM (single human), these don't matter much for generation

2. **`phi_mean[recipe][spice]`**: Recipe-specific preferences (Level 3)
   - These are **sampled** from the fixed theta, not fixed themselves
   - Each recipe gets its own sample of phi values

## How Preferences Are Generated

### Step-by-Step Process

1. **Fixed Parameters Setup** (once per hidden HBM):
   ```python
   # Create hidden HBM with fixed theta for each spice
   theta_params = {
       "salt": 2.0,      # Human prefers adding salt
       "pepper": -1.5,   # Human prefers robot adds pepper
       "garlic": 0.5,    # Slight human preference
       ...
   }
   hidden_hbm = _create_hidden_hbm(spices, recipes, theta_params=theta_params)
   ```

2. **For Each Recipe Episode** (called when `env.reset()` happens):
   
   When `SpiceEnv.reset()` is called, it calls `__sample_preferences_from_hbm()`:
   
   ```python
   # For each spice in the recipe:
   for spice in recipe.spices:
       # Get fixed theta_mean for this spice
       theta_mean = hidden_hbm.theta_mean[spice]  # e.g., 2.0 for "salt"
       
       # Sample phi from: phi ~ N(theta_mean, sigma_r²)
       phi = rng.normal(theta_mean, hidden_hbm.sigma_r)
       # Example: if theta_mean=2.0, sigma_r=1.0, might sample phi=2.3 or phi=1.7
       
       # Convert phi to actor preference using sigmoid
       p_human = 1.0 / (1.0 + exp(-phi))
       # If phi=2.0, p_human ≈ 0.88 (88% chance of preferring human)
       # If phi=-1.5, p_human ≈ 0.18 (18% chance of preferring human)
       
       # Sample the actual preference
       preferred_actor = "human" if rng.random() < p_human else "robot"
   ```

### Key Points:

- **Each recipe gets NEW sampled preferences** - even for the same spice in the same recipe
- The preferences are sampled from the **same fixed theta**, so they're consistent on average
- The randomness comes from:
  1. Sampling `phi ~ N(theta, sigma_r²)` (recipe-level variance)
  2. Converting `phi` to probability and sampling the actor

## Multiple Spices

Each spice has its **own fixed theta_mean**:

```python
theta_params = {
    "salt": 2.0,        # Fixed: human strongly prefers salt
    "pepper": -1.5,     # Fixed: human prefers robot for pepper
    "garlic": 0.5,      # Fixed: slight human preference for garlic
    "cumin": -0.3,      # Fixed: slight robot preference for cumin
    ...
}
```

For a recipe with spices `["salt", "pepper", "garlic"]`:
1. Sample `phi_salt ~ N(2.0, sigma_r²)` → likely prefer human
2. Sample `phi_pepper ~ N(-1.5, sigma_r²)` → likely prefer robot
3. Sample `phi_garlic ~ N(0.5, sigma_r²)` → slight preference for human

Each spice is **independent** - the preferences for salt don't affect preferences for pepper.

## What the Learning Algorithm Tries to Do

The learning algorithm (the learned HBM) starts with:
- All preferences at 0 (no prior knowledge)
- Through observations (satisfaction feedback), it tries to infer:
  - `phi` (recipe-specific) → learns quickly from individual episodes
  - `theta` (human-level) → learns by pooling across recipes
  - `mu` (global) → learns by pooling across all humans (if multiple)

**Goal**: The learned `theta_mean[spice]` should converge to match the hidden HBM's `theta_mean[spice]`

## Example: Convergence Over Time

### Hidden HBM (Fixed):
```python
theta_mean = {
    "salt": 2.0,
    "pepper": -1.5,
    "garlic": 0.5
}
```

### Learned HBM (Starts at 0, learns over time):

**Episode 1:**
- Learned theta: `{"salt": 0.0, "pepper": 0.0, "garlic": 0.0}`
- Sees recipe with salt → human preferred → satisfaction +1
- Updates phi for salt → slightly positive
- After episode: `theta = {"salt": 0.1, ...}` (still close to 0)

**Episode 10:**
- Learned theta: `{"salt": 1.2, "pepper": -0.8, "garlic": 0.3}`
- Getting closer to hidden values!

**Episode 100:**
- Learned theta: `{"salt": 1.9, "pepper": -1.4, "garlic": 0.5}`
- Very close to hidden theta! ✅

## Important Design Decisions

### Why Fix Theta, Not Phi?

We fix `theta` (human-level) because:
- `theta` represents the **person's underlying preference** - this is what we want to learn
- `phi` (recipe-specific) has natural variation - even the same person might prefer slightly differently in different recipes
- By fixing `theta` and sampling `phi`, we create realistic variation while maintaining consistency

### Why Not Fix Mu (Global)?

For a single human experiment:
- `mu` represents preferences across **all humans** (population level)
- For a single hidden HBM, `mu` doesn't affect generation (phi is sampled from theta, not mu)
- We could fix `mu` if we want, but it's not necessary for single-human experiments

For multiple humans:
- Each human has their own fixed `theta`
- The global `mu` would represent the average across all humans
- We typically don't fix `mu` because it represents a population statistic, not an individual's truth

## Current Implementation Details

### Where Parameters Are Fixed

**File**: `tests/envs/test_spices_csp.py`
- Function: `_create_hidden_hbm()` (line 38)
- Sets: `hbm.theta_mean[spice] = theta_params[spice]` (line 73)

**File**: `src/multitask_personalization/envs/spices/spices_env.py`
- Function: `__sample_preferences_from_hbm()` (line 492)
- Uses fixed `theta_mean` to sample preferences (line 508)

### Sampling Formula

```python
# Fixed theta from hidden HBM
theta_mean = hidden_hbm.theta_mean[spice]  # Fixed parameter

# Sample recipe-specific preference
phi = rng.normal(theta_mean, sigma_r)  # sigma_r = 1.0 by default

# Convert to probability and sample actor
p_human = sigmoid(phi) = 1 / (1 + exp(-phi))
preferred_actor = "human" if rng.random() < p_human else "robot"
```

### Visualizing Convergence

The plots show:
- **Hidden HBM theta**: The fixed true values (should be horizontal lines)
- **Learned HBM theta**: The values learned over time (should converge to hidden values)
- **Error bars**: Show variance across multiple seed runs

## Summary

1. **Fixed**: `theta_mean[spice]` for each spice (human-level preferences)
2. **Sampled**: `phi[recipe][spice]` for each recipe episode (recipe-specific preferences)
3. **Goal**: Learned HBM's `theta_mean` should converge to hidden HBM's `theta_mean`
4. **Each spice is independent** - has its own fixed theta parameter
5. **Each recipe gets new samples** - phi is resampled each episode, but centered on fixed theta

