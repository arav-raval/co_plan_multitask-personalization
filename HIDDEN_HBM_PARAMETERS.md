# Where We Fix True Values and Set Deviation Parameters

## 1. Where True Theta Values Are Fixed

### Location 1: Single Recipe Test (`test_spices_csp_single_recipe`)

**File**: `tests/envs/test_spices_csp.py`  
**Line**: **1570**

```python
# Create hidden HBM if requested
hidden_hbm = None
if use_hidden_hbm:
    # Create a fixed hidden HBM with some default theta parameters
    theta_params = {spice: 1.5 if i % 2 == 0 else -1.5 for i, spice in enumerate(spices)}
    hidden_hbm = _create_hidden_hbm(spices, [recipe_name], theta_params=theta_params)
```

**What happens here:**
- Creates a dictionary mapping each spice to a theta value
- Pattern: Alternates between `1.5` and `-1.5` for spices
- Example: `{"salt": 1.5, "pepper": -1.5, "garlic": 1.5, ...}`

### Location 2: Multiple Humans Test (`_create_multiple_humans`)

**File**: `tests/envs/test_spices_csp.py`  
**Lines**: **1170-1191**

```python
for human_idx in range(num_humans):
    # Create different theta parameters for each human
    theta_params = {}
    for spice in spices:
        if human_idx == 0:
            # Human 0: prefers human for first half of spices alphabetically, robot for rest
            theta_params[spice] = 2.0 if spices.index(spice) < len(spices) // 2 else -2.0
        elif human_idx == 1:
            # Human 1: opposite pattern
            theta_params[spice] = -2.0 if spices.index(spice) < len(spices) // 2 else 2.0
        else:
            # Human 2+: random but consistent preferences
            rng_human = np.random.default_rng(seed + human_idx * 1000)
            theta_params[spice] = rng_human.normal(0.0, 2.0)
    
    hidden_hbm = _create_hidden_hbm(spices, recipes, theta_params=theta_params)
    humans.append(hidden_hbm)
```

**What happens here:**
- Human 0: First half of spices get `theta = 2.0`, second half get `theta = -2.0`
- Human 1: Opposite pattern
- Human 2+: Random normal distribution with mean=0, std=2.0

### Location 3: Where Theta Values Are Actually Set in the HBM

**File**: `tests/envs/test_spices_csp.py`  
**Function**: `_create_hidden_hbm`  
**Lines**: **69-75**

```python
# Set fixed theta parameters if provided
if theta_params is not None:
    for spice, theta_mean in theta_params.items():
        if spice in hbm.theta_mean:
            hbm.theta_mean[spice] = theta_mean  # ← THIS IS WHERE IT'S FIXED
            # Update variance accordingly
            hbm.theta_var[spice] = sigma_h**2 + sigma_r**2
```

**This is the critical line**: `hbm.theta_mean[spice] = theta_mean`

This directly sets the theta_mean value in the hidden HBM, making it fixed for all future sampling.

## 2. How Deviation Values Are Decided

### Current Implementation: Hardcoded Defaults

**File**: `tests/envs/test_spices_csp.py`  
**Function**: `_create_hidden_hbm`  
**Lines**: **38-40**

```python
def _create_hidden_hbm(spices: list[str], recipes: list[str], theta_params: dict[str, float] | None = None, 
                       mu0: float = 0.0, 
                       sigma0: float = 1.0,      # ← Global variance (Level 1)
                       sigma_h: float = 1.0,     # ← Human-level variance (Level 2)
                       sigma_r: float = 1.0,     # ← Recipe-level variance (Level 3)
                       base_satisfaction_bias: float = 3.0) -> HierarchicalPreferenceModel:
```

**Current default values:**
- `sigma0 = 1.0`: Variance at global level (Level 1: μ)
- `sigma_h = 1.0`: Variance at human level (Level 2: θ)
- `sigma_r = 1.0`: Variance at recipe level (Level 3: φ) ← **This controls sampling variance**
- `sigma_obs = 1.0`: Observation noise (hardcoded in HierarchicalPreferenceModel)

### Where These Are Used

1. **For sampling preferences** (`spices_env.py`, line 508):
   ```python
   # Sample phi from N(theta_mean, sigma_r^2)
   phi = self._rng.normal(theta_mean, hidden_hbm.sigma_r)
   ```
   - `sigma_r` controls how much variation there is between recipes
   - Larger `sigma_r` → more variation in preferences across recipes
   - Smaller `sigma_r` → more consistent preferences across recipes

2. **For initializing variances** (`spices_hbm.py`, lines 56-71):
   ```python
   self.mu_var: Dict[str, float] = {s: sigma0**2 for s in self.spices}
   self.theta_var: Dict[str, float] = {s: sigma0**2 + sigma_h**2 for s in self.spices}
   self.phi_var: Dict[str, Dict[str, float]] = defaultdict(
       lambda: {s: sigma_r**2 for s in self.spices}
   )
   ```

### How These Values Are Currently NOT Customizable

**Problem**: When `_create_hidden_hbm` is called (lines 1571, 1191), the sigma values are **not passed**, so they default to 1.0:

```python
# Current calls - sigma values use defaults (all 1.0)
hidden_hbm = _create_hidden_hbm(spices, [recipe_name], theta_params=theta_params)
# sigma0=1.0, sigma_h=1.0, sigma_r=1.0 (hardcoded defaults)
```

### What These Values Mean

| Parameter | Level | Meaning | Effect |
|-----------|-------|---------|--------|
| `sigma0` | Global (μ) | Variance in preferences across all humans | Not used for single-human experiments |
| `sigma_h` | Human (θ) | Variance in human-level preferences | Controls uncertainty in theta |
| `sigma_r` | Recipe (φ) | Variance in recipe-specific preferences | **Most important**: Controls how much preferences vary per recipe |
| `sigma_obs` | Observation | Noise in satisfaction feedback | Controls learning rate/sensitivity |

### Recommended Values

Based on the satisfaction bias (`base_satisfaction_bias = 3.0`):

- **`sigma_r = 1.0`**: Good default - allows moderate variation between recipes
  - With `theta = 2.0`, `phi` will typically be in range [0.7, 3.3] (±2σ)
  - This gives good separation between human/robot preferences

- **`sigma_h = 1.0`**: Reasonable - moderate uncertainty at human level

- **`sigma_r = 0.5`**: Tighter consistency - preferences very similar across recipes
- **`sigma_r = 2.0`**: More variation - preferences can vary significantly

### Making Deviation Values Configurable

To make these configurable, you could:

**Option 1: Add to PARAMETERS dict**
```python
PARAMETERS = {
    ...
    "sigma_r": 1.0,  # Recipe-level variance
    "sigma_h": 1.0,  # Human-level variance
    ...
}
```

**Option 2: Pass as arguments when creating hidden HBM**
```python
hidden_hbm = _create_hidden_hbm(
    spices, 
    [recipe_name], 
    theta_params=theta_params,
    sigma_r=PARAMETERS.get("sigma_r", 1.0),
    sigma_h=PARAMETERS.get("sigma_h", 1.0),
)
```

## Summary

1. **True theta values are fixed**:
   - Single recipe: Line 1570 - alternating pattern (1.5, -1.5)
   - Multiple humans: Lines 1170-1191 - different patterns per human
   - Actually set in: Line 73 (`hbm.theta_mean[spice] = theta_mean`)

2. **Deviation values are currently hardcoded**:
   - All default to `1.0` (lines 39-40)
   - Not configurable from test functions
   - `sigma_r` is most important - controls recipe-level variation

3. **To customize**:
   - Modify theta values at lines 1570 or 1170-1191
   - Modify sigma defaults at line 39-40, or pass them explicitly when calling `_create_hidden_hbm`

