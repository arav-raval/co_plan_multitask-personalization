# Convergence Analysis: Why Learning is Slow

## Issues Identified

After analyzing the HBM learning mechanism, we've identified **5 key factors** causing slow convergence:

### 1. **Very Small Learning Rate** ⚠️
- **Current**: Base learning rate = `0.05`, maximum = `0.1`
- **Impact**: Updates are extremely conservative - only 5-10% of the signal is incorporated per observation
- **Location**: `spices_hbm.py` line 240-242
```python
base_lr = 0.05  
confidence_boost = (neutral_conf - neutral_threshold) / (1.0 - neutral_threshold) 
learning_rate = base_lr + (0.1 - base_lr) * confidence_boost  # Max 0.1
```

### 2. **Heavy EMA Smoothing** ⚠️
- **Current**: EMA alpha = `0.1` (only 10% of new information per update)
- **Impact**: Even when phi updates, the EMA smooths it heavily, making convergence very slow
- **Location**: `spices_hbm.py` line 77, 176
```python
self.ema_alpha = 0.1
smoothed_mean = self.ema_alpha * post_mean + (1 - self.ema_alpha) * current_ema
```

### 3. **Neutral Threshold Filtering** ⚠️
- **Current**: Updates only occur when `P(neutral) >= 0.5`
- **Impact**: Many observations are discarded, reducing learning opportunities
- **Location**: `spices_hbm.py` line 237
```python
if neutral_conf >= neutral_threshold:  # Only updates if confident in neutral
    # ... update phi
```

### 4. **Single Recipe Limitation** ⚠️
- **Current**: Theta updates pool across all recipes: `y = mean(phi_mean[r, s] for r in recipes)`
- **Impact**: With only 1 recipe, theta updates are based on a single phi value, making updates very conservative
- **Location**: `spices_hbm.py` line 201
```python
y = float(np.mean(phis))  # With 1 recipe, this is just phi_mean[recipe][spice]
```

### 5. **Large Prior Variance** ⚠️
- **Current**: Prior variance for theta = `sigma0^2 + sigma_h^2 = 1 + 1 = 2`
- **Impact**: Large prior variance makes Bayesian updates conservative (high uncertainty = small updates)
- **Location**: `spices_hbm.py` line 203
```python
prior_var = self.sigma0**2 + self.sigma_h**2  # = 2.0
```

## Mathematical Analysis

### Update Formula for Theta
```
post_mean = post_var * (prior_mean/prior_var + y/sigma_obs2)
post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
```

With:
- `prior_var = 2.0` (large)
- `sigma_obs2 = sigma_r^2 = 1.0`
- `y = phi_mean[recipe][spice]` (single recipe)

The update is:
```
post_var = 1.0 / (1.0/2.0 + 1.0/1.0) = 1.0 / 1.5 = 0.667
post_mean = 0.667 * (mu_mean/2.0 + phi_mean/1.0)
```

This means theta moves only **2/3 of the way** toward phi_mean, and since phi_mean itself is updated slowly (due to EMA and low learning rate), convergence is very slow.

## Recommendations

### Quick Fixes (Low Risk)
1. **Increase learning rate**: `base_lr = 0.2`, `max_lr = 0.5`
2. **Increase EMA alpha**: `ema_alpha = 0.3` to `0.5`
3. **Lower neutral threshold**: `neutral_threshold = 0.3` to `0.4`

### Medium Changes (Moderate Risk)
4. **Adaptive learning rate**: Scale learning rate with confidence and episode number
5. **Remove EMA for early episodes**: Use direct updates for first N episodes

### Advanced Changes (Higher Risk)
6. **Single-recipe optimization**: Special case for single recipe to update theta more directly
7. **Reduce prior variance**: Use smaller `sigma0` and `sigma_h` for faster convergence

## Expected Impact

With the quick fixes:
- **Learning rate 0.2-0.5**: 4-10x faster convergence
- **EMA alpha 0.3-0.5**: 3-5x faster convergence  
- **Lower threshold 0.3**: 2x more learning opportunities

**Combined**: Should see convergence in ~200-300 episodes instead of 1000+

## Diagnostic Function

The `diagnose_slow_convergence()` function will:
- Show error reduction over time
- Identify if convergence has plateaued
- Display specific issues for each spice
- Provide recommendations

Run your test and check the logs for the diagnostic output!

