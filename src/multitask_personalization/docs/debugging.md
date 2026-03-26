# Debugging Guide

Common failure modes, how to diagnose them, and how to fix them.
Read this before adding any new heuristics or workarounds.

---

## phi not converging / staying near zero

**Symptom:** After 20 episodes, `get_phi` is still close to 0.0 even when
training data clearly shows a preference.

**Diagnosis:**
```python
# Check if observations are accumulating
print(hbm._obs_count[h][recipe][spice])  # should grow each episode

# Check if ELBO is changing at all
print(hbm._elbo_history[-10:])           # should not be constant

# Check if m_phi tensor has gradient
m = hbm._phi_m[h][recipe][spice]
print(m.requires_grad)                   # must be True
print(m.grad)                            # should be non-None after a backward pass
```

**Causes and fixes:**
1. `requires_grad=False` on m_phi — check initialization in `register_recipe`
2. Learning rate too small — try `LR_PHI = 0.05` (default 0.03)
3. `n_phi_steps` too small — try 20 (default 10)
4. log_v_phi clamped at lower bound — print `_phi_logv[h][r][s].item()`
   if it's at LOG_VAR_MIN (-13.8), the variance collapsed and gradients are tiny

---

## phi variance not shrinking

**Symptom:** `get_phi_var` stays near initial value even after many observations.

**Diagnosis:**
```python
logv = hbm._phi_logv[h][recipe][spice].item()
print(f"log_v_phi = {logv:.3f}, v_phi = {exp(logv):.4f}")
```

**Causes and fixes:**
1. KL term dominating — the prior is pulling variance back up as fast as
   the likelihood is pushing it down. Check: is there actually enough data?
   For a 1D Gaussian prior, you need ~5 observations to see 50% variance reduction.
2. sigma_r is very large (> 3.0) — the prior variance is huge and dominates.
   Check `hbm.get_learned_sigmas()["sigma_r"]`.
3. LR_PHI too large causing log_v_phi to oscillate — reduce to 1e-2.

---

## ELBO is NaN or -inf

**Symptom:** `hbm._elbo_history` contains NaN or -inf.

**Diagnosis:** Run a single observation and print intermediate values:
```python
# In _elbo_phi, add print statements:
print(f"m_phi={m_phi.item():.3f}, log_v_phi={log_v_phi.item():.3f}")
print(f"phi_samples={phi_samples}")
print(f"log_sigma_obs={log_sigma_obs.item():.3f}")
```

**Causes and fixes:**
1. `log_v_phi` not clamped — exploded to very negative, causing `exp(0.5*log_v)=0`,
   then `log(0)=-inf` in the KL. Check that the clamp is active.
2. `log_sigma_obs` too negative (sigma_obs → 0) — the Gaussian term explodes.
   Check: `exp(hbm.log_sigma_obs.item())` should be >= 0.05.
3. Satisfaction value outside [-1, +1] — clip inputs before calling `observe`.
4. `phi_samples` values very large (|phi| > 50) — the sigmoid saturates to
   exactly 0 or 1, causing `log(0)`. Add gradient clipping or clip phi to [-10, 10].

---

## phi converges to wrong sign

**Symptom:** Training shows human adding spice with positive satisfaction, but
`preferred_actor` returns "robot".

**Diagnosis:**
```python
# Check sign conventions
actor = "human"
sign = 1.0 if actor == "human" else -1.0   # should be +1

# Check the full ELBO gradient direction
phi = hbm._phi_m[h][r][s]
loss = -_elbo_phi(observations, phi, ...)
loss.backward()
print(phi.grad)   # should be negative (gradient ascent pushes phi positive)
```

**Causes and fixes:**
1. Sign convention inverted somewhere — check `_log_likelihood_single`:
   `logit = sign * phi_samples` where sign = +1 for human.
2. Satisfaction values have wrong sign — check that sat > 0 means happy,
   sat < 0 means unhappy. If your environment inverts this, multiply by -1
   before calling `observe`.
3. Wrong actor string — check that actor is literally "human" or "robot",
   not "Human" or "HUMAN".

---

## Cold-start transfer not working (new recipe phi near zero)

**Symptom:** After learning Alice's preferences on SimpleDal, a new recipe
`get_phi("alice", "SweetCurry", "turmeric")` returns ≈ 0.0 instead of ≈ theta.

**Diagnosis:**
```python
theta = hbm.get_theta("alice", "turmeric")
phi_new = hbm.get_phi("alice", "SweetCurry", "turmeric")
print(f"theta={theta:.3f}, phi_new={phi_new:.3f}")  # should be close
```

**Cause:** `register_recipe` initializes phi from theta, but theta hasn't
been updated yet when the recipe was registered.

**Fix:** Always call `end_episode` (which calls `update_theta_and_mu`) before
registering a new recipe. If you register recipes upfront at init, call
`flush_theta_mu()` after the first few episodes to propagate learning to theta,
then re-register any recipes that were initialized from stale theta values.

The cleaner pattern: register recipes lazily (let `_ensure_registered` handle it)
rather than pre-registering. Lazy registration happens after observations have
accumulated and theta is meaningful.

---

## sigma_obs not adapting

**Symptom:** `get_learned_sigmas()["sigma_obs"]` stays near initial value.

**Diagnosis:**
```python
# Check that log_sigma_obs is in the optimizer
phi_opt = hbm._build_phi_optimizer(...)   # check params list
print([p.shape for p in phi_opt.param_groups[0]['params']])
```

**Causes and fixes:**
1. `log_sigma_obs` not in the phi optimizer params — check `_update_phi_elbo`
   optimizer initialization includes `self.log_sigma_obs`.
2. Learning rate for sigma_obs too small — it's in the phi optimizer at LR_PHI,
   which may be appropriate. If sigma_obs needs to adapt faster, give it its
   own optimizer at LR_HYPER.
3. Satisfaction values are all near ±1 (strongly polarized) — tanh(logit) ≈ ±1
   means residuals (sat - tanh(logit)) are always near zero, giving no gradient
   signal for sigma_obs. This is correct behavior, not a bug.

---

## Multiple humans: one human's phi affecting another's

**Symptom:** After Bob learns to prefer robot for turmeric, Alice's phi for
turmeric also shifts negative.

**Diagnosis:**
```python
# Check that mu is not being over-updated
print(hbm.mu_mean["turmeric"])  # should shift only slightly toward Bob's theta
print(hbm.get_theta("alice", "turmeric"))  # should not mirror Bob's theta
```

**Cause:** `update_theta_and_mu` runs for all humans together. If Bob has
many observations and Alice has few, Bob's theta precision dominates the mu
update, which then pulls Alice's theta toward Bob's via the prior.

**Fix:** This is correct Bayesian behavior — the population mean learns from
all humans. If it's too aggressive, increase `sigma_h` (allow more
human-to-human variation) or reduce `update_theta_mu_every_n_episodes`
to update mu less frequently. Don't try to isolate humans from each other —
the hierarchical pooling IS the mechanism for efficient cold-start for new humans.

---

## Tests failing intermittently (non-deterministic)

**Symptom:** Tests pass sometimes but fail others with slightly different values.

**Cause:** The ELBO uses Monte Carlo sampling (`torch.randn(N_MC_SAMPLES)`).
Tests that check exact values will fail intermittently.

**Fix:** Set a fixed seed before training in tests:
```python
torch.manual_seed(42)
np.random.seed(42)
```
Or redesign tests to check directional trends and final value ranges rather
than exact numbers. See `test_spices_hbm_stage1.py` for examples of
robust test patterns.

---

## Runtime too slow for real-time operation

**Symptom:** `observe()` takes > 50ms, making the robot sluggish.

**Profile first:**
```python
import cProfile
cProfile.run('hbm.observe(h, r, s, "human", 0.8)', sort='cumulative')
```

**Typical bottlenecks and fixes:**
1. `n_phi_steps` too large — reduce from 10 to 5. Test whether phi still
   converges in the same number of episodes.
2. `N_MC_SAMPLES` too large — reduce from 8 to 4. ELBO estimates get noisier
   but training still works.
3. Adam overhead — consider switching the phi update to SGD (no momentum
   accumulation) for faster per-step updates.
4. Tensor creation overhead — if `torch.randn(N_MC_SAMPLES)` is called very
   frequently, pre-allocate and reuse.
5. As a last resort, switch the phi update to the Laplace approximation
   (3 Newton steps, no autograd). See `design_decisions.md` for the tradeoffs.
   This should only happen after profiling confirms it's necessary.
