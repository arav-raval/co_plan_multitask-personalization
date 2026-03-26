# Interfaces

This document specifies the contracts between the three subsystems:
HBM (preference inference), SSM (transient state), and CSP (planning).

The interfaces are intentionally narrow. Each subsystem can be rewritten
independently as long as these contracts are preserved.

---

## HBM → CSP interface

The CSP queries the HBM before each planning step. The HBM returns
plain Python floats — never PyTorch tensors.

### Current (Stages 1-5)

```python
# Point estimate of preference (used for soft constraint value)
hbm.get_phi(human_id: str, recipe_name: str, spice: str) -> float
# Sign: + means human prefers to do it, - means prefers robot
# Scale: logit space. phi=+2 → P(human)≈0.88, phi=-2 → P(human)≈0.12

# Posterior variance (used for entropy-based active exploration)
hbm.get_phi_var(human_id: str, recipe_name: str, spice: str) -> float
# Added in Stage 1. High var → explore this dimension. Low var → exploit.
# Stage 3 will start using this in the CSP entropy criterion.

# Log probability of actor assignment (direct CSP soft constraint input)
hbm.log_prob_prefer(human_id: str, recipe_name: str, spice: str, actor: str) -> float
# Returns log P(actor | phi_mean). Always <= 0. Used as CSP cost term.

# Point prediction for actor assignment
hbm.preferred_actor(human_id: str, recipe_name: str, spice: str) -> str
# Returns "human" if phi >= 0, else "robot".
```

### Stage 6+ (SSM active): same interface, different internals

Once the SSM is active, `get_phi` returns `m_phi + C_d · z_t` (preference
plus current session state contribution). The CSP doesn't need to know
this — the interface is identical. The HBM internally combines stable
preferences and transient state before exposing them.

```python
# The CSP always calls these same methods.
# After Stage 6, the HBM internally computes:
mean = m_phi + C_d @ z_mean       # SSM contribution added here
var  = v_phi + C_d @ P @ C_d.T    # SSM uncertainty added here
```

### What the CSP must NOT do

- Never access `_phi_m`, `_phi_logv`, `log_sigma_h`, or any other internal
  HBM state. Only use the public getter methods.
- Never pass PyTorch tensors to the HBM. All inputs are plain Python types.
- Never call `update_theta_and_mu` or `_update_phi_elbo` directly. These
  are triggered by `observe()` and `end_episode()`.

---

## CSP → HBM interface (data flow back)

The CSP reports observations to the HBM after each step:

```python
hbm.observe(
    human_id: str,
    recipe_name: str,
    spice: str,
    actor: str,          # "human" or "robot"
    satisfaction: float, # in [-1, +1]
    force_neutral_mood: bool = False,  # kept for compat; no longer gates updates
) -> None
```

At episode end:
```python
hbm.end_episode(
    human_id: str,
    neutral_threshold: float = 0.5,  # kept for compat; no longer gates updates
) -> None
```

---

## HBM internal subsystem contracts

### phi update contract

`_update_phi_elbo` is called once per observation from `observe()`.
It takes:
- The full list of observations for this `(human, recipe, spice)` accumulated
  in `_episode_data` so far this episode.
- The current `m_theta` (detached — treated as fixed prior mean).
- The shared hyperparameters `log_sigma_r`, `log_sigma_obs`.

It mutates:
- `_phi_m[human_id][recipe_name][spice]`
- `_phi_logv[human_id][recipe_name][spice]`
- `log_sigma_obs` (jointly optimized with phi)

It does NOT mutate:
- theta, mu, sigma_h (those are episode-level updates)

### theta update contract

`update_theta_and_mu` is called from `end_episode()`.
It uses the current phi posteriors as "noisy observations" of theta.
It mutates theta, log_v_theta, sigma_h, sigma_r, mu_mean.
It does NOT mutate phi.

### Separation of timescales

```
Per-step:    phi, log_sigma_obs
Per-episode: theta, log_v_theta, mu, log_sigma_h, log_sigma_r
```

This separation is not arbitrary — it matches the model structure:
- phi is context-specific, updates from individual observations
- theta is human-specific, needs to aggregate across contexts
- sigma_h describes population-level variation, needs multiple humans

---

## Dimension Registry (Stage 4+)

When moving to multiple preference dimensions, the Dimension Registry
is the shared schema that all subsystems read from. It is defined once
(either hand-crafted or LLM-proposed) and never changed at runtime.

```python
@dataclass
class PreferenceDimension:
    dim_id: int
    name: str               # e.g. "chop_ownership"
    csp_variable: str       # e.g. "subtask_assignment"
    observation_proxy: str  # e.g. "corrections"
    prior_mean: float       # usually 0.0
    prior_range: float      # expected |phi| range, used to set sigma_0

DIMENSION_REGISTRY: List[PreferenceDimension] = [...]
```

The HBM uses `dim_id` as the index for phi. The CSP uses `csp_variable`
to know which CSP variable to apply the phi constraint to. The `observation_proxy`
field tells the likelihood function which observation channel primarily informs
this dimension.

---

## What stays the same across all stages

These are the invariants that must be preserved through every migration:

1. `log_prob_prefer` always returns a float <= 0
2. `preferred_actor` always returns "human" or "robot"
3. `get_phi` always returns a float (the posterior mean in logit space)
4. `observe` and `end_episode` have the same signature
5. `register_human` is idempotent (safe to call multiple times)
6. `register_recipe` initializes phi from theta (not from zero, not from mu)
   — this is the cold-start transfer mechanism and must never be broken

Invariant 6 is the most commonly broken. When refactoring registration code,
always verify that `get_phi` for a brand new recipe returns approximately
`get_theta` for that human, not 0.0.
