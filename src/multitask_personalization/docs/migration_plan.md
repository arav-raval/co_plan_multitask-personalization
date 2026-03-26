# Migration Plan

## Overview

The original `spices_hbm.py` has several correctness problems that are being
fixed in a staged migration. Each stage produces a working, testable system.
**Never skip a stage** — the stages are ordered so each one validates the
assumptions the next stage builds on.

---

## Stage 1 — Fix the likelihood and move to PyTorch ELBO

**Status: COMPLETE**
**Files:** `spices_hbm.py` (merged from `spices_hbm_stage1.py`), `tests/envs/test_spices_hbm_stage1.py`

### What was wrong

The original `_pseudo_obs_weighted` and `_update_phi` methods collapsed the
entire probabilistic observation model into a single transform:
```python
g = sign(actor) * satisfaction * base_satisfaction_bias
```
Then updated phi via a manual precision-weighted Normal-Normal update using
hardcoded sigma_obs.

This is wrong for four reasons:
1. It destroys the probabilistic interpretation — you cannot compute a valid
   ELBO from g because g is not a probability distribution.
2. The Normal-Normal update assumes a Gaussian likelihood. Actor choice is
   Bernoulli. Using a Gaussian likelihood misses the saturation behavior:
   when phi is already large, additional confirming observations should have
   diminishing impact. The Gaussian model keeps updating linearly.
3. sigma_h, sigma_r, sigma_obs were hardcoded. This makes the
   precision-weighted hierarchy numerically arbitrary — you're doing a
   weighted average with weights that don't mean anything.
4. EMA, annealing, and variance-adaptive learning rate in `_update_phi`
   were hand-tuned heuristics compensating for the incorrect likelihood.
   The ELBO naturally handles what these were trying to do.

### What Stage 1 does

Replaces the pseudo-observation + manual update with:
- Explicit two-term joint likelihood: Bernoulli (actor) + Gaussian (satisfaction)
- ELBO objective optimized by Adam with reparameterization trick
- All variational parameters (m_phi, log_v_phi, m_theta, log_v_theta) as
  PyTorch tensors with requires_grad=True
- Hyperparameters (log_sigma_obs, log_sigma_h, log_sigma_r) also learned
- Mood confidence gate **removed** — phi updates every observation

### What Stage 1 does NOT change

- CSP interface: `log_prob_prefer`, `preferred_actor`, `get_phi`, `get_theta`
- Registration: `register_human`, `register_recipe`
- Mood inference: `_update_mood_posterior`, `_loglik_feedback_given_mood`
  (kept for monitoring; will be used structurally in Stage 2)
- Episode structure: `observe`, `end_episode` signatures unchanged
- `update_theta_and_mu` structure (phi->theta->mu propagation direction)

### Tests to pass before Stage 2

Run: `pytest tests/envs/test_spices_hbm_stage1.py -v`

All 7 test classes must pass:
- `TestPhiConvergence`: phi converges toward correct sign
- `TestVarianceShrinkage`: phi_var decreases with data
- `TestELBOImprovement`: ELBO trends upward during training
- `TestHyperparamAdaptation`: sigma_obs, sigma_h move from initial values
- `TestCSPInterface`: log_prob_prefer and preferred_actor correct
- `TestColdStartTransfer`: new recipe phi initialized from theta, not zero
- `TestMultiHuman`: different humans learn different phi; mu reflects population

### Known issues to watch

- Stochastic ELBO (N_MC_SAMPLES=8) causes noisy phi trajectories. Expected.
  Tests check final values and trends, not strict monotonicity.
- If ELBO oscillates wildly: reduce LR_PHI from 3e-2 to 1e-2.
- If phi doesn't move: increase n_phi_steps from 10 to 20.
- First few observations: phi is initialized at theta_mean and phi_var is
  initialized from sigma_r. The first Adam steps may look unstable because
  the gradient is computed from very few data points. This stabilizes quickly.

---

## Stage 2 — Replace mood gate with scalar psi latent variable

**Status: COMPLETE**
**Files:** `spices_hbm.py`, `tests/envs/test_spices_hbm_stage2.py`

### The problem with the mood gate (original and Stage 1)

Stage 1 removes the mood gate but doesn't add any structured transient
effects model. This means mood-driven behavior still corrupts phi — it's
just no longer gated out, it's absorbed directly into phi.

Stage 2 adds the proper fix: a separate session-level latent variable psi
that absorbs transient effects.

### What to add

```python
# Per-episode session offset (scalar, same for all spice dims)
psi_{h,sess} ~ N(0, sigma_mood²)

# Modified likelihood:
logit = sign(actor) * (phi + psi)

# Modified ELBO:
ELBO_phi = E_q[log p(y|phi,psi)] - KL(q(phi)||p(phi|theta)) - KL(q(psi)||N(0,sigma_mood²))
```

Variational parameters to add:
- `m_psi: Dict[str, float]` — per-human session mean (reset each episode)
- `log_v_psi: Dict[str, float]` — per-human session log variance (reset each episode)

Episode end behavior:
```python
# Aggressive decay (nearly full reset)
m_psi[human_id] *= 0.05
log_v_psi[human_id] = log(sigma_mood²)   # reset variance fully
```

### WHY this works without a gate

Within an episode, moving psi costs less KL than moving phi:
- psi's prior is N(0, sigma_mood²) — fresh every episode
- phi's prior is anchored to previous episodes via theta

The optimizer preferentially routes transient signals through psi because
it's the "cheaper" explanation. Persistent signals can't stay in psi
(it resets) so they accumulate in phi over multiple episodes.

No observations are discarded. The math does the routing.

### Tests to add

- Bad mood episode (all robot assignments, negative satisfaction) should not
  significantly change phi after episode end
- Neutral episodes before and after a bad mood episode should show phi
  recovering / not corrupted
- psi_mean should be near zero between episodes

---

## Stage 3 — Pass uncertainty to CSP for active exploration

**Status: COMPLETE**
**Files:** `spices_hbm.py`, `spices_csp.py`, `tests/envs/test_spices_hbm_stage3.py`

### What to change in the CSP

The current `log_prob_prefer` returns a point estimate. Extend to also return
or expose the uncertainty:

```python
# Current (Stage 1):
def log_prob_prefer(human_id, recipe_name, spice, actor) -> float:
    # returns log P(actor | phi_mean)

# Stage 3 addition:
def preference_posterior(human_id, recipe_name, spice) -> tuple[float, float]:
    mean = hbm.get_phi(human_id, recipe_name, spice)
    var  = hbm.get_phi_var(human_id, recipe_name, spice)
    return mean, var
```

Exploration mode in CSP (entropy criterion from CBTL):
```python
entropy = H(Bernoulli(sigmoid(mean))) * var
# Choose action that maximizes entropy subject to hard constraints
```

### WHY this is stage 3 not stage 1

The entropy criterion only makes sense once variance is calibrated. In Stage 1
the variance is real (it reflects actual uncertainty). In Stage 0 it was
meaningless (hardcoded). So the CSP integration requires Stage 1 to be correct.

---

## Stage 4 — Vector psi + ARD prior for dimension pruning

**Status: Not started**
**Files:** Modify HBM

### What changes

- psi: scalar → vector, one entry per spice/dimension
- sigma_mood: scalar → per-dimension learnable parameter
- sigma_h: scalar → per-dimension (ARD prior)

ARD (Automatic Relevance Determination): place a HalfCauchy prior on sigma_h_d.
After training, dimensions where humans don't actually vary will have
sigma_h_d → 0. Inspect these to identify which spices have real preference
variation vs which are irrelevant.

This enables the LLM dimension proposal in Stage 7: propose 10-15 candidate
dimensions, ARD prunes to the 3-5 that actually explain variance.

---

## Stage 5 — Port to Overcooked

**Status: Not started**

### What changes at the model level

Almost nothing in the HBM math changes. The index changes:
- Spice env:    phi[(human, recipe_name, spice_name)]
- Overcooked:   phi[(human, recipe_type, subtask_type + dim)]

Context = recipe_type (not specific recipe_name) — this enables transfer
across recipes of the same type.

### New preference dimensions to define

```python
PREFERENCE_DIMS = {
    "ownership_chop":    0,   # prefers to chop themselves
    "ownership_plate":   1,   # prefers to plate themselves
    "ownership_deliver": 2,   # prefers to deliver themselves
    "pacing":            3,   # preferred time between handoffs
    "initiative":        4,   # how much robot should act unprompted
}
```

### New CSP variables and constraint functions

```python
# subtask_assignment ∈ {human, robot}
# → actorPreferred(assignment, phi_ownership_{subtask})

# handoff_timing ∈ ℝ+
# → timingPreferred(t, phi_pacing)

# robot_initiative ∈ {wait, suggest, act}
# → initiativeLevel(level, phi_initiative)
```

### New likelihood terms

Add a timing proxy (Gaussian on handoff intervals):
```python
log p(timing | phi_pacing) = log N(timing; preferred_interval(phi_pacing), sigma_t²)
```

Add correction signal (human override) as a separate strong-signal channel:
```python
log p(correction | phi) ≈ 2 * log p(actor | phi)   # weight corrections 2x
```

### Key test

Does theta transfer from recipe type A to recipe type B without cold start?
This is the hierarchy's core empirical claim.

---

## Stage 6 — Diagonal SSM (Kalman filter for transient state)

**Status: Not started**

### When to do this

Only if Overcooked sessions are long enough that within-episode drift is
real and measurable. Run Stage 5 first and check whether episodes longer
than ~20 steps show systematic mid-session preference changes.

### What changes

Replace the per-episode psi scalar with a continuous latent state z_t:

```python
# State equation
z_t = A * z_{t-1} + w_t,   w ~ N(0, Q)
A = diag(alpha_d)           # diagonal: one decay rate per dimension

# Observation equation (modified likelihood)
logit_{d,t} = phi_{h,c,d} + z_t[d]   # direct emission (C=I for now)
```

Implementation: `KalmanFilter` class with `predict()` and `update()` methods.
The update uses Extended Kalman Filter (EKF) because the likelihood is
non-Gaussian (Bernoulli+Gaussian requires linearizing the sigmoid/tanh).

Between sessions: `z *= alpha_session` (strong decay, not full reset).

### Vector psi is a special case of SSM

This is important to understand:
```
SSM(A=0, C=I) = vector psi    (set A to zero = no dynamics = iid per episode)
```
So Stage 4 (vector psi) and Stage 6 (SSM) are the same architecture — Stage 6
just adds the dynamics. The code can be structured as one class with A
controlling behavior:
- A=0: reduces to vector psi (Stage 4 behavior)
- A=diag(alpha): diagonal SSM (Stage 6 behavior)
- A=full matrix: fully structured SSM (Stage 7 optional)

### Key decision point

If diagonal SSM doesn't significantly outperform vector psi on Overcooked,
the complexity cost may not be worth it. Let the data decide. Metrics:
- Prediction accuracy in the second half of long episodes
- Number of episodes to convergence
- ELBO comparison

---

## Stage 7 — Structured SSM + LLM dimension proposal (optional)

**Status: Not started — thesis contribution frontier**

### Full SSM

- A: full matrix (cross-dimension dynamics, e.g. stress feeds into fatigue)
- C: learned emission matrix (z is lower-dimensional than phi — one latent
  cause affects multiple preference dimensions)
- Observable proxies → z (response latency, correction rate directly inform z)

Only pursue if you have 10+ humans, 20+ sessions each — otherwise A and C
are not identifiable from data.

### LLM dimension proposal

For new environments where you don't want to hand-craft dimensions:

```python
prompt = """
Given environment: {env_description}
CSP variables: {csp_variables}
Observations available: {observation_types}

Propose 6-8 preference dimensions. For each output:
  name | csp_variable | observation_proxy | prior_mean | prior_range
  (interpretation: + means human prefers to do it, - means prefers robot)
"""
```

LLM output populates the Dimension Registry. ARD pruning (Stage 4) then
identifies which proposed dimensions actually explain variance in the data.

**Integration constraint:** LLM runs offline (once per environment), not
at inference time. This preserves the closed-form ELBO and doesn't
introduce LLM latency into the planning loop.
