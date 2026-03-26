# Design Decisions

This document records the reasoning behind every major design choice.
When you're tempted to change something, read the relevant entry here first.

---

## Why VI instead of MCMC

MCMC samples from the true posterior and is asymptotically exact. VI
approximates the posterior with a parametric family and is faster but biased.

We chose VI because:
1. **Latency.** The robot needs to update its model between steps (or at
   episode end). MCMC chains need thousands of steps to mix across the
   posterior. VI runs until ELBO convergence, which is much faster and can
   be warm-started from the previous posterior.
2. **Online updates.** Stochastic VI (running Adam on mini-batches of data)
   naturally supports streaming observations. MCMC does not.
3. **Differentiability.** The full model (HBM + SSM + hyperparameters) is
   a differentiable program. VI exploits this; MCMC does not require it.

The cost: mean-field VI systematically underestimates uncertainty (variance
too tight). This means the CSP may over-exploit. Monitor entropy curves —
if they drop faster than prediction accuracy improves, increase prior variance
or switch to a richer variational family.

---

## Why ELBO instead of Laplace approximation

The Laplace approximation finds the MAP of the posterior and fits a Gaussian
to the local curvature. It's faster than VI (3 Newton steps vs 10+ Adam steps)
but has two problems for this setting:

1. **Cannot learn hyperparameters jointly.** sigma_h, sigma_r, sigma_obs
   appear in the prior/likelihood but not in the Laplace update formula.
   Learning them requires an outer EM loop — which is essentially VI.
2. **Wrong approximation family.** Laplace fits a Gaussian centered at the
   MAP. When the true posterior is skewed (common with sparse data and
   logistic likelihoods), the Laplace approximation is systematically wrong
   in ways that are hard to detect.

ELBO jointly optimizes variational parameters AND hyperparameters in one
gradient step. It's the principled choice and the one used in standard
VI libraries (Pyro, NumPyro, PyTorch distributions).

The runtime cost is real (~10 Adam steps vs 3 Newton steps per observation).
If it becomes a bottleneck in real-time robotics, optimize then — but don't
pre-optimize at the cost of correctness.

---

## Why PyTorch instead of NumPy for inference

Three reasons:
1. **Autograd.** Computing ELBO gradients by hand for the Bernoulli+Gaussian
   joint likelihood with hyperparameters is error-prone and hard to maintain.
   PyTorch autograd handles it automatically and correctly.
2. **Future stages.** Stage 6 (SSM) adds a Kalman filter whose parameters
   (A, C, Q) are learned via gradient descent. Stage 7 adds a neural encoder.
   Both require autograd. Starting with PyTorch means no refactor later.
3. **Ecosystem.** Standard debugging tools (gradient checking, loss curves,
   `torch.autograd.gradcheck`) work out of the box.

PyTorch is used **only inside the inference methods**. The public interface
(everything the CSP calls) returns plain Python floats. This isolation means
the CSP never needs to know about PyTorch.

---

## Why mean-field VI (independent q) instead of structured VI

Mean-field assumes q(phi, theta) = q(phi) * q(theta) — no correlation between
the two levels. The true posterior has correlations (phi and theta co-vary).

We chose mean-field because:
1. Structured VI (e.g., full-covariance Gaussian over phi AND theta jointly)
   has O(d²) parameters where d is the number of latent variables. For many
   humans, many recipes, and many spices, this is prohibitive.
2. The correlation between phi and theta is partially captured by the
   hierarchical update schedule: phi updates online, theta updates at episode
   end using phi posteriors. This isn't exact structured VI but it propagates
   information in the right direction.

The main consequence: posterior variance is underestimated. If you see the
robot converging to wrong answers with high confidence, this is likely the cause.
Fix: increase the prior variances (sigma_h, sigma_r) to act as regularizers.

---

## Why Bernoulli + Gaussian joint likelihood instead of just one term

Several alternatives were considered:

**Just Bernoulli (actor choice):** Loses the information in satisfaction magnitude.
A correction with satisfaction -0.9 and one with satisfaction -0.1 would update
phi identically. Satisfaction magnitude is real signal.

**Just Gaussian (satisfaction as pseudo-observation):** This was the original
approach (`g = sign * sat * bias`). Problem: actor choice is binary, not
Gaussian. Treating ±1 as a Gaussian observation introduces an arbitrary noise
parameter sigma that has no principled value. Also misses the saturation
behavior (Gaussian keeps updating linearly; Bernoulli saturates at extremes).

**Bernoulli only for corrections, Gaussian for accepted assignments:** Tempting
but complicated. The likelihood would be conditional on whether the assignment
was accepted or corrected. This introduces selection bias — you only observe
corrections when the robot was wrong, so the correction distribution is not
representative of the overall preference distribution.

**Joint (chosen approach):** Both terms are always present for every observation.
The Bernoulli term captures the binary assignment direction. The Gaussian term
captures the magnitude of satisfaction. They're approximately conditionally
independent given phi (the main shared driver), so adding them is valid.

---

## Why satisfaction enters the Gaussian as tanh(logit) not as phi directly

The expected satisfaction is `tanh(sign(actor) * phi)`, not `phi` directly.

Three reasons:
1. **Boundedness.** Satisfaction is in [-1, +1]. `tanh` maps the real line
   to (-1, +1). Using phi directly would allow expected satisfaction > 1,
   which is nonsense.
2. **Monotonicity.** Higher |phi| → higher |expected_sat|. tanh preserves
   this: tanh(3) ≈ 0.995, tanh(0.5) ≈ 0.46. A strong preference should
   predict strong satisfaction.
3. **Consistency with Bernoulli term.** Both terms derive their signal from
   the same logit `sign * phi`. The Bernoulli term uses `sigmoid(logit)` to
   model the binary outcome; the Gaussian term uses `tanh(logit) = 2*sigmoid(2*logit) - 1`
   to model the continuous outcome. They're compatible transformations of the
   same underlying logit.

---

## Why psi resets at episode end (not gradual decay)

The episode boundary in the spice env is meaningful: each episode is a
distinct cooking session. Between sessions the human's mood genuinely resets —
they come back fresh the next day.

For the spice env: full reset (`m_psi *= 0.05`, `v_psi = sigma_mood²`).

For Overcooked and longer-horizon robotics: the episode boundary may be
artificial. Stage 6's SSM replaces the hard reset with a parameterized
decay rate A that the model learns. This is the right generalization for
settings where "episode" is a planning artifact, not a real-world reset.

---

## Why theta is not variational in Stage 1 (updated analytically)

Theta's update in Stage 1 runs N_THETA_STEPS Adam steps of ELBO_theta,
but mu is still updated analytically (precision-weighted Gaussian pooling).

**Why not make mu variational too?**
With 3 humans in the spice env, there's very little data to estimate the
population distribution. A variational mu would have a very uncertain posterior
that changes noisily with each human added. The analytic update is more stable
with small samples and produces the correct Bayesian update when the likelihood
is Gaussian (which it approximately is at the mu level — phi values
look roughly Gaussian distributed around their true means).

Mu becomes variational in Stage 4 when we introduce ARD priors on sigma_h
per dimension — that requires the full VI treatment because HalfCauchy is
not conjugate to Gaussian.

---

## Why cold-start uses theta (not mu) for new recipe initialization

When Alice starts recipe 5 (never seen before):
```python
# Wrong: initialize from population mean
phi_new = mu                     # cold — ignores everything learned about Alice

# Right: initialize from Alice's human-level estimate
phi_new = theta[Alice]           # warm — starts from Alice's personal history
phi_var_new = theta_var + sigma_r²  # extra uncertainty for new context
```

This is the primary benefit of the three-level hierarchy over a two-level model.
After 4 recipes, Alice's theta reflects her actual preferences. Any new recipe
inherits that — the robot doesn't need to re-learn from scratch.

Verify this invariant is preserved by checking:
`abs(hbm.get_phi(h, new_recipe, s) - hbm.get_theta(h, s)) < 0.3`
immediately after `register_recipe`.

---

## Why the mood gate was removed (not just fixed)

The original gate:
```python
if neutral_conf >= 0.5:
    update phi
else:
    discard observation
```

Problems:
1. **Information loss.** Every observation carries information about phi,
   even if mood is uncertain. Discarding it wastes data.
2. **Hard threshold.** The gate creates a discontinuity: an episode with
   confidence 0.49 discards all data; one with confidence 0.51 uses all of it.
   Small noise in mood inference causes large swings in phi learning.
3. **Wrong mechanism.** The mood gate was a heuristic approximation of
   the correct mechanism, which is: let the ELBO route signals to phi or psi
   based on which explanation is cheaper (lower KL cost). The gate approximated
   this by asking "is mood likely?" and making a binary decision. The ELBO
   does it continuously and correctly.

Stage 2 adds psi as the proper replacement. The gate is not needed once psi exists.

---

## Why separate phi update and theta update timescales

It would be simpler to run one big ELBO over all latent variables jointly.
We don't do this for three reasons:

1. **Computational cost.** A joint update would require back-propagating
   through the full hierarchy for every single observation. With many spices,
   many recipes, and many humans, this scales poorly.
2. **Stability.** phi can validly fluctuate episode-to-episode (different
   recipes genuinely differ). theta should not fluctuate that fast — it
   represents a stable human trait. Separate timescales enforce this.
3. **Correctness.** In the generative model, theta is sampled once per human
   and determines the prior for ALL of that human's phi values. Updating theta
   after every single observation would be treating it as if it could change
   with each recipe step, which violates the model structure.

---

## Why CSP variables and preference dimensions are not the same thing

One preference dimension can apply to multiple CSP variables across different
planning problems. Example:

`occlusion_sensitivity` (one scalar) constrains:
- `robotConf ∈ ℝ⁷` in the feeding robot CSP
- `drinkPose ∈ SE(2)` in a different CSP
- `platePose ∈ SE(2)` in yet another CSP

The preference is stable across all three. The CSP variables change depending
on what's currently being planned. The constraint function `c_theta(csp_var, phi)`
is the bridge — it applies the same phi to different variable domains.

**Consequence:** When defining new environments, use co-generation prompting
to define (csp_var, pref_dim, constraint_fn) triples jointly. A one-to-one
mapping between CSP variables and preference dimensions is too restrictive
and loses cross-context generalization.

---

## Why log_v_phi is clamped to [LOG_VAR_MIN, LOG_VAR_MAX]

Adam can drive log_v_phi to extreme values:
- Very negative: variance → 0. This causes NaN in `1/v` in the KL term and
  produces a delta function posterior (certainty with no data) — wrong.
- Very positive: variance → ∞. Gradients become negligible; learning stalls.

The clamp `[log(1e-6), log(10.0)]` corresponds to variance in [1e-6, 10.0].
These bounds are generous — legitimate variance values are in [0.01, 2.0] —
so the clamp only activates in pathological cases.

The clamp is applied with `torch.no_grad()` so it doesn't affect the gradient
computation for that step. It's a constraint, not a soft regularizer.
