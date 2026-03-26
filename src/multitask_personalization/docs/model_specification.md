# Model Specification

This document is the single source of truth for the mathematical model.
Every equation here has a WHY. Read this before changing any inference code.

---

## 1. The problem being solved

We observe a human collaborating with a robot over many episodes. At each
timestep the robot must decide who performs a task (human or robot). The
human has **stable preferences** (they generally like to chop things themselves)
and **transient states** (today they're tired and want the robot to do more).

We need to:
- Infer stable preferences from noisy, sparse observations
- Separate stable preferences from transient mood effects
- Transfer knowledge across recipes/contexts and across humans
- Quantify uncertainty so the CSP planner knows when to explore

---

## 2. The hierarchical structure

Three levels, from most general to most specific:

```
mu_d           ~ N(0, sigma_0²)            Level 3: global population mean
                                            per preference dimension d
theta_{h,d}    ~ N(mu_d, sigma_h²)         Level 2: human h's stable preference
                                            for dimension d, across all contexts
phi_{h,c,d}    ~ N(theta_{h,d}, sigma_c²)  Level 1: human h's preference for
                                            dimension d in context c
```

**WHY three levels and not two?**
Two levels (global + individual) would mean every new recipe starts from
the population mean. Three levels means a new recipe starts from *that
human's* theta — which is a much better cold start after the first few
recipes. The hierarchy buys cross-context transfer within a person.

**WHY Gaussian priors at each level?**
Conjugacy: Gaussian prior + Gaussian likelihood = Gaussian posterior
(closed form). More importantly, the ELBO KL term between two Gaussians
has a closed form, making the optimization tractable without Monte Carlo
for the KL piece. Only the likelihood expectation requires MC samples.

**What "dimension" means in the spice env:**
In the spice env, each spice is its own preference dimension indexed by
`(human, recipe, spice)`. In Overcooked this generalizes to
`(human, recipe_type, subtask_type)`. The math is identical; only the
index changes.

---

## 3. The variational posteriors

We cannot compute the true posterior `p(phi, theta | data)` because the
denominator `p(data)` requires integrating over all possible phi and theta
values — intractable in high dimensions.

Instead, we approximate with a mean-field Gaussian variational family:

```
q(phi_{h,c,d})  = N(m_phi,   exp(log_v_phi))
q(theta_{h,d})  = N(m_theta, exp(log_v_theta))
```

**WHY mean-field (independent q)?**
Mean-field assumes phi and theta are independent under q. This is wrong
(they're correlated in the true posterior) but makes optimization tractable.
The cost is that mean-field systematically underestimates posterior variance
— the posterior tends to be too tight. In practice this means the CSP may
think it's more confident than it should be, potentially under-exploring.
Monitor entropy curves: if they drop too fast before prediction accuracy
improves, the mean-field approximation is hurting you.

**WHY store log_v instead of v?**
Two reasons:
1. Variance must be strictly positive. `exp(log_v)` is always positive
   regardless of what Adam does to `log_v`. Direct variance parameterization
   would allow Adam to step v to negative values, causing NaN in `sqrt(v)`.
2. Gradient scaling. Variances range from ~0.01 (confident) to ~2.0 (prior).
   In raw space the gradient landscape is steep near zero and flat at large
   values. In log space the range is [-4.6, 0.7] — compact and well-scaled.

**WHY store m_phi and log_v_phi as `requires_grad=True` PyTorch tensors?**
These are the parameters Adam optimizes. PyTorch autograd computes gradients
of the ELBO with respect to them automatically. This is the correct way to
implement VI — the same pattern used in every modern VI library (Pyro, NumPyro,
PyTorch distributions). It also means the same code pattern scales directly
to Stages 4-7 with only the model components changing.

---

## 4. The likelihood

Each observation is `y_t = (actor_t, satisfaction_t)`. The joint likelihood
factorizes into two terms:

```
log p(y_t | phi) = log p(actor_t | phi) + log p(sat_t | phi, actor_t)
```

**Term 1 — Actor choice (Bernoulli with logistic link):**
```
p(actor=human | phi) = sigmoid(phi)
p(actor=robot  | phi) = sigmoid(-phi)

log p(actor | phi) = log sigmoid(sign(actor) * phi)
                   = log_sigmoid(sign * phi)    [numerically stable form]
```

**WHY Bernoulli and not Gaussian for the actor signal?**
Actor choice is binary. The natural model for a binary outcome is Bernoulli.
Using a Gaussian likelihood would require treating the actor as a continuous
value (e.g., ±1) and introducing an arbitrary observation noise sigma. This
sigma has no principled interpretation and would need tuning. More
importantly, the Bernoulli model has built-in **saturation**: when phi is
already large (+2.5), an additional "human did it" observation barely moves
the posterior because sigmoid(2.5) ≈ 0.92 — you're already confident. The
Gaussian model has no such saturation and would keep updating phi even when
you're already very confident, producing over-confident posteriors.

**Term 2 — Satisfaction magnitude (Gaussian with tanh link):**
```
expected_sat = tanh(sign(actor) * phi)
log p(sat | phi, actor) = log N(sat; expected_sat, sigma_obs²)
                        = -0.5 * ((sat - tanh(sign*phi)) / sigma_obs)²
                          - log(sigma_obs) - 0.5*log(2*pi)
```

**WHY tanh as the link function?**
Satisfaction is bounded in [-1, +1]. The tanh function maps the real-valued
logit to [-1, +1], so expected satisfaction is always in the valid range.
A raw Gaussian would assign positive probability to satisfaction = 3.7,
which is impossible. The tanh link enforces the constraint.

**WHY a separate Gaussian term instead of just using the Bernoulli?**
Satisfaction carries information about *how much* the human cared about the
assignment, not just whether they were happy or not. A correction with
satisfaction -0.9 is a much stronger signal than one with satisfaction -0.1.
The Bernoulli term alone treats all corrections as equivalent. The Gaussian
term correctly weights observations by their magnitude.

**WHY conditional independence between the two terms?**
Given phi, the actor choice and the satisfaction rating are approximately
independent — phi explains both. This is an approximation (satisfaction
depends on actor choice via tanh(sign*phi)) but it simplifies the joint
to a product, which is standard in preference learning models.

---

## 5. The ELBO

The Evidence Lower BOund is the objective we maximize:

```
ELBO = E_q[log p(y | phi)] - KL(q(phi) || p(phi | theta))
```

**WHY maximize ELBO instead of directly minimizing KL(q || p_true)?**
The true posterior `p(phi | y)` is intractable — we can't compute it.
But we can show that:

```
log p(y) = ELBO + KL(q(phi) || p(phi | y))
```

Since `log p(y)` is a constant and KL >= 0, maximizing ELBO is equivalent
to minimizing the KL divergence between q and the true posterior. We maximize
ELBO because we can compute it; we can't compute the KL directly.

**The two terms have opposing incentives — this tension IS the inference:**
- `E_q[log p(y | phi)]`: pull q toward regions of high likelihood
  (fit the data). Pushes m_phi toward wherever the observations point.
- `KL(q || prior)`: pull q toward the prior (don't overfit).
  Pushes m_phi back toward theta, and pushes v_phi toward sigma_r².

The optimal q balances these two forces. With few observations, the KL
dominates and q stays near the prior. With many observations, the likelihood
term dominates and q concentrates on the true preference. This automatic
interpolation IS the Bayesian inference you want.

**The KL term for two Gaussians has a closed form:**
```
KL(N(m_q, v_q) || N(m_p, v_p))
    = 0.5 * [log(v_p/v_q) + v_q/v_p + (m_q - m_p)²/v_p - 1]
```
This is exact — no approximation needed. Only the likelihood expectation
requires Monte Carlo samples via the reparameterization trick.

---

## 6. The reparameterization trick

To compute gradients of `E_q[log p(y | phi)]` with respect to m_phi and
log_v_phi, we need gradients through a sampling operation. Sampling is
not differentiable, but we can rewrite it as:

```
phi = m_phi + exp(0.5 * log_v_phi) * eps,   eps ~ N(0, 1)
```

Now phi is a deterministic function of (m_phi, log_v_phi) and a fixed
noise variable eps. Gradients flow through m_phi and log_v_phi via autograd.
eps is sampled once and treated as a constant during the backward pass.

In code:
```python
eps = torch.randn(N_MC_SAMPLES)
phi_samples = m_phi + torch.exp(0.5 * log_v_phi) * eps
```

We use N_MC_SAMPLES=8 samples to estimate the expectation. This introduces
variance in the gradient estimate but 8 samples is enough for the signal
to be useful in practice.

---

## 7. The transient state model (Stages 1-5: psi; Stage 6+: SSM)

**The core problem:** A single observation is ambiguous. "Alice rejected the
robot's chopping" could mean she prefers to chop herself (stable preference)
or she's in a bad mood today (transient state). We cannot separate these
from one observation.

**How separation happens:** Through their different statistical signatures:
- Stable preferences show consistent patterns across episodes and specific
  to certain dimensions.
- Transient mood affects all dimensions uniformly within one episode, then
  disappears.
- The model routes signals to the cheapest explanation in terms of KL cost.

**Stage 1-5: Scalar psi (episode-scoped offset)**
```
psi_{h,sess} ~ N(0, sigma_mood²)    [reset each episode]
logit_{d,t}  = phi_{h,c,d} + psi
```
Within an episode, moving psi costs less KL (fresh prior) than moving phi
(anchored to previous episodes). So the ELBO optimizer preferentially uses
psi to explain within-episode deviations. Across episodes, psi resets so
only phi accumulates.

**Stage 6+: State-space model (Kalman filter)**
Replaces the episode-scoped psi with a continuous latent state vector z_t
that evolves over time:
```
z_t = A * z_{t-1} + w_t,    w ~ N(0, Q)
logit_{d,t} = phi_{h,c,d} + C_d · z_t
```
This adds three capabilities psi cannot provide:
1. Within-episode drift (fatigue building mid-session)
2. Cross-session persistence (controlled by A's decay rate)
3. Cross-dimension inference (observing initiative updates pacing prediction
   via the shared z and emission matrix C)

**WHY not SSM from the start?**
The SSM requires specifying A and C. With few humans and short episodes,
A and C are not identifiable from data. The psi model is the correctly
simpler choice until you have enough data to learn the dynamics.

---

## 8. The update schedule (three timescales)

```
Per observation (every step):
  phi update — 10 Adam steps on ELBO_phi
  Includes: m_phi, log_v_phi, log_sigma_obs

Per episode end:
  theta update — 20 Adam steps on ELBO_theta
  mu update    — analytic precision-weighted (not variational yet)
  Includes: m_theta, log_v_theta, log_sigma_h, log_sigma_r

Cross-episode (batched, every N episodes):
  Governed by config.hbm.update_theta_mu_every_n_episodes
```

**WHY different timescales?**
This matches the structure of the model. phi is context-specific and
updates quickly from new observations. theta is human-specific and needs
to aggregate across multiple recipe observations before it's meaningful.
mu is population-level and needs multiple humans before it's stable.
Forcing all updates at the same rate would either make phi too slow or
make theta unstable from individual noisy observations.

**WHY fresh Adam per update call (not persistent optimizer state)?**
In online VI, each update call runs to approximate convergence on recent
data, not tracking gradient momentum across the history of all observations.
Persistent Adam state would accumulate stale momentum from earlier
observations and bias the current update direction.

---

## 9. What the CSP receives

The CSP queries the HBM for each preference dimension before making a decision:

```python
mean, var = hbm.get_phi(h, c, d), hbm.get_phi_var(h, c, d)
# + psi/SSM contribution when those are active

# mean → soft constraint: action that maximizes mean preference
# var  → entropy criterion: action that maximizes uncertainty resolution
#        H(Bernoulli(sigmoid(mean))) scaled by var
```

**WHY pass variance to the CSP?**
The original system had no uncertainty — it passed a point estimate and
always exploited. This means if the initial phi was slightly wrong (as it
always is), the robot would confidently keep doing the wrong thing. Passing
variance enables the CBTL entropy criterion: deliberately choose actions
that resolve uncertainty about under-explored dimensions. This is the
primary mechanism for efficient learning.

**The exploration-exploitation tradeoff:**
- Exploit: choose action with highest mean preference score
- Explore: choose action that maximizes entropy H(Bernoulli(sigmoid(mean)))
  weighted by variance
- CBTL's insight: both happen within the CSP null space, so exploration
  never violates safety constraints.
