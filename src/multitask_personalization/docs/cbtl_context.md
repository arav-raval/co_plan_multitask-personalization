# CBTL Paper Context

## What CBTL is

"Coloring Between the Lines" (Silver et al., Cornell 2025, arXiv:2505.15503)
proposes a framework for robot personalization built on three ideas:

1. **Robots make decisions by solving CSPs.** Every decision (which sauce to use,
   how to position a plate, which ready signal to use) is a CSP variable with
   a domain. Safety and feasibility constraints define the feasible set.

2. **The CSP null space is the personalization space.** Many CSP solutions are
   all equally safe and competent. The robot should choose solutions that
   match the user's preferences — "coloring between the lines" of the constraints.

3. **Personalized constraints are learned incrementally.** Each personalization
   axis has a parameterized constraint c_theta. The robot actively gathers
   data to learn theta, using an entropy-based criterion to choose actions
   that maximally resolve uncertainty.

## What CBTL does NOT do that this project adds

**CBTL's parameter learning is ad-hoc.** The paper uses:
- Supervised classification from labeled data for scalar theta
- LLM summarization for text theta (e.g., "prefers ketchup with fried foods")

Neither approach quantifies uncertainty correctly. The LLM approach especially
has no principled uncertainty representation — theta is a text string, not a
probability distribution. This means:
- The entropy criterion has no principled uncertainty to compute entropy over
- Cold-start for new users / new contexts has no principled initialization
- Population-level learning (what we know about humans in general) is absent

**This project's contribution:** Replace CBTL's ad-hoc parameter learning with
a proper hierarchical Bayesian model and VI inference. This gives:
- Calibrated uncertainty → correct entropy criterion
- Hierarchical pooling → cold-start transfer within a person and across people
- Learned hyperparameters → the model discovers how much humans vary

## How the systems connect

```
CBTL framework:
  CSP generator → produces (V, C) for each decision
  Personalized constraint generators → fire on relevant CSP variables
  Active learning criterion → chooses which solution to pick

This project replaces:
  Parameter learning → replaced by HBM + VI
  Uncertainty representation → phi posterior variance
  Cold-start → hierarchical initialization from theta

The interface:
  CBTL asks: "what is the constraint c_theta(v) for this variable?"
  This project answers: log_prob_prefer(h, r, d, actor) = log sigmoid(sign * phi_mean)
  
  CBTL asks: "how uncertain are we about this constraint?"
  This project answers: phi_var = v_phi + SSM_contribution
```

## What the paper hand-crafts (and what stays hand-crafted here)

CBTL requires human engineers to specify:
1. Which CSP variables are personalizable (vs physics-only)
2. The form of the personalized constraint function c_theta
3. The initiation condition for each constraint generator

This project optionally automates item 1 via LLM dimension proposal (Stage 7),
but items 2 and 3 still require engineering judgment for each new environment.

The key insight from the paper's structure: CSP variables and preference
dimensions are NOT the same thing. A single preference dimension (phi) can
apply as a constraint to multiple CSP variables across different planning
problems. The constraint function c_theta is the bridge.

## Active learning (entropy criterion)

CBTL's active learning chooses CSP solutions that maximize uncertainty about
the personalized constraint parameters:

```
max_v  (1/|C_p|) * sum_{C_p} H(C_p(v))
subject to: C(v) = True  for all hard constraints C
```

Where H(C_p(v)) = entropy of the Bernoulli distribution P(C_p(v) = True).

In this project's implementation (Stage 3):
```python
entropy = H(Bernoulli(sigmoid(phi_mean))) * phi_var
# phi_mean from HBM, phi_var from HBM
# High entropy + high variance = prioritize this dimension for exploration
```

The variance weighting is an addition beyond the paper's formulation — it
prevents the robot from repeatedly probing a dimension that's uncertain because
the model has no data vs a dimension that's uncertain because preferences
genuinely vary (truly ambiguous). The variance term helps distinguish these.

## Related work to cite

- CBTL (Silver et al. 2025) — the planning framework this extends
- VPL (Poddar et al. 2024, arXiv:2408.10075) — variational preference learning
  for RLHF, most directly related to the HBM approach
- Bradley-Terry model — the pairwise preference likelihood (Bernoulli with
  logistic link) used in this project's actor-choice term
- Mean-field VI — the inference algorithm; cite Blei et al. 2017 review
- Reparameterization trick — Kingma & Welling 2013 (VAE paper)
