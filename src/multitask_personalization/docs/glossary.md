# Glossary and Notation

A reference for all variable names, mathematical symbols, and domain terms
used across the codebase and documentation.

---

## Mathematical symbols

| Symbol | Code name | Meaning |
|--------|-----------|---------|
| `mu_d` / `mu_s` | `mu_mean[spice]` | Global population mean for preference dimension d (or spice s) |
| `theta_{h,d}` | `_theta_m[h][s]` | Human h's stable preference for dimension d, across all contexts |
| `phi_{h,c,d}` | `_phi_m[h][recipe][s]` | Human h's preference for dimension d in context c |
| `psi_{h,sess}` | `m_psi[h]` | Session-level transient offset (Stage 2+) |
| `z_t` | `z_mean` | SSM latent state at timestep t (Stage 6+) |
| `sigma_0` | `sigma0` | Prior std for mu (global level uncertainty) |
| `sigma_h` | `exp(log_sigma_h)` | Prior std controlling human-to-human variation |
| `sigma_r` or `sigma_c` | `exp(log_sigma_r)` | Prior std controlling recipe/context variation |
| `sigma_obs` | `exp(log_sigma_obs)` | Observation noise std for satisfaction ratings |
| `sigma_mood` | `sigma_mood` | Prior std for session-level psi offset |
| `m_phi` | `_phi_m[h][r][s]` (as tensor) | Variational mean of q(phi) |
| `log_v_phi` | `_phi_logv[h][r][s]` | Log variational variance of q(phi) |
| `v_phi` | `exp(_phi_logv[h][r][s])` | Variational variance of q(phi) |
| `A` | `A` | SSM transition matrix (Stage 6+) |
| `C` | `C` | SSM emission matrix mapping z to preference logit offsets (Stage 6+) |
| `Q` | `Q` | SSM process noise covariance (Stage 6+) |

---

## Index variables

| Index | Meaning | Example values |
|-------|---------|----------------|
| `h` | Human identifier | "alice", "bob", DEFAULT_HUMAN |
| `s` or `d` | Spice or preference dimension | "turmeric", "cumin"; or 0, 1, 2... |
| `r` or `c` | Recipe or context | "SimpleDal", "SweetCurry" |
| `t` | Timestep within episode | 0, 1, 2, ... |
| `sess` | Session/episode index | implicit (resets each episode) |

---

## Key quantities

| Term | Definition | Sign convention |
|------|-----------|----------------|
| `logit` | `sign(actor) * (phi + psi)` | + when assignment matches preference |
| `sign(actor)` | +1 for "human", -1 for "robot" | |
| `P(human)` | `sigmoid(phi)` | phi > 0 → P(human) > 0.5 |
| `expected_sat` | `tanh(sign * (phi + psi))` | bounded in (-1, +1) |
| ELBO | Evidence Lower BOund | maximized during training |
| KL | KL divergence `KL(q||p)` | always >= 0, penalizes q far from prior |

---

## Phi interpretation

The phi scale is in logit space:

| phi value | P(human) | Interpretation |
|-----------|----------|----------------|
| +3.0 | 0.95 | Very strong preference for human |
| +1.5 | 0.82 | Clear preference for human |
| +0.5 | 0.62 | Mild preference for human |
| 0.0 | 0.50 | Indifferent |
| -0.5 | 0.38 | Mild preference for robot |
| -1.5 | 0.18 | Clear preference for robot |
| -3.0 | 0.05 | Very strong preference for robot |

---

## Code conventions

| Convention | Meaning |
|------------|---------|
| `_phi_m[h][r][s]` | PyTorch tensor, requires_grad=True |
| `get_phi(h, r, s)` | Returns `.item()` — plain float, no grad |
| `log_sigma_*` | Log of a sigma value (always stored in log space) |
| `log_v_*` | Log of a variance value (always stored in log space) |
| `detach()` | Treat a tensor as a fixed constant (no gradient flows through) |
| `N_MC_SAMPLES` | Number of reparameterization samples for ELBO expectation |
| `n_phi_steps` | Number of Adam steps per per-observation phi update |
| `n_theta_steps` | Number of Adam steps per end-of-episode theta update |

---

## Environment terms

| Term | Meaning |
|------|---------|
| Episode | One cooking session (spice env) or collaborative task (Overcooked) |
| Context | The recipe or task type — the "c" index for phi |
| Dimension | One axis of preference variation — the "d" index for phi |
| CSP | Constraint Satisfaction Problem — the planner's representation of a decision |
| Null space | The set of valid CSP solutions (satisfying hard constraints) |
| Personalized constraint | A CSP constraint whose parameters theta are learned from interaction |
| Soft constraint | A preference that influences which CSP solution is chosen but doesn't rule out others |
| Hard constraint | A constraint that must be satisfied (safety, feasibility) |
| Active exploration | Choosing actions to maximize information gain, not just preference match |
| Entropy criterion | CBTL's method for active exploration: choose action maximizing H(Bernoulli(sigmoid(phi))) |

---

## Moods (spice env specific)

| Mood | Meaning | Effect on behavior |
|------|---------|-------------------|
| `neutral` | Base preferences apply | phi directly drives behavior |
| `all_self` | Human wants to do everything | Strong positive bias on all human assignments |
| `none_self` | Human wants robot to do everything | Strong negative bias on human assignments |

Moods are episode-level. They are inferred (not observed) from the pattern
of (actor, satisfaction) observations during the episode.

---

## CBTL paper terminology mapped to this codebase

| CBTL term | This codebase | Notes |
|-----------|--------------|-------|
| Personalized constraint parameter θ | `phi_{h,c,d}` | Our phi is the HBM's posterior over CBTL's theta |
| Constraint generator GEN_p | `SpicesAssignCSPGenerator` | Produces constraints from phi |
| Initiation condition ι_p | Implicit in generator (always fires for actor variable) | |
| CONSTRAINT_PROMPT | `log_prob_prefer` | Maps phi to constraint satisfaction probability |
| LEARNING_PROMPT | `observe` + `end_episode` | Updates phi from interaction history |
| Entropy-based active learning | Stage 3 CSP modification | Uses `get_phi_var` from HBM |
| Null space | Set of valid `actor` assignments | {human, robot} in spice env |
