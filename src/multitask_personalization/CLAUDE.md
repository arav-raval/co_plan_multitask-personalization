# CLAUDE.md — Hierarchical Bayesian Preference Learning for HRI

This file gives Claude Code the full context needed to continue development
of a hierarchical Bayesian model (HBM) for personalized human-robot collaboration,
integrated with a CSP-based planning framework extending the CBTL paper.

**Read this file first, then read every file in `docs/` before touching any code.**

---

## What this project is

A senior thesis system for **long-term preference learning in human-robot
collaboration**. The robot learns what a specific human prefers (who should
perform which tasks, at what pace, with how much initiative) and adapts its
behavior accordingly over many episodes.

The system extends the **CBTL paper** (Silver et al., 2025 — "Coloring Between
the Lines: Personalization in the Null Space of Planning Constraints",
arXiv:2505.15503). That paper uses CSPs for robot planning and learns
parameterized personalized constraints. This project adds:

1. A proper hierarchical Bayesian model for the preference parameters
2. Principled VI inference replacing CBTL's ad-hoc parameter updates
3. A session-level latent state model for transient effects (mood, fatigue)
4. Optional LLM-assisted preference dimension discovery

**Current environment:** A toy "spice env" where a robot and human take turns
adding spices to a recipe. The action at each step is assigning a spice to
human or robot. This is the testbed — the architecture is designed to scale
to Overcooked and 3D robotics.

---

## Repository layout (expected)

```
src/multitask_personalization/envs/spices/
├── spices_env.py          # Gymnasium environment (DO NOT CHANGE in Stage 1)
├── spices_hbm.py          # Original HBM — being replaced by spices_hbm_stage1.py
├── spices_hbm_stage1.py   # Stage 1 implementation (PyTorch ELBO — see docs/)
├── spices_csp.py          # CSP generator (Stage 3 will modify this)
├── recipes.py             # Recipe DAGs
└── config/
    ├── spices_config.py   # HBMConfig, MoodConfig, SatisfactionConfig
    ├── hidden_hbm_configs.py
    └── test_configs.py

tests/envs/
├── test_spices_csp.py
├── test_spices_hbm_multi_human.py
└── test_spices_hbm_stage1.py   # Stage 1 validation tests
```

---

## The 7-stage migration plan (current status: Stages 1–3 complete, Stage 4 next)

See `docs/migration_plan.md` for full details. Brief summary:

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Fix likelihood, replace pseudo-obs with ELBO+PyTorch | **COMPLETE** |
| 2 | Replace mood gate with scalar psi latent variable | **COMPLETE** |
| 3 | Pass uncertainty (variance) to CSP for active exploration | **COMPLETE** |
| 4 | Vector psi + ARD prior for dimension pruning | Not started |
| 5 | Port to Overcooked | Not started |
| 6 | Diagonal SSM upgrade (Kalman filter for transient state) | Not started |
| 7 | Structured SSM + optional LLM dimension proposal | Not started |

**Do not jump ahead.** Each stage is independently testable and must pass
its tests before the next stage begins.

---

## The single most important architectural principle

The HBM, the SSM (transient state model), and the CSP planner are **three
separate subsystems** that communicate through a well-defined interface.
Never mix their concerns:

- **HBM** owns stable preference posteriors `q(phi)` and `q(theta)`.
  It speaks PyTorch internally but exposes plain floats to everything else.
- **SSM** (added in Stage 6) owns transient session state `z_t`.
  It uses a Kalman filter, not VI.
- **CSP** asks the HBM for `(mean, var)` per preference dimension and uses
  `mean` for soft constraints and `var` for entropy-based active exploration.

The CSP **never** sees PyTorch tensors. The HBM **never** knows about CSP
variables. The interface is defined in `docs/interfaces.md`.

---

## What NOT to do

These are the mistakes the original implementation made. Do not reintroduce them:

1. **Do not use pseudo-observations** (`g = sign(actor) * satisfaction * bias`).
   This collapses the likelihood into a single number, destroying the
   probabilistic interpretation. Use `_log_likelihood` with the proper
   Bernoulli+Gaussian joint likelihood instead.

2. **Do not gate phi updates on mood confidence**. The original code only
   updated phi when mood posterior was confidently neutral. This throws away
   real data. The psi/SSM model handles mood automatically through the KL
   penalty — no gate needed.

3. **Do not hardcode sigma_h, sigma_r, sigma_obs**. These must be learned.
   Hardcoded variances make the precision-weighted updates in the hierarchy
   numerically meaningless (you're doing a weighted average with arbitrary weights).

4. **Do not use the Laplace approximation instead of ELBO**. We chose ELBO
   because it jointly optimizes variational parameters AND hyperparameters
   (sigma_h, sigma_r, sigma_obs) via the same Adam step. Laplace finds the
   MAP and approximates curvature locally — it cannot learn sigma_h without
   an outer EM loop, which is essentially VI anyway.

5. **Do not store variances directly — always store log_v**. Variances must
   be strictly positive. Storing log_v and recovering v = exp(log_v) enforces
   this constraint and gives better gradient scaling across the wide range
   of variance values the model will encounter.

---

## Critical file to read before any code changes

`docs/model_specification.md` — contains the full mathematical model,
every equation, and the WHY behind each design choice.
