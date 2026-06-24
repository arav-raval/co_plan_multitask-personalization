"""
overcooked_config.py — Centralized configuration for the Overcooked HBM.

Mirrors spices_config.py but replaces:
  - MoodConfig  →  SessionConfig  (fatigue / frustration as session-level offsets)
  - SatisfactionConfig  →  TaskConfig  (order-completion-based feedback)

The HBMConfig is identical to the spices version; all VI hyperparameters are
shared across environments.

Key difference from spices: psi is a *vector* (one entry per subtask) rather
than a scalar.  The psi_dim field in HBMConfig controls this.  Vector psi
enables ARD-style pruning in Stage 4 and is the primary motivation for
porting to Overcooked before implementing Stage 4 on spices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class HBMConfig:
    """Configuration for the Hierarchical Bayesian Model (HBM).

    Identical fields to spices HBMConfig with two additions:
      - psi_dim: dimensionality of the per-episode psi vector (one per subtask
                 category; set to 1 for scalar-psi compatibility).
      - ard_prior_alpha / ard_prior_beta: shape/rate of Gamma ARD prior on
                 each psi dimension's precision (Stage 4 hook).
    """

    # Prior means and variances
    mu0: float = 0.0
    sigma0: float = 1.0
    sigma_h: float = 1.0        # human-level variance
    sigma_r: float = 0.5        # layout-level variance (analogous to recipe)
    sigma_obs: float = 1.0      # observation noise

    # Variance-adaptive updates (kept for compatibility, not used in VI path)
    use_variance_adaptive_lr: bool = True
    variance_adaptive_min_scale: float = 0.3
    variance_adaptive_max_scale: float = 1.0
    variance_converged_threshold: float = 0.3
    base_learning_rate: float = 0.3
    use_lr_annealing: bool = True
    lr_annealing_start: int = 100
    lr_annealing_end: int = 500
    lr_annealing_min: float = 0.1
    preference_mismatch_threshold: float = 1.0
    preference_mismatch_penalty: float = 1.0
    update_theta_mu_every_n_episodes: int = 2

    # Vector psi: per-episode session-offset vector.
    # psi_dim should equal len(subtasks) for the layout being trained on,
    # or be set explicitly here.  Default 5 covers the full ALL_SUBTASKS list.
    psi_dim: int = 5

    # Prior std for each psi dimension.  Fixed (not learned) to prevent
    # psi prior collapse, same reasoning as scalar-psi in spices.
    # Prior std for psi. Wide enough to absorb session effects without
    # the KL penalty pulling psi to 0 and contaminating phi.
    sigma_session: float = 1.0

    # Aggressive decay at episode end (same as spices psi_decay).
    psi_decay: float = 0.05

    # Stage 4 hook: ARD Gamma prior on 1/sigma_session per dimension.
    # alpha=1, beta=1 is a non-informative prior; tune to encourage sparsity.
    ard_prior_alpha: float = 1.0
    ard_prior_beta: float = 1.0

    # ELBO optimizer
    n_mc_samples: int = 8
    n_phi_steps: int = 12
    n_theta_steps: int = 12
    lr_phi: float = 3e-2
    lr_theta: float = 1e-2
    lr_hyper: float = 5e-3
    log_var_min: float = math.log(1e-6)
    log_var_max: float = math.log(10.0)


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for session-level transient state modeling.

    Replaces MoodConfig.  In Overcooked the transient state is interpreted
    as fatigue or frustration rather than discrete mood categories.

    Since we use vector psi (no discrete mood categories), inference is
    purely through the ELBO — no Bayesian mood-posterior filter needed.

    Session profiles
    ----------------
    Non-neutral sessions apply *differentiated* effects per subtask via
    named session profiles (see config/session_profiles.py).  Each profile
    defines per-subtask sensitivity weights controlling how strongly
    fatigue/energy affects each subtask's claiming probability:

        psi_true[d] ~ N(weight[d] * ±mean_abs, std²)

    This makes vector psi strictly more expressive than scalar psi: a scalar
    model cannot capture the heterogeneous session response across subtasks.
    """

    # Prior for session-level psi (one std per subtask dimension).
    session_prior_std: float = 0.5

    # Base magnitude of non-neutral session effects.
    session_nonneutral_mean_abs: float = 2.0
    session_nonneutral_std: float = 0.5
    session_neutral_std: float = 0.2

    # Named session profile controlling per-subtask sensitivity weights.
    # See config/session_profiles.py for available profiles and their semantics.
    # "PhysicalFatigue" (default): physically demanding tasks more affected.
    # "TiredOfWalking": navigation tasks heavily affected, stationary tasks spared.
    # "Uniform": all tasks equally affected (equivalent to scalar psi baseline).
    session_profile_name: str = "PhysicalFatigue"

    # Session type prior. Higher non-neutral rate makes vector psi more
    # valuable — the model must absorb per-subtask fatigue/energy effects.
    prob_neutral_session: float = 0.6


@dataclass(frozen=True)
class TaskConfig:
    """Configuration for the task-level feedback (satisfaction) model.

    Generates a continuous satisfaction signal in [-1, +1] that encodes
    preference magnitude, session effects, and coordination quality.
    Mirrors SatisfactionConfig from spices.

    The generative model:
        phi_latent = pref_sign * phi_mag
        phi_mag    = |hidden_theta(subtask)| when available, else base_task_bias
        logit      = actor_sign * (phi_latent + psi_true[subtask_dim])
        p          = (tanh(logit / T) + 1) / 2    maps logit to (0, 1)
        sat      ~ Beta(p * kappa, (1 - p) * kappa) rescaled to [-1, +1]

    Coordination cost (additive penalty):
        When the robot mispredicts (conflict or missed pass), coordination_cost
        is subtracted.  This makes satisfaction sensitive to robot prediction
        quality independently of preference strength.
    """

    # Preference signal strength (analogous to base_satisfaction_bias).
    base_task_bias: float = 1.5

    # Temperature for tanh(logit / T) mapping logit → expected score.
    task_logit_temperature: float = 1.0

    # Concentration for Beta-distributed satisfaction (higher = less noise).
    task_beta_kappa: float = 10.0

    # Flat penalty subtracted from satisfaction when robot mispredicts.
    coordination_cost: float = 0.5

    # Minimum-effort threshold: when the human has no preferred tasks available,
    # they'll still accept a task if phi + psi > min_effort_threshold.
    # -1.0 = very reluctant (only mildly disliked tasks)
    # -3.0 = will do almost anything if necessary
    # float('inf') = never does disliked tasks (pure preference-driven)
    min_effort_threshold: float = -1.0

    # Normalisation range for raw Overcooked shaped reward → [-1, +1].
    max_shaped_reward: float = 20.0


@dataclass(frozen=True)
class OvercookedConfig:
    """Top-level config combining all sub-configs."""

    hbm: HBMConfig = HBMConfig()
    session: SessionConfig = SessionConfig()
    task: TaskConfig = TaskConfig()

    @property
    def session_prior_std(self) -> float:
        return self.session.session_prior_std

    def get_session_type_prior(self) -> Dict[str, float]:
        """Return probability of each session type."""
        p_neutral = self.session.prob_neutral_session
        p_other = (1.0 - p_neutral) / 2.0
        return {
            "energised": p_other,
            "neutral": p_neutral,
            "fatigued": p_other,
        }


# Default configuration instance
DEFAULT_CONFIG = OvercookedConfig()
