"""
spices_hbm.py — Stage 2 migration of HierarchicalPreferenceModel.

Stage 2 adds a per-episode scalar latent variable psi to absorb transient
session effects (mood, fatigue) without gating or discarding observations.

Key changes from Stage 1:
  - psi_{h,sess} ~ N(0, sigma_mood²): a scalar session offset per human.
  - The likelihood logit is now sign(actor) * (phi + psi) instead of
    sign(actor) * phi. Psi explains session-level deviations cheaply (its
    KL prior resets each episode), leaving phi to encode stable preferences.
  - _elbo_phi now jointly optimizes (m_phi, log_v_phi, m_psi, log_v_psi).
  - end_episode aggressively decays psi (0.05× mean, full variance reset)
    so persistent signals cannot accumulate in psi across episodes.
  - sigma_mood is a fixed hyperparameter (not learned) — making it learnable
    risks the psi prior collapsing to zero, which would negate its purpose.
  - The generative model in SpicesEnv is updated to match: mood_adj is
    replaced by a scalar psi_true sampled once per episode.

Architecture (Stage 2)
-----------------------
  Variational posteriors:
    q(phi_{h,r,s})  = N(m_phi,  exp(log_v_phi))
    q(theta_{h,s})  = N(m_theta, exp(log_v_theta))
    q(psi_{h,sess}) = N(m_psi,  exp(log_v_psi))    ← new

  Priors:
    phi_{h,r,s}  ~ N(theta_{h,s}, exp(2*log_sigma_r))
    theta_{h,s}  ~ N(mu_s,        exp(2*log_sigma_h))
    psi_{h,sess} ~ N(0,           sigma_mood²)       ← new, reset each episode
    mu_s         ~ N(0,           sigma0^2)           [not variational yet]

  Likelihood (joint per observation):
    log p(actor, sat | phi, psi) =
        log sigmoid(sign(actor) * (phi + psi))          [Bernoulli term]
      + log N(sat; tanh(sign*(phi+psi)), sigma_obs²)    [Gaussian term]

  ELBO (per spice, per human+recipe context):
    ELBO = E_q[sum_t log p(y_t | phi, psi)]
           - KL(q(phi) || p(phi | theta))
           - KL(q(psi) || N(0, sigma_mood²))

  The theta ELBO is unchanged from Stage 1.

Update schedule (coordinate ascent at episode end):
  - psi (Phase 1): N_PHI_STEPS Adam steps at episode end with phi fixed
  - phi (Phase 2): N_PHI_STEPS Adam steps at episode end with psi fixed at inferred value
  - theta + mu + sigma_h/r/obs: N_THETA_STEPS Adam steps at episode end
  - psi decay: aggressive reset (0.05×) between episodes

PyTorch is used only inside the ELBO update methods.
All bookkeeping, registration, and CSP interface remain in plain Python/numpy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .config.spices_config import DEFAULT_CONFIG, SpicesConfig

MOODS = ("all_self", "neutral", "none_self")
DEFAULT_HUMAN = "human"

# Centralized HBM runtime defaults (owned by spices_config.py).
_HBM_CFG = DEFAULT_CONFIG.hbm
N_MC_SAMPLES = _HBM_CFG.n_mc_samples
N_PHI_STEPS = _HBM_CFG.n_phi_steps
N_THETA_STEPS = _HBM_CFG.n_theta_steps
LR_PHI = _HBM_CFG.lr_phi
LR_THETA = _HBM_CFG.lr_theta
LR_HYPER = _HBM_CFG.lr_hyper
LOG_VAR_MIN = _HBM_CFG.log_var_min
LOG_VAR_MAX = _HBM_CFG.log_var_max


# ---------------------------------------------------------------------------
# Mood utilities (unchanged from original)
# ---------------------------------------------------------------------------

def sample_episode_mood(
    rng: np.random.Generator,
    prior: Optional[np.ndarray] = None,
) -> str:
    if prior is None:
        prior = np.array(DEFAULT_CONFIG.mood.mood_prior, dtype=float)
    return str(rng.choice(MOODS, p=prior))


def compute_mood_bias(mood: str, actor: str) -> float:
    bias_dict = DEFAULT_CONFIG.get_mood_bias()
    return bias_dict.get(mood, {}).get(actor, 0.0)


class MoodModel:
    """Unchanged from original."""
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.current_mood: Optional[str] = None

    def sample_mood(self) -> str:
        self.current_mood = sample_episode_mood(self.rng)
        return self.current_mood


# ---------------------------------------------------------------------------
# ELBO computation (pure PyTorch, no side effects)
# ---------------------------------------------------------------------------

def _gaussian_kl(m_q: torch.Tensor, log_v_q: torch.Tensor,
                 m_p: torch.Tensor, log_v_p: torch.Tensor) -> torch.Tensor:
    """
    KL(q || p) for two Gaussians, both parameterized by (mean, log_variance).

    KL(N(m_q, v_q) || N(m_p, v_p))
        = 0.5 * [log(v_p/v_q) + v_q/v_p + (m_q - m_p)²/v_p - 1]

    All tensors can be scalars or batched — shapes must broadcast.
    """
    v_q = torch.exp(log_v_q)
    v_p = torch.exp(log_v_p)
    return 0.5 * (
        log_v_p - log_v_q
        + v_q / v_p
        + (m_q - m_p) ** 2 / v_p
        - 1.0
    )


def _obs_to_tensors(
    observations: List[Tuple[str, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-convert a list of (actor, satisfaction) observations to tensors.

    Returns:
        signs:  float tensor of shape [T], +1 for human, -1 for robot
        sats:   float tensor of shape [T], satisfaction values in [0, 1]

    Call this once before the Adam loop so the conversion is not repeated
    on every gradient step.
    """
    signs = torch.tensor(
        [1.0 if a == "human" else -1.0 for a, _ in observations],
        dtype=torch.float32,
    )
    sats = torch.tensor([s for _, s in observations], dtype=torch.float32)
    return signs, sats


def _ell_phi_psi(
    signs: torch.Tensor,          # [T]
    sats: torch.Tensor,           # [T]
    phi_samples: torch.Tensor,    # [N_MC]  (may be scalar broadcast if deterministic)
    psi_offset: torch.Tensor,     # scalar or [N_MC]
    log_sigma_obs: torch.Tensor,
    sat_logit_temperature: float,
) -> torch.Tensor:
    """
    Expected log likelihood shared by both phi and psi ELBO steps.
    logit = sign * (phi + psi): shape [T, N_MC].
    """
    logits = signs.unsqueeze(1) * (
        phi_samples.unsqueeze(0) + psi_offset.unsqueeze(0)
    )
    sigma_obs = torch.exp(log_sigma_obs)
    log_p_actor = torch.nn.functional.logsigmoid(logits)
    temp = max(float(sat_logit_temperature), 1e-6)
    expected_sat = torch.tanh(logits / temp)
    log_p_sat = (
        -0.5 * ((sats.unsqueeze(1) - expected_sat) / sigma_obs) ** 2
        - log_sigma_obs
        - 0.5 * math.log(2 * math.pi)
    )
    return (log_p_actor + log_p_sat).sum(dim=0).mean()


def _elbo_phi_step(
    signs: torch.Tensor,
    sats: torch.Tensor,
    m_phi: torch.Tensor,
    log_v_phi: torch.Tensor,
    m_psi_fixed: torch.Tensor,    # detached — psi held at its current mean
    m_theta: torch.Tensor,
    log_sigma_r: torch.Tensor,
    log_sigma_obs: torch.Tensor,
    sat_logit_temperature: float,
) -> torch.Tensor:
    """
    ELBO for the phi update step (coordinate ascent Phase 1).

    Psi is held deterministic at m_psi_fixed (no MC noise from psi). This prevents
    psi's uncertainty from inflating the variance gradient estimate for phi, which
    would otherwise cause sigma_r to drift upward via the theta optimizer.

    ELBO_phi = E_phi[sum_t log p(y_t | phi, psi=m_psi)] - KL(q(phi) || N(theta, sigma_r²))
    """
    if signs.numel() == 0:
        return torch.tensor(0.0)

    eps_phi = torch.randn(N_MC_SAMPLES)
    phi_samples = m_phi + torch.exp(0.5 * log_v_phi) * eps_phi  # [N_MC]

    # psi is deterministic — broadcast as [1] so unsqueeze gives [1, 1]
    ell = _ell_phi_psi(
        signs,
        sats,
        phi_samples,
        m_psi_fixed.expand(N_MC_SAMPLES),
        log_sigma_obs,
        sat_logit_temperature,
    )
    kl_phi = _gaussian_kl(m_phi, log_v_phi, m_theta, 2.0 * log_sigma_r)
    return ell - kl_phi


def _elbo_psi_step(
    signs: torch.Tensor,
    sats: torch.Tensor,
    phi_anchors: torch.Tensor,    # [T] — per-observation phi, detached
    m_psi: torch.Tensor,
    log_sigma_mood: torch.Tensor, # fixed (no grad) — prior std = posterior std
    log_sigma_obs: torch.Tensor,
    sat_logit_temperature: float,
) -> torch.Tensor:
    """
    ELBO for the psi update step (coordinate ascent Phase 2).

    phi_anchors is a [T] tensor of the actual phi posterior mean for each
    observation's spice. Using per-observation phi (rather than a blended
    average) ensures psi only explains the residual deviation that phi alone
    cannot, which is the correct interpretation of psi as a session offset.

    log_v_psi is fixed at sigma_mood² (q_var = p_var), so the KL simplifies to
    just the mean-squared penalty: 0.5 * m_psi² / sigma_mood².

    ELBO_psi = E_psi[sum_t log p(y_t | phi_t, psi)] - KL(q(psi) || N(0, sigma_mood²))
    """
    if signs.numel() == 0:
        return torch.tensor(0.0)

    sigma_mood = torch.exp(log_sigma_mood)
    sigma_obs = torch.exp(log_sigma_obs)
    eps_psi = torch.randn(N_MC_SAMPLES)
    psi_samples = m_psi + sigma_mood * eps_psi  # [N_MC] — sample with prior std

    # logit[t, k] = sign_t * (phi_t + psi_k): shape [T, N_MC]
    logits = signs.unsqueeze(1) * (phi_anchors.unsqueeze(1) + psi_samples.unsqueeze(0))
    log_p_actor = torch.nn.functional.logsigmoid(logits)
    temp = max(float(sat_logit_temperature), 1e-6)
    expected_sat = torch.tanh(logits / temp)
    log_p_sat = (
        -0.5 * ((sats.unsqueeze(1) - expected_sat) / sigma_obs) ** 2
        - log_sigma_obs
        - 0.5 * math.log(2 * math.pi)
    )
    ell = (log_p_actor + log_p_sat).sum(dim=0).mean()

    # KL when q_var == p_var == sigma_mood²: reduces to 0.5 * m_psi² / sigma_mood²
    kl_psi = 0.5 * (m_psi / sigma_mood) ** 2
    return ell - kl_psi


def _elbo_phi(
    signs: torch.Tensor,            # [T]  pre-converted observation signs
    sats: torch.Tensor,             # [T]  pre-converted satisfaction values
    m_phi: torch.Tensor,            # scalar variational mean
    log_v_phi: torch.Tensor,        # scalar log variational variance
    m_psi: torch.Tensor,            # scalar session offset mean
    log_v_psi: torch.Tensor,        # scalar session offset log variance (fixed, no grad)
    m_theta: torch.Tensor,          # scalar prior mean (from theta)
    log_sigma_r: torch.Tensor,      # scalar log prior std (recipe level)
    log_sigma_obs: torch.Tensor,    # scalar log obs noise std
    log_sigma_mood: torch.Tensor,   # scalar log psi prior std (fixed, no grad)
    sat_logit_temperature: float,
) -> torch.Tensor:
    """
    Full joint ELBO for diagnostic use (compute_elbo_snapshot).

    Not used in the training update — training uses the coordinate ascent
    steps _elbo_phi_step and _elbo_psi_step to avoid phi-psi gradient interference.

    ELBO = E_q[sum_t log p(y_t | phi, psi)]
           - KL(q(phi) || N(theta, sigma_r²))
           - KL(q(psi) || N(0, sigma_mood²))
    """
    if signs.numel() == 0:
        return torch.tensor(0.0)

    eps_phi = torch.randn(N_MC_SAMPLES)
    phi_samples = m_phi + torch.exp(0.5 * log_v_phi) * eps_phi

    eps_psi = torch.randn(N_MC_SAMPLES)
    psi_samples = m_psi + torch.exp(0.5 * log_v_psi) * eps_psi

    ell = _ell_phi_psi(
        signs, sats, phi_samples, psi_samples, log_sigma_obs, sat_logit_temperature
    )
    kl_phi = _gaussian_kl(m_phi, log_v_phi, m_theta, 2.0 * log_sigma_r)
    kl_psi = _gaussian_kl(m_psi, log_v_psi, torch.zeros_like(m_psi), 2.0 * log_sigma_mood)
    return ell - kl_phi - kl_psi


def _elbo_theta(
    phi_posteriors: List[Tuple[torch.Tensor, torch.Tensor]],  # [(m_phi, log_v_phi)]
    m_theta: torch.Tensor,
    log_v_theta: torch.Tensor,
    mu: float,                  # global mean (scalar, not variational yet)
    log_sigma_h: torch.Tensor,  # scalar log prior std at human level
    log_sigma_r: torch.Tensor,  # scalar log prior std at recipe level
) -> torch.Tensor:
    """
    ELBO for q(theta) treating each phi posterior as a noisy observation of theta.

    ELBO_theta = E_q[sum_r log p(phi_{r} | theta)] - KL(q(theta) || N(mu, sigma_h²))

    p(phi | theta) = N(phi; theta, sigma_r²).
    We use the phi posterior mean as a point estimate of phi for efficiency
    (this is the standard "posterior predictive" approximation in hierarchical VI).
    """
    if not phi_posteriors:
        return torch.tensor(0.0)

    # Reparameterize theta
    eps = torch.randn(N_MC_SAMPLES)
    theta_samples = m_theta + torch.exp(0.5 * log_v_theta) * eps  # [N_MC_SAMPLES]

    # Expected log p(phi | theta) for each recipe's phi posterior
    ell = torch.tensor(0.0)
    log_v_r = 2.0 * log_sigma_r
    v_r = torch.exp(log_v_r)

    for m_phi, log_v_phi in phi_posteriors:
        # E[log N(m_phi; theta, sigma_r²)] averaged over theta samples
        # Using phi posterior mean as point estimate of phi
        log_p = (
            -0.5 * ((m_phi - theta_samples) ** 2) / v_r
            - 0.5 * log_v_r
            - 0.5 * math.log(2 * math.pi)
        )
        ell = ell + log_p.mean()

    # KL(q(theta) || N(mu, sigma_h²))
    m_prior = torch.tensor(float(mu))
    log_v_prior = 2.0 * log_sigma_h
    kl = _gaussian_kl(m_theta, log_v_theta, m_prior, log_v_prior)

    return ell - kl


# ---------------------------------------------------------------------------
# Main HBM class
# ---------------------------------------------------------------------------

class HierarchicalPreferenceModel:
    """
    Hierarchical Bayesian preference model with VI inference via PyTorch ELBO.

    Structure (unchanged from Stage 0):
        mu_s         ~ N(0, sigma0²)             global, per spice
        theta_{h,s}  ~ N(mu_s, sigma_h²)         human-specific
        phi_{h,r,s}  ~ N(theta_{h,s}, sigma_r²)  human+recipe-specific

    What changed:
        - phi updates use ELBO + Adam instead of pseudo-observations.
        - theta/mu updates use ELBO + Adam instead of manual precision pooling.
        - sigma_h, sigma_r, sigma_obs are now learned (not fixed).
        - Mood confidence gate removed; all observations update phi.
        - Mood inference retained for monitoring.

    Public interface to CSP (unchanged):
        log_prob_prefer(human_id, recipe_name, spice, actor) -> float
        preferred_actor(human_id, recipe_name, spice) -> str
        get_phi / get_theta / get_mu -> float
        observe(human_id, recipe_name, spice, actor, satisfaction)
        end_episode(human_id)
    """

    def __init__(
        self,
        spices: List[str],
        recipes: Optional[List[str]] = None,
        mu0: float = 0.0,
        sigma0: float = 1.0,
        sigma_h: Optional[float] = None,
        sigma_r: Optional[float] = None,
        sigma_obs: Optional[float] = None,
        sigma_mood: Optional[float] = None,
        config: Optional[SpicesConfig] = None,
        n_phi_steps: Optional[int] = None,
        n_theta_steps: Optional[int] = None,
        lr_phi: Optional[float] = None,
        lr_theta: Optional[float] = None,
        lr_hyper: Optional[float] = None,
        enable_mood_learning: bool = True,
    ) -> None:
        self.spices = list(spices)
        self.config = config if config is not None else DEFAULT_CONFIG
        self.mu0 = mu0
        self.sigma0 = sigma0

        # Initial values for learnable hyperparams
        _sigma_h = sigma_h if sigma_h is not None else self.config.hbm.sigma_h
        _sigma_r = sigma_r if sigma_r is not None else self.config.hbm.sigma_r
        _sigma_obs = sigma_obs if sigma_obs is not None else self.config.hbm.sigma_obs

        # Stage 2: sigma_mood is fixed (not learned). Making it learnable risks
        # the psi prior collapsing toward zero, which would defeat psi's purpose.
        _sigma_mood = sigma_mood if sigma_mood is not None else self.config.hbm.sigma_mood
        self.log_sigma_mood = torch.tensor(
            math.log(_sigma_mood), dtype=torch.float32, requires_grad=False
        )

        # --- Shared learnable hyperparameters (PyTorch, require grad) ---
        # sigma_h: how much humans deviate from the global mean
        # sigma_r: how much recipes deviate from the human mean
        # sigma_obs: observation noise on satisfaction ratings
        self.log_sigma_h = torch.tensor(
            math.log(_sigma_h), dtype=torch.float32, requires_grad=True
        )
        self.log_sigma_r = torch.tensor(
            math.log(_sigma_r), dtype=torch.float32, requires_grad=True
        )
        self.log_sigma_obs = torch.tensor(
            math.log(_sigma_obs), dtype=torch.float32, requires_grad=True
        )

        # Optimization settings (defaults now come from centralized config).
        self.n_phi_steps = int(
            n_phi_steps if n_phi_steps is not None else self.config.hbm.n_phi_steps
        )
        self.n_psi_steps = int(self.config.hbm.n_psi_steps)
        self.n_theta_steps = int(
            n_theta_steps if n_theta_steps is not None else self.config.hbm.n_theta_steps
        )
        self.lr_phi = float(lr_phi if lr_phi is not None else self.config.hbm.lr_phi)
        self.lr_psi = float(self.config.hbm.lr_psi)
        self.lr_theta = float(
            lr_theta if lr_theta is not None else self.config.hbm.lr_theta
        )
        self.lr_hyper = float(
            lr_hyper if lr_hyper is not None else self.config.hbm.lr_hyper
        )
        self.log_var_min = float(self.config.hbm.log_var_min)
        self.log_var_max = float(self.config.hbm.log_var_max)
        self._enable_mood_learning = bool(enable_mood_learning)

        # Shared hyperparameter optimizer for log_sigma_h and log_sigma_r.
        # Stepped ONCE per update_theta_and_mu call (after all per-spice theta loops),
        # so each shared scalar receives one gradient step per episode regardless of
        # how many (human, spice) pairs were processed. This prevents sigma_h and
        # sigma_r from collapsing due to over-accumulation of gradient steps.
        # log_sigma_obs is NOT here — it has no obs-noise term in _elbo_theta and
        # would receive zero gradient. It is updated in _update_sigma_obs_episode.
        self._hyper_optimizer = optim.Adam(
            [self.log_sigma_h, self.log_sigma_r],
            lr=self.lr_hyper,
        )

        # --- Global mu (not variational in Stage 2; updated analytically) ---
        # Will become variational in Stage 4+ when we add the full hierarchy.
        self.mu_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.mu_var: Dict[str, float] = {s: sigma0 ** 2 for s in self.spices}

        # --- Per-human variational parameters (PyTorch tensors) ---
        # Indexed as _theta_m[human_id][spice], _theta_logv[human_id][spice]
        self._theta_m: Dict[str, Dict[str, torch.Tensor]] = {}
        self._theta_logv: Dict[str, Dict[str, torch.Tensor]] = {}

        # phi variational parameters: _phi_m[human_id][recipe][spice]
        self._phi_m: Dict[str, Any] = {}
        self._phi_logv: Dict[str, Any] = {}

        # Stage 2: psi variational parameters (scalar per human, reset each episode)
        # _psi_m[human_id], _psi_logv[human_id]
        self._psi_m: Dict[str, torch.Tensor] = {}
        self._psi_logv: Dict[str, torch.Tensor] = {}

        # Stage 3: running psi estimate (updated per-observation for mid-episode CSP adaptation).
        # Separate from _psi_m so it cannot contaminate phi's reference psi at episode end.
        # Reset to 0 at episode start, grows within episode as mood signal accumulates,
        # exposed via get_running_psi() and used in log_prob_prefer + preferred_actor.
        self._running_psi_m: Dict[str, torch.Tensor] = {}

        # Per-human episode state (plain Python)
        self._mood_posterior: Dict[str, np.ndarray] = {}
        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        # Stage 2: tracks (recipe_name, spice) pairs seen this episode for batch phi update
        self._episode_recipe_spices: Dict[str, List[Tuple[str, str]]] = {}
        self._current_recipe: Dict[str, Optional[str]] = {}
        self._phi_updated: Dict[str, bool] = {}
        self._log_lik_accum: Dict[str, np.ndarray] = {}
        self._obs_count: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._total_observations: Dict[str, int] = {}
        self._recipe_total_obs: Dict[str, Dict[str, int]] = {}
        self._episode_count: Dict[str, int] = {}

        # ELBO history for monitoring
        self._elbo_history: List[float] = []

        # Mood config (unchanged)
        self.mood_prior = np.array(self.config.mood_prior_array, dtype=float)
        self.mood_bias = self.config.get_mood_bias()
        self.base_satisfaction_bias = self.config.satisfaction.base_satisfaction_bias
        self.sat_logit_temperature = max(
            float(self.config.satisfaction.satisfaction_logit_temperature), 1e-6
        )

        # Register default human
        self.register_human(DEFAULT_HUMAN)
        if recipes:
            for r in recipes:
                self.register_recipe(DEFAULT_HUMAN, r)

    # ------------------------------------------------------------------
    # Registration (structure identical to original)
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        """Register a new human, initializing theta from current global mu."""
        if human_id in self._theta_m:
            return

        # Theta variational parameters — initialized from mu, high variance
        self._theta_m[human_id] = {
            s: torch.tensor(self.mu_mean[s], dtype=torch.float32, requires_grad=True)
            for s in self.spices
        }
        self._theta_logv[human_id] = {
            s: torch.tensor(
                math.log(math.exp(self.log_sigma_h.item()) ** 2),
                dtype=torch.float32,
                requires_grad=True,
            )
            for s in self.spices
        }

        # Phi storage (populated lazily on first recipe registration)
        self._phi_m[human_id] = defaultdict(dict)
        self._phi_logv[human_id] = defaultdict(dict)

        # Stage 2: psi initialized at zero mean, variance fixed at prior sigma_mood².
        # We only learn psi's mean (not variance) because within a single episode there
        # isn't enough data to reliably estimate both without the variance wandering into
        # regimes that corrupt the psi direction signal.
        self._psi_m[human_id] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        self._psi_logv[human_id] = torch.tensor(
            2.0 * self.log_sigma_mood.item(), dtype=torch.float32, requires_grad=False
        )
        # Stage 3: running psi starts at 0 (no mid-episode signal yet)
        self._running_psi_m[human_id] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        # Episode state
        self._mood_posterior[human_id] = self.mood_prior.copy()
        self._log_lik_accum[human_id] = np.zeros(len(MOODS))
        self._episode_data[human_id] = []
        self._episode_recipe_spices[human_id] = []
        self._current_recipe[human_id] = None
        self._phi_updated[human_id] = False
        self._obs_count[human_id] = {}
        self._total_observations[human_id] = 0
        self._recipe_total_obs[human_id] = {}
        self._episode_count[human_id] = 0

    def register_recipe(self, human_id: str, recipe_name: str) -> None:
        """Register a new recipe for a human, initializing phi from theta."""
        if human_id not in self._theta_m:
            self.register_human(human_id)
        if recipe_name in self._phi_logv.get(human_id, {}):
            return

        # Initialize phi mean from current theta mean (cold-start transfer)
        # Initialize phi log variance from sigma_r (prior uncertainty)
        for s in self.spices:
            theta_val = self._theta_m[human_id][s].item()
            log_v_init = math.log(math.exp(self.log_sigma_r.item()) ** 2)
            self._phi_m[human_id][recipe_name][s] = torch.tensor(
                theta_val, dtype=torch.float32, requires_grad=True
            )
            self._phi_logv[human_id][recipe_name][s] = torch.tensor(
                log_v_init, dtype=torch.float32, requires_grad=True
            )

        self._obs_count[human_id][recipe_name] = {s: 0 for s in self.spices}
        self._recipe_total_obs[human_id][recipe_name] = 0

    def _ensure_registered(self, human_id: str, recipe_name: str) -> None:
        if human_id not in self._theta_m:
            self.register_human(human_id)
        if recipe_name not in self._phi_logv.get(human_id, {}):
            self.register_recipe(human_id, recipe_name)

    # ------------------------------------------------------------------
    # Properties for default human (backward compatibility)
    # ------------------------------------------------------------------

    @property
    def theta_mean(self) -> Dict[str, float]:
        return {s: self._theta_m[DEFAULT_HUMAN][s].item() for s in self.spices}

    @property
    def theta_var(self) -> Dict[str, float]:
        return {
            s: math.exp(self._theta_logv[DEFAULT_HUMAN][s].item())
            for s in self.spices
        }

    @property
    def phi_mean(self) -> Any:
        """Recipe→spice phi mean dict for the default human (plain floats)."""
        out: Dict[str, Dict[str, float]] = {}
        for r in self._phi_m[DEFAULT_HUMAN]:
            out[r] = {
                s: self._phi_m[DEFAULT_HUMAN][r][s].item()
                for s in self._phi_m[DEFAULT_HUMAN][r]
            }
        return out

    @property
    def mood_posterior(self) -> np.ndarray:
        return self._mood_posterior[DEFAULT_HUMAN]

    @mood_posterior.setter
    def mood_posterior(self, value: np.ndarray) -> None:
        self._mood_posterior[DEFAULT_HUMAN] = value

    # ------------------------------------------------------------------
    # Mood inference (unchanged from original — kept for monitoring)
    # ------------------------------------------------------------------

    def _loglik_feedback_given_mood(
        self,
        human_id: str,
        actor: str,
        spice: str,
        satisfaction: float,
        recipe_name: str,
    ) -> np.ndarray:
        """Unchanged from original. Returns log P(sat | mood) for each mood."""
        phi = self._phi_m[human_id][recipe_name].get(spice, {})
        phi_val = phi.item() if isinstance(phi, torch.Tensor) else 0.0
        if abs(phi_val) < 1e-6 and spice in self._theta_m[human_id]:
            phi_val = self._theta_m[human_id][spice].item()
        phi_val = float(np.clip(phi_val, -self.base_satisfaction_bias, self.base_satisfaction_bias))

        sign_actor = 1.0 if actor == "human" else -1.0
        pref_expectation = (sign_actor * phi_val) > 0
        matches_preference = pref_expectation == (satisfaction > 0)
        pref_weight_match = self.config.mood.non_neutral_pref_weight_match
        pref_weight_mismatch = self.config.mood.non_neutral_pref_weight_mismatch
        sigma_sat = self.config.mood.satisfaction_sigma

        logits = np.zeros(3)
        for i, m in enumerate(MOODS):
            if m == "neutral":
                logits[i] = sign_actor * phi_val + self.mood_bias[m][actor]
            else:
                mood_bias_val = self.mood_bias[m][actor]
                pref_weight = pref_weight_match if matches_preference else pref_weight_mismatch
                logits[i] = mood_bias_val + sign_actor * phi_val * pref_weight

        p = 1.0 / (1.0 + np.exp(-logits))
        sat_expected = 2.0 * p - 1.0
        log_lik = -0.5 * ((satisfaction - sat_expected) / sigma_sat) ** 2
        return np.clip(
            log_lik,
            self.config.mood.satisfaction_loglik_min,
            self.config.mood.satisfaction_loglik_max,
        )

    def _update_mood_posterior(
        self,
        human_id: str,
        recipe_name: str,
        actor: str,
        spice: str,
        satisfaction: float,
    ) -> None:
        """Unchanged from original — incremental mood posterior update."""
        delta = self._loglik_feedback_given_mood(
            human_id, actor, spice, satisfaction, recipe_name
        )
        self._log_lik_accum[human_id] += delta

        prior_weight = self.config.mood.mood_prior_weight
        logps = np.log(self.mood_prior) * prior_weight + self._log_lik_accum[human_id]
        logps -= np.max(logps)
        ps = np.exp(logps)
        ps /= ps.sum()

        smoothing_alpha = self.config.mood.mood_smoothing_alpha
        ps = smoothing_alpha * ps + (1 - smoothing_alpha) * self._mood_posterior[human_id]
        ps /= ps.sum()
        self._mood_posterior[human_id] = ps

    # ------------------------------------------------------------------
    # ELBO-based phi update (replaces _pseudo_obs_weighted + _update_phi)
    # ------------------------------------------------------------------

    def _update_phi_elbo(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
    ) -> float:
        """
        Update the variational posterior q(phi_{h,r,s}) for one new observation
        using Adam gradient ascent on the ELBO.

        We maintain a sliding window of the last N_WINDOW observations for this
        (human, recipe, spice) to avoid unbounded memory growth. This is the
        standard online VI approach: use recent data to keep the local ELBO
        estimate fresh without reprocessing all history.

        Returns the final ELBO value (for monitoring).
        """
        # Accumulate this observation in episode_data (already done in observe())
        # Collect all observations for this specific spice in this recipe
        spice_obs: List[Tuple[str, float]] = [
            (act, sat)
            for act, sp, sat in self._episode_data[human_id]
            if sp == spice
        ]

        # Pre-convert observations to tensors once, outside the Adam loop.
        # String→float conversion is otherwise repeated n_phi_steps times.
        signs, sats = _obs_to_tensors(spice_obs)

        # Retrieve variational parameters (require grad)
        m_phi = self._phi_m[human_id][recipe_name][spice]
        log_v_phi = self._phi_logv[human_id][recipe_name][spice]
        m_theta = self._theta_m[human_id][spice].detach()  # treat as fixed prior

        # Stage 2: psi is inferred once per episode (in end_episode), not per-observation.
        # During the episode, phi is updated with psi held fixed at the episode-start value
        # (~0 after decay). This preserves Stage 1's phi convergence properties because
        # psi never saturates the phi gradient within an episode.
        m_psi_fixed = self._psi_m[human_id].detach()

        # log_sigma_obs is intentionally excluded here: it is updated once per episode
        # in _update_sigma_obs_episode with its own dedicated Adam optimizer. Including
        # it here would give it accumulated momentum from every (recipe, spice) pair
        # at the high phi lr, causing over-inflated obs-noise estimates.
        # log_sigma_obs still enters _elbo_phi_step as a read-only input for gradient
        # flow into log_v_phi (the phi variance correctly shrinks with more data).
        optimizer_phi = optim.Adam(
            [m_phi, log_v_phi],
            lr=self.lr_phi,
        )
        elbo_val = 0.0
        for _ in range(self.n_phi_steps):
            optimizer_phi.zero_grad()
            with torch.no_grad():
                log_v_phi.clamp_(self.log_var_min, self.log_var_max)
            elbo = _elbo_phi_step(
                signs=signs,
                sats=sats,
                m_phi=m_phi,
                log_v_phi=log_v_phi,
                m_psi_fixed=m_psi_fixed,
                m_theta=m_theta,
                log_sigma_r=self.log_sigma_r.detach(),
                log_sigma_obs=self.log_sigma_obs.detach(),
                sat_logit_temperature=self.sat_logit_temperature,
            )
            (-elbo).backward()
            optimizer_phi.step()
            elbo_val = elbo.item()

        # Write back phi (psi is written back in end_episode after batch inference)
        self._phi_m[human_id][recipe_name][spice] = m_phi
        self._phi_logv[human_id][recipe_name][spice] = log_v_phi

        return elbo_val

    # ------------------------------------------------------------------
    # Hierarchical pooling: theta and mu (ELBO-based)
    # ------------------------------------------------------------------

    def update_theta_and_mu(self) -> None:
        """
        Update q(theta_{h,s}) and mu_s at end of episode.

        For each (human, spice):
          - Collect phi posteriors from all recipes with observations.
          - Run N_THETA_STEPS Adam steps on the theta ELBO treating phi
            posteriors as noisy observations of theta.
          - Jointly update log_sigma_h (how much humans vary from mu).

        Then update mu_s analytically as the precision-weighted mean of
        all theta posteriors (same as original — mu is not variational yet).
        """
        registered_humans = list(self._theta_m.keys())

        # Count observed (human, spice) pairs for gradient normalization below.
        n_active = sum(
            1
            for h in registered_humans
            for s in self.spices
            if any(
                self._obs_count[h].get(r, {}).get(s, 0) > 0
                for r in self._phi_logv.get(h, {})
            )
        )
        # Avoid division by zero; fall back to 1 if nothing observed yet.
        n_active = max(n_active, 1)

        self._hyper_optimizer.zero_grad()

        for h in registered_humans:
            for s in self.spices:
                # Collect phi posteriors from recipes with actual observations
                phi_posteriors: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for r in self._phi_logv.get(h, {}):
                    if self._obs_count[h].get(r, {}).get(s, 0) > 0:
                        m_phi = self._phi_m[h][r][s].detach()
                        log_v_phi = self._phi_logv[h][r][s].detach()
                        phi_posteriors.append((m_phi, log_v_phi))

                if not phi_posteriors:
                    continue

                m_theta = self._theta_m[h][s]
                log_v_theta = self._theta_logv[h][s]

                optimizer = optim.Adam(
                    [m_theta, log_v_theta],
                    lr=self.lr_theta,
                )

                for _ in range(self.n_theta_steps):
                    optimizer.zero_grad()
                    with torch.no_grad():
                        log_v_theta.clamp_(self.log_var_min, self.log_var_max)

                    elbo = _elbo_theta(
                        phi_posteriors=phi_posteriors,
                        m_theta=m_theta,
                        log_v_theta=log_v_theta,
                        mu=self.mu_mean[s],
                        log_sigma_h=self.log_sigma_h,
                        log_sigma_r=self.log_sigma_r,
                    )
                    # Normalize by number of active (human, spice) pairs so the
                    # gradient magnitude on shared log_sigma_h / log_sigma_r is
                    # independent of vocabulary size. Without this, a 94-spice
                    # vocabulary produces ~94x larger hyperparameter gradients than
                    # a 5-spice vocabulary, causing sigma_r to collapse.
                    (-elbo / n_active).backward()
                    optimizer.step()

                self._theta_m[h][s] = m_theta
                self._theta_logv[h][s] = log_v_theta

        # Single hyperparameter step after all (human, spice) theta updates.
        self._hyper_optimizer.step()

        # Update mu analytically (precision-weighted mean of theta posteriors)
        # This is the same as the original implementation.
        # mu will become variational in Stage 4.
        for s in self.spices:
            theta_means: List[float] = []
            theta_precisions: List[float] = []
            for h in registered_humans:
                if s in self._theta_m[h]:
                    m = self._theta_m[h][s].item()
                    v = math.exp(self._theta_logv[h][s].item())
                    theta_means.append(m)
                    theta_precisions.append(1.0 / v)

            if not theta_means:
                continue

            total_prec = sum(theta_precisions)
            weighted_sum = sum(p * m for p, m in zip(theta_precisions, theta_means))
            prior_prec = 1.0 / (self.sigma0 ** 2)

            post_var = 1.0 / (prior_prec + total_prec)
            post_mean = post_var * (self.mu0 * prior_prec + weighted_sum)

            self.mu_mean[s] = post_mean
            self.mu_var[s] = post_var

    def flush_theta_mu(self) -> None:
        """Force immediate theta/mu update. Call before eval."""
        self.update_theta_and_mu()

    def set_theta(
        self,
        human_id: str,
        theta_dict: Dict[str, float],
        sigma_h: Optional[float] = None,
    ) -> None:
        """
        Directly set theta values for a human (used to construct hidden/ground-truth HBMs).

        Overwrites _theta_m tensors with the given float values.
        If sigma_h is provided, also sets the theta variance (log_v_theta = 2*log(sigma_h)).
        This is the Stage 1 replacement for directly assigning _theta_mean/_theta_var dicts.
        """
        if human_id not in self._theta_m:
            self.register_human(human_id)
        for s, v in theta_dict.items():
            if s in self._theta_m[human_id]:
                self._theta_m[human_id][s] = torch.tensor(
                    float(v), dtype=torch.float32, requires_grad=True
                )
                if sigma_h is not None:
                    self._theta_logv[human_id][s] = torch.tensor(
                        2.0 * math.log(max(sigma_h, 1e-6)),
                        dtype=torch.float32,
                        requires_grad=True,
                    )

    def set_phi(
        self,
        human_id: str,
        recipe_name: str,
        phi_dict: Dict[str, float],
    ) -> None:
        """
        Directly set phi mean values for a (human, recipe) pair.

        Used to initialise per-recipe ground-truth theta overrides in the hidden HBM
        (Option A recipe-conflict configs).  Only the spices present in phi_dict are
        updated; other spices keep their theta-initialised values from register_recipe.
        """
        self._ensure_registered(human_id, recipe_name)
        for s, v in phi_dict.items():
            if s in self._phi_m.get(human_id, {}).get(recipe_name, {}):
                self._phi_m[human_id][recipe_name][s] = torch.tensor(
                    float(v), dtype=torch.float32, requires_grad=True
                )

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def observe(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        force_neutral_mood: bool = False,  # kept for interface compat; no longer gates phi
    ) -> None:
        """
        Buffer one observation for batch processing at episode end.

        Stage 2 change: phi is no longer updated per-observation. Instead, all
        phi updates are deferred to end_episode() where psi is inferred first
        (with phi fixed at pre-episode values), and then phi is updated with psi
        fixed at the newly inferred value. This achieves mood absorption:
        a contradictory session is explained by psi, leaving phi unchanged.
        """
        self._ensure_registered(human_id, recipe_name)
        self._episode_data[human_id].append((actor, spice, satisfaction))
        self._episode_recipe_spices[human_id].append((recipe_name, spice))
        self._current_recipe[human_id] = recipe_name
        self._obs_count[human_id][recipe_name][spice] += 1
        self._total_observations[human_id] += 1
        self._recipe_total_obs[human_id][recipe_name] = (
            self._recipe_total_obs[human_id].get(recipe_name, 0) + 1
        )

        # Mood posterior (kept for monitoring)
        if force_neutral_mood:
            neutral_idx = MOODS.index("neutral")
            self._mood_posterior[human_id] = np.zeros(3)
            self._mood_posterior[human_id][neutral_idx] = 1.0
        elif self._enable_mood_learning:
            self._update_mood_posterior(human_id, recipe_name, actor, spice, satisfaction)

        # Stage 3: update running psi estimate after each observation.
        # This enables mid-episode CSP adaptation: log_prob_prefer uses phi + running_psi,
        # so if a mood is detected mid-episode the CSP shifts its assignments immediately.
        if self._enable_mood_learning:
            self._update_running_psi(human_id)

    def _update_running_psi(self, human_id: str) -> None:
        """
        Incrementally update the running psi estimate using all episode data so far.

        Called after each observation in observe(). Provides a per-step psi signal
        that the CSP uses via log_prob_prefer(phi + running_psi). Unlike _psi_m
        (the batch estimate used by phi updates at episode end), this running estimate
        is kept separate so it cannot contaminate phi's reference psi in _update_phi_episode.

        Uses N_RUNNING_PSI_STEPS = n_phi_steps // 2 Adam steps (fewer than the batch
        estimate since the running estimate is updated frequently and converges incrementally).
        """
        all_obs: List[Tuple[str, float]] = [
            (act, sat) for act, _sp, sat in self._episode_data[human_id]
        ]
        if not all_obs:
            return

        signs, sats = _obs_to_tensors(all_obs)
        # Build per-observation phi anchor using the correct (recipe, spice) pair.
        # _episode_recipe_spices is parallel to _episode_data (both appended in observe()).
        phi_anchor_list: List[float] = []
        for recipe_name, sp in self._episode_recipe_spices[human_id]:
            phi_val = 0.0
            if (recipe_name and human_id in self._phi_m
                    and recipe_name in self._phi_m[human_id]
                    and sp in self._phi_m[human_id][recipe_name]):
                phi_val = self._phi_m[human_id][recipe_name][sp].item()
            phi_anchor_list.append(phi_val)
        phi_anchors = torch.tensor(phi_anchor_list, dtype=torch.float32)

        m_psi_running = self._running_psi_m[human_id]
        n_steps = max(1, self.n_psi_steps // 4)  # running estimate: fewer steps (called per-obs)
        optimizer = optim.Adam([m_psi_running], lr=self.lr_phi)
        for _ in range(n_steps):
            optimizer.zero_grad()
            elbo = _elbo_psi_step(
                signs=signs,
                sats=sats,
                phi_anchors=phi_anchors,
                m_psi=m_psi_running,
                log_sigma_mood=self.log_sigma_mood,
                log_sigma_obs=self.log_sigma_obs.detach(),
                sat_logit_temperature=self.sat_logit_temperature,
            )
            (-elbo).backward()
            optimizer.step()
        self._running_psi_m[human_id] = m_psi_running

    def _update_psi_episode(self, human_id: str) -> None:
        """
        Batch-infer psi from all episode observations, with phi posteriors fixed.

        Called once at episode end. Collects all (actor, satisfaction) pairs from
        the episode across ALL spices, then runs n_psi_steps Adam steps on m_psi.

        Using all spice observations jointly gives a better psi estimate than
        per-observation updates, and avoids saturating phi's per-observation gradient.

        Neutral-episode gate: if the mood posterior is already confident that this
        episode is neutral (neutral_prob >= psi_skip_neutral_threshold), skip the
        ELBO entirely.  On neutral episodes psi_true ≈ 0 and the KL would pull
        m_psi back to 0 anyway, but the MC likelihood gradient introduces noise
        that biases phi's subsequent update.  Skipping keeps m_psi ≈ 0 cleanly and
        gives phi an uncontaminated signal.
        """
        all_obs: List[Tuple[str, float]] = [
            (act, sat) for act, _sp, sat in self._episode_data[human_id]
        ]
        if not all_obs:
            return

        # Gate: skip if the episode is confidently neutral.
        threshold = self.config.hbm.psi_skip_neutral_threshold
        mood_post = self._mood_posterior.get(human_id)
        if mood_post is not None:
            neutral_idx = MOODS.index("neutral")
            if float(mood_post[neutral_idx]) >= threshold:
                return

        signs, sats = _obs_to_tensors(all_obs)
        # Build per-observation phi anchor using the correct (recipe, spice) pair.
        # _episode_recipe_spices is parallel to _episode_data (both appended in observe()).
        phi_anchor_list: List[float] = []
        for recipe_name, sp in self._episode_recipe_spices[human_id]:
            phi_val = 0.0
            if (recipe_name and human_id in self._phi_m
                    and recipe_name in self._phi_m[human_id]
                    and sp in self._phi_m[human_id][recipe_name]):
                phi_val = self._phi_m[human_id][recipe_name][sp].item()
            phi_anchor_list.append(phi_val)
        phi_anchors = torch.tensor(phi_anchor_list, dtype=torch.float32)

        m_psi = self._psi_m[human_id]
        optimizer_psi = optim.Adam([m_psi], lr=self.lr_psi)
        for _ in range(self.n_psi_steps):
            optimizer_psi.zero_grad()
            elbo = _elbo_psi_step(
                signs=signs,
                sats=sats,
                phi_anchors=phi_anchors,
                m_psi=m_psi,
                log_sigma_mood=self.log_sigma_mood,
                log_sigma_obs=self.log_sigma_obs.detach(),
                sat_logit_temperature=self.sat_logit_temperature,
            )
            (-elbo).backward()
            optimizer_psi.step()

        self._psi_m[human_id] = m_psi

    def _update_phi_episode(self, human_id: str) -> None:
        """
        Batch update phi for all (recipe, spice) pairs observed this episode.

        Called after _update_psi_episode() so self._psi_m already holds the
        episode's inferred mood offset. _update_phi_elbo() reads psi via
        self._psi_m[human_id].detach(), so phi learns only from the residual
        signal after mood has been subtracted.

        This is coordinate ascent Phase 2: phi update conditioned on inferred psi.
        """
        seen_pairs = set(self._episode_recipe_spices.get(human_id, []))
        for recipe_name, spice in seen_pairs:
            elbo_val = self._update_phi_elbo(human_id, recipe_name, spice, "", 0.0)
            self._elbo_history.append(elbo_val)

    def _update_sigma_obs_episode(self, human_id: str) -> None:
        """
        Update log_sigma_obs once per episode using all episode observations.

        log_sigma_obs is excluded from optimizer_phi (to avoid accumulating momentum
        from O(n_spices * n_steps) updates per episode) and excluded from _hyper_optimizer
        (which uses _elbo_theta, which has no obs-noise term). This dedicated pass runs
        a few Adam steps on log_sigma_obs alone, with phi and psi fixed at their
        post-update values, so sigma_obs can track the true observation noise level.
        """
        all_obs: List[Tuple[str, float]] = [
            (act, sat) for act, _sp, sat in self._episode_data[human_id]
        ]
        if not all_obs:
            return

        signs, sats = _obs_to_tensors(all_obs)
        phi_anchor_list: List[float] = []
        for recipe_name, sp in self._episode_recipe_spices[human_id]:
            phi_val = 0.0
            if (recipe_name and human_id in self._phi_m
                    and recipe_name in self._phi_m[human_id]
                    and sp in self._phi_m[human_id][recipe_name]):
                phi_val = self._phi_m[human_id][recipe_name][sp].item()
            phi_anchor_list.append(phi_val)
        phi_anchors = torch.tensor(phi_anchor_list, dtype=torch.float32)
        m_psi_fixed = self._psi_m[human_id].detach()

        # Limit to 4 steps so sigma_obs adapts slowly — too many steps causes
        # it to collapse to near-zero, which flattens the phi ELBO gradient and
        # freezes learning after ~500 training steps.
        # Also clamp from below: Beta noise floor std ≈ 1/(2*sqrt(kappa)) ≈ 0.16
        # for kappa=10, so sigma_obs < 0.25 is over-fit.
        log_sigma_obs_min = math.log(0.25)
        optimizer_obs = optim.Adam([self.log_sigma_obs], lr=self.lr_hyper)
        for _ in range(4):
            optimizer_obs.zero_grad()
            with torch.no_grad():
                self.log_sigma_obs.clamp_(min=log_sigma_obs_min)
            sigma_obs = torch.exp(self.log_sigma_obs)
            logits = signs * (phi_anchors + m_psi_fixed)
            temp = max(self.sat_logit_temperature, 1e-6)
            expected_sat = torch.tanh(logits / temp)
            log_p_sat = (
                -0.5 * ((sats - expected_sat) / sigma_obs) ** 2
                - self.log_sigma_obs
                - 0.5 * math.log(2 * math.pi)
            )
            ell = log_p_sat.sum()
            (-ell).backward()
            optimizer_obs.step()
        with torch.no_grad():
            self.log_sigma_obs.clamp_(min=log_sigma_obs_min)

    def end_episode(self, human_id: str, neutral_threshold: float = 0.5) -> None:
        """
        End-of-episode update.

        Changes from original:
          - No neutral_threshold gate (parameter kept for interface compat).
          - Calls update_theta_and_mu unconditionally (phi was updated online).
          - Resets episode state.

        The neutral_threshold parameter is intentionally kept so that any
        existing callsites (tests, CSP code) continue to work unchanged.
        """
        self._episode_count[human_id] = self._episode_count.get(human_id, 0) + 1
        self._phi_updated[human_id] = True  # always updated now

        # Stage 2: coordinate ascent at episode end.
        # Phase 1 (psi): infer psi from all episode observations with phi fixed.
        #   Psi gets first credit for the session-level mood offset.
        # Phase 2 (phi): update phi conditioned on the inferred psi.
        #   Phi only learns from the residual after psi has been subtracted,
        #   preventing mood from corrupting the persistent preference signal.
        # This ordering is correct: psi must be identified before phi can learn,
        # because phi is updated with m_psi_fixed = self._psi_m (current inferred value).
        if self._enable_mood_learning:
            self._update_psi_episode(human_id)
        self._update_phi_episode(human_id)
        self._update_sigma_obs_episode(human_id)

        batch_size = self.config.hbm.update_theta_mu_every_n_episodes
        if batch_size <= 1 or self._episode_count[human_id] % batch_size == 0:
            self.update_theta_and_mu()

        # Stage 2: aggressive psi mean decay (log_v_psi is fixed, no reset needed).
        # Persistent signals accumulate in phi; transient signals cannot persist in psi.
        if self._enable_mood_learning:
            psi_decay = self.config.hbm.psi_decay
            self._psi_m[human_id] = torch.tensor(
                self._psi_m[human_id].item() * psi_decay,
                dtype=torch.float32, requires_grad=True,
            )
        else:
            self._psi_m[human_id] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

        # Reset episode state
        self._episode_data[human_id] = []
        self._episode_recipe_spices[human_id] = []
        self._log_lik_accum[human_id] = np.zeros(len(MOODS))
        self._mood_posterior[human_id] = self.mood_prior.copy()
        self._current_recipe[human_id] = None
        # Stage 3: reset running psi — next episode starts with no mid-episode signal
        self._running_psi_m[human_id] = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

    def observe_eval(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        done: bool,
    ) -> None:
        """
        Eval-time psi adaptation: accumulate observations and update running_psi only.

        Does NOT touch phi, theta, mu, obs counts, or any training state.
        Enables the CSP to adapt mid-episode to mood signals at eval time via
        log_prob_prefer(phi + running_psi), without any learning side-effects.

        At done=True, resets _episode_data and running_psi to avoid bleeding
        into the next eval episode.
        """
        if not self._enable_mood_learning:
            return
        self._ensure_registered(human_id, recipe_name)
        self._episode_data[human_id].append((actor, spice, satisfaction))
        self._episode_recipe_spices[human_id].append((recipe_name, spice))
        self._current_recipe[human_id] = recipe_name
        self._update_running_psi(human_id)
        if done:
            self._episode_data[human_id] = []
            self._episode_recipe_spices[human_id] = []
            self._running_psi_m[human_id] = torch.tensor(
                0.0, dtype=torch.float32, requires_grad=True
            )

    # ------------------------------------------------------------------
    # Public update (kept for backward compat with any direct callers)
    # ------------------------------------------------------------------

    def update_phi(
        self, human_id: str, recipe_name: str, spice: str, g: float
    ) -> None:
        """
        Backward-compatible direct phi update.
        In Stage 1 this converts g back to a synthetic observation and
        runs the ELBO update. Callers should prefer observe() directly.
        g > 0 is treated as actor=human, g < 0 as actor=robot.
        satisfaction is set to |g| / base_satisfaction_bias, clipped to [0,1].
        """
        self._ensure_registered(human_id, recipe_name)
        actor = "human" if g >= 0 else "robot"
        sat = float(np.clip(abs(g) / max(self.base_satisfaction_bias, 1e-6), 0.0, 1.0))
        # Add a synthetic entry so _update_phi_elbo has data to work with
        self._episode_data[human_id].append((actor, spice, sat))
        self._update_phi_elbo(human_id, recipe_name, spice, actor, sat)

    # ------------------------------------------------------------------
    # Getters (all unchanged from original — CSP interface preserved)
    # ------------------------------------------------------------------

    def get_phi(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Return phi posterior mean. Returns 0.0 if unregistered."""
        try:
            return self._phi_m[human_id][recipe_name][spice].item()
        except (KeyError, AttributeError):
            return 0.0

    def get_phi_var(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Return phi posterior variance. New in Stage 1 — used by CSP for entropy."""
        try:
            return math.exp(self._phi_logv[human_id][recipe_name][spice].item())
        except (KeyError, AttributeError):
            return math.exp(self.log_sigma_r.item()) ** 2

    def get_phi_entropy(self, human_id: str, recipe_name: str, spice: str) -> float:
        """
        Return the variance-weighted Bernoulli entropy for a spice preference.

        Exploration value = H(Bernoulli(sigmoid(phi_mean))) * phi_var

        Stage 3 interpretation:
          - H(B(sigma(phi))) measures how informative a new observation would be
            given the current phi estimate: maximized at phi=0, zero at |phi|→∞.
          - phi_var scales the exploration value by our residual uncertainty:
            if the posterior is tight (var≈0), the model is already converged
            and no exploration is needed even if phi happens to be near zero.
          - The product naturally transitions from high (uncertain, unexplored)
            to low (confident, exploitable) as episodes accumulate, without
            any explicit annealing schedule.

        Returns a float >= 0. Larger values mean more exploration benefit.
        """
        phi_mean = self.get_phi(human_id, recipe_name, spice)
        phi_var = self.get_phi_var(human_id, recipe_name, spice)
        # log P(human | phi_mean) for bernoulli_entropy — same as log_prob_prefer(human)
        logit = phi_mean
        # log sigmoid(logit) — numerically stable form
        log_p_human = -math.log1p(math.exp(-abs(logit))) - max(0.0, -logit)
        p_human = math.exp(log_p_human)
        p_robot = 1.0 - p_human
        # H(Bernoulli) in nats; clamp probs away from 0/1 for numerical safety
        p_human = max(p_human, 1e-10)
        p_robot = max(p_robot, 1e-10)
        H = -p_human * math.log(p_human) - p_robot * math.log(p_robot)
        return H * phi_var

    def get_theta(self, human_id: str, spice: str) -> float:
        return self._theta_m[human_id][spice].item()

    def get_theta_var(self, human_id: str, spice: str) -> float:
        """New in Stage 1 — used for cold-start initialization of phi."""
        return math.exp(self._theta_logv[human_id][spice].item())

    def get_mu(self, spice: str) -> float:
        return self.mu_mean[spice]

    def get_mu_var(self, spice: str) -> float:
        return self.mu_var[spice]

    def get_psi_m(self, human_id: str) -> float:
        """Return the batch psi posterior mean (inferred at episode end, decays between episodes)."""
        try:
            return self._psi_m[human_id].item()
        except KeyError:
            return 0.0

    def get_running_psi(self, human_id: str) -> float:
        """
        Return the mid-episode running psi estimate for a human.

        Stage 3: updated after each observation in observe(). Within an episode
        this accumulates the inferred session offset from observations seen so far,
        giving the CSP a real-time view of psi without waiting for episode end.
        Resets to 0 at the start of each new episode.
        """
        if not self._enable_mood_learning:
            return 0.0
        try:
            return self._running_psi_m[human_id].item()
        except KeyError:
            return 0.0

    def preference_posterior(
        self, human_id: str, recipe_name: str, spice: str
    ) -> tuple:
        """
        Return (phi_mean, phi_var) for a spice preference.

        Stage 3 interface: the CSP uses mean for exploitation and var (via
        get_phi_entropy) for exploration. Both are plain Python floats.
        """
        return self.get_phi(human_id, recipe_name, spice), self.get_phi_var(human_id, recipe_name, spice)

    def preferred_actor(self, human_id: str, recipe_name: str, spice: str) -> str:
        """
        Returns 'human' if phi + running_psi >= 0, else 'robot'.

        Stage 3: includes the running psi so mid-episode mood signals shift the
        preferred actor without waiting for episode end.
        """
        phi = self.get_phi(human_id, recipe_name, spice)
        psi = self.get_running_psi(human_id)
        return "human" if (phi + psi) >= 0 else "robot"

    def log_prob_prefer(
        self, human_id: str, recipe_name: str, spice: str, actor: str
    ) -> float:
        """
        Returns log P(actor | phi_mean + running_psi).

        Stage 3: uses phi + running_psi as the effective logit so the CSP
        adapts mid-episode as mood signals accumulate. Outside of an episode
        running_psi = 0 so behavior is identical to the Stage 1/2 version.
        """
        phi = self.get_phi(human_id, recipe_name, spice)
        psi = self.get_running_psi(human_id)
        sign_actor = 1.0 if actor == "human" else -1.0
        logit = sign_actor * (phi + psi)
        log_p = -math.log1p(math.exp(-abs(logit))) - max(0.0, -logit)
        return float(max(log_p, -20.0))   # floor for numerical stability

    def sample_episode_preferences(
        self,
        recipe_spices: List[str],
        rng: np.random.Generator,
        human_id: str = DEFAULT_HUMAN,
        stochastic: bool = False,
        recipe_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Return the preferred actor for each spice.

        If recipe_name is provided and per-recipe phi overrides were set via set_phi
        (Option A recipe-conflict configs), those override the global theta for the
        spices in that recipe.  This gives the hidden HBM true recipe-level ground
        truth that the learning HBM's per-recipe phi hierarchy must track.

        stochastic=False (default): uses the effective theta/phi mean sign directly.
        stochastic=True: samples phi ~ N(mean, sigma_r²) and uses its sign.
          Borderline spices flip across episodes, exposing CBTL's inability to track
          per-recipe uncertainty (it averages flipped labels to ~0).

        Spices with no registered theta default to "robot" (neutral / no preference).
        """
        sigma_r = math.exp(self.log_sigma_r.item())
        preferences: Dict[str, str] = {}
        for spice in recipe_spices:
            if spice not in self._theta_m.get(human_id, {}):
                preferences[spice] = "robot"
                continue
            # Use per-recipe phi override if available (Option A conflict config).
            phi_recipes = self._phi_m.get(human_id, {})
            if (recipe_name and recipe_name in phi_recipes
                    and spice in phi_recipes[recipe_name]):
                mean_val = phi_recipes[recipe_name][spice].item()
            else:
                mean_val = self._theta_m[human_id][spice].item()
            if stochastic:
                sampled = float(rng.normal(mean_val, sigma_r))
                preferred_actor = "human" if sampled >= 0 else "robot"
            else:
                preferred_actor = "human" if mean_val >= 0 else "robot"
            preferences[spice] = preferred_actor
        return preferences

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_elbo_snapshot(
        self, human_id: str, recipe_name: str, spice: str
    ) -> float:
        """
        Compute the current ELBO for one (human, recipe, spice) cell.
        Uses all episode data seen so far for this spice.
        Useful for monitoring convergence in tests.
        """
        spice_obs: List[Tuple[str, float]] = [
            (act, sat)
            for act, sp, sat in self._episode_data[human_id]
            if sp == spice
        ]
        if not spice_obs or recipe_name not in self._phi_m.get(human_id, {}):
            return float("nan")

        signs, sats = _obs_to_tensors(spice_obs)
        with torch.no_grad():
            elbo = _elbo_phi(
                signs=signs,
                sats=sats,
                m_phi=self._phi_m[human_id][recipe_name][spice],
                log_v_phi=self._phi_logv[human_id][recipe_name][spice],
                m_psi=self._psi_m[human_id],
                log_v_psi=self._psi_logv[human_id],
                m_theta=self._theta_m[human_id][spice],
                log_sigma_r=self.log_sigma_r,
                log_sigma_obs=self.log_sigma_obs,
                log_sigma_mood=self.log_sigma_mood,
                sat_logit_temperature=self.sat_logit_temperature,
            )
        return elbo.item()

    def get_psi(self, human_id: str) -> float:
        """Return current session psi mean. Near 0 between episodes after reset."""
        try:
            return self._psi_m[human_id].item()
        except KeyError:
            return 0.0

    def get_psi_var(self, human_id: str) -> float:
        """Return current session psi variance."""
        try:
            return math.exp(self._psi_logv[human_id].item())
        except KeyError:
            return math.exp(self.log_sigma_mood.item()) ** 2

    def get_learned_sigmas(self) -> Dict[str, float]:
        """Return current learned hyperparameter values. Useful for monitoring."""
        return {
            "sigma_h":    math.exp(self.log_sigma_h.item()),
            "sigma_r":    math.exp(self.log_sigma_r.item()),
            "sigma_obs":  math.exp(self.log_sigma_obs.item()),
            "sigma_mood": math.exp(self.log_sigma_mood.item()),
        }