"""
overcooked_hbm.py — Hierarchical Bayesian Preference Model for Overcooked.

Closely mirrors spices_hbm.py (Stage 3) with one key extension: the
per-episode session-offset variable psi is a *vector* rather than a scalar.

Scalar psi (spices): one offset shared across all spices in an episode.
Vector psi (overcooked): one offset per subtask dimension, e.g.
    psi[0] = session effect on "fetch_onion" preference
    psi[1] = session effect on "fetch_dish" preference
    ...

This matters in Overcooked because fatigue or frustration may affect certain
task types disproportionately (e.g. a tired human dislikes delivery but is
fine with chopping). A scalar psi cannot capture this; a vector psi over
subtask categories can.

Architecture
------------
Level 1 (global):     mu_s ~ N(0, sigma0²)
Level 2 (human):      theta_{h,s} ~ N(mu_s, sigma_h²)
Level 3 (layout):     phi_{h,L,s} ~ N(theta_{h,s}, sigma_r²)
Session (vector):     psi_{h,sess} ~ N(0, diag(sigma_session²))  ← vector

Likelihood per observation (subtask s, actor a, task_score y):
    log p(a, y | phi, psi) =
        log sigmoid(sign(a) * (phi_s + psi_s))          [Bernoulli]
      + log N(y; tanh(sign*(phi_s + psi_s)), sigma_obs²) [Gaussian]

The only difference from the scalar case in the ELBO is that psi_offset in
_ell_phi_psi receives the scalar psi[s] for each observation's subtask index
rather than a single shared psi.

Public interface to CSP (identical to spices HBM):
    log_prob_prefer(human_id, layout_name, subtask, actor) -> float
    preferred_actor(human_id, layout_name, subtask) -> str
    preference_posterior(human_id, layout_name, subtask) -> (mean, var)
    get_phi_entropy(human_id, layout_name, subtask) -> float
    observe(human_id, layout_name, subtask, actor, task_score)
    end_episode(human_id)

All PyTorch is used only inside update methods.
All getters and CSP interface return plain Python floats.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from .config.overcooked_config import DEFAULT_CONFIG, OvercookedConfig
from .layouts import ALL_SUBTASKS

DEFAULT_HUMAN = "human"


# ---------------------------------------------------------------------------
# Module-level runtime constants (from centralized config)
# ---------------------------------------------------------------------------

_HBM_CFG = DEFAULT_CONFIG.hbm
N_MC_SAMPLES: int = _HBM_CFG.n_mc_samples
N_PHI_STEPS: int = _HBM_CFG.n_phi_steps
N_THETA_STEPS: int = _HBM_CFG.n_theta_steps
LR_PHI: float = _HBM_CFG.lr_phi
LR_THETA: float = _HBM_CFG.lr_theta
LR_HYPER: float = _HBM_CFG.lr_hyper
LOG_VAR_MIN: float = _HBM_CFG.log_var_min
LOG_VAR_MAX: float = _HBM_CFG.log_var_max


# ---------------------------------------------------------------------------
# ELBO computation (pure PyTorch, no side effects)
# ---------------------------------------------------------------------------

def _gaussian_kl(
    m_q: torch.Tensor,
    log_v_q: torch.Tensor,
    m_p: torch.Tensor,
    log_v_p: torch.Tensor,
) -> torch.Tensor:
    """KL(N(m_q, exp(log_v_q)) || N(m_p, exp(log_v_p))). Shapes must broadcast."""
    v_q = torch.exp(log_v_q)
    v_p = torch.exp(log_v_p)
    return 0.5 * (log_v_p - log_v_q + v_q / v_p + (m_q - m_p) ** 2 / v_p - 1.0)


def _obs_to_tensors(
    observations: List[Tuple[str, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert [(actor, task_score)] to (signs [T], scores [T]) tensors."""
    signs = torch.tensor(
        [1.0 if a == "human" else -1.0 for a, _ in observations],
        dtype=torch.float32,
    )
    scores = torch.tensor([s for _, s in observations], dtype=torch.float32)
    return signs, scores


def _ell_phi_psi(
    signs: torch.Tensor,        # [T]
    scores: torch.Tensor,       # [T]
    phi_samples: torch.Tensor,  # [N_MC]
    psi_offset: torch.Tensor,   # [N_MC] or scalar
    log_sigma_obs: torch.Tensor,
    task_logit_temperature: float,
) -> torch.Tensor:
    """
    Expected log-likelihood shared by phi and psi ELBO steps.
    logit[t, k] = sign_t * (phi_k + psi_k), shape [T, N_MC].
    """
    logits = signs.unsqueeze(1) * (
        phi_samples.unsqueeze(0) + psi_offset.unsqueeze(0)
    )
    sigma_obs = torch.exp(log_sigma_obs)
    log_p_actor = torch.nn.functional.logsigmoid(logits)
    temp = max(float(task_logit_temperature), 1e-6)
    expected_score = torch.tanh(logits / temp)
    log_p_score = (
        -0.5 * ((scores.unsqueeze(1) - expected_score) / sigma_obs) ** 2
        - log_sigma_obs
        - 0.5 * math.log(2 * math.pi)
    )
    return (log_p_actor + log_p_score).sum(dim=0).mean()


def _elbo_phi_step(
    signs: torch.Tensor,
    scores: torch.Tensor,
    m_phi: torch.Tensor,
    log_v_phi: torch.Tensor,
    psi_fixed: torch.Tensor,      # scalar — psi[subtask_dim] held fixed
    m_theta: torch.Tensor,
    log_sigma_r: torch.Tensor,
    log_sigma_obs: torch.Tensor,
    task_logit_temperature: float,
) -> torch.Tensor:
    """ELBO for phi update (coordinate ascent). Psi is held deterministic."""
    if signs.numel() == 0:
        return torch.tensor(0.0)
    eps_phi = torch.randn(N_MC_SAMPLES)
    phi_samples = m_phi + torch.exp(0.5 * log_v_phi) * eps_phi
    ell = _ell_phi_psi(
        signs,
        scores,
        phi_samples,
        psi_fixed.expand(N_MC_SAMPLES),
        log_sigma_obs,
        task_logit_temperature,
    )
    kl_phi = _gaussian_kl(m_phi, log_v_phi, m_theta, 2.0 * log_sigma_r)
    return ell - kl_phi


def _elbo_psi_vec_step(
    signs_by_dim: List[torch.Tensor],   # [D] each [T_d]
    scores_by_dim: List[torch.Tensor],  # [D] each [T_d]
    phi_anchors_by_dim: List[torch.Tensor],  # [D] each [T_d]
    m_psi: torch.Tensor,                # [D]
    log_sigma_session: torch.Tensor,    # [D] fixed, no grad
    log_sigma_obs: torch.Tensor,
    task_logit_temperature: float,
) -> torch.Tensor:
    """
    ELBO for the vector-psi update (coordinate ascent).

    For each subtask dimension d, we compute the per-dimension ELL using the
    scalar m_psi[d] and the observations that belong to subtask d.  The KL
    per dimension is identical to the scalar case (q_var == p_var = sigma_session²).

    ELBO_psi = sum_d [ E_{psi_d}[ sum_t log p(y_t^d | phi_t^d, psi_d) ]
                       - KL(q(psi_d) || N(0, sigma_session_d²)) ]
    """
    total_ell = torch.tensor(0.0)
    total_kl = torch.tensor(0.0)

    D = m_psi.shape[0]
    sigma_session = torch.exp(log_sigma_session)  # [D]

    for d in range(D):
        signs_d = signs_by_dim[d]
        scores_d = scores_by_dim[d]
        phi_d = phi_anchors_by_dim[d]

        if signs_d.numel() == 0:
            continue

        sigma_d = sigma_session[d]
        eps_psi = torch.randn(N_MC_SAMPLES)
        psi_d_samples = m_psi[d] + sigma_d * eps_psi  # [N_MC]

        logits = signs_d.unsqueeze(1) * (phi_d.unsqueeze(1) + psi_d_samples.unsqueeze(0))
        log_p_actor = torch.nn.functional.logsigmoid(logits)
        temp = max(float(task_logit_temperature), 1e-6)
        expected_score = torch.tanh(logits / temp)
        sigma_obs = torch.exp(log_sigma_obs)
        log_p_score = (
            -0.5 * ((scores_d.unsqueeze(1) - expected_score) / sigma_obs) ** 2
            - log_sigma_obs
            - 0.5 * math.log(2 * math.pi)
        )
        ell_d = (log_p_actor + log_p_score).sum(dim=0).mean()
        total_ell = total_ell + ell_d

        # KL when q_var == p_var == sigma_session_d²: 0.5 * m_psi_d² / sigma_session_d²
        kl_d = 0.5 * (m_psi[d] / sigma_d) ** 2
        total_kl = total_kl + kl_d

    return total_ell - total_kl


def _elbo_theta(
    phi_posteriors: List[Tuple[torch.Tensor, torch.Tensor]],
    m_theta: torch.Tensor,
    log_v_theta: torch.Tensor,
    mu: float,
    log_sigma_h: torch.Tensor,
    log_sigma_r: torch.Tensor,
) -> torch.Tensor:
    """ELBO for q(theta), identical to spices version."""
    if not phi_posteriors:
        return torch.tensor(0.0)
    eps = torch.randn(N_MC_SAMPLES)
    theta_samples = m_theta + torch.exp(0.5 * log_v_theta) * eps
    ell = torch.tensor(0.0)
    log_v_r = 2.0 * log_sigma_r
    v_r = torch.exp(log_v_r)
    for m_phi, log_v_phi in phi_posteriors:
        log_p = (
            -0.5 * ((m_phi - theta_samples) ** 2) / v_r
            - 0.5 * log_v_r
            - 0.5 * math.log(2 * math.pi)
        )
        ell = ell + log_p.mean()
    m_prior = torch.tensor(float(mu))
    log_v_prior = 2.0 * log_sigma_h
    kl = _gaussian_kl(m_theta, log_v_theta, m_prior, log_v_prior)
    return ell - kl


# ---------------------------------------------------------------------------
# Main HBM class
# ---------------------------------------------------------------------------

class OvercookedPreferenceModel:
    """
    Hierarchical Bayesian preference model for Overcooked with vector psi.

    Structure:
        mu_s         ~ N(0, sigma0²)               global, per subtask
        theta_{h,s}  ~ N(mu_s, sigma_h²)           human-specific
        phi_{h,L,s}  ~ N(theta_{h,s}, sigma_r²)    human+layout-specific
        psi_{h,sess} ~ N(0, diag(sigma_session²))  per-episode vector ← new

    The "recipe" → "layout", "spice" → "subtask" renaming is the only
    structural difference from spices_hbm.HierarchicalPreferenceModel.

    Vector psi enables subtask-specific session offsets (e.g. fatigue
    affects delivery more than chopping).  Each dimension uses an
    independent scalar ELBO; the vector structure is enforced by grouping
    episode observations by subtask dimension before the psi update.

    Public interface to CSP (all return plain floats):
        register_human(human_id)
        register_layout(human_id, layout_name)
        observe(human_id, layout_name, subtask, actor, task_score)
        end_episode(human_id)
        log_prob_prefer(human_id, layout_name, subtask, actor) -> float
        preferred_actor(human_id, layout_name, subtask) -> str
        preference_posterior(human_id, layout_name, subtask) -> (float, float)
        get_phi_entropy(human_id, layout_name, subtask) -> float
        set_theta(human_id, theta_dict)
    """

    def __init__(
        self,
        subtasks: List[str],
        layouts: Optional[List[str]] = None,
        mu0: float = 0.0,
        sigma0: float = 1.0,
        sigma_h: Optional[float] = None,
        sigma_r: Optional[float] = None,
        sigma_obs: Optional[float] = None,
        sigma_session: Optional[float] = None,
        config: Optional[OvercookedConfig] = None,
        n_phi_steps: Optional[int] = None,
        n_theta_steps: Optional[int] = None,
        lr_phi: Optional[float] = None,
        lr_theta: Optional[float] = None,
        lr_hyper: Optional[float] = None,
        scalar_psi: bool = False,
    ) -> None:
        self.subtasks = list(subtasks)
        self.config = config if config is not None else DEFAULT_CONFIG
        self.mu0 = mu0
        self._scalar_psi = scalar_psi
        self.sigma0 = sigma0

        cfg = self.config.hbm
        _sigma_h = sigma_h if sigma_h is not None else cfg.sigma_h
        _sigma_r = sigma_r if sigma_r is not None else cfg.sigma_r
        _sigma_obs = sigma_obs if sigma_obs is not None else cfg.sigma_obs
        _sigma_session = sigma_session if sigma_session is not None else cfg.sigma_session

        # Subtask index map for psi dimension lookup.
        # Vector psi: each subtask gets its own dimension.
        # Scalar psi: all subtasks share dimension 0.
        if self._scalar_psi:
            self._subtask_index: Dict[str, int] = {s: 0 for s in self.subtasks}
            self._psi_dim = 1
        else:
            self._subtask_index = {s: i for i, s in enumerate(self.subtasks)}
            self._psi_dim = len(self.subtasks)

        # Fixed (not learned) psi prior std per dimension.
        # Stored as log_sigma_session [D] tensor, no grad.
        self.log_sigma_session = torch.full(
            (self._psi_dim,),
            math.log(_sigma_session),
            dtype=torch.float32,
            requires_grad=False,
        )

        # Learned hyperparameters
        self.log_sigma_h = torch.tensor(
            math.log(_sigma_h), dtype=torch.float32, requires_grad=True
        )
        self.log_sigma_r = torch.tensor(
            math.log(_sigma_r), dtype=torch.float32, requires_grad=True
        )
        self.log_sigma_obs = torch.tensor(
            math.log(_sigma_obs), dtype=torch.float32, requires_grad=True
        )

        # Optimization settings
        self.n_phi_steps = int(n_phi_steps if n_phi_steps is not None else cfg.n_phi_steps)
        self.n_theta_steps = int(n_theta_steps if n_theta_steps is not None else cfg.n_theta_steps)
        self.lr_phi = float(lr_phi if lr_phi is not None else cfg.lr_phi)
        self.lr_theta = float(lr_theta if lr_theta is not None else cfg.lr_theta)
        self.lr_hyper = float(lr_hyper if lr_hyper is not None else cfg.lr_hyper)
        self.log_var_min = float(cfg.log_var_min)
        self.log_var_max = float(cfg.log_var_max)
        self.task_logit_temperature = max(
            float(self.config.task.task_logit_temperature), 1e-6
        )
        self.base_task_bias = float(self.config.task.base_task_bias)

        self._hyper_optimizer = optim.Adam(
            [self.log_sigma_h, self.log_sigma_r],
            lr=self.lr_hyper,
        )

        # Global mu (not variational; updated analytically)
        self.mu_mean: Dict[str, float] = {s: mu0 for s in self.subtasks}
        self.mu_var: Dict[str, float] = {s: sigma0 ** 2 for s in self.subtasks}

        # Per-human variational parameters
        self._theta_m: Dict[str, Dict[str, torch.Tensor]] = {}
        self._theta_logv: Dict[str, Dict[str, torch.Tensor]] = {}

        # phi: [human_id][layout_name][subtask]
        self._phi_m: Dict[str, Any] = {}
        self._phi_logv: Dict[str, Any] = {}

        # Vector psi: [human_id] → [D] tensor (batch, inferred at episode end)
        self._psi_m: Dict[str, torch.Tensor] = {}

        # Running psi (mid-episode, for CSP adaptation)
        self._running_psi_m: Dict[str, torch.Tensor] = {}

        # Episode state
        # _episode_data: (actor, subtask, task_score)
        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._episode_layout_subtasks: Dict[str, List[Tuple[str, str]]] = {}
        self._obs_count: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._total_observations: Dict[str, int] = {}
        self._layout_total_obs: Dict[str, Dict[str, int]] = {}
        self._episode_count: Dict[str, int] = {}
        self._elbo_history: List[float] = []

        # Register default human
        self.register_human(DEFAULT_HUMAN)
        if layouts:
            for L in layouts:
                self.register_layout(DEFAULT_HUMAN, L)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        """Register a new human, initialising theta from global mu."""
        if human_id in self._theta_m:
            return

        self._theta_m[human_id] = {
            s: torch.tensor(self.mu_mean[s], dtype=torch.float32, requires_grad=True)
            for s in self.subtasks
        }
        self._theta_logv[human_id] = {
            s: torch.tensor(
                2.0 * self.log_sigma_h.item(),
                dtype=torch.float32,
                requires_grad=True,
            )
            for s in self.subtasks
        }
        self._phi_m[human_id] = defaultdict(dict)
        self._phi_logv[human_id] = defaultdict(dict)

        # Vector psi: [D] means, all zero; variance fixed at sigma_session²
        self._psi_m[human_id] = torch.zeros(
            self._psi_dim, dtype=torch.float32, requires_grad=True
        )
        self._running_psi_m[human_id] = torch.zeros(
            self._psi_dim, dtype=torch.float32, requires_grad=True
        )

        self._episode_data[human_id] = []
        self._episode_layout_subtasks[human_id] = []
        self._obs_count[human_id] = {}
        self._total_observations[human_id] = 0
        self._layout_total_obs[human_id] = {}
        self._episode_count[human_id] = 0

    def register_layout(self, human_id: str, layout_name: str) -> None:
        """Register a layout for a human, initialising phi from theta."""
        if human_id not in self._theta_m:
            self.register_human(human_id)
        if layout_name in self._phi_logv.get(human_id, {}) and \
           layout_name in self._phi_m.get(human_id, {}) and \
           len(self._phi_m[human_id][layout_name]) == len(self.subtasks):
            return

        for s in self.subtasks:
            theta_val = self._theta_m[human_id][s].item()
            log_v_init = 2.0 * self.log_sigma_r.item()
            self._phi_m[human_id][layout_name][s] = torch.tensor(
                theta_val, dtype=torch.float32, requires_grad=True
            )
            self._phi_logv[human_id][layout_name][s] = torch.tensor(
                log_v_init, dtype=torch.float32, requires_grad=True
            )

        self._obs_count[human_id][layout_name] = {s: 0 for s in self.subtasks}
        self._layout_total_obs[human_id][layout_name] = 0

    def _ensure_registered(self, human_id: str, layout_name: str) -> None:
        if human_id not in self._theta_m:
            self.register_human(human_id)
        if layout_name not in self._phi_logv.get(human_id, {}):
            self.register_layout(human_id, layout_name)

    # ------------------------------------------------------------------
    # ELBO-based phi update
    # ------------------------------------------------------------------

    def _update_phi_elbo(
        self,
        human_id: str,
        layout_name: str,
        subtask: str,
    ) -> float:
        """Update q(phi_{h,L,s}) for one subtask using Adam on the ELBO."""
        spice_obs: List[Tuple[str, float]] = [
            (act, score)
            for act, st, score in self._episode_data[human_id]
            if st == subtask
        ]
        if not spice_obs:
            return 0.0

        signs, scores = _obs_to_tensors(spice_obs)

        m_phi = self._phi_m[human_id][layout_name][subtask]
        log_v_phi = self._phi_logv[human_id][layout_name][subtask]
        m_theta = self._theta_m[human_id][subtask].detach()

        # Use the per-dimension psi entry for this subtask
        dim = self._subtask_index.get(subtask, 0)
        psi_fixed = self._psi_m[human_id][dim].detach()

        optimizer_phi = optim.Adam([m_phi, log_v_phi], lr=self.lr_phi)
        elbo_val = 0.0
        for _ in range(self.n_phi_steps):
            optimizer_phi.zero_grad()
            with torch.no_grad():
                log_v_phi.clamp_(self.log_var_min, self.log_var_max)
            elbo = _elbo_phi_step(
                signs=signs,
                scores=scores,
                m_phi=m_phi,
                log_v_phi=log_v_phi,
                psi_fixed=psi_fixed,
                m_theta=m_theta,
                log_sigma_r=self.log_sigma_r.detach(),
                log_sigma_obs=self.log_sigma_obs.detach(),
                task_logit_temperature=self.task_logit_temperature,
            )
            (-elbo).backward()
            optimizer_phi.step()
            elbo_val = elbo.item()

        self._phi_m[human_id][layout_name][subtask] = m_phi
        self._phi_logv[human_id][layout_name][subtask] = log_v_phi
        return elbo_val

    # ------------------------------------------------------------------
    # Theta + mu update
    # ------------------------------------------------------------------

    def update_theta_and_mu(self) -> None:
        """Update q(theta) and mu analytically. Same structure as spices."""
        registered_humans = list(self._theta_m.keys())

        n_active = sum(
            1
            for h in registered_humans
            for s in self.subtasks
            if any(
                self._obs_count[h].get(L, {}).get(s, 0) > 0
                for L in self._phi_logv.get(h, {})
            )
        )
        n_active = max(n_active, 1)

        self._hyper_optimizer.zero_grad()

        for h in registered_humans:
            for s in self.subtasks:
                phi_posteriors: List[Tuple[torch.Tensor, torch.Tensor]] = []
                for L in self._phi_logv.get(h, {}):
                    if self._obs_count[h].get(L, {}).get(s, 0) > 0:
                        phi_posteriors.append((
                            self._phi_m[h][L][s].detach(),
                            self._phi_logv[h][L][s].detach(),
                        ))

                if not phi_posteriors:
                    continue

                m_theta = self._theta_m[h][s]
                log_v_theta = self._theta_logv[h][s]
                optimizer = optim.Adam([m_theta, log_v_theta], lr=self.lr_theta)

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
                    (-elbo / n_active).backward()
                    optimizer.step()

                self._theta_m[h][s] = m_theta
                self._theta_logv[h][s] = log_v_theta

        self._hyper_optimizer.step()

        # Analytical mu update (precision-weighted mean of theta posteriors)
        for s in self.subtasks:
            theta_means: List[float] = []
            theta_precs: List[float] = []
            for h in registered_humans:
                if s in self._theta_m[h]:
                    theta_means.append(self._theta_m[h][s].item())
                    theta_precs.append(1.0 / math.exp(self._theta_logv[h][s].item()))

            if not theta_means:
                continue

            total_prec = sum(theta_precs)
            weighted_sum = sum(p * m for p, m in zip(theta_precs, theta_means))
            prior_prec = 1.0 / (self.sigma0 ** 2)
            post_var = 1.0 / (prior_prec + total_prec)
            post_mean = post_var * (self.mu0 * prior_prec + weighted_sum)
            self.mu_mean[s] = post_mean
            self.mu_var[s] = post_var

    # ------------------------------------------------------------------
    # Episode psi update (vector)
    # ------------------------------------------------------------------

    def _build_psi_inputs(
        self, human_id: str
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Group episode observations by subtask dimension for the vector psi update.

        Returns (signs_by_dim, scores_by_dim, phi_anchors_by_dim), each a list
        of length D = psi_dim.  Empty tensors are returned for unobserved dims.
        """
        D = self._psi_dim
        signs_by_dim: List[List[float]] = [[] for _ in range(D)]
        scores_by_dim: List[List[float]] = [[] for _ in range(D)]
        phi_anchors_by_dim: List[List[float]] = [[] for _ in range(D)]

        for act, st, score in self._episode_data[human_id]:
            dim = self._subtask_index.get(st)
            if dim is None:
                continue
            sign = 1.0 if act == "human" else -1.0
            signs_by_dim[dim].append(sign)
            scores_by_dim[dim].append(score)

            # phi anchor: look up phi across all registered layouts
            phi_val = 0.0
            for L in self._phi_m.get(human_id, {}):
                if st in self._phi_m[human_id][L]:
                    phi_val = self._phi_m[human_id][L][st].item()
                    break
            phi_anchors_by_dim[dim].append(phi_val)

        return (
            [torch.tensor(s, dtype=torch.float32) for s in signs_by_dim],
            [torch.tensor(s, dtype=torch.float32) for s in scores_by_dim],
            [torch.tensor(p, dtype=torch.float32) for p in phi_anchors_by_dim],
        )

    def _update_psi_episode(self, human_id: str) -> None:
        """Infer vector psi from all episode observations at episode end."""
        if not self._episode_data[human_id]:
            return

        signs_by_dim, scores_by_dim, phi_by_dim = self._build_psi_inputs(human_id)

        m_psi = self._psi_m[human_id]
        if not m_psi.requires_grad:
            m_psi = m_psi.detach().clone().requires_grad_(True)
        optimizer_psi = optim.Adam([m_psi], lr=self.lr_phi)
        for _ in range(self.n_phi_steps):
            optimizer_psi.zero_grad()
            elbo = _elbo_psi_vec_step(
                signs_by_dim=signs_by_dim,
                scores_by_dim=scores_by_dim,
                phi_anchors_by_dim=phi_by_dim,
                m_psi=m_psi,
                log_sigma_session=self.log_sigma_session,
                log_sigma_obs=self.log_sigma_obs.detach(),
                task_logit_temperature=self.task_logit_temperature,
            )
            if not elbo.requires_grad:
                break
            (-elbo).backward()
            optimizer_psi.step()
        self._psi_m[human_id] = m_psi

    def _update_running_psi(self, human_id: str) -> None:
        """
        Incrementally update running vector psi after each observation.
        Provides mid-episode psi signal for real-time CSP adaptation.
        Kept separate from _psi_m so it cannot contaminate phi updates.
        """
        if not self._episode_data[human_id]:
            return

        signs_by_dim, scores_by_dim, phi_by_dim = self._build_psi_inputs(human_id)

        m_psi_running = self._running_psi_m[human_id]
        if not m_psi_running.requires_grad:
            m_psi_running = m_psi_running.detach().clone().requires_grad_(True)
        n_steps = max(1, self.n_phi_steps // 2)  # half steps for running psi (fast mid-episode tracking)
        optimizer = optim.Adam([m_psi_running], lr=self.lr_phi)
        for _ in range(n_steps):
            optimizer.zero_grad()
            elbo = _elbo_psi_vec_step(
                signs_by_dim=signs_by_dim,
                scores_by_dim=scores_by_dim,
                phi_anchors_by_dim=phi_by_dim,
                m_psi=m_psi_running,
                log_sigma_session=self.log_sigma_session,
                log_sigma_obs=self.log_sigma_obs.detach(),
                task_logit_temperature=self.task_logit_temperature,
            )
            if not elbo.requires_grad:
                break
            (-elbo).backward()
            optimizer.step()
        self._running_psi_m[human_id] = m_psi_running

    def _update_phi_episode(self, human_id: str) -> None:
        """Batch-update phi for all (layout, subtask) pairs observed this episode."""
        seen_pairs = set(self._episode_layout_subtasks.get(human_id, []))
        for layout_name, subtask in seen_pairs:
            # Ensure phi exists for this (human, layout, subtask).
            if subtask not in self._phi_m.get(human_id, {}).get(layout_name, {}):
                self._ensure_registered(human_id, layout_name)
                if subtask not in self._phi_m.get(human_id, {}).get(layout_name, {}):
                    continue  # subtask not in this model's subtask list
            elbo_val = self._update_phi_elbo(human_id, layout_name, subtask)
            self._elbo_history.append(elbo_val)

    def _update_sigma_obs_episode(self, human_id: str) -> None:
        """Update log_sigma_obs once per episode with phi+psi fixed."""
        if not self._episode_data[human_id]:
            return

        signs_list, phi_list, scores_list = [], [], []
        for act, st, score in self._episode_data[human_id]:
            dim = self._subtask_index.get(st, 0)
            signs_list.append(1.0 if act == "human" else -1.0)
            scores_list.append(score)
            phi_val = 0.0
            for L in self._phi_m.get(human_id, {}):
                if st in self._phi_m[human_id][L]:
                    phi_val = self._phi_m[human_id][L][st].item()
                    break
            phi_list.append(phi_val + self._psi_m[human_id][dim].item())

        signs = torch.tensor(signs_list, dtype=torch.float32)
        scores = torch.tensor(scores_list, dtype=torch.float32)
        effective_phi = torch.tensor(phi_list, dtype=torch.float32)

        optimizer_obs = optim.Adam([self.log_sigma_obs], lr=self.lr_hyper)
        for _ in range(self.n_phi_steps):
            optimizer_obs.zero_grad()
            sigma_obs = torch.exp(self.log_sigma_obs)
            logits = signs * effective_phi
            temp = max(self.task_logit_temperature, 1e-6)
            expected_score = torch.tanh(logits / temp)
            log_p = (
                -0.5 * ((scores - expected_score) / sigma_obs) ** 2
                - self.log_sigma_obs
                - 0.5 * math.log(2 * math.pi)
            )
            (-log_p.sum()).backward()
            optimizer_obs.step()

    # ------------------------------------------------------------------
    # Episode interface
    # ------------------------------------------------------------------

    def observe(
        self,
        human_id: str,
        layout_name: str,
        subtask: str,
        actor: str,
        task_score: float,
    ) -> None:
        """
        Buffer one observation for batch processing at episode end.

        task_score: normalised order-completion feedback in [-1, +1].
        """
        self._ensure_registered(human_id, layout_name)
        self._episode_data[human_id].append((actor, subtask, task_score))
        self._episode_layout_subtasks[human_id].append((layout_name, subtask))
        if layout_name not in self._obs_count.get(human_id, {}):
            # Safety: ensure _obs_count is populated even if register_layout
            # was called before _obs_count was initialised for this human.
            if human_id not in self._obs_count:
                self._obs_count[human_id] = {}
            self._obs_count[human_id][layout_name] = {s: 0 for s in self.subtasks}
        self._obs_count[human_id][layout_name][subtask] = (
            self._obs_count[human_id][layout_name].get(subtask, 0) + 1
        )
        self._total_observations[human_id] = self._total_observations.get(human_id, 0) + 1
        if human_id not in self._layout_total_obs:
            self._layout_total_obs[human_id] = {}
        self._layout_total_obs[human_id][layout_name] = (
            self._layout_total_obs[human_id].get(layout_name, 0) + 1
        )
        # Update running psi for mid-episode CSP adaptation
        self._update_running_psi(human_id)

    def end_episode(self, human_id: str) -> None:
        """
        End-of-episode coordinate ascent update.

        Phase 1: infer vector psi from all episode observations (phi fixed).
        Phase 2: update phi for all observed (layout, subtask) pairs (psi fixed).
        Phase 3: update sigma_obs.
        Phase 4 (batched): update theta + mu.
        Then: aggressively decay psi mean toward zero.
        """
        self._episode_count[human_id] = self._episode_count.get(human_id, 0) + 1

        self._update_psi_episode(human_id)
        self._update_phi_episode(human_id)
        self._update_sigma_obs_episode(human_id)

        batch_size = self.config.hbm.update_theta_mu_every_n_episodes
        if batch_size <= 1 or self._episode_count[human_id] % batch_size == 0:
            self.update_theta_and_mu()

        # Aggressive psi decay: 95% of session signal discarded between episodes
        psi_decay = self.config.hbm.psi_decay
        self._psi_m[human_id] = torch.tensor(
            self._psi_m[human_id].detach().numpy() * psi_decay,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Reset episode state
        self._episode_data[human_id] = []
        self._episode_layout_subtasks[human_id] = []
        self._running_psi_m[human_id] = torch.zeros(
            self._psi_dim, dtype=torch.float32, requires_grad=True
        )

    def end_episode_eval(self, human_id: str) -> None:
        """Eval-time end-of-episode: update psi only, no phi/theta/mu learning.

        Infers psi from the episode's observations (for logging/diagnostics),
        then decays psi and resets episode state. This allows session-effect
        tracking during eval without modifying learned preference parameters.
        """
        self._update_psi_episode(human_id)

        # Aggressive psi decay (same as training)
        psi_decay = self.config.hbm.psi_decay
        self._psi_m[human_id] = torch.tensor(
            self._psi_m[human_id].detach().numpy() * psi_decay,
            dtype=torch.float32,
            requires_grad=True,
        )

        # Reset episode state
        self._episode_data[human_id] = []
        self._episode_layout_subtasks[human_id] = []
        self._running_psi_m[human_id] = torch.zeros(
            self._psi_dim, dtype=torch.float32, requires_grad=True
        )

    # ------------------------------------------------------------------
    # Public query interface (CSP-facing, all return plain floats)
    # ------------------------------------------------------------------

    def get_phi(self, human_id: str, layout_name: str, subtask: str) -> float:
        try:
            return self._phi_m[human_id][layout_name][subtask].item()
        except (KeyError, AttributeError):
            return 0.0

    def get_phi_var(self, human_id: str, layout_name: str, subtask: str) -> float:
        try:
            return math.exp(self._phi_logv[human_id][layout_name][subtask].item())
        except (KeyError, AttributeError):
            return math.exp(self.log_sigma_r.item()) ** 2

    def get_phi_entropy(self, human_id: str, layout_name: str, subtask: str) -> float:
        """Variance-weighted Bernoulli entropy. Higher → more exploration benefit."""
        phi_mean = self.get_phi(human_id, layout_name, subtask)
        phi_var = self.get_phi_var(human_id, layout_name, subtask)
        psi_val = self.get_running_psi(human_id, subtask)
        logit = phi_mean + psi_val
        log_p_human = -math.log1p(math.exp(-abs(logit))) - max(0.0, -logit)
        p_human = math.exp(log_p_human)
        p_robot = 1.0 - p_human
        p_human = max(p_human, 1e-10)
        p_robot = max(p_robot, 1e-10)
        H = -p_human * math.log(p_human) - p_robot * math.log(p_robot)
        return H * phi_var

    def get_theta(self, human_id: str, subtask: str) -> float:
        return self._theta_m[human_id][subtask].item()

    def get_theta_var(self, human_id: str, subtask: str) -> float:
        return math.exp(self._theta_logv[human_id][subtask].item())

    def get_mu(self, subtask: str) -> float:
        return self.mu_mean[subtask]

    def get_mu_var(self, subtask: str) -> float:
        return self.mu_var[subtask]

    def get_psi_vec(self, human_id: str) -> List[float]:
        """Return the full batch psi vector (inferred at episode end)."""
        try:
            return self._psi_m[human_id].detach().tolist()
        except KeyError:
            return [0.0] * self._psi_dim

    def get_running_psi_vec(self, human_id: str) -> List[float]:
        """Return the full running psi vector (mid-episode estimate)."""
        try:
            return self._running_psi_m[human_id].detach().tolist()
        except KeyError:
            return [0.0] * self._psi_dim

    def get_running_psi(self, human_id: str, subtask: str) -> float:
        """Return the running psi scalar for a specific subtask dimension."""
        dim = self._subtask_index.get(subtask, 0)
        try:
            return float(self._running_psi_m[human_id][dim].item())
        except KeyError:
            return 0.0

    def preference_posterior(
        self, human_id: str, layout_name: str, subtask: str
    ) -> Tuple[float, float]:
        """Return (phi_mean, phi_var). Used by CSP for exploitation + exploration."""
        return self.get_phi(human_id, layout_name, subtask), self.get_phi_var(
            human_id, layout_name, subtask
        )

    def preferred_actor(
        self, human_id: str, layout_name: str, subtask: str
    ) -> str:
        """Return 'human' or 'robot' based on phi + running_psi[subtask]."""
        phi = self.get_phi(human_id, layout_name, subtask)
        psi = self.get_running_psi(human_id, subtask)
        return "human" if (phi + psi) >= 0 else "robot"

    def log_prob_prefer(
        self, human_id: str, layout_name: str, subtask: str, actor: str
    ) -> float:
        """Return log P(actor | phi + running_psi[subtask])."""
        phi = self.get_phi(human_id, layout_name, subtask)
        psi = self.get_running_psi(human_id, subtask)
        sign_actor = 1.0 if actor == "human" else -1.0
        logit = sign_actor * (phi + psi)
        log_p = -math.log1p(math.exp(-abs(logit))) - max(0.0, -logit)
        return float(max(log_p, -20.0))

    def get_learned_sigmas(self) -> Dict[str, float]:
        """Return current learned hyperparameter values."""
        return {
            "sigma_h": math.exp(self.log_sigma_h.item()),
            "sigma_r": math.exp(self.log_sigma_r.item()),
            "sigma_obs": math.exp(self.log_sigma_obs.item()),
            "sigma_session": math.exp(self.log_sigma_session[0].item()),
        }

    def set_theta(
        self,
        human_id: str,
        theta_dict: Dict[str, float],
        sigma_h: Optional[float] = None,
    ) -> None:
        """Directly set theta values (for constructing hidden ground-truth HBMs)."""
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

    def sample_episode_preferences(
        self,
        subtasks: List[str],
        rng: np.random.Generator,
        human_id: str = DEFAULT_HUMAN,
    ) -> Dict[str, str]:
        """Return preferred actor per subtask based on theta mean (deterministic sign)."""
        preferences: Dict[str, str] = {}
        for st in subtasks:
            if st in self._theta_m.get(human_id, {}):
                theta_mean = self._theta_m[human_id][st].item()
                preferences[st] = "human" if theta_mean >= 0 else "robot"
            else:
                preferences[st] = "robot"
        return preferences

    def flush_theta_mu(self) -> None:
        """Force immediate theta/mu update. Call before eval."""
        self.update_theta_and_mu()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_elbo_snapshot(
        self, human_id: str, layout_name: str, subtask: str
    ) -> float:
        """Compute the current ELBO for one (human, layout, subtask) cell."""
        spice_obs: List[Tuple[str, float]] = [
            (act, score)
            for act, st, score in self._episode_data[human_id]
            if st == subtask
        ]
        if not spice_obs:
            return 0.0
        signs, scores = _obs_to_tensors(spice_obs)
        m_phi = self._phi_m[human_id][layout_name][subtask]
        log_v_phi = self._phi_logv[human_id][layout_name][subtask]
        dim = self._subtask_index.get(subtask, 0)
        m_psi = self._psi_m[human_id][dim]
        log_v_psi = self.log_sigma_session[dim] * 2.0

        phi_samples = m_phi + torch.exp(0.5 * log_v_phi) * torch.randn(N_MC_SAMPLES)
        psi_samples = m_psi + torch.exp(0.5 * log_v_psi) * torch.randn(N_MC_SAMPLES)

        ell = _ell_phi_psi(
            signs, scores, phi_samples, psi_samples,
            self.log_sigma_obs, self.task_logit_temperature
        )
        kl_phi = _gaussian_kl(
            m_phi, log_v_phi,
            self._theta_m[human_id][subtask].detach(),
            2.0 * self.log_sigma_r,
        )
        kl_psi = 0.5 * (m_psi / torch.exp(self.log_sigma_session[dim])) ** 2
        return float((ell - kl_phi - kl_psi).item())

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist all variational parameters to disk."""
        import pickle
        state = {
            "phi_m": {
                h: {L: {s: p.detach().cpu().item() for s, p in sdict.items()}
                     for L, sdict in ldict.items()}
                for h, ldict in self._phi_m.items()
            },
            "phi_logv": {
                h: {L: {s: p.detach().cpu().item() for s, p in sdict.items()}
                     for L, sdict in ldict.items()}
                for h, ldict in self._phi_logv.items()
            },
            "theta_m": {
                h: {s: p.detach().cpu().item() for s, p in sdict.items()}
                for h, sdict in self._theta_m.items()
            },
            "theta_logv": {
                h: {s: p.detach().cpu().item() for s, p in sdict.items()}
                for h, sdict in self._theta_logv.items()
            },
            "mu_mean": {s: float(v) for s, v in self.mu_mean.items()},
            "mu_var": {s: float(v) for s, v in self.mu_var.items()},
            "log_sigma_obs": self.log_sigma_obs.detach().cpu().item(),
            "log_sigma_r": self.log_sigma_r.detach().cpu().item(),
        }
        with open(path / "overcooked_hbm.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        """Load variational parameters from disk."""
        import pickle
        p = path / "overcooked_hbm.pkl"
        if not p.exists():
            return
        try:
            with open(p, "rb") as f:
                state = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, Exception):
            return  # corrupted file — skip load
        for h, ldict in state["phi_m"].items():
            if h not in self._phi_m:
                self.register_human(h)
            for L, sdict in ldict.items():
                if L not in self._phi_m.get(h, {}):
                    self.register_layout(h, L)
                for s, v in sdict.items():
                    if s in self._phi_m.get(h, {}).get(L, {}):
                        self._phi_m[h][L][s] = torch.tensor(
                            v, dtype=torch.float32, requires_grad=True
                        )
        for h, ldict in state["phi_logv"].items():
            for L, sdict in ldict.items():
                for s, v in sdict.items():
                    if s in self._phi_logv.get(h, {}).get(L, {}):
                        self._phi_logv[h][L][s] = torch.tensor(
                            v, dtype=torch.float32, requires_grad=True
                        )
        for h, sdict in state["theta_m"].items():
            for s, v in sdict.items():
                if s in self._theta_m.get(h, {}):
                    self._theta_m[h][s] = torch.tensor(
                        v, dtype=torch.float32, requires_grad=True
                    )
        for h, sdict in state["theta_logv"].items():
            for s, v in sdict.items():
                if s in self._theta_logv.get(h, {}):
                    self._theta_logv[h][s] = torch.tensor(
                        v, dtype=torch.float32, requires_grad=True
                    )
        for s, v in state.get("mu_mean", {}).items():
            if s in self.mu_mean:
                self.mu_mean[s] = float(v)
        for s, v in state.get("mu_var", {}).items():
            if s in self.mu_var:
                self.mu_var[s] = float(v)
        self.log_sigma_obs = torch.tensor(
            state["log_sigma_obs"], dtype=torch.float32, requires_grad=True
        )
        self.log_sigma_r = torch.tensor(
            state["log_sigma_r"], dtype=torch.float32, requires_grad=True
        )

        # Re-initialize phi for layouts that were registered BEFORE theta was
        # loaded (e.g., eval layout registered in constructor with theta=0).
        # These layouts have stale phi=0; re-register to initialize from loaded theta.
        saved_layouts = set()
        for h, ldict in state.get("phi_m", {}).items():
            saved_layouts.update(ldict.keys())
        for h in list(self._phi_m.keys()):
            for L in list(self._phi_m[h].keys()):
                if L not in saved_layouts:
                    # This layout wasn't in the saved model — re-init from theta.
                    del self._phi_m[h][L]
                    del self._phi_logv[h][L]
                    self.register_layout(h, L)
