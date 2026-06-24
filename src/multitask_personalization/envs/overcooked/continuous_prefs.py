"""Continuous preference learning for Overcooked (phase 1: ingredient count).

============================================================================
STATUS: FUTURE WORK — skeleton only, not used in the published thesis.
============================================================================

This module implements a prototype continuous-preference learner for
Overcooked (preferred ingredient count before triggering cooking). It is
fully gated behind ``OvercookedSceneSpec.continuous_prefs_enabled``, which
is only set to True by the dedicated ``overcooked_continuous.yaml`` Hydra
config. Every other env/approach/experiment config in the repo leaves this
branch inert (no CSP variable, no satisfaction blending, no model build).

**Why it is future work**:
  1. ``action.ingredient_count`` is currently only a *label* for the
     hidden satisfaction function — the game mechanics still load 3 onions
     every time. A publishable version needs real early-cooking hooks via
     ``begin_cooking()`` after N<MAX onions have been placed.
  2. The single-context experiment reproduces SpiceEnv Main: DBTL ≈ Flat
     because there is no hierarchy to pool over, and CBTL's
     ``Bounded1DClassifier`` beats the Gaussian HBM simply because the
     trapezoidal non-parametric fit is more sample-efficient for a single
     1-D preference function than a mis-specified Gaussian likelihood.
  3. A fair hierarchical test would need (i) real game-mechanic wiring
     and (ii) a multi-layout config with heterogeneous preferred counts so
     that ``μ_global → μ_h → μ_{h,L}`` has something to pool.

The skeleton is left in place for a future extension of the thesis.

----------------------------------------------------------------------------

Adds a parallel continuous-preference learner alongside the existing binary
HBM and CBTL baselines. The model learns a Gaussian acceptance function over
a single scalar parameter (ingredient count) per (human, layout) pair, with
optional hierarchical pooling across layouts via a human-level mean.

Architecture:
    mu_p[h]        ~ N(mu0, sigma_h^2)   # human-level preferred count
    mu_p[h, L]     ~ N(mu_p[h], sigma_r^2)  # layout-specific preferred count

    Observation likelihood:
        s_t ~ N(acceptance(count; mu_p[h, L], sigma_p), sigma_obs^2)
        where acceptance(x) = 2*exp(-(x-mu)^2/(2*sigma_p^2)) - 1

The CBTL baseline uses Bounded1DClassifier (trapezoidal) from the original
CBTL paper. Both models learn from the same satisfaction signal. CBTL is
pooled across humans (matching the binary baseline policy).
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian preference HBM (hierarchical)
# ---------------------------------------------------------------------------

@dataclass
class _GaussianPrefPosterior:
    """Posterior for a single (human, layout) continuous preference.

    Uses a simple closed-form Gaussian update in the latent ideal-value space.
    We treat the chosen count x with satisfaction s as a noisy observation:
        expected_acceptance(x) = 2*exp(-(x - mu)^2 / (2*sigma_p^2)) - 1
    and update by gradient on the Gaussian log-likelihood.
    """
    mu: float  # posterior mean of preferred count
    var: float  # posterior variance of preferred count
    sigma_p: float  # fixed width of acceptance function (generative)
    n_obs: int = 0


class ContinuousPreferenceHBM:
    """Hierarchical Gaussian-preference model for a single continuous param.

    Structure:
        mu_global                         # learned from all observations
        mu_human[h]   ~ N(mu_global, s_h^2)   # per-human preferred value
        mu_context[h,L] ~ N(mu_human[h], s_r^2)  # per-(human, layout) preferred

    Learning: gradient descent on the Gaussian log-likelihood of observed
    satisfaction values against the predicted acceptance function. At episode
    end, updates mu_context via gradient, then updates mu_human and mu_global
    via precision-weighted averaging.

    This is SEPARATE from the binary HBM — doesn't touch phi/theta/mu/psi.
    Only handles the single continuous dimension (ingredient count).
    """

    def __init__(
        self,
        param_min: float = 1.0,
        param_max: float = 4.0,
        sigma_p: float = 1.0,   # width of acceptance function
        sigma_r: float = 0.5,   # hierarchical prior std (per-context)
        sigma_h: float = 0.5,   # hierarchical prior std (per-human)
        sigma_obs: float = 0.3, # observation noise on satisfaction
        lr: float = 0.05,
        n_steps: int = 20,
    ) -> None:
        self.param_min = param_min
        self.param_max = param_max
        self.sigma_p = sigma_p
        self.sigma_r = sigma_r
        self.sigma_h = sigma_h
        self.sigma_obs = sigma_obs
        self.lr = lr
        self.n_steps = n_steps

        # Global prior mean (center of param range)
        self._mu_global: float = (param_min + param_max) / 2.0

        # Per-human posterior: mu_human[h] = mean
        self._mu_human: Dict[str, float] = {}

        # Per-context posterior: mu_ctx[h][layout] = (mean, var)
        self._mu_ctx: Dict[str, Dict[str, Tuple[float, float]]] = {}

        # Episode data: list of (layout, count, satisfaction) per human
        self._episode_data: Dict[str, List[Tuple[str, int, float]]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_human(self, human_id: str) -> None:
        if human_id not in self._mu_human:
            self._mu_human[human_id] = self._mu_global
            self._mu_ctx[human_id] = {}
            self._episode_data[human_id] = []

    def register_layout(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        if layout_name not in self._mu_ctx[human_id]:
            # Warm-start context from human-level mean
            self._mu_ctx[human_id][layout_name] = (
                self._mu_human[human_id],
                self.sigma_r ** 2,  # initial variance = prior
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def get_preferred_count(self, human_id: str, layout_name: str) -> float:
        """Return the posterior mean preferred count."""
        if human_id not in self._mu_ctx:
            return self._mu_global
        if layout_name not in self._mu_ctx[human_id]:
            return self._mu_human.get(human_id, self._mu_global)
        return self._mu_ctx[human_id][layout_name][0]

    def get_variance(self, human_id: str, layout_name: str) -> float:
        """Return posterior variance of preferred count."""
        if human_id not in self._mu_ctx:
            return self.sigma_h ** 2 + self.sigma_r ** 2
        if layout_name not in self._mu_ctx[human_id]:
            return self.sigma_r ** 2
        return self._mu_ctx[human_id][layout_name][1]

    def acceptance_prob(self, count: int, human_id: str, layout_name: str) -> float:
        """Predicted acceptance probability for a given count.

        Returns the Gaussian acceptance value mapped to [0, 1]:
            p(accept) = exp(-(count - mu_ctx)^2 / (2 * sigma_p^2))
        """
        mu = self.get_preferred_count(human_id, layout_name)
        return float(np.exp(-((count - mu) ** 2) / (2.0 * self.sigma_p ** 2)))

    def log_prob_accept(self, count: int, human_id: str, layout_name: str) -> float:
        """Log of acceptance probability."""
        p = self.acceptance_prob(count, human_id, layout_name)
        return float(np.log(max(p, 1e-9)))

    def best_count(self, human_id: str, layout_name: str) -> int:
        """Return the discrete count in [param_min, param_max] closest to posterior mean."""
        mu = self.get_preferred_count(human_id, layout_name)
        # Clip and round to nearest integer in range
        mu_clipped = max(self.param_min, min(self.param_max, mu))
        return int(round(mu_clipped))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(
        self,
        human_id: str,
        layout_name: str,
        count: int,
        satisfaction: float,
    ) -> None:
        """Buffer a (count, satisfaction) observation for this (human, layout)."""
        self.register_layout(human_id, layout_name)
        self._episode_data[human_id].append((layout_name, count, satisfaction))

    def end_episode(self, human_id: str) -> None:
        """Run gradient updates on mu_ctx, then propagate up the hierarchy."""
        if human_id not in self._episode_data:
            return
        obs = self._episode_data[human_id]
        if not obs:
            self._episode_data[human_id] = []
            return

        # Group by layout
        by_layout: Dict[str, List[Tuple[int, float]]] = {}
        for layout, count, sat in obs:
            by_layout.setdefault(layout, []).append((count, sat))

        # Update mu_ctx for each layout via gradient descent
        for layout, pairs in by_layout.items():
            self.register_layout(human_id, layout)
            mu_ctx, var_ctx = self._mu_ctx[human_id][layout]
            mu_prior = self._mu_human[human_id]

            # Gradient steps on Gaussian log-likelihood
            # Expected sat given count x: e(x) = 2*exp(-(x-mu)^2/(2*sigma_p^2)) - 1
            # log p(sat | mu) = -0.5 * ((sat - e(x)) / sigma_obs)^2  (Gaussian)
            # d(log p)/d(mu) = (sat - e(x)) * de/d(mu) / sigma_obs^2
            # de/d(mu) = 2*exp(-(x-mu)^2/(2*sigma_p^2)) * (x-mu)/sigma_p^2
            for _ in range(self.n_steps):
                grad = 0.0
                for count, sat in pairs:
                    resid = (count - mu_ctx)
                    expo = np.exp(-(resid ** 2) / (2.0 * self.sigma_p ** 2))
                    expected = 2.0 * expo - 1.0
                    de_dmu = 2.0 * expo * resid / (self.sigma_p ** 2)
                    grad += (sat - expected) * de_dmu / (self.sigma_obs ** 2)
                # KL prior toward mu_human
                grad -= (mu_ctx - mu_prior) / (self.sigma_r ** 2)
                mu_ctx = mu_ctx + self.lr * grad
                # Clip to valid range
                mu_ctx = max(self.param_min, min(self.param_max, mu_ctx))

            # Update variance via simple rule: more data = lower variance
            n = len(pairs)
            new_var = max(1e-6, 1.0 / (1.0 / (self.sigma_r ** 2) + n / (self.sigma_obs ** 2)))
            self._mu_ctx[human_id][layout] = (mu_ctx, new_var)

        # Update mu_human via precision-weighted average of context means
        if self._mu_ctx[human_id]:
            ctx_means = [v[0] for v in self._mu_ctx[human_id].values()]
            ctx_vars = [v[1] for v in self._mu_ctx[human_id].values()]
            precisions = [1.0 / max(v, 1e-6) for v in ctx_vars]
            weighted_sum = sum(p * m for p, m in zip(precisions, ctx_means))
            total_prec = sum(precisions) + 1.0 / (self.sigma_h ** 2)
            self._mu_human[human_id] = (
                weighted_sum + self._mu_global / (self.sigma_h ** 2)
            ) / total_prec

        # Update mu_global via precision-weighted average of human means
        if self._mu_human:
            human_means = list(self._mu_human.values())
            prec = 1.0 / (self.sigma_h ** 2)
            total = prec * len(human_means)
            weighted = sum(prec * m for m in human_means)
            # Weak prior toward range center
            center = (self.param_min + self.param_max) / 2.0
            weak_prec = 0.01
            self._mu_global = (weighted + weak_prec * center) / (total + weak_prec)

        # Clear episode buffer
        self._episode_data[human_id] = []

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        state = {
            "mu_global": self._mu_global,
            "mu_human": self._mu_human,
            "mu_ctx": self._mu_ctx,
            "param_min": self.param_min,
            "param_max": self.param_max,
            "sigma_p": self.sigma_p,
            "sigma_r": self.sigma_r,
            "sigma_h": self.sigma_h,
        }
        with open(path / "continuous_hbm.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "continuous_hbm.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._mu_global = state["mu_global"]
        self._mu_human = state["mu_human"]
        self._mu_ctx = state["mu_ctx"]


# ---------------------------------------------------------------------------
# Bounded1DClassifier for CBTL baseline
# ---------------------------------------------------------------------------

class Bounded1DClassifier:
    """Faithful replica of CBTL's Bounded1DClassifier (trapezoidal interval model).

    Given positive examples (accepted values) and negative examples (rejected),
    learns a trapezoidal probability function defined by four breakpoints:
        x1 : maximum of negative examples below positives (or -inf)
        x2 : minimum of positive examples
        x3 : maximum of positive examples
        x4 : minimum of negative examples above positives (or +inf)

    The probability ramps linearly 0→1 on [x1, x2], is 1 on [x2, x3], and
    ramps 1→0 on [x3, x4]. This is the original CBTL paper's approach.

    Non-parametric: refits breakpoints from all observed data on each update.
    """

    def __init__(
        self,
        a_lo: float = 1.0,
        b_hi: float = 4.0,
    ) -> None:
        self.a_lo = a_lo
        self.b_hi = b_hi
        # Initial state: uniform uncertainty across the full range.
        self.x1 = a_lo
        self.x2 = a_lo
        self.x3 = b_hi
        self.x4 = b_hi
        self._X: List[float] = []
        self._Y: List[bool] = []

    def fit_incremental(self, X: List[float], Y: List[bool]) -> None:
        """Add new observations and re-fit breakpoints."""
        self._X.extend(X)
        self._Y.extend(Y)
        self._refit()

    def _refit(self) -> None:
        """Re-compute breakpoints from accumulated data."""
        X_pos = sorted({x for x, y in zip(self._X, self._Y) if y})
        X_neg = sorted({x for x, y in zip(self._X, self._Y) if not y})
        if not X_pos:
            # No positive data — stay at uniform uncertainty
            self.x1 = self.a_lo
            self.x2 = self.a_lo
            self.x3 = self.b_hi
            self.x4 = self.b_hi
            return
        # Pivot at the mid of positive examples
        pos_mid = X_pos[len(X_pos) // 2]
        X_neg_lo = [x for x in X_neg if x < pos_mid]
        X_neg_hi = [x for x in X_neg if x >= pos_mid]
        self.x1 = max([self.a_lo] + X_neg_lo)
        self.x2 = X_pos[0]
        self.x3 = X_pos[-1]
        self.x4 = min([self.b_hi] + X_neg_hi)

    def predict_proba(self, x: float) -> float:
        """Return P(accept) for a given value."""
        if x < self.x1 or x > self.x4:
            return 0.0
        if self.x1 <= x < self.x2:
            if self.x2 - self.x1 <= 0:
                return 1.0
            return (x - self.x1) / (self.x2 - self.x1)
        if self.x2 <= x <= self.x3:
            return 1.0
        if self.x3 < x <= self.x4:
            if self.x4 - self.x3 <= 0:
                return 1.0
            return 1.0 - (x - self.x3) / (self.x4 - self.x3)
        return 0.0


class ContinuousPreferenceCBTL:
    """CBTL-faithful baseline for continuous preferences.

    Uses Bounded1DClassifier per (pool_key, layout). When pooled_across_humans
    is True (default), pool_key is '__pooled__' and all humans share one
    classifier per layout. When False, each human has its own classifier.

    Updates from binary accept/reject labels derived from satisfaction sign:
        satisfaction > 0 → accepted
        satisfaction <= 0 → rejected
    """

    def __init__(
        self,
        param_min: float = 1.0,
        param_max: float = 4.0,
        pooled_across_humans: bool = True,
    ) -> None:
        self.param_min = param_min
        self.param_max = param_max
        self._pooled = pooled_across_humans
        # classifiers[pool_key][layout] = Bounded1DClassifier
        self._classifiers: Dict[str, Dict[str, Bounded1DClassifier]] = {}
        # Episode buffers per human (independent of pooling)
        self._episode_data: Dict[str, List[Tuple[str, int, float]]] = {}

    def _pool_key(self, human_id: str) -> str:
        return "__pooled__" if self._pooled else human_id

    def register_human(self, human_id: str) -> None:
        key = self._pool_key(human_id)
        if key not in self._classifiers:
            self._classifiers[key] = {}
        if human_id not in self._episode_data:
            self._episode_data[human_id] = []

    def register_layout(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        key = self._pool_key(human_id)
        if layout_name not in self._classifiers[key]:
            self._classifiers[key][layout_name] = Bounded1DClassifier(
                a_lo=self.param_min, b_hi=self.param_max
            )

    def observe(
        self,
        human_id: str,
        layout_name: str,
        count: int,
        satisfaction: float,
    ) -> None:
        self.register_layout(human_id, layout_name)
        self._episode_data[human_id].append((layout_name, count, satisfaction))

    def end_episode(self, human_id: str) -> None:
        key = self._pool_key(human_id)
        if human_id not in self._episode_data:
            return
        # Group observations by layout
        by_layout: Dict[str, List[Tuple[int, float]]] = {}
        for layout, count, sat in self._episode_data[human_id]:
            by_layout.setdefault(layout, []).append((count, sat))
        # Update each layout's classifier with binary labels
        for layout, pairs in by_layout.items():
            self.register_layout(human_id, layout)
            clf = self._classifiers[key][layout]
            Xs = [float(c) for c, _ in pairs]
            Ys = [s > 0 for _, s in pairs]  # accept if satisfaction > 0
            clf.fit_incremental(Xs, Ys)
        self._episode_data[human_id] = []

    def acceptance_prob(self, count: int, human_id: str, layout_name: str) -> float:
        key = self._pool_key(human_id)
        if key not in self._classifiers or layout_name not in self._classifiers[key]:
            return 0.5  # uniform uncertainty before any data
        return self._classifiers[key][layout_name].predict_proba(float(count))

    def log_prob_accept(self, count: int, human_id: str, layout_name: str) -> float:
        p = self.acceptance_prob(count, human_id, layout_name)
        return float(np.log(max(p, 1e-9)))

    def best_count(self, human_id: str, layout_name: str) -> int:
        """Return the discrete count with the highest acceptance probability."""
        best = int(round((self.param_min + self.param_max) / 2.0))
        best_p = -1.0
        for c in range(int(self.param_min), int(self.param_max) + 1):
            p = self.acceptance_prob(c, human_id, layout_name)
            if p > best_p:
                best_p = p
                best = c
        return best

    def save(self, path: Path) -> None:
        state = {
            "classifiers": {
                k: {l: (c.x1, c.x2, c.x3, c.x4, c._X, c._Y)
                    for l, c in lyts.items()}
                for k, lyts in self._classifiers.items()
            },
            "param_min": self.param_min,
            "param_max": self.param_max,
            "pooled": self._pooled,
        }
        with open(path / "continuous_cbtl.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "continuous_cbtl.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._classifiers = {}
        for k, lyts in state["classifiers"].items():
            self._classifiers[k] = {}
            for l, (x1, x2, x3, x4, X, Y) in lyts.items():
                c = Bounded1DClassifier(
                    a_lo=self.param_min, b_hi=self.param_max,
                )
                c.x1, c.x2, c.x3, c.x4 = x1, x2, x3, x4
                c._X, c._Y = list(X), list(Y)
                self._classifiers[k][l] = c


# ---------------------------------------------------------------------------
# Flat (per-context, no transfer) baseline
# ---------------------------------------------------------------------------

class ContinuousPreferenceFlat:
    """Flat Bayesian baseline for continuous preferences.

    Maintains a Gaussian posterior over the preferred count per
    (human, layout) triple with NO hierarchical pooling. Each context
    starts from a broad prior and is updated independently.
    """

    def __init__(
        self,
        param_min: float = 1.0,
        param_max: float = 4.0,
        sigma_p: float = 1.0,
        sigma_obs: float = 0.3,
        lr: float = 0.05,
        n_steps: int = 20,
    ) -> None:
        self.param_min = param_min
        self.param_max = param_max
        self.sigma_p = sigma_p
        self.sigma_obs = sigma_obs
        self.lr = lr
        self.n_steps = n_steps

        # posteriors[human][layout] = mu
        self._mu: Dict[str, Dict[str, float]] = {}
        self._episode_data: Dict[str, List[Tuple[str, int, float]]] = {}
        self._prior_mu: float = (param_min + param_max) / 2.0

    def register_human(self, human_id: str) -> None:
        if human_id not in self._mu:
            self._mu[human_id] = {}
            self._episode_data[human_id] = []

    def register_layout(self, human_id: str, layout_name: str) -> None:
        self.register_human(human_id)
        if layout_name not in self._mu[human_id]:
            self._mu[human_id][layout_name] = self._prior_mu

    def observe(
        self,
        human_id: str,
        layout_name: str,
        count: int,
        satisfaction: float,
    ) -> None:
        self.register_layout(human_id, layout_name)
        self._episode_data[human_id].append((layout_name, count, satisfaction))

    def end_episode(self, human_id: str) -> None:
        obs = self._episode_data.get(human_id, [])
        by_layout: Dict[str, List[Tuple[int, float]]] = {}
        for layout, count, sat in obs:
            by_layout.setdefault(layout, []).append((count, sat))
        for layout, pairs in by_layout.items():
            self.register_layout(human_id, layout)
            mu = self._mu[human_id][layout]
            for _ in range(self.n_steps):
                grad = 0.0
                for count, sat in pairs:
                    resid = count - mu
                    expo = np.exp(-(resid ** 2) / (2.0 * self.sigma_p ** 2))
                    expected = 2.0 * expo - 1.0
                    de_dmu = 2.0 * expo * resid / (self.sigma_p ** 2)
                    grad += (sat - expected) * de_dmu / (self.sigma_obs ** 2)
                mu = mu + self.lr * grad
                mu = max(self.param_min, min(self.param_max, mu))
            self._mu[human_id][layout] = mu
        self._episode_data[human_id] = []

    def acceptance_prob(self, count: int, human_id: str, layout_name: str) -> float:
        mu = self._mu.get(human_id, {}).get(layout_name, self._prior_mu)
        return float(np.exp(-((count - mu) ** 2) / (2.0 * self.sigma_p ** 2)))

    def log_prob_accept(self, count: int, human_id: str, layout_name: str) -> float:
        p = self.acceptance_prob(count, human_id, layout_name)
        return float(np.log(max(p, 1e-9)))

    def best_count(self, human_id: str, layout_name: str) -> int:
        mu = self._mu.get(human_id, {}).get(layout_name, self._prior_mu)
        mu_clipped = max(self.param_min, min(self.param_max, mu))
        return int(round(mu_clipped))

    def save(self, path: Path) -> None:
        state = {"mu": self._mu, "prior_mu": self._prior_mu}
        with open(path / "continuous_flat.pkl", "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path) -> None:
        p = path / "continuous_flat.pkl"
        if not p.exists():
            return
        with open(p, "rb") as f:
            state = pickle.load(f)
        self._mu = state["mu"]
        self._prior_mu = state["prior_mu"]
