from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


# Moods
MOODS = ("all_self", "neutral", "none_self")

# Mood biases
MOOD_BIAS = {
    "all_self":   {"human": +6.0, "robot": -6.0},
    "neutral":    {"human":  0.0, "robot":  0.0},
    "none_self":  {"human": -6.0, "robot": +6.0},
}


class HierarchicalPreferenceModel:
    """
    Hierarchical Bayesian model of human preferences:
        μ_s      ~ N(μ0, σ0^2)
        θ_s      ~ N(μ_s, σ_h^2)
        φ_{r,s}  ~ N(θ_s, σ_r^2)

    Mood m_t is latent and inferred per episode.

    Observations update:
      - posterior over mood P(m | data)
      - level-3 φ (recipe-specific preferences)
      - periodically θ (human-level) and μ (global)

    TODO: add multiple humans support
    """

    def __init__(
        self,
        spices: List[str],
        recipes: List[str],
        mu0: float = 0.0,
        sigma0: float = 1.0,
        sigma_h: float = 1.0,
        sigma_r: float = 1.0,
        sigma_obs: float = 1.0,
    ) -> None:

        self.spices = list(spices)
        self.recipes = list(recipes)

        # Variances
        self.sigma0 = sigma0
        self.sigma_h = sigma_h
        self.sigma_r = sigma_r
        self.sigma_obs = sigma_obs

        # ---------------- Level 1: μ_s ----------------
        self.mu_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.mu_var: Dict[str, float] = {s: sigma0**2 for s in self.spices}

        # ---------------- Level 2: θ_s ----------------
        self.theta_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.theta_var: Dict[str, float] = {
            s: sigma0**2 + sigma_h**2 for s in self.spices
        }

        # ---------------- Level 3: φ_{r,s} ----------------
        self.phi_mean: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: 0.0 for s in self.spices}
        )
        self.phi_var: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: sigma_r**2 for s in self.spices}
        )

        # Per-step data for mood inference (actor, spice, satisfaction)
        self.episode_data: List[Tuple[str, str, float]] = []

        # Prior over mood
        self.mood_prior = np.array([1/3, 1/3, 1/3], dtype=float)
        self.mood_posterior = self.mood_prior.copy()

    # --- Mood inference ---
    def _loglik_feedback_given_mood(
        self, actor: str, spice: str, satisfaction: float, m: str, recipe_name: str
    ) -> float:
        """
        Compute log P(y | actor, spice, φ, m)
        using the generative model logit = sign(actor)*phi + mood_bias
        """
        phi = self.phi_mean[recipe_name][spice]
        sign_actor = +1.0 if actor == "human" else -1.0

        logit = sign_actor * phi + MOOD_BIAS[m][actor]
        p = 1.0 / (1.0 + np.exp(-logit))

        if satisfaction > 0:
            return np.log(max(p, 1e-9))
        else:
            return np.log(max(1 - p, 1e-9))

    def _update_mood_posterior(self, recipe_name: str):
        """
        Update P(m | all feedback so far in the episode)
        """
        logps = []
        for m in MOODS:
            lp = np.log(self.mood_prior[MOODS.index(m)])
            for (actor, spice, sat) in self.episode_data:
                lp += self._loglik_feedback_given_mood(actor, spice, sat, m, recipe_name)
            logps.append(lp)

        # Normalize log probs
        logps = np.array(logps)
        logps -= np.max(logps)
        ps = np.exp(logps)
        ps /= ps.sum()

        self.mood_posterior = ps

    # Recipe level updates ----------------------------------------------------
    def _pseudo_obs_weighted(self, actor: str, satisfaction: float) -> float:
        """
        Convert (actor, satisfaction) to a signal g
        weighted by mood posterior.
        """
        g_m = []

        # Compute expected g under each mood
        for m in MOODS:
            b = MOOD_BIAS[m][actor]
            if m == "neutral":
                sign_actor = +1.0 if actor == "human" else -1.0
                sign_sat = +1.0 if satisfaction > 0 else -1.0
                g = sign_actor * sign_sat
            else:
                g = 0.0

            g_m.append(g)

        g_expectation = float(np.dot(self.mood_posterior, g_m))
        return g_expectation

    def _update_phi(self, recipe_name: str, spice: str, g: float):
        """
        Normal-Normal update:
            g ~ N(φ, σ_obs^2)
            φ ~ N(theta_s, σ_r^2)
        """

        theta_mean = self.theta_mean[spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2

        # Prior
        prior_mean = theta_mean
        prior_var = sigma_r2

        # Posterior
        post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
        post_mean = post_var * (prior_mean/prior_var + g/sigma_obs2)

        self.phi_mean[recipe_name][spice] = post_mean
        self.phi_var[recipe_name][spice] = post_var

    # --- θ and μ updates (hierarchical pooling) ---
    def update_theta_and_mu(self):
        """
        After each episode, update:
            θ_s from all φ_{r,s}
            μ_s from all θ_s
        using Gaussian pooling.
        """

        # Update θ_s
        for s in self.spices:
            # gather φ_mean(r,s) over all recipes
            phis = []
            for r in self.recipes:
                if s in self.phi_mean[r]:
                    phis.append(self.phi_mean[r][s])

            if len(phis) == 0:
                continue

            y = float(np.mean(phis))
            prior_mean = self.mu_mean[s]
            prior_var = self.sigma0**2 + self.sigma_h**2
            sigma_obs2 = self.sigma_r**2

            post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
            post_mean = post_var * (prior_mean/prior_var + y/sigma_obs2)

            self.theta_mean[s] = post_mean
            self.theta_var[s] = post_var

        # Update μ_s
        for s in self.spices:
            y = self.theta_mean[s]
            prior_mean = 0.0
            prior_var = self.sigma0**2
            sigma_obs2 = self.sigma_h**2

            post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
            post_mean = post_var * (prior_mean/prior_var + y/sigma_obs2)

            self.mu_mean[s] = post_mean
            self.mu_var[s] = post_var

    def observe(self, recipe_name: str, spice: str, actor: str, satisfaction: float) -> None:
        """
        Process a transition (s, a, y) from the environment.
        Mood is hidden; we infer it internally.

        Steps:
          1. Add (actor, spice, sat) to episode_data
          2. Update mood posterior
          3. Compute expected preference signal E[g]
          4. Update φ
        """
        self.episode_data.append((actor, spice, satisfaction))
        self._update_mood_posterior(recipe_name)
        g = self._pseudo_obs_weighted(actor, satisfaction)
        self._update_phi(recipe_name, spice, g)

    def end_episode(self):
        """Clear episode buffers and update θ, μ."""
        self.update_theta_and_mu()
        self.episode_data = []
        self.mood_posterior = self.mood_prior.copy()

    def get_phi(self, recipe_name: str, spice: str) -> float:
        return self.phi_mean[recipe_name][spice]

    def preferred_actor(self, recipe_name: str, spice: str) -> str:
        phi = self.get_phi(recipe_name, spice)
        return "human" if phi >= 0 else "robot"

    def log_prob_prefer(self, recipe_name: str, spice: str, actor: str) -> float:
        phi = self.get_phi(recipe_name, spice)
        sign_actor = +1.0 if actor == "human" else -1.0
        logit = sign_actor * phi
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(np.log(max(p, 1e-9)))
