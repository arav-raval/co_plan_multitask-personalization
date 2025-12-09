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
        base_satisfaction_bias: float = 3.0, # pull from SpiceEnv
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
        
        # Exponential moving average for smoothing updates
        self.phi_ema: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: 0.0 for s in self.spices}
        )
        self.ema_alpha = 0.1

        # Per-step data for mood inference (actor, spice, satisfaction)
        self.episode_data: List[Tuple[str, str, float]] = []

        # Prior over mood
        self.mood_prior = np.array([0.05, 0.9, 0.05], dtype=float) # bias towards neutral mood
        self.mood_posterior = self.mood_prior.copy()

        self.base_satisfaction_bias = base_satisfaction_bias
        self.mood_bias_strength = base_satisfaction_bias * 2.0
        self.mood_bias = {
            "all_self":   {"human": +self.mood_bias_strength, "robot": -self.mood_bias_strength},
            "neutral":    {"human": 0.0,  "robot": 0.0},
            "none_self":  {"human": -self.mood_bias_strength, "robot": +self.mood_bias_strength},
        }

    # --- Mood inference ---
    def _loglik_feedback_given_mood(
        self, actor: str, spice: str, satisfaction: float, m: str, recipe_name: str
    ) -> float:
        """
        Compute log-probability of satisfaction given mood, actor, and spice using the 
        generative model logit = sign(actor)*phi + mood_bias
        """
        phi = self.phi_mean[recipe_name][spice]
        sign_actor = +1.0 if actor == "human" else -1.0

        logit = sign_actor * phi + self.mood_bias[m][actor]
        p = 1.0 / (1.0 + np.exp(-logit))

        if satisfaction > 0:
            return np.log(max(p, 1e-9))
        else:
            return np.log(max(1 - p, 1e-9))

    def _update_mood_posterior(self, recipe_name: str):
        """
        Update mood posterior from all observations in current episode.
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
    def _pseudo_obs_weighted(self, actor: str, satisfaction: float, neutral_threshold: float = 0.5) -> float:
        """
        Convert (actor, satisfaction) to a preference signal weighted by mood posterior.
        Weighted by neutral confident to reduce noise from mood-biased episodes.
        """
        neutral_idx = MOODS.index("neutral")
        neutral_conf = self.mood_posterior[neutral_idx]
        
        sign_actor = +1.0 if actor == "human" else -1.0
        sign_sat = +1.0 if satisfaction > 0 else -1.0
        g_neutral = sign_actor * sign_sat  # Preference signal if mood is neutral
        
        # Use full signal (neutral mood)
        if neutral_conf >= neutral_threshold:
            return float(g_neutral)
        # Weight by neutral confident to reduce noise 
        else:
            return float(neutral_conf * g_neutral)

    def _update_phi(self, recipe_name: str, spice: str, g: float, learning_rate: float = 1.0):
        """
        Normal-Normal update:
            g ~ N(φ, σ_obs^2)
            φ ~ N(theta_s, σ_r^2)
        """

        theta_mean = self.theta_mean[spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2 / learning_rate

        # Prior
        prior_mean = theta_mean
        prior_var = sigma_r2

        # Posterior
        post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
        post_mean = post_var * (prior_mean/prior_var + g/sigma_obs2)

        # Smoothing update
        current_ema = self.phi_ema[recipe_name][spice]
        if current_ema == 0.0 and self.phi_mean[recipe_name][spice] == 0.0:
            smoothed_mean = post_mean
        else:
            smoothed_mean = self.ema_alpha * post_mean + (1 - self.ema_alpha) * current_ema
        
        self.phi_mean[recipe_name][spice] = smoothed_mean
        self.phi_ema[recipe_name][spice] = smoothed_mean
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

    def observe(self, recipe_name: str, spice: str, actor: str, satisfaction: float, neutral_threshold: float = 0.5) -> None:
        """
        Process a transition (s, a, y) from the environment.
        (Update mood, compute preference signal, update φ)
        """
        self.episode_data.append((actor, spice, satisfaction))
        self._update_mood_posterior(recipe_name)
        
        # Only compute signal if confident in neutral mood
        neutral_idx = MOODS.index("neutral")
        neutral_conf = self.mood_posterior[neutral_idx]
        
        if neutral_conf >= neutral_threshold:
            g = self._pseudo_obs_weighted(actor, satisfaction, neutral_threshold)
            
            base_lr = 0.05  
            confidence_boost = (neutral_conf - neutral_threshold) / (1.0 - neutral_threshold) 
            learning_rate = base_lr + (0.1 - base_lr) * confidence_boost 
            self._update_phi(recipe_name, spice, g, learning_rate=learning_rate)

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
