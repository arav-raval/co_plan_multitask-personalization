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
        self.ema_alpha = 0.4  # Increased from 0.1 to 0.4 for faster convergence (40% of new info per update)

        # Per-step data for mood inference (actor, spice, satisfaction)
        self.episode_data: List[Tuple[str, str, float]] = []
        
        # Track recipe name for current episode (needed for batch updates)
        self._current_recipe_name: str | None = None
        
        # Track whether phi was updated this episode (to avoid unnecessary hierarchical updates)
        self._phi_updated_this_episode = False

        # Prior over mood - STRONGER bias towards neutral to prevent wild swings
        self.mood_prior = np.array([0.02, 0.96, 0.02], dtype=float) # Much stronger bias towards neutral mood
        self.mood_posterior = self.mood_prior.copy()
        
        # Track mood posterior history for smoothing (prevent rapid swings)
        self._mood_posterior_history = []
        
        # Track number of observations for adaptive EMA
        self._total_observations = 0

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
        
        CONTINUOUS SATISFACTION: satisfaction is now continuous in [-1, +1].
        We model it using a Beta likelihood for better handling of continuous values.
        """
        phi = self.phi_mean[recipe_name][spice]
        sign_actor = +1.0 if actor == "human" else -1.0

        logit = sign_actor * phi + self.mood_bias[m][actor]
        p = 1.0 / (1.0 + np.exp(-logit))

        # For continuous satisfaction, model it as a scaled Beta distribution
        # Map satisfaction from [-1, +1] to [0, 1] probability space
        p_obs = (satisfaction + 1.0) / 2.0  # Map [-1, +1] → [0, 1]
        p_obs = np.clip(p_obs, 1e-6, 1.0 - 1e-6)  # Avoid log(0)
        
        # CRITICAL FIX: Use a more stable likelihood for continuous satisfaction
        # The Beta likelihood with kappa=10 was too sensitive and caused wild mood swings
        # Instead, use a Gaussian approximation which is more stable
        
        # Map expected probability p to expected satisfaction: p → sat_expected
        # p = 0.0 → sat = -1.0, p = 0.5 → sat = 0.0, p = 1.0 → sat = +1.0
        sat_expected = 2.0 * p - 1.0
        
        # Use Gaussian likelihood: satisfaction ~ N(sat_expected, sigma_sat^2)
        # This is more stable than Beta and prevents extreme log-likelihoods
        sigma_sat = 0.3  # Reasonable variance for satisfaction
        log_lik = -0.5 * ((satisfaction - sat_expected) / sigma_sat)**2
        
        # Add small constant to prevent extreme values
        log_lik = np.clip(log_lik, -10.0, 0.0)
        
        return float(log_lik)

    def _update_mood_posterior(self, recipe_name: str):
        """
        Update mood posterior from all observations in current episode.
        
        CRITICAL FIX: Add smoothing to prevent rapid mood swings.
        """
        logps = []
        for m in MOODS:
            # Use stronger prior to prevent wild swings
            prior_weight = 2.0  # Weight prior more heavily
            lp = np.log(self.mood_prior[MOODS.index(m)]) * prior_weight
            
            for (actor, spice, sat) in self.episode_data:
                lp += self._loglik_feedback_given_mood(actor, spice, sat, m, recipe_name)
            logps.append(lp)

        # Normalize log probs
        logps = np.array(logps)
        logps -= np.max(logps)
        ps = np.exp(logps)
        ps /= ps.sum()
        
        # CRITICAL FIX: Smooth with previous posterior to prevent rapid swings
        # Use exponential moving average with high retention (0.7 = keep 70% of old, 30% new)
        if len(self._mood_posterior_history) > 0:
            smoothing_alpha = 0.3  # Only 30% new info per update
            ps = smoothing_alpha * ps + (1 - smoothing_alpha) * self.mood_posterior
            ps /= ps.sum()  # Renormalize
        
        self.mood_posterior = ps
        self._mood_posterior_history.append(ps.copy())

    # Recipe level updates ----------------------------------------------------
    def _pseudo_obs_weighted(self, actor: str, satisfaction: float, recipe_name: str, spice: str, neutral_threshold: float = 0.5) -> float:
        """
        Convert (actor, satisfaction) to a preference signal weighted by mood posterior.
        Weighted by neutral confident to reduce noise from mood-biased episodes.
        
        CONTINUOUS SATISFACTION: Now satisfaction is continuous in [-1, +1], so we preserve
        magnitude information. Strong satisfaction (e.g., +0.9) produces stronger updates than
        weak satisfaction (e.g., +0.2).
        
        FIX 1: Scale the preference signal to match expected phi range.
        FIX A: De-mood satisfaction signal - subtract expected mood contribution
        FIX C: Use preference mismatch as signal - reduce update if satisfaction doesn't match preference
        """
        neutral_idx = MOODS.index("neutral")
        all_self_idx = MOODS.index("all_self")
        none_self_idx = MOODS.index("none_self")
        
        neutral_conf = self.mood_posterior[neutral_idx]
        all_self_conf = self.mood_posterior[all_self_idx]
        none_self_conf = self.mood_posterior[none_self_idx]
        
        sign_actor = +1.0 if actor == "human" else -1.0
        
        # CRITICAL FIX: Only de-mood if we're NOT confident in neutral mood
        # If we're confident in neutral (>= threshold), satisfaction should reflect preferences directly
        # De-mooding when confident in neutral would incorrectly remove valid preference signals!
        demooded_satisfaction = satisfaction
        if neutral_conf < neutral_threshold:
            # Only de-mood if we're uncertain about neutral mood
            # If we're confident in a non-neutral mood, subtract expected mood contribution
            expected_mood_contribution = 0.0
            if all_self_conf > 0.3:  # Confident in all_self
                mood_bias = self.mood_bias["all_self"][actor]
                # Map mood bias to expected satisfaction contribution
                # mood_bias = ±6.0, normalize to [-1, +1] range
                # Use sigmoid to map logit space to satisfaction space
                expected_mood_logit = mood_bias / self.base_satisfaction_bias  # Normalize
                expected_mood_contribution = np.tanh(expected_mood_logit) * 0.3  # Scale down more conservatively
            elif none_self_conf > 0.3:  # Confident in none_self
                mood_bias = self.mood_bias["none_self"][actor]
                expected_mood_logit = mood_bias / self.base_satisfaction_bias
                expected_mood_contribution = np.tanh(expected_mood_logit) * 0.3
            
            # De-mood satisfaction: subtract expected mood contribution
            demooded_satisfaction = satisfaction - expected_mood_contribution
            demooded_satisfaction = np.clip(demooded_satisfaction, -1.0, 1.0)
        
        # Continuous satisfaction: use (possibly de-mooded) value (preserves magnitude)
        # Scale to match expected phi range
        g_neutral = sign_actor * demooded_satisfaction * self.base_satisfaction_bias
        
        # FIX C: Check if satisfaction matches preference expectation
        # BUT: Only apply this check if we have a well-learned preference (avoid blocking early learning)
        current_phi = self.phi_mean[recipe_name].get(spice, 0.0)
        if abs(current_phi) > 1.0:  # Only check if preference is well-learned
            pref_logit = sign_actor * current_phi
            pref_expectation = pref_logit > 0  # positive if preference suggests satisfaction
            sat_positive = demooded_satisfaction > 0  # Use de-mooded satisfaction
            
            # If satisfaction doesn't match preference, likely mood contamination
            # BUT: Don't block updates completely, just reduce strength
            if pref_expectation != sat_positive:
                # Mismatch: reduce update strength (likely mood, not preference)
                g_neutral = g_neutral * 0.3  # Reduced from 0.1 to 0.3 - less aggressive
        
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
        
        FIX D: Adaptive EMA - less smoothing early in learning
        """
        self._total_observations += 1

        theta_mean = self.theta_mean[spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2 / learning_rate

        # Prior
        prior_mean = theta_mean
        prior_var = sigma_r2

        # Posterior
        post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
        post_mean = post_var * (prior_mean/prior_var + g/sigma_obs2)

        # FIX D: Adaptive EMA smoothing - less smoothing early in learning
        current_ema = self.phi_ema[recipe_name][spice]
        if current_ema == 0.0 and self.phi_mean[recipe_name][spice] == 0.0:
            smoothed_mean = post_mean
        else:
            # Adaptive EMA: less smoothing early (faster recovery from mistakes)
            if self._total_observations < 20:
                ema_alpha = 0.7  # Less smoothing (70% new info) early
            elif self._total_observations < 50:
                ema_alpha = 0.5  # Medium smoothing
            else:
                ema_alpha = self.ema_alpha  # Normal smoothing (40% new info) later
            smoothed_mean = ema_alpha * post_mean + (1 - ema_alpha) * current_ema
        
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

            # FIX 3: For single recipe case, use much stronger updates
            # Reduce both prior variance and observation variance for faster convergence
            if len(self.recipes) == 1:
                # Much stronger updates: reduce both variances
                prior_var = (self.sigma0**2 + self.sigma_h**2) * 0.25  # 0.5 instead of 2.0 (4x reduction)
                sigma_obs2 = self.sigma_r**2 * 0.25  # 0.25 instead of 1.0 (4x reduction)
                # This makes theta move much more directly toward phi
                # CRITICAL FIX: For single recipe, mu (prior_mean) tends to stay near 0 and pull theta back
                # Instead of using mu as prior, use a weaker prior centered at current theta to allow more movement
                # This prevents mu from pulling theta back toward 0
                prior_mean = self.theta_mean[s] * 0.3 + self.mu_mean[s] * 0.7  # Weighted: mostly current theta, some mu
                post_var = 1.0 / (1.0/prior_var + 1.0/sigma_obs2)
                post_mean = post_var * (prior_mean/prior_var + y/sigma_obs2)
            else:
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
        (Update mood posterior only - DON'T update φ yet, wait until end_episode for batch updates)
        
        BATCH UPDATE APPROACH: Collect observations during episode, update preferences
        at end of episode if confident in neutral mood.
        """
        self.episode_data.append((actor, spice, satisfaction))
        self._current_recipe_name = recipe_name  # Track recipe for batch updates
        self._update_mood_posterior(recipe_name)
        self._phi_updated_this_episode = False  # Reset flag at start of episode
        
        # DON'T update phi here - wait until end_episode() for batch processing

    def end_episode(self, neutral_threshold: float = 0.5):
        """
        Clear episode buffers and update θ, μ.
        
        BATCH UPDATE APPROACH: Process all episode observations at once if confident
        in neutral mood at the end. This uses better mood inference (based on all
        episode data) to decide whether to update preferences.
        
        FIX B: Stricter filtering with confidence weighting
        """
        # Check final mood confidence (after seeing all episode observations)
        neutral_idx = MOODS.index("neutral")
        neutral_conf = self.mood_posterior[neutral_idx]
        
        # FIX B: Stricter threshold - only update if very confident in neutral mood
        effective_threshold = max(neutral_threshold, 0.7)  # At least 0.7 confidence required
        
        # Batch update preferences using all episode observations if confident in neutral mood
        if neutral_conf >= effective_threshold and self._current_recipe_name:
            # Weight all observations by final confidence
            confidence_weight = (neutral_conf - effective_threshold) / (1.0 - effective_threshold)
            confidence_weight = max(confidence_weight, 0.1)  # Minimum 10% weight
            
            # Process all observations from the episode
            for (actor, spice, satisfaction) in self.episode_data:
                g = self._pseudo_obs_weighted(actor, satisfaction, self._current_recipe_name, spice, neutral_threshold)
                
                # Scale by confidence: more confident = stronger updates
                g_weighted = g * confidence_weight
                
                # Learning rate based on confidence
                base_lr = 0.15
                learning_rate = base_lr * confidence_weight
                
                self._update_phi(self._current_recipe_name, spice, g_weighted, learning_rate=learning_rate)
                self._phi_updated_this_episode = True  # Mark that phi was updated
        
        # CRITICAL FIX: Only update hierarchical parameters (theta, mu) if phi was actually updated
        # This prevents theta/mu from drifting when no new preference information is available
        # (e.g., during non-neutral episodes where phi doesn't change)
        if self._phi_updated_this_episode:
            self.update_theta_and_mu()
        
        # Clear episode buffers
        self.episode_data = []
        # CRITICAL: Don't reset mood posterior to prior - keep it for next episode
        # This maintains continuity and prevents wild swings at episode boundaries
        # Only reset if we want to start fresh (commented out for now)
        # self.mood_posterior = self.mood_prior.copy()
        self._current_recipe_name = None

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
