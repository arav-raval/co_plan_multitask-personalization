from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict

from .spices_config import DEFAULT_CONFIG, SpicesConfig

# Moods
MOODS = ("all_self", "neutral", "none_self")

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
        sigma0: float | None = None,
        sigma_h: float | None = None,
        sigma_r: float | None = None,
        sigma_obs: float | None = None,
        config: SpicesConfig | None = None,
    ) -> None:

        self.spices = list(spices)
        self.recipes = list(recipes)
        
        # Load configuration
        self.config = config if config is not None else DEFAULT_CONFIG

        # Variances (use config if not explicitly provided, otherwise use provided values)
        self.sigma0 = sigma0 if sigma0 is not None else self.config.hbm.sigma0
        self.sigma_h = sigma_h if sigma_h is not None else self.config.hbm.sigma_h
        self.sigma_r = sigma_r if sigma_r is not None else self.config.hbm.sigma_r
        self.sigma_obs = sigma_obs if sigma_obs is not None else self.config.hbm.sigma_obs

        # ---------------- Level 1: μ_s ----------------
        self.mu_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.mu_var: Dict[str, float] = {s: self.sigma0**2 for s in self.spices}

        # ---------------- Level 2: θ_s ----------------
        self.theta_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.theta_var: Dict[str, float] = {
            s: self.sigma0**2 + self.sigma_h**2 for s in self.spices
        }

        # ---------------- Level 3: φ_{r,s} ----------------
        self.phi_mean: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: 0.0 for s in self.spices}
        )
        self.phi_var: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: self.sigma_r**2 for s in self.spices}
        )
        
        # Exponential moving average for smoothing updates
        self.phi_ema: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {s: 0.0 for s in self.spices}
        )
        self.ema_alpha = self.config.hbm.ema_alpha

        # Per-step data for mood inference (actor, spice, satisfaction)
        self.episode_data: List[Tuple[str, str, float]] = []
        
        # Track recipe name for current episode (needed for batch updates)
        self._current_recipe_name: str | None = None
        
        # Track whether phi was updated this episode (to avoid unnecessary hierarchical updates)
        self._phi_updated_this_episode = False

        # Prior over mood (from config)
        self.mood_prior = np.array(self.config.mood_prior_array, dtype=float)
        self.mood_posterior = self.mood_prior.copy()
        
        # Track mood posterior history for smoothing (prevent rapid swings)
        self._mood_posterior_history = []
        
        # Track number of observations for adaptive EMA
        self._total_observations = 0
        
        # Track observation count per spice for early initialization
        self._spice_observation_count: Dict[str, int] = defaultdict(int)

        self.base_satisfaction_bias = base_satisfaction_bias
        self.mood_bias_strength = self.config.mood_bias_strength
        self.mood_bias = self.config.get_mood_bias()

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
        sigma_sat = self.config.mood.satisfaction_sigma
        log_lik = -0.5 * ((satisfaction - sat_expected) / sigma_sat)**2
        
        # Add small constant to prevent extreme values
        log_lik = np.clip(log_lik, self.config.mood.satisfaction_loglik_min, 
                         self.config.mood.satisfaction_loglik_max)
        
        return float(log_lik)

    def _update_mood_posterior(self, recipe_name: str):
        """
        Update mood posterior from all observations in current episode.
        
        CRITICAL FIX: Add smoothing to prevent rapid mood swings.
        """
        logps = []
        for m in MOODS:
            # Use stronger prior to prevent wild swings
            prior_weight = self.config.mood.mood_prior_weight
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
        # Use exponential moving average with high retention
        if len(self._mood_posterior_history) > 0:
            smoothing_alpha = self.config.mood.mood_smoothing_alpha
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
            demood_threshold = self.config.mood.demood_confidence_threshold
            if all_self_conf > demood_threshold:  # Confident in all_self
                mood_bias = self.mood_bias["all_self"][actor]
                # Map mood bias to expected satisfaction contribution
                expected_mood_logit = mood_bias / self.base_satisfaction_bias  # Normalize
                expected_mood_contribution = np.tanh(expected_mood_logit) * self.config.mood.demood_scale_factor
            elif none_self_conf > demood_threshold:  # Confident in none_self
                mood_bias = self.mood_bias["none_self"][actor]
                expected_mood_logit = mood_bias / self.base_satisfaction_bias
                expected_mood_contribution = np.tanh(expected_mood_logit) * self.config.mood.demood_scale_factor
            
            # De-mood satisfaction: subtract expected mood contribution
            demooded_satisfaction = satisfaction - expected_mood_contribution
            demooded_satisfaction = np.clip(demooded_satisfaction, -1.0, 1.0)
        
        # Continuous satisfaction: use (possibly de-mooded) value (preserves magnitude)
        # Scale to match expected phi range
        g_neutral = sign_actor * demooded_satisfaction * self.base_satisfaction_bias
        
        # FIX C: Check if satisfaction matches preference expectation
        # BUT: Only apply this check if we have a well-learned preference (avoid blocking early learning)
        current_phi = self.phi_mean[recipe_name].get(spice, 0.0)
        mismatch_threshold = self.config.hbm.preference_mismatch_threshold
        if abs(current_phi) > mismatch_threshold:  # Only check if preference is well-learned
            pref_logit = sign_actor * current_phi
            pref_expectation = pref_logit > 0  # positive if preference suggests satisfaction
            sat_positive = demooded_satisfaction > 0  # Use de-mooded satisfaction
            
            # If satisfaction doesn't match preference, likely mood contamination
            # BUT: Don't block updates completely, just reduce strength
            if pref_expectation != sat_positive:
                # Mismatch: reduce update strength (likely mood, not preference)
                g_neutral = g_neutral * self.config.hbm.preference_mismatch_penalty
        
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
        FIX 2: Early initialization - direct phi initialization for first few observations
        """
        self._total_observations += 1
        self._spice_observation_count[spice] += 1
        obs_count = self._spice_observation_count[spice]

        # FIX 2: Early initialization - directly set phi from first few observations
        # This breaks the circular dependency between mood inference and preference learning
        if obs_count <= 5:
            # Direct initialization: phi = sign(actor) * satisfaction * base_bias
            # No smoothing, no prior - just direct observation
            # g already contains: sign(actor) * satisfaction * base_bias
            # For early init, we want to directly use g as phi estimate
            phi_init = g  # Direct initialization
            self.phi_mean[recipe_name][spice] = phi_init
            self.phi_ema[recipe_name][spice] = phi_init
            self.phi_var[recipe_name][spice] = self.sigma_r**2
            return

        theta_mean = self.theta_mean[spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2 / learning_rate

        # Prior
        prior_mean = theta_mean
        prior_var = sigma_r2

        # Learning rate annealing (reduce LR over time for stability)
        anneal_factor = 1.0
        if self.config.hbm.use_lr_annealing:
            if self._total_observations > self.config.hbm.lr_annealing_start:
                if self._total_observations >= self.config.hbm.lr_annealing_end:
                    anneal_factor = self.config.hbm.lr_annealing_min / self.config.hbm.base_learning_rate
                else:
                    # Linear interpolation
                    progress = (self._total_observations - self.config.hbm.lr_annealing_start) / (
                        self.config.hbm.lr_annealing_end - self.config.hbm.lr_annealing_start)
                    anneal_factor = 1.0 - progress * (1.0 - self.config.hbm.lr_annealing_min / self.config.hbm.base_learning_rate)
        
        # Get current variance (before update) for variance-adaptive scaling and convergence detection
        current_var = self.phi_var[recipe_name].get(spice, sigma_r2)
        is_converged = current_var < self.config.hbm.variance_converged_threshold
        
        # Variance-adaptive learning rate scaling (to reduce oscillations when converged)
        # When variance is low (high confidence), make updates smaller to reduce oscillations
        # When variance is high (low confidence), allow larger updates for faster learning
        lr_scale = 1.0
        if self.config.hbm.use_variance_adaptive_lr:
            # Normalize: high var → 1.0 (allow updates), low var → min_scale (smaller updates)
            # When variance is very low, we're converged, so reduce update magnitude
            var_normalized = min(1.0, max(0.0, np.sqrt(current_var / sigma_r2)))
            lr_scale = (self.config.hbm.variance_adaptive_min_scale * (1 - var_normalized) + 
                       self.config.hbm.variance_adaptive_max_scale * var_normalized)
        
        # Apply both annealing and variance scaling to the effective observation variance
        # This scales how much we trust new observations vs. prior
        effective_sigma_obs2 = sigma_obs2 / (anneal_factor * lr_scale)

        # Posterior (with scaled observation variance)
        post_var = 1.0 / (1.0/prior_var + 1.0/effective_sigma_obs2)
        post_mean = post_var * (prior_mean/prior_var + g/effective_sigma_obs2)

        # FIX D: Adaptive EMA smoothing - less smoothing early, more when converged
        current_ema = self.phi_ema[recipe_name][spice]
        if current_ema == 0.0 and self.phi_mean[recipe_name][spice] == 0.0:
            smoothed_mean = post_mean
        else:
            
            # Adaptive EMA: less smoothing early (faster recovery), more when converged (stability)
            if self._total_observations < self.config.hbm.ema_early_threshold:
                ema_alpha = self.config.hbm.ema_early_alpha
            elif self._total_observations < self.config.hbm.ema_medium_threshold:
                ema_alpha = self.config.hbm.ema_medium_alpha
            elif is_converged:
                ema_alpha = self.config.hbm.ema_alpha_converged  # More smoothing when converged
            else:
                ema_alpha = self.ema_alpha  # Normal smoothing
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
                prior_var_mult = self.config.hbm.single_recipe_prior_var_multiplier
                obs_var_mult = self.config.hbm.single_recipe_obs_var_multiplier
                prior_var = (self.sigma0**2 + self.sigma_h**2) * prior_var_mult
                sigma_obs2 = self.sigma_r**2 * obs_var_mult
                # This makes theta move much more directly toward phi
                # CRITICAL FIX: For single recipe, mu (prior_mean) tends to stay near 0 and pull theta back
                # Instead of using mu as prior, use current theta as prior (momentum-based)
                # This completely removes mu's pull-back effect, allowing theta to converge freely
                prior_mean = self.theta_mean[s]  # Use current theta, not mu!
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
        effective_threshold = max(neutral_threshold, self.config.update.effective_threshold_min)
        
        # Batch update preferences using all episode observations if confident in neutral mood
        if neutral_conf >= effective_threshold and self._current_recipe_name:
            # Weight all observations by final confidence
            confidence_weight = (neutral_conf - effective_threshold) / (1.0 - effective_threshold)
            confidence_weight = max(confidence_weight, self.config.update.confidence_weight_min)
            
            # Process all observations from the episode
            for (actor, spice, satisfaction) in self.episode_data:
                g = self._pseudo_obs_weighted(actor, satisfaction, self._current_recipe_name, spice, neutral_threshold)
                
                # Scale by confidence: more confident = stronger updates
                g_weighted = g * confidence_weight
                
                # Learning rate based on confidence
                base_lr = self.config.hbm.base_learning_rate
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


class MultiHumanHierarchicalPreferenceModel:
    """
    Multi-human hierarchical Bayesian model of preferences.

    This is a "pure" preference model without mood inference, with three levels:

        μ_s           ~ N(μ0, σ0^2)                     (global, across humans & recipes)
        θ_{h,s}       ~ N(μ_s, σ_h^2)                   (human-specific)
        φ_{h,r,s}     ~ N(θ_{h,s}, σ_r^2)               (human+recipe-specific)

    where:
        - s indexes spices
        - r indexes recipes
        - h indexes humans

    The intended semantics are:
        - μ_s captures population-level structure shared across humans and recipes.
        - θ_{h,s} captures how human h deviates from the population mean.
        - φ_{h,r,s} captures how human h deviates on recipe r from their own θ_{h,s}.

    This class is designed to be independent from the environment / CSP code so
    that it can be unit-tested in isolation. Integration with `SpicesAssignCSPGenerator`
    is expected to pass in a `human_id` and `recipe_name` when observing data.
    """

    def __init__(
        self,
        spices: List[str],
        recipes: List[str],
        human_ids: List[str],
        mu0: float = 0.0,
        sigma0: float | None = None,
        sigma_h: float | None = None,
        sigma_r: float | None = None,
        sigma_obs: float | None = None,
        config: SpicesConfig | None = None,
    ) -> None:
        self.spices = list(spices)
        self.recipes = list(recipes)
        self.humans = list(human_ids)

        # Load configuration
        self.config = config if config is not None else DEFAULT_CONFIG

        # Variances (use config if not explicitly provided, otherwise use provided values)
        self.sigma0 = sigma0 if sigma0 is not None else self.config.hbm.sigma0
        self.sigma_h = sigma_h if sigma_h is not None else self.config.hbm.sigma_h
        self.sigma_r = sigma_r if sigma_r is not None else self.config.hbm.sigma_r
        self.sigma_obs = sigma_obs if sigma_obs is not None else self.config.hbm.sigma_obs

        # ---------------- Level 1: μ_s (global across humans & recipes) ----------------
        self.mu_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.mu_var: Dict[str, float] = {s: self.sigma0**2 for s in self.spices}

        # ---------------- Level 2: θ_{h,s} (human-specific) ----------------
        # theta_mean[human_id][spice]
        self.theta_mean: Dict[str, Dict[str, float]] = {
            h: {s: mu0 for s in self.spices} for h in self.humans
        }
        # theta_var[human_id][spice]
        self.theta_var: Dict[str, Dict[str, float]] = {
            h: {s: self.sigma0**2 + self.sigma_h**2 for s in self.spices}
            for h in self.humans
        }

        # ---------------- Level 3: φ_{h,r,s} (human+recipe-specific) ----------------
        # phi_mean[human_id][recipe_name][spice]
        self.phi_mean: Dict[str, Dict[str, Dict[str, float]]] = {
            h: {
                r: {s: 0.0 for s in self.spices}
                for r in self.recipes
            }
            for h in self.humans
        }
        # phi_var[human_id][recipe_name][spice]
        self.phi_var: Dict[str, Dict[str, Dict[str, float]]] = {
            h: {
                r: {s: self.sigma_r**2 for s in self.spices}
                for r in self.recipes
            }
            for h in self.humans
        }

    # -------------------------------------------------------------------------
    # Low-level update primitives
    # -------------------------------------------------------------------------
    def update_phi(self, human_id: str, recipe_name: str, spice: str, g: float) -> None:
        """
        Update φ_{h,r,s} for a single pseudo-observation g.

        Observation model:
            g ~ N(φ_{h,r,s}, σ_obs^2)
            φ_{h,r,s} ~ N(θ_{h,s}, σ_r^2)

        Posterior (Normal-Normal):
            post_var  = 1 / (1/σ_r^2 + 1/σ_obs^2)
            post_mean = post_var * (θ_{h,s}/σ_r^2 + g/σ_obs^2)
        """
        assert human_id in self.humans, f"Unknown human_id: {human_id}"
        assert recipe_name in self.recipes, f"Unknown recipe_name: {recipe_name}"
        assert spice in self.spices, f"Unknown spice: {spice}"

        theta_hs = self.theta_mean[human_id][spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2

        # Prior for φ
        prior_mean = theta_hs
        prior_var = sigma_r2

        # Posterior
        post_var = 1.0 / (1.0 / prior_var + 1.0 / sigma_obs2)
        post_mean = post_var * (prior_mean / prior_var + g / sigma_obs2)

        self.phi_mean[human_id][recipe_name][spice] = post_mean
        self.phi_var[human_id][recipe_name][spice] = post_var

    def update_theta_and_mu(self) -> None:
        """
        Update θ_{h,s} from all φ_{h,r,s}, then update μ_s from all θ_{h,s}.

        - For each human h, spice s:
              φ_{h,r,s} pooled over recipes r to update θ_{h,s}.
        - For each spice s:
              θ_{h,s} pooled over humans h to update μ_s.
        """
        # --- Update θ_{h,s} from φ_{h,r,s} ---
        for h in self.humans:
            for s in self.spices:
                # Gather φ over recipes for this human and spice
                phis: List[float] = []
                for r in self.recipes:
                    if s in self.phi_mean[h][r]:
                        phis.append(self.phi_mean[h][r][s])
                if not phis:
                    continue

                y = float(np.mean(phis))
                prior_mean = self.mu_mean[s]
                prior_var = self.sigma0**2 + self.sigma_h**2
                sigma_obs2 = self.sigma_r**2

                post_var = 1.0 / (1.0 / prior_var + 1.0 / sigma_obs2)
                post_mean = post_var * (prior_mean / prior_var + y / sigma_obs2)

                self.theta_mean[h][s] = post_mean
                self.theta_var[h][s] = post_var

        # --- Update μ_s from θ_{h,s} across humans ---
        for s in self.spices:
            thetas: List[float] = []
            for h in self.humans:
                if s in self.theta_mean[h]:
                    thetas.append(self.theta_mean[h][s])
            if not thetas:
                continue

            # Pool θ across humans
            y = float(np.mean(thetas))
            prior_mean = 0.0
            prior_var = self.sigma0**2
            sigma_obs2 = self.sigma_h**2

            post_var = 1.0 / (1.0 / prior_var + 1.0 / sigma_obs2)
            post_mean = post_var * (prior_mean / prior_var + y / sigma_obs2)

            self.mu_mean[s] = post_mean
            self.mu_var[s] = post_var

    # -------------------------------------------------------------------------
    # Convenience getters
    # -------------------------------------------------------------------------
    def get_mu(self, spice: str) -> float:
        """Get global preference μ_s."""
        return self.mu_mean[spice]

    def get_theta(self, human_id: str, spice: str) -> float:
        """Get human-level preference θ_{h,s}."""
        return self.theta_mean[human_id][spice]

    def get_phi(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Get human+recipe-specific preference φ_{h,r,s}."""
        return self.phi_mean[human_id][recipe_name][spice]
