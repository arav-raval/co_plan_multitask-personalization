from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from .config.spices_config import DEFAULT_CONFIG, SpicesConfig

MOODS = ("all_self", "neutral", "none_self")
DEFAULT_HUMAN = "human"


# ----- Mood utilities (single source of truth for generation and inference) -----


def sample_episode_mood(
    rng: np.random.Generator,
    prior: np.ndarray | None = None,
) -> str:
    """Sample a mood for the current episode. Uses DEFAULT_CONFIG prior if not specified."""
    if prior is None:
        prior = np.array(DEFAULT_CONFIG.mood.mood_prior, dtype=float)
    mood = rng.choice(MOODS, p=prior)
    return str(mood)


def compute_mood_bias(mood: str, actor: str) -> float:
    """
    Compute mood bias that overrides base preferences (generative model).

    Mood semantics:
        "all_self"  — human wants to do everything; satisfaction only when human acts.
        "none_self" — human doesn't want to do anything; satisfaction only when robot acts.
        "neutral"   — no mood override; base preferences apply.
    """
    bias_dict = DEFAULT_CONFIG.get_mood_bias()
    return bias_dict.get(mood, {}).get(actor, 0.0)


class MoodModel:
    """
    Handles sampling the current episode's mood.

    Moods and prior probabilities are taken from MOODS and MoodConfig so that
    the generative distribution and the HBM's inference prior are always in sync.
    """
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.current_mood: str | None = None

    def sample_mood(self) -> str:
        """Sample a mood for the current episode."""
        self.current_mood = sample_episode_mood(self.rng)
        return self.current_mood


class HierarchicalPreferenceModel:
    """
    Unified hierarchical Bayesian model of human preferences.

    Structure
    ---------
        μ_s           ~ N(μ0, σ0²)           — global, shared across all humans
        θ_{h,s}       ~ N(μ_s, σ_h²)         — human-specific
        φ_{h,r,s}     ~ N(θ_{h,s}, σ_r²)     — human+recipe-specific

    where h indexes humans, r indexes recipes, s indexes spices.
    Mood m_t is latent and inferred per episode, per human
    """

    def __init__(
        self,
        spices: List[str],
        recipes: List[str] | None = None,
        mu0: float = 0.0,
        sigma0: float | None = None,
        sigma_h: float | None = None,
        sigma_r: float | None = None,
        sigma_obs: float | None = None,
        config: SpicesConfig | None = None,
    ) -> None:
        self.spices = list(spices)
        self.config = config if config is not None else DEFAULT_CONFIG

        self.sigma0 = sigma0 if sigma0 is not None else self.config.hbm.sigma0
        self.sigma_h = sigma_h if sigma_h is not None else self.config.hbm.sigma_h
        self.sigma_r = sigma_r if sigma_r is not None else self.config.hbm.sigma_r
        self.sigma_obs = sigma_obs if sigma_obs is not None else self.config.hbm.sigma_obs

        # Level 1: global μ_s (one entry per spice, shared across all humans).
        self.mu_mean: Dict[str, float] = {s: mu0 for s in self.spices}
        self.mu_var: Dict[str, float] = {s: self.sigma0**2 for s in self.spices}

        # Per-human state (keyed by human_id).  Populated via register_human().
        self._theta_mean: Dict[str, Dict[str, float]] = {}
        self._theta_var: Dict[str, Dict[str, float]] = {}
        self._phi_mean: Dict[str, Any] = {}
        self._phi_var: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Per-human episode state
        self._mood_posterior: Dict[str, np.ndarray] = {}
        self._episode_data: Dict[str, List[Tuple[str, str, float]]] = {}
        self._current_recipe: Dict[str, str | None] = {}
        self._phi_updated: Dict[str, bool] = {}
        # Running per-mood log-likelihood accumulator (incremental mood inference).
        self._log_lik_accum: Dict[str, np.ndarray] = {}

        # Per-human, per-recipe, per-spice observation counts
        self._obs_count: Dict[str, Dict[str, Dict[str, int]]] = {}
        # Per-human total observations (lifetime counter, kept for reference).
        self._total_observations: Dict[str, int] = {}
        # Per-human, per-recipe total observations.
        # Used for LR annealing and EMA thresholds instead of the lifetime
        # counter so that training on many recipes does not exhaust the annealing
        # schedule before the agent reaches a new (e.g. test) recipe.
        self._recipe_total_obs: Dict[str, Dict[str, int]] = {}
        # Per-human episode count for batch update_theta_and_mu.
        self._episode_count: Dict[str, int] = {}

        self.base_satisfaction_bias = self.config.satisfaction.base_satisfaction_bias
        self.mood_prior = np.array(self.config.mood_prior_array, dtype=float)
        self.mood_bias = self.config.get_mood_bias()
        self.ema_alpha = self.config.hbm.ema_alpha

        # Register default human (single human tests) and associated recipes
        self.register_human(DEFAULT_HUMAN)
        if recipes:
            for r in recipes:
                self.register_recipe(DEFAULT_HUMAN, r)


    # ----- Registration -----
    def register_human(self, human_id: str) -> None:
        """Register a new human, deriving θ from the current global μ"""
        if human_id in self._theta_mean:
            return

        # Thetas
        self._theta_mean[human_id] = {s: self.mu_mean[s] for s in self.spices}
        self._theta_var[human_id] = {s: self.sigma_h**2 for s in self.spices}

        # Phi
        self._phi_mean[human_id] = defaultdict(dict)
        self._phi_var[human_id] = {}

        # Mood Posteriors / Episode Data
        self._mood_posterior[human_id] = self.mood_prior.copy()
        self._log_lik_accum[human_id] = np.zeros(len(MOODS))
        self._episode_data[human_id] = []
        self._current_recipe[human_id] = None
        self._phi_updated[human_id] = False
        self._obs_count[human_id] = {}
        self._total_observations[human_id] = 0
        self._recipe_total_obs[human_id] = {}
        self._episode_count[human_id] = 0

    def register_recipe(self, human_id: str, recipe_name: str) -> None:
        """Register a new recipe for a human, deriving φ from current θ"""
        if human_id not in self._theta_mean:
            self.register_human(human_id)
        if recipe_name not in self._phi_var.get(human_id, {}):
            # Theta
            theta = self._theta_mean[human_id]

            # Phi
            self._phi_mean[human_id][recipe_name] = {s: theta[s] for s in self.spices}
            self._phi_var[human_id][recipe_name] = {s: self.sigma_r**2 for s in self.spices}

            # Observation Count
            self._obs_count[human_id][recipe_name] = {s: 0 for s in self.spices}
            self._recipe_total_obs[human_id][recipe_name] = 0

    def _ensure_registered(self, human_id: str, recipe_name: str) -> None:
        if human_id not in self._theta_mean:
            self.register_human(human_id)
        if recipe_name not in self._phi_var.get(human_id, {}):
            self.register_recipe(human_id, recipe_name)

    @property
    def theta_mean(self) -> Dict[str, float]:
        """Human-level θ for the default human"""
        return self._theta_mean[DEFAULT_HUMAN]

    @property
    def theta_var(self) -> Dict[str, float]:
        """Human-level θ variance for the default human"""
        return self._theta_var[DEFAULT_HUMAN]

    @property
    def phi_mean(self) -> Any:
        """Recipe→spice phi dict for the default human"""
        return self._phi_mean[DEFAULT_HUMAN]

    @property
    def mood_posterior(self) -> np.ndarray:
        """Mood posterior for the default human."""
        return self._mood_posterior[DEFAULT_HUMAN]

    @mood_posterior.setter
    def mood_posterior(self, value: np.ndarray) -> None:
        self._mood_posterior[DEFAULT_HUMAN] = value


    # ------ Mood inference ------ 
    def _loglik_feedback_given_mood(
        self,
        human_id: str,
        actor: str,
        spice: str,
        satisfaction: float,
        recipe_name: str,
    ) -> np.ndarray:
        """
        Compute log P(satisfaction | mood, actor, spice) for each mood.

        Returns array of length 3: [all_self, neutral, none_self] matching MOODS order.

        Generative model:
            logit  = sign(actor) * phi + mood_bias[m][actor]          (neutral)
            logit  = mood_bias[m][actor] + sign(actor)*phi*pref_weight (non-neutral)
            p      = sigmoid(logit)
            sat    ~ N(2p - 1, sigma_sat²)

        For non-neutral moods the preference phi is down-weighted (pref_weight < 1)
        because mood dominates.  The weight is further reduced when satisfaction
        disagrees with the preference expectation.

        When recipe-specific phi ≈ 0 (early learning), falls back to the
        human-level theta to provide a non-zero signal.
        """
        phi = self._phi_mean[human_id][recipe_name].get(spice, 0.0)
        if abs(phi) < 1e-6 and spice in self._theta_mean[human_id]:
            phi = self._theta_mean[human_id][spice]
        phi = float(np.clip(phi, -self.base_satisfaction_bias, self.base_satisfaction_bias))

        sign_actor = 1.0 if actor == "human" else -1.0
        pref_expectation = (sign_actor * phi) > 0
        matches_preference = pref_expectation == (satisfaction > 0)
        pref_weight_match = self.config.mood.non_neutral_pref_weight_match
        pref_weight_mismatch = self.config.mood.non_neutral_pref_weight_mismatch

        sigma_sat = self.config.mood.satisfaction_sigma
        loglik_min = self.config.mood.satisfaction_loglik_min
        loglik_max = self.config.mood.satisfaction_loglik_max

        logits = np.zeros(3)
        for i, m in enumerate(MOODS):
            if m == "neutral":
                logits[i] = sign_actor * phi + self.mood_bias[m][actor]
            else:
                mood_bias_val = self.mood_bias[m][actor]
                pref_weight = pref_weight_match if matches_preference else pref_weight_mismatch
                logits[i] = mood_bias_val + sign_actor * phi * pref_weight

        p = 1.0 / (1.0 + np.exp(-logits))
        sat_expected = 2.0 * p - 1.0
        log_lik = -0.5 * ((satisfaction - sat_expected) / sigma_sat) ** 2
        return np.clip(log_lik, loglik_min, loglik_max)

    def _update_mood_posterior(
        self,
        human_id: str,
        recipe_name: str,
        actor: str,
        spice: str,
        satisfaction: float,
    ) -> None:
        """
        Incrementally update the mood posterior for one new observation.

        Instead of re-processing all episode observations each step (O(N²) total),
        we maintain a running log-likelihood accumulator and add only the new
        observation's contribution (O(N) total).  The resulting posterior is
        identical to the batch computation.
        """
        # Accumulate per-mood log-likelihood contribution of the new observation.
        delta = self._loglik_feedback_given_mood(
            human_id, actor, spice, satisfaction, recipe_name
        )
        self._log_lik_accum[human_id] += delta

        # Full log-posterior = weighted log-prior + accumulated log-likelihoods.
        prior_weight = self.config.mood.mood_prior_weight
        logps = np.log(self.mood_prior) * prior_weight + self._log_lik_accum[human_id]
        logps -= np.max(logps)
        ps = np.exp(logps)
        ps /= ps.sum()

        # EMA smoothing to prevent abrupt swings.
        smoothing_alpha = self.config.mood.mood_smoothing_alpha
        ps = smoothing_alpha * ps + (1 - smoothing_alpha) * self._mood_posterior[human_id]
        ps /= ps.sum()

        self._mood_posterior[human_id] = ps

    # ----- Recipe-level preference updates -----
    def _pseudo_obs_weighted(
        self,
        human_id: str,
        actor: str,
        satisfaction: float,
        recipe_name: str,
        spice: str,
        neutral_threshold: float = 0.5,
    ) -> float:
        """
        Convert (actor, satisfaction) to a preference signal weighted by mood confidence.

        When the mood is confidently neutral the full satisfaction signal is used.
        When mood is uncertain the signal is down-weighted and optionally de-mooded.
        """
        neutral_idx = MOODS.index("neutral")
        all_self_idx = MOODS.index("all_self")
        none_self_idx = MOODS.index("none_self")

        neutral_conf = self._mood_posterior[human_id][neutral_idx]
        all_self_conf = self._mood_posterior[human_id][all_self_idx]
        none_self_conf = self._mood_posterior[human_id][none_self_idx]

        sign_actor = +1.0 if actor == "human" else -1.0

        demooded_satisfaction = satisfaction
        if neutral_conf < neutral_threshold:
            expected_mood_contribution = 0.0
            demood_threshold = self.config.mood.demood_confidence_threshold
            if all_self_conf > demood_threshold:
                mood_bias = self.mood_bias["all_self"][actor]
                expected_mood_logit = mood_bias / self.base_satisfaction_bias
                expected_mood_contribution = (
                    np.tanh(expected_mood_logit) * self.config.mood.demood_scale_factor
                )
            elif none_self_conf > demood_threshold:
                mood_bias = self.mood_bias["none_self"][actor]
                expected_mood_logit = mood_bias / self.base_satisfaction_bias
                expected_mood_contribution = (
                    np.tanh(expected_mood_logit) * self.config.mood.demood_scale_factor
                )
            demooded_satisfaction = np.clip(
                satisfaction - expected_mood_contribution, -1.0, 1.0
            )

        g_neutral = sign_actor * demooded_satisfaction * self.base_satisfaction_bias

        current_phi = self._phi_mean[human_id][recipe_name].get(spice, 0.0)
        mismatch_threshold = self.config.hbm.preference_mismatch_threshold
        if abs(current_phi) > mismatch_threshold:
            pref_expectation = (sign_actor * current_phi) > 0
            sat_positive = demooded_satisfaction > 0
            if pref_expectation != sat_positive:
                g_neutral = g_neutral * self.config.hbm.preference_mismatch_penalty

        if neutral_conf >= neutral_threshold:
            return float(g_neutral)
        return float(neutral_conf * g_neutral)

    def _update_phi(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        g: float,
        learning_rate: float = 1.0,
    ) -> None:
        """
        Update φ_{h,r,s} via Normal-Normal posterior with adaptive EMA.

        Early observations (per human+recipe+spice) are applied directly without
        smoothing to break the cold-start dependency on theta.
        """
        self._total_observations[human_id] += 1
        self._obs_count[human_id][recipe_name][spice] += 1
        obs_count = self._obs_count[human_id][recipe_name][spice]

        # Recipe-local observation count: used for all learning-rate and EMA
        # decisions so that multi-recipe training does not exhaust the annealing
        # schedule before the agent reaches a new recipe.
        recipe_obs = self._recipe_total_obs[human_id].get(recipe_name, 0)
        self._recipe_total_obs[human_id][recipe_name] = recipe_obs + 1

        # Direct initialization for the first few per-recipe observations.
        if obs_count <= 5:
            self._phi_mean[human_id][recipe_name][spice] = g
            self._phi_var[human_id][recipe_name][spice] = self.sigma_r**2
            return

        theta_mean = self._theta_mean[human_id][spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2 / learning_rate

        # Learning-rate annealing (per-recipe count, not lifetime total).
        anneal_factor = 1.0
        if self.config.hbm.use_lr_annealing:
            if recipe_obs > self.config.hbm.lr_annealing_start:
                if recipe_obs >= self.config.hbm.lr_annealing_end:
                    anneal_factor = (
                        self.config.hbm.lr_annealing_min / self.config.hbm.base_learning_rate
                    )
                else:
                    progress = (recipe_obs - self.config.hbm.lr_annealing_start) / (
                        self.config.hbm.lr_annealing_end - self.config.hbm.lr_annealing_start
                    )
                    anneal_factor = 1.0 - progress * (
                        1.0 - self.config.hbm.lr_annealing_min / self.config.hbm.base_learning_rate
                    )

        current_var = self._phi_var[human_id][recipe_name].get(spice, sigma_r2)
        is_converged = current_var < self.config.hbm.variance_converged_threshold

        # Variance-adaptive LR scaling.
        lr_scale = 1.0
        if self.config.hbm.use_variance_adaptive_lr:
            var_normalized = min(1.0, max(0.0, np.sqrt(current_var / sigma_r2)))
            lr_scale = (
                self.config.hbm.variance_adaptive_min_scale * (1 - var_normalized)
                + self.config.hbm.variance_adaptive_max_scale * var_normalized
            )

        effective_sigma_obs2 = sigma_obs2 / (anneal_factor * lr_scale)
        post_var = 1.0 / (1.0 / sigma_r2 + 1.0 / effective_sigma_obs2)
        post_mean = post_var * (theta_mean / sigma_r2 + g / effective_sigma_obs2)

        # Adaptive EMA: skip smoothing on the first real update after registration.
        # EMA phase thresholds also use per-recipe count (same reasoning as annealing).
        current_phi = self._phi_mean[human_id][recipe_name][spice]
        if current_phi == theta_mean:
            smoothed_mean = post_mean
        else:
            if recipe_obs < self.config.hbm.ema_early_threshold:
                ema_alpha = self.config.hbm.ema_early_alpha
            elif recipe_obs < self.config.hbm.ema_medium_threshold:
                ema_alpha = self.config.hbm.ema_medium_alpha
            elif is_converged:
                ema_alpha = self.config.hbm.ema_alpha_converged
            else:
                ema_alpha = self.ema_alpha
            smoothed_mean = ema_alpha * post_mean + (1 - ema_alpha) * current_phi

        self._phi_mean[human_id][recipe_name][spice] = smoothed_mean
        self._phi_var[human_id][recipe_name][spice] = post_var

    # ----- Hierarchical Pooling -----
    def update_theta_and_mu(self) -> None:
        """
        Update θ_{h,s} from each human's φ_{h,r,s}, then update μ_s from all θ_{h,s}.

        Both pooling steps use precision-weighted combination so that well-estimated
        parameters (low variance) contribute more than uncertain ones.  The posterior
        is a proper Normal-Normal update:

            post_var  = 1 / (1/prior_var + Σ 1/obs_var_i)
            post_mean = post_var * (prior_mean/prior_var + Σ obs_mean_i/obs_var_i)
        """
        registered_humans = list(self._theta_mean.keys())

        # Update θ_{h,s} from φ_{h,r,s} precision-weighted over recipes.
        #
        # IMPORTANT: only include (recipe, spice) pairs where at least one actual
        # observation has been made.  Uninitialised pairs carry phi = 0.0 (from
        # register_recipe) with phi_var = sigma_r².  Including them would inject
        # spurious zero-valued pseudo-observations that dilute the theta estimate
        # toward 0, causing sign flips for spices with few training recipes.
        # E.g. a single-recipe spice with true theta = +0.8 ends up at theta ≈ +0.05
        # because 18 zero-phi recipes overwhelm the 1 signal recipe.
        for h in registered_humans:
            for s in self.spices:
                phi_means: List[float] = []
                phi_precisions: List[float] = []
                for r in self._phi_var.get(h, {}):
                    if s in self._phi_mean[h][r]:
                        if self._obs_count[h].get(r, {}).get(s, 0) > 0:
                            phi_means.append(self._phi_mean[h][r][s])
                            phi_precisions.append(1.0 / self._phi_var[h][r][s])
                if not phi_means:
                    continue

                total_phi_precision = sum(phi_precisions)
                weighted_phi_sum = sum(p * m for p, m in zip(phi_precisions, phi_means))

                prior_mean = self.mu_mean[s]
                prior_var = self.sigma0**2 + self.sigma_h**2

                post_var = 1.0 / (1.0 / prior_var + total_phi_precision)
                post_mean = post_var * (prior_mean / prior_var + weighted_phi_sum)

                self._theta_mean[h][s] = post_mean
                self._theta_var[h][s] = post_var

        # Update μ_s from θ_{h,s} precision-weighted over humans.
        for s in self.spices:
            theta_means: List[float] = []
            theta_precisions: List[float] = []
            for h in registered_humans:
                if s in self._theta_mean[h]:
                    theta_means.append(self._theta_mean[h][s])
                    theta_precisions.append(1.0 / self._theta_var[h][s])
            if not theta_means:
                continue

            total_theta_precision = sum(theta_precisions)
            weighted_theta_sum = sum(p * m for p, m in zip(theta_precisions, theta_means))

            prior_var = self.sigma0**2

            post_var = 1.0 / (1.0 / prior_var + total_theta_precision)
            post_mean = post_var * weighted_theta_sum  # global prior mean = 0.0

            self.mu_mean[s] = post_mean
            self.mu_var[s] = post_var

    def flush_theta_mu(self) -> None:
        """
        Force an immediate θ/μ update. Call before eval (or at training end) to ensure
        θ and μ reflect all accumulated φ updates, including those in the last partial batch.
        """
        self.update_theta_and_mu()

    # ----- Episode Interface -----
    def observe(
        self,
        human_id: str,
        recipe_name: str,
        spice: str,
        actor: str,
        satisfaction: float,
        force_neutral_mood: bool = False,
    ) -> None:
        """
        Process a single transition for the given human and recipe.

        Registers the human and recipe lazily on first call.
        Updates the mood posterior incrementally; defers φ updates until end_episode().
        When force_neutral_mood is True, skips mood inference and sets posterior to neutral.
        """
        self._ensure_registered(human_id, recipe_name)
        self._episode_data[human_id].append((actor, spice, satisfaction))
        self._current_recipe[human_id] = recipe_name

        if force_neutral_mood:
            neutral_idx = MOODS.index("neutral")
            self._mood_posterior[human_id] = np.zeros(3)
            self._mood_posterior[human_id][neutral_idx] = 1.0
        else:
            self._update_mood_posterior(human_id, recipe_name, actor, spice, satisfaction)

    def end_episode(self, human_id: str, neutral_threshold: float = 0.5) -> None:
        """
        Batch-update φ for this human if confident in neutral mood, propagate
        updates to θ and μ (every N episodes when batching), then reset episode state.

        Each episode samples a fresh mood (sample_episode_mood), so
        the mood posterior and log-likelihood accumulator are reset to the prior at end.
        """
        self._phi_updated[human_id] = False
        self._episode_count[human_id] = self._episode_count.get(human_id, 0) + 1

        neutral_idx = MOODS.index("neutral")
        neutral_conf = self._mood_posterior[human_id][neutral_idx]

        if neutral_conf >= neutral_threshold and self._current_recipe[human_id] and self._episode_data[human_id]:
            recipe_name = self._current_recipe[human_id]
            confidence_weight = (neutral_conf - neutral_threshold) / (
                1.0 - neutral_threshold
            )
            confidence_weight = max(confidence_weight, self.config.update.confidence_weight_min)

            for actor, spice, satisfaction in self._episode_data[human_id]:
                g = self._pseudo_obs_weighted(
                    human_id, actor, satisfaction, recipe_name, spice, neutral_threshold
                )
                g_weighted = g * confidence_weight
                learning_rate = self.config.hbm.base_learning_rate * confidence_weight
                self._update_phi(
                    human_id, recipe_name, spice, g_weighted, learning_rate=learning_rate
                )
            self._phi_updated[human_id] = True

        batch_size = self.config.hbm.update_theta_mu_every_n_episodes
        if self._phi_updated[human_id] and (
            batch_size <= 1 or self._episode_count[human_id] % batch_size == 0
        ):
            self.update_theta_and_mu()

        # Reset all episode state and mood posterior to prior.
        self._episode_data[human_id] = []
        self._log_lik_accum[human_id] = np.zeros(len(MOODS))
        self._mood_posterior[human_id] = self.mood_prior.copy()
        self._current_recipe[human_id] = None

    # ---- Public updates -----
    def update_phi(
        self, human_id: str, recipe_name: str, spice: str, g: float
    ) -> None:
        """
        Update φ_{h,r,s} with a pseudo-observation g via a simple Normal-Normal
        Bayesian update (no EMA or annealing).
        """
        self._ensure_registered(human_id, recipe_name)

        theta_hs = self._theta_mean[human_id][spice]
        sigma_r2 = self.sigma_r**2
        sigma_obs2 = self.sigma_obs**2

        post_var = 1.0 / (1.0 / sigma_r2 + 1.0 / sigma_obs2)
        post_mean = post_var * (theta_hs / sigma_r2 + g / sigma_obs2)

        self._phi_mean[human_id][recipe_name][spice] = post_mean
        self._phi_var[human_id][recipe_name][spice] = post_var
        self._obs_count[human_id][recipe_name][spice] += 1

    # ----- Getters ----- 
    def get_phi(self, human_id: str, recipe_name: str, spice: str) -> float:
        """Return φ_{h,r,s}.  Returns 0.0 for unregistered human/recipe/spice."""
        return self._phi_mean[human_id][recipe_name].get(spice, 0.0)

    def get_theta(self, human_id: str, spice: str) -> float:
        """Return θ_{h,s}."""
        return self._theta_mean[human_id][spice]

    def get_mu(self, spice: str) -> float:
        """Return μ_s (global mean preference for this spice)."""
        return self.mu_mean[spice]

    def get_mu_var(self, spice: str) -> float:
        """Return the posterior variance of μ_s (global uncertainty for this spice)."""
        return self.mu_var[spice]

    def preferred_actor(self, human_id: str, recipe_name: str, spice: str) -> str:
        phi = self.get_phi(human_id, recipe_name, spice)
        return "human" if phi >= 0 else "robot"

    def sample_episode_preferences(
        self,
        recipe_spices: List[str],
        rng: np.random.Generator,
        human_id: str = DEFAULT_HUMAN,
    ) -> Dict[str, str]:
        """
        Sample episode-level preferred_actor for each spice from the HBM's θ posterior.

        For each spice: φ ~ N(θ_mean, θ_var + σ_r²), then p_human = sigmoid(φ),
        preferred_actor ~ Bernoulli(p_human). Spices not in θ fall back to uniform.
        """
        preferences: Dict[str, str] = {}
        for spice in recipe_spices:
            if spice in self._theta_mean.get(human_id, {}):
                theta_mean = self._theta_mean[human_id][spice]
                theta_var = self._theta_var[human_id].get(
                    spice, self.sigma_h**2 + self.sigma_r**2
                )
                phi_std = float(np.sqrt(theta_var + self.sigma_r**2))
                phi = rng.normal(theta_mean, phi_std)
                p_human = 1.0 / (1.0 + np.exp(-phi))
                preferred_actor = "human" if rng.random() < p_human else "robot"
            else:
                preferred_actor = str(rng.choice(["human", "robot"]))
            preferences[spice] = preferred_actor
        return preferences

    def log_prob_prefer(
        self, human_id: str, recipe_name: str, spice: str, actor: str
    ) -> float:
        phi = self.get_phi(human_id, recipe_name, spice)
        sign_actor = +1.0 if actor == "human" else -1.0
        logit = sign_actor * phi
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(np.log(max(p, 1e-9)))
