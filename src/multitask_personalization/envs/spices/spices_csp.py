"""CSP Elements for the spices environment."""

from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Any, Collection, List, Tuple
import logging
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import RadiusNeighborsClassifier
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_generation import (
    CSPGenerator,
    CSPConstraintGenerator,
)

from multitask_personalization.envs.spices.spices_env import SpiceAction, SpiceState
from multitask_personalization.envs.spices.spices_hbm import HierarchicalPreferenceModel
from multitask_personalization.envs.spices.spices_config import DEFAULT_CONFIG, SpicesConfig

from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPConstraint,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)

class _SpiceCSPPolicy(CSPPolicy[SpiceState, SpiceAction]):
    def __init__(self, csp_variables: Collection[CSPVariable], seed: int = 0, verbose: bool = False) -> None:
        super().__init__(csp_variables, seed)
        self._rng = np.random.default_rng(seed)
        self._actor: str | None = None
        self._done_emitted = False
        self._verbose = verbose

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._actor = self._get_value("actor")
        self._done_emitted = False

    def step(self, obs: SpiceState) -> SpiceAction:
        # Emit done 
        if (not obs.current_spice) or (len(obs.feasible_next) == 0 and len(obs.remaining_spices) == 0):
            if not self._done_emitted:
                self._done_emitted = True
            return (1, None)
        
        # Assign the selected actor for the current spice
        assert self._actor in ("human", "robot")
        return (0, self._actor)

    def check_termination(self, obs: SpiceState) -> bool:
        return self._done_emitted
    
class _AssignPreferenceGenerator(CSPConstraintGenerator[SpiceState, SpiceAction]):
    def __init__(
        self, 
        spice_list: list[str],
        recipe_list: list[str] | None = None,  # All recipes seen so far
        base_satisfaction_bias: float = 3.0,
        neutral_confidence_threshold: float = 0.5, # only learn spice preferences when P(neutral) > threshold
        use_hbm: bool = True,  # Use HBM for continuous learning
        seed: int = 0, 
        verbose: bool = False,
        config: SpicesConfig | None = None) -> None:
        super().__init__(seed)
        self._spice_to_index = {spice: i for i, spice in enumerate(spice_list)}
        self._actor_to_index = {actor: i for i, actor in enumerate(["human", "robot"])}
        
        # Load configuration
        self.config = config if config is not None else DEFAULT_CONFIG

        # HBM for continuous preference learning
        self._use_hbm = use_hbm
        self._recipe_list = recipe_list if recipe_list is not None else []
        
        # Always initialize classifier attributes (for backward compatibility)
        self._classifier: RadiusNeighborsClassifier | None = None
        self._training_inputs: list[NDArray] = []
        self._training_outputs: list[bool] = []
        self._neutral_training_inputs: list[NDArray] = []
        self._neutral_training_outputs: list[bool] = []
        
        if self._use_hbm:
            self._hbm = HierarchicalPreferenceModel(
                spices=spice_list,
                recipes=self._recipe_list,
                base_satisfaction_bias=base_satisfaction_bias,
                mu0=self.config.hbm.mu0,
                sigma0=self.config.hbm.sigma0,
                sigma_h=self.config.hbm.sigma_h,
                sigma_r=self.config.hbm.sigma_r,
                sigma_obs=self.config.hbm.sigma_obs,
                config=self.config,
            )
            # Sync HBM mood posterior with our mood inference
            self._hbm.mood_posterior = np.array(self.config.mood_prior_array, dtype=float).copy()
        else:
            self._hbm = None

        # Mood bias
        self._base_satisfaction_bias = base_satisfaction_bias
        self._mood_bias_strength = self.config.mood_bias_strength
        self._neutral_confidence_threshold = neutral_confidence_threshold

        self._mood_bias = self.config.get_mood_bias()

        # Mood inference state (per episode)
        self._mood_prior = np.array(self.config.mood_prior_array, dtype=float)
        self._mood_posterior = self._mood_prior.copy()

        # Per-episode buffers for mood inference (still needed for our mood inference)
        self._episode_observations: List[Tuple[str, str, float]] = []  # (actor, spice, satisfaction)
        self._current_recipe_name: str | None = None  # Track current recipe for HBM
        
        self._verbose = verbose

    def _get_preference_phi(self, spice: str, actor: str, recipe_name: str | None = None) -> float:
        """Get preference phi value for a spice-actor pair.
        
        Returns phi estimate: positive if actor matches preference, negative otherwise.
        Uses HBM if available, otherwise falls back to classifier or returns 0.
        """
        if self._use_hbm and self._hbm is not None:
            # In _get_preference_phi(), add fallback:
            if spice not in self._hbm.theta_mean:
                # Spice not seen in training, use global average or 0
                return 0.0
                
            # Use HBM's recipe-specific preference
            if recipe_name is None:
                # If no recipe specified, use the most recent recipe or first available
                recipe_name = self._recipe_list[-1] if self._recipe_list else None
                if recipe_name is None:
                    return 0.0
            
            # Get phi from HBM (recipe-specific preference)
            phi = self._hbm.get_phi(recipe_name, spice)

            # If phi is 0 (uninitialized), use theta as fallback
            if abs(phi) < 1e-6 and spice in self._hbm.theta_mean:
                phi = self._hbm.theta_mean[spice]  # Use human-level preference
            
            # Convert to actor-specific preference signal
            # phi > 0 means prefer human, phi < 0 means prefer robot
            sign_actor = +1.0 if actor == "human" else -1.0
            phi_est = sign_actor * phi
            
            # Clamp to reasonable range
            phi_est = np.clip(phi_est, -self._base_satisfaction_bias, self._base_satisfaction_bias)
            return phi_est
        
        # Fallback to classifier (legacy mode)
        if hasattr(self, '_classifier') and self._classifier is not None:
            x = self._featurize(spice, actor)
            p_pref = self._safe_predict_proba(x)
            sign_actor = +1.0 if actor == "human" else -1.0
            logit_pref = np.log(p_pref / (1 - p_pref)) if p_pref not in (0, 1) else 0.0
            phi_est = sign_actor * logit_pref
            phi_est = np.clip(phi_est, -self._base_satisfaction_bias, self._base_satisfaction_bias)
            return phi_est
        
        return 0.0  # No preference information yet
    
    def _update_mood_posterior(self) -> None:
        """Update mood posterior from all observations in current episode.
        
        Uses learned preferences to better distinguish mood effects from preference matches.
        Key insight: if satisfaction matches preference → could be neutral or mood.
        If satisfaction doesn't match preference → more likely mood effect.
        """
        if not self._episode_observations:
            return
        
        MOODS = ("all_self", "neutral", "none_self")
        logps = []
        
        for m in MOODS:
            # Use stronger prior to prevent wild swings
            prior_weight = self.config.mood.mood_prior_weight
            lp = np.log(self._mood_prior[MOODS.index(m)]) * prior_weight
            
            # Compute likelihood under this mood hypothesis
            for (actor, spice, sat) in self._episode_observations:
                # Get preference phi (learned from neutral episodes)
                phi_pref = self._get_preference_phi(spice, actor, self._current_recipe_name)
                
                # Compute logit under this mood hypothesis
                sign_actor = +1.0 if actor == "human" else -1.0
                
                if m == "neutral":
                    # Neutral mood: satisfaction depends primarily on preference match
                    # phi_pref already encodes the preference (positive if preferred, negative if not)
                    logit = sign_actor * phi_pref + self._mood_bias[m][actor]
                else:
                    # Non-neutral mood: mood bias is strong, but preference provides signal
                    # If satisfaction doesn't match preference, it's stronger evidence for mood
                    # If satisfaction matches preference, it's weaker evidence (could be either)
                    mood_bias = self._mood_bias[m][actor]
                    
                    # Check if satisfaction matches preference expectation
                    # CONTINUOUS SATISFACTION: Use magnitude, not just sign
                    pref_logit = sign_actor * phi_pref
                    pref_expectation = pref_logit > 0  # positive if preference suggests satisfaction
                    sat_positive = sat > 0
                    matches_preference = pref_expectation == sat_positive
                    
                    if matches_preference:
                        # Satisfaction matches preference: could be mood OR preference
                        # Use mood bias but with some preference influence
                        pref_weight = self.config.mood.non_neutral_pref_weight_match
                        logit = mood_bias + sign_actor * phi_pref * pref_weight
                    else:
                        # Satisfaction doesn't match preference: strong evidence for mood
                        # Mood bias dominates
                        pref_weight = self.config.mood.non_neutral_pref_weight_mismatch
                        logit = mood_bias + sign_actor * phi_pref * pref_weight
                
                p = 1.0 / (1.0 + np.exp(-logit))
                
                # CONTINUOUS SATISFACTION: Use Beta likelihood similar to HBM
                # Map satisfaction from [-1, +1] to [0, 1] probability space
                p_obs = (sat + 1.0) / 2.0  # Map [-1, +1] → [0, 1]
                p_obs = np.clip(p_obs, 1e-6, 1.0 - 1e-6)  # Avoid log(0)
                
                # CRITICAL FIX: Use Gaussian likelihood instead of Beta for stability
                # Map expected probability p to expected satisfaction
                sat_expected = 2.0 * p - 1.0
                sigma_sat = self.config.mood.satisfaction_sigma
                log_lik = -0.5 * ((sat - sat_expected) / sigma_sat)**2
                log_lik = np.clip(log_lik, self.config.mood.satisfaction_loglik_min,
                                 self.config.mood.satisfaction_loglik_max)
                lp += log_lik
            
            logps.append(lp)
        
        # Normalize
        logps = np.array(logps)
        logps -= np.max(logps)
        ps = np.exp(logps)
        ps /= ps.sum()
        
        # CRITICAL FIX: Smooth with previous posterior to prevent rapid swings
        # Use exponential moving average with high retention
        smoothing_alpha = self.config.mood.mood_smoothing_alpha
        ps = smoothing_alpha * ps + (1 - smoothing_alpha) * self._mood_posterior
        ps /= ps.sum()  # Renormalize
        
        self._mood_posterior = ps
    
    def get_expected_mood(self) -> float:
        """Return expected mood value: -1.0 (none_self) to +1.0 (all_self)."""
        MOODS = ("all_self", "neutral", "none_self")
        mood_values = {"all_self": +1.0, "neutral": 0.0, "none_self": -1.0}
        return sum(self._mood_posterior[i] * mood_values[m] for i, m in enumerate(MOODS))
    
    def get_mood_posterior_breakdown(self) -> dict[str, float]:
        """Return mood posterior probabilities for each mood."""
        MOODS = ("all_self", "neutral", "none_self")
        return {mood: float(self._mood_posterior[i]) for i, mood in enumerate(MOODS)}

    def get_most_likely_mood(self) -> tuple[str, float]:
        """Return the most likely mood and its probability."""
        MOODS = ("all_self", "neutral", "none_self")
        idx = int(np.argmax(self._mood_posterior))
        return MOODS[idx], float(self._mood_posterior[idx])

    def is_confident_neutral(self) -> bool:
        """Check if we're confident we're in neutral mood."""
        MOODS = ("all_self", "neutral", "none_self")
        neutral_idx = MOODS.index("neutral")
        return self._mood_posterior[neutral_idx] >= self._neutral_confidence_threshold

    def save(self, model_dir: Path) -> None:
        if self._use_hbm and self._hbm is not None:
            # Save HBM state (would need to implement save/load in HBM)
            if self._verbose:
                logging.info("[Save] HBM model (save not yet implemented)")
        else:
            # Save classifier
            outfile = model_dir / "assign_preference_classifier.pkl"
            with open(outfile, "wb") as f:
                pkl.dump(self._classifier, f)
    
    def load(self, model_dir: Path) -> None:
        if self._use_hbm and self._hbm is not None:
            # Load HBM state (would need to implement save/load in HBM)
            if self._verbose:
                logging.info("[Load] HBM model (load not yet implemented)")
        else:
            # Load classifier
            outfile = model_dir / "assign_preference_classifier.pkl"
            with open(outfile, "rb") as f:
                self._classifier = pkl.load(f)

    def generate(self, obs: SpiceState, variables: list[CSPVariable], name: str) -> CSPConstraint:
        (actor_vars, ) = variables
        current = obs.current_spice

        def _logprob(actor: str) -> float:
            # Use HBM
            if self._use_hbm and self._hbm is not None and self._current_recipe_name:
                return self._hbm.log_prob_prefer(self._current_recipe_name, current, actor)
            # Fallback to classifier TODO: REMOVE THIS [LEGACY]                
            elif hasattr(self, '_classifier') and self._classifier is not None:
                x = self._featurize(current, actor)
                p = self._safe_predict_proba(x)
                return np.log(p)
            else:
                return np.log(0.5)  # Uniform prior

        return LogProbCSPConstraint(name, [actor_vars], _logprob, threshold=np.log(0.3))

    def learn_from_transition(
        self, obs: SpiceState, act: SpiceAction, next_obs: SpiceState, done: bool, info: dict[str, Any]
    ) -> None:
        """Learn from transition, updating mood posterior per-step.
        
        With HBM: continuously updates preferences weighted by mood posterior.
        Without HBM: buffers observations for all-or-nothing learning at episode end.
        """
        if info.get("last_spice") is None or info.get("last_actor") is None:
            return
        
        # Get recipe name from info if available
        recipe_name = info.get("recipe_name") or self._current_recipe_name
        if recipe_name and recipe_name not in self._recipe_list:
            # Add new recipe to HBM if using HBM
            if self._use_hbm and self._hbm is not None:
                self._recipe_list.append(recipe_name)
                self._hbm.recipes = list(self._recipe_list)
        self._current_recipe_name = recipe_name
        
        # Collect observation
        actor = str(info["last_actor"])
        spice = str(info["last_spice"])
        satisfaction = float(info["satisfaction"])
        
        # Buffer for mood inference
        self._episode_observations.append((actor, spice, satisfaction))
        
        # Update mood posterior (per-step inference for online replanning)
        self._update_mood_posterior()
        
        # With HBM: continuously update preferences weighted by mood
        if self._use_hbm and self._hbm is not None and recipe_name:
            # Sync mood posterior to HBM (HBM uses its own, but we keep ours for CSP logic)
            self._hbm.mood_posterior = self._mood_posterior.copy()
            # HBM handles continuous updates weighted by mood
            self._hbm.observe(recipe_name, spice, actor, satisfaction)
        
        # Finalize episode at end
        if done:
            self._finalize_episode()
    
    def _featurize(self, spice: str, actor: str) -> NDArray:
        return np.array([self._spice_to_index[spice], self._actor_to_index[actor]], dtype=float)
    
    def _safe_predict_proba(self, x: NDArray) -> float:
        if self._classifier is None:
            return 0.5
        try: 
            proba = self._classifier.predict_proba([x])
            if proba.size == 0 or len(proba[0]) == 0:
                return 0.5
            return float(np.clip(proba[0][1], 1e-6, 1.0 - 1e-6))
        except (ValueError, IndexError):
            return 0.5

    def _finalize_episode(self) -> None:
        """Finalize episode: with HBM, update hierarchical preferences. Without HBM, all-or-nothing learning."""
        if self._use_hbm and self._hbm is not None:
            # HBM handles batch updates at end of episode (processes all observations if confident in neutral mood)
            self._hbm.end_episode(neutral_threshold=self._neutral_confidence_threshold)
            if self._verbose:
                logging.info(f"[Episode] HBM updated hierarchical preferences (θ, μ)")
        else:
            # Legacy all-or-nothing learning (only if not using HBM)
            MOODS = ("all_self", "neutral", "none_self")
            neutral_idx = MOODS.index("neutral")
            neutral_conf = self._mood_posterior[neutral_idx]
            
            most_likely_mood, most_likely_conf = self.get_most_likely_mood()
            is_neutral_most_likely = (most_likely_mood == "neutral")
            is_above_threshold = (neutral_conf >= self._neutral_confidence_threshold)
            legacy_threshold = self.config.update.legacy_learning_threshold
            should_learn = (is_neutral_most_likely or is_above_threshold) and neutral_conf >= legacy_threshold
            
            if should_learn:
                if hasattr(self, '_episode_features') and hasattr(self, '_episode_labels'):
                    self._neutral_training_inputs.extend(self._episode_features)
                    self._neutral_training_outputs.extend(self._episode_labels)
                    self._update_constraint_parameters()
                    if self._verbose:
                        logging.info(f"[Episode] Learning from neutral episode: {len(self._episode_features)} samples")
        
        # Reset episode buffers
        self._episode_observations.clear()
        if hasattr(self, '_episode_features'):
            self._episode_features.clear()
        if hasattr(self, '_episode_labels'):
            self._episode_labels.clear()
        self._mood_posterior = self._mood_prior.copy()
        if self._use_hbm and self._hbm is not None:
            self._hbm.mood_posterior = self._mood_prior.copy()

    def _update_constraint_parameters(self) -> None:
        """Train classifier only on neutral-confident episodes."""
        # Check if we have both positive and negative examples
        unique_labels = set(self._neutral_training_outputs)
        if len(unique_labels) < 2:
            if self._verbose:
                logging.info(
                    f"[Learning] Not enough label diversity to train "
                    f"(only {unique_labels}). Need both positive and negative examples."
                )
            return
        
        # Check minimum data requirement
        if len(self._neutral_training_inputs) < 4:
            if self._verbose:
                logging.info(
                    f"[Learning] Not enough examples to train ({len(self._neutral_training_inputs)}). "
                    f"Need at least 4 examples."
                )
            return
        
        # Train on all neutral-confident episodes
        self._classifier = RadiusNeighborsClassifier(
            radius=1.5,
            weights="distance",
            algorithm="auto",
            p=2,
            metric="minkowski",
        )
        
        self._classifier.fit(self._neutral_training_inputs, self._neutral_training_outputs)
        
        # Compute training accuracy for debugging
        if len(self._neutral_training_inputs) > 0:
            predictions = self._classifier.predict(self._neutral_training_inputs)
            train_accuracy = np.mean(predictions == self._neutral_training_outputs)
            pos_examples = sum(self._neutral_training_outputs)
            neg_examples = len(self._neutral_training_outputs) - pos_examples
        else:
            train_accuracy = 0.0
            pos_examples = 0
            neg_examples = 0
        
        if self._verbose:
            logging.info(
                f"[Learning] Trained on {len(self._neutral_training_inputs)} samples "
                f"({pos_examples} positive, {neg_examples} negative). "
                f"Training accuracy: {train_accuracy:.3f}"
            )
    
    def get_metrics(self) -> dict[str, float]:
        """Return metrics including mood posterior."""
        MOODS = ("all_self", "neutral", "none_self")
        metrics = {
            "expected_mood": self.get_expected_mood(),
            "neutral_confidence": self._mood_posterior[MOODS.index("neutral")],
            "neutral_confident_examples": len(self._neutral_training_inputs),
        }
        
        # Add classifier diagnostics
        if self._classifier is not None:
            metrics["classifier_trained"] = 1.0
            if len(self._neutral_training_inputs) > 0:
                predictions = self._classifier.predict(self._neutral_training_inputs)
                train_accuracy = float(np.mean(predictions == self._neutral_training_outputs))
                metrics["classifier_train_accuracy"] = train_accuracy
            else:
                metrics["classifier_train_accuracy"] = 0.0
        else:
            metrics["classifier_trained"] = 0.0
            metrics["classifier_train_accuracy"] = 0.0
        
        return metrics

class SpicesAssignCSPGenerator(CSPGenerator[SpiceState, SpiceAction]):
    """CSP: choose the actor for the environment's current spice; learn the preferences"""

    def __init__(self, spice_list: list[str], recipe_list: list[str] | None = None, base_satisfaction_bias: float = 3.0, neutral_confidence_threshold: float = 0.75, use_hbm: bool = True, verbose: bool = False, config: SpicesConfig | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._spices = list(spice_list)
        self._pref_gen = _AssignPreferenceGenerator(
            self._spices, 
            recipe_list=recipe_list,
            base_satisfaction_bias=base_satisfaction_bias, 
            neutral_confidence_threshold=neutral_confidence_threshold,
            use_hbm=use_hbm,
            seed=self._seed, 
            verbose=verbose,
            config=config
        )

        # Separate RNG that won't be reset
        self._init_rng = np.random.default_rng(self._seed)

    def save(self, model_dir: Path) -> None:
        self._pref_gen.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._pref_gen.load(model_dir)

    def get_pref_snapshot(self) -> dict[str, dict[str, float]]:
        """Return current P(prefer=True) for each spice/actor."""
        probs = {}
        recipe_name = self._pref_gen._current_recipe_name or (self._pref_gen._recipe_list[-1] if self._pref_gen._recipe_list else None)
        
        for spice in self._pref_gen._spice_to_index.keys():
            spice_probs = {}
            for actor in ["human", "robot"]:
                if self._pref_gen._use_hbm and self._pref_gen._hbm is not None and recipe_name:
                    # Use HBM probability
                    logp = self._pref_gen._hbm.log_prob_prefer(recipe_name, spice, actor)
                    p = np.exp(logp)
                    spice_probs[actor] = float(np.clip(p, 1e-6, 1.0 - 1e-6))
                elif hasattr(self._pref_gen, '_classifier') and self._pref_gen._classifier is not None:
                    # Fallback to classifier
                    x = self._pref_gen._featurize(spice, actor)
                    p = self._pref_gen._classifier.predict_proba([x])[0][1]
                    spice_probs[actor] = float(np.clip(p, 1e-6, 1.0 - 1e-6))
                else:
                    spice_probs[actor] = 0.5
            total = sum(spice_probs.values())
            spice_probs = {k: round(v / total, 3) for k, v in spice_probs.items()}
            probs[spice] = spice_probs
        return probs

    def _generate_variables(self, obs: SpiceState) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        actor = CSPVariable("actor", EnumSpace(["human", "robot"]))
        variables = [actor]
        initialization = {actor: self._init_rng.choice(["human", "robot"])}

        return variables, initialization

    def _generate_personal_constraints(self, obs: SpiceState, variables: list[CSPVariable]) -> list[CSPConstraint]:
        """Generate constraints that respect mood: force actor assignment if mood is confident."""
        actor_var = variables[0]
        constraints: list[CSPConstraint] = []
        
        # Check current mood belief
        mood, conf = self._pref_gen.get_most_likely_mood()
        
        # Force actor if confident on non-neutral mood
        if mood == "all_self" and conf >= self._pref_gen._neutral_confidence_threshold:
            constraints.append(
                FunctionalCSPConstraint(
                    "respect_all_self_mood",
                    [actor_var],
                    lambda actor: actor == "human",
                )
            )
            return constraints
        
        if mood == "none_self" and conf >= self._pref_gen._neutral_confidence_threshold:
            constraints.append(
                FunctionalCSPConstraint(
                    "respect_none_self_mood",
                    [actor_var],
                    lambda actor: actor == "robot",
                )
            )
            return constraints
        
        # Use learned preferences (neutral mood)
        user_preference_constraint = self._pref_gen.generate(obs, variables, "user_preference")
        constraints.append(user_preference_constraint)
        return constraints
    
    def _generate_nonpersonal_constraints(self, obs: SpiceState, variables: list[CSPVariable]) -> list[CSPConstraint]:
        # Feasibility of spice enforced in the environment
        return []
    
    def _generate_exploit_cost(self, obs: SpiceState, variables: list[CSPVariable]) -> CSPCost | None:
        """Use negative log-probability as cost to prefer higher probability solutions."""
        actor = variables[0]
        current = obs.current_spice

        # Check for prefernces
        has_preferences = False
        if self._pref_gen._use_hbm and self._pref_gen._hbm is not None:
            has_preferences = True
        elif hasattr(self._pref_gen, '_classifier') and self._pref_gen._classifier is not None:
            has_preferences = True
        
        if not has_preferences:
            return None

        def _cost_fn(actor_val: str) -> float:
            # Use HBM log probability
            if self._pref_gen._use_hbm and self._pref_gen._hbm is not None and self._pref_gen._current_recipe_name:
                logp = self._pref_gen._hbm.log_prob_prefer(self._pref_gen._current_recipe_name, current, actor_val)
                return -logp
            # Fallback to classifier TODO: REMOVE THIS
            else:
                x = self._pref_gen._featurize(current, actor_val)
                p = self._pref_gen._safe_predict_proba(x)
                return -np.log(p)
        
        return CSPCost("maximize_preference", [actor], _cost_fn)
    
    def _generate_samplers(self, obs: SpiceState, csp: CSP) -> list[CSPSampler]:
        actor = csp.variables[0]
        current_spice = obs.current_spice

        def _sample_actor(sol: dict[CSPVariable, Any], rng: np.random.Generator) -> dict[CSPVariable, Any]:
            # Sample according to HBM preference probabilities for this spice
            probs = []
            for a in ["human", "robot"]:
                if self._pref_gen._use_hbm and self._pref_gen._hbm is not None and self._pref_gen._current_recipe_name:
                    logp = self._pref_gen._hbm.log_prob_prefer(
                        self._pref_gen._current_recipe_name, current_spice, a
                    )
                    p = np.exp(logp)
                else:
                    p = 0.5 
                probs.append(max(p, 1e-6))

            probs = np.array(probs)
            probs /= probs.sum()
            chosen = rng.choice(["human", "robot"], p=probs)
            return {actor: chosen}

        return [FunctionalCSPSampler(_sample_actor, csp, {actor})]

    def _generate_policy(self, obs: SpiceState, csp_variables: Collection[CSPVariable]) -> CSPPolicy:
        return _SpiceCSPPolicy(csp_variables, seed=self._seed)

    def observe_transition(self, obs: SpiceState, act: SpiceAction, next_obs: SpiceState, done: bool, info: dict[str, Any]) -> None:
        if not self._disable_learning:
            self._pref_gen.learn_from_transition(obs, act, next_obs, done, info)
        
        # Add mood information to info dict (after updating mood posterior)
        mood_metrics = self._pref_gen.get_metrics()
        mood_breakdown = self._pref_gen.get_mood_posterior_breakdown()
        
        info["expected_mood"] = mood_metrics["expected_mood"]
        info["neutral_confidence"] = mood_metrics["neutral_confidence"]
        info["mood_posterior"] = mood_breakdown  # dict with "all_self", "neutral", "none_self" probabilities
        info["is_confident_neutral"] = self._pref_gen.is_confident_neutral()
        info["neutral_confident_examples"] = mood_metrics["neutral_confident_examples"]

    def get_metrics(self) -> dict[str, float]:
        return self._pref_gen.get_metrics()
