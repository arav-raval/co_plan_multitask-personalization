"""CSP Elements for the spices environment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Collection

import numpy as np
from tomsutils.spaces import EnumSpace

from multitask_personalization.csp_generation import (
    CSPConstraintGenerator,
    CSPGenerator,
)
from multitask_personalization.envs.spices.spices_config import DEFAULT_CONFIG, SpicesConfig
from multitask_personalization.envs.spices.spices_env import SpiceAction, SpiceState
from multitask_personalization.envs.spices.spices_hbm import (
    DEFAULT_HUMAN,
    MOODS,
    HierarchicalPreferenceModel,
)
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
    def __init__(self, csp_variables: Collection[CSPVariable], seed: int = 0) -> None:
        super().__init__(csp_variables, seed)
        self._actor: str | None = None
        self._done_emitted = False

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._actor = self._get_value("actor")
        self._done_emitted = False

    def step(self, obs: SpiceState) -> SpiceAction:
        if (not obs.current_spice) or (
            len(obs.feasible_next) == 0 and len(obs.remaining_spices) == 0
        ):
            self._done_emitted = True
            return (1, None)

        assert self._actor in ("human", "robot")
        return (0, self._actor)

    def check_termination(self, obs: SpiceState) -> bool:
        return self._done_emitted


class _AssignPreferenceGenerator(CSPConstraintGenerator[SpiceState, SpiceAction]):
    def __init__(
        self,
        spice_list: list[str],
        neutral_confidence_threshold: float,
        human_id: str = DEFAULT_HUMAN,
        recipe_list: list[str] | None = None,
        seed: int = 0,
        verbose: bool = False,
        config: SpicesConfig | None = None,
    ) -> None:
        super().__init__(seed)

        self.config = config if config is not None else DEFAULT_CONFIG
        self._human_id = human_id
        self._recipe_list: list[str] = recipe_list if recipe_list is not None else []
        self._neutral_confidence_threshold = neutral_confidence_threshold
        self._verbose = verbose

        self._hbm = HierarchicalPreferenceModel(
            spices=spice_list,
            recipes=self._recipe_list,
            mu0=self.config.hbm.mu0,
            sigma0=self.config.hbm.sigma0,
            sigma_h=self.config.hbm.sigma_h,
            sigma_r=self.config.hbm.sigma_r,
            sigma_obs=self.config.hbm.sigma_obs,
            config=self.config,
        )
        self._current_recipe_name: str | None = None

    def get_expected_mood(self) -> float:
        """Return expected mood value: -1.0 (none_self) to +1.0 (all_self)."""
        mp = self._hbm._mood_posterior[self._human_id]
        mood_values = {"all_self": +1.0, "neutral": 0.0, "none_self": -1.0}
        return sum(mp[i] * mood_values[m] for i, m in enumerate(MOODS))

    def get_mood_posterior_breakdown(self) -> dict[str, float]:
        """Return mood posterior probabilities for each mood."""
        mp = self._hbm._mood_posterior[self._human_id]
        return {mood: float(mp[i]) for i, mood in enumerate(MOODS)}

    def get_most_likely_mood(self) -> tuple[str, float]:
        """Return the most likely mood and its probability."""
        mp = self._hbm._mood_posterior[self._human_id]
        idx = int(np.argmax(mp))
        return MOODS[idx], float(mp[idx])

    def save(self, model_dir: Path) -> None:
        if self._verbose:
            logging.info("[Save] HBM model (save not yet implemented)")

    def load(self, model_dir: Path) -> None:
        if self._verbose:
            logging.info("[Load] HBM model (load not yet implemented)")

    def generate(self, obs: SpiceState, variables: list[CSPVariable], name: str) -> CSPConstraint:
        (actor_var,) = variables
        current = obs.current_spice

        def _logprob(actor: str) -> float:
            if self._current_recipe_name:
                return self._hbm.log_prob_prefer(
                    self._human_id, self._current_recipe_name, current, actor
                )
            return np.log(0.5)

        return LogProbCSPConstraint(name, [actor_var], _logprob, threshold=np.log(0.3))

    def learn_from_transition(
        self,
        obs: SpiceState,
        act: SpiceAction,
        next_obs: SpiceState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update mood posterior and HBM on each observed transition."""
        if info.get("last_spice") is None or info.get("last_actor") is None:
            return

        recipe_name = info.get("recipe_name") or self._current_recipe_name
        if recipe_name and recipe_name not in self._recipe_list:
            self._recipe_list.append(recipe_name)
            self._hbm.recipes = list(self._recipe_list)
        self._current_recipe_name = recipe_name

        actor = str(info["last_actor"])
        spice = str(info["last_spice"])
        satisfaction = float(info["satisfaction"])

        if recipe_name:
            self._hbm.observe(self._human_id, recipe_name, spice, actor, satisfaction)

        # Capture mood AFTER the observation update but BEFORE the episode reset
        # so callers always see the inferred mood for the current step.
        info["mood_posterior"] = self.get_mood_posterior_breakdown()
        info["expected_mood"] = self.get_expected_mood()

        if done:
            self._finalize_episode()

    def _finalize_episode(self) -> None:
        """Update hierarchical HBM preferences and reset episode state.

        end_episode handles batch phi updates, theta/mu propagation, and mood
        posterior reset to prior (since each episode draws a fresh mood).
        """
        self._hbm.end_episode(self._human_id, neutral_threshold=self._neutral_confidence_threshold)
        if self._verbose:
            logging.info("[Episode] HBM updated hierarchical preferences (θ, μ)")

    def get_metrics(self) -> dict[str, float]:
        """Return mood inference metrics."""
        mp = self._hbm._mood_posterior[self._human_id]
        return {
            "expected_mood": self.get_expected_mood(),
            "neutral_confidence": float(mp[MOODS.index("neutral")]),
        }


class SpicesAssignCSPGenerator(CSPGenerator[SpiceState, SpiceAction]):
    """CSP: choose the actor for the current spice; learn preferences via HBM."""

    def __init__(
        self,
        spice_list: list[str],
        recipe_list: list[str] | None = None,
        neutral_confidence_threshold: float = 0.75,
        human_id: str = DEFAULT_HUMAN,
        verbose: bool = False,
        config: SpicesConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._spices = list(spice_list)
        self._pref_gen = _AssignPreferenceGenerator(
            self._spices,
            neutral_confidence_threshold=neutral_confidence_threshold,
            human_id=human_id,
            recipe_list=recipe_list,
            seed=self._seed,
            verbose=verbose,
            config=config,
        )
        self._init_rng = np.random.default_rng(self._seed)

    def save(self, model_dir: Path) -> None:
        self._pref_gen.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._pref_gen.load(model_dir)

    def get_pref_snapshot(self) -> dict[str, dict[str, float]]:
        """Return current P(prefer actor) for each spice from HBM."""
        human_id = self._pref_gen._human_id
        recipe_name = self._pref_gen._current_recipe_name or (
            self._pref_gen._recipe_list[-1] if self._pref_gen._recipe_list else None
        )
        probs: dict[str, dict[str, float]] = {}
        for spice in self._pref_gen._hbm.spices:
            spice_probs: dict[str, float] = {}
            for actor in ["human", "robot"]:
                if recipe_name:
                    logp = self._pref_gen._hbm.log_prob_prefer(
                        human_id, recipe_name, spice, actor
                    )
                    p = float(np.exp(logp))
                else:
                    p = 0.5
                spice_probs[actor] = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            total = sum(spice_probs.values())
            probs[spice] = {k: round(v / total, 3) for k, v in spice_probs.items()}
        return probs

    def _generate_variables(
        self, obs: SpiceState
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        actor = CSPVariable("actor", EnumSpace(["human", "robot"]))
        initialization = {actor: self._init_rng.choice(["human", "robot"])}
        return [actor], initialization

    def _generate_personal_constraints(
        self, obs: SpiceState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        """Hard mood constraints when confident; HBM preference constraint otherwise."""
        actor_var = variables[0]
        mood, conf = self._pref_gen.get_most_likely_mood()
        threshold = self._pref_gen._neutral_confidence_threshold

        if mood == "all_self" and conf >= threshold:
            return [
                FunctionalCSPConstraint(
                    "respect_all_self_mood", [actor_var], lambda a: a == "human"
                )
            ]
        if mood == "none_self" and conf >= threshold:
            return [
                FunctionalCSPConstraint(
                    "respect_none_self_mood", [actor_var], lambda a: a == "robot"
                )
            ]

        return [self._pref_gen.generate(obs, variables, "user_preference")]

    def _generate_nonpersonal_constraints(
        self, obs: SpiceState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        return []

    def _generate_exploit_cost(
        self, obs: SpiceState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        """Minimize negative HBM log-probability of actor for the current spice."""
        actor = variables[0]
        current = obs.current_spice

        def _cost_fn(actor_val: str) -> float:
            if self._pref_gen._current_recipe_name:
                return -self._pref_gen._hbm.log_prob_prefer(
                    self._pref_gen._human_id,
                    self._pref_gen._current_recipe_name,
                    current,
                    actor_val,
                )
            return 0.0

        return CSPCost("maximize_preference", [actor], _cost_fn)

    def _generate_samplers(self, obs: SpiceState, csp: CSP) -> list[CSPSampler]:
        actor = csp.variables[0]
        current_spice = obs.current_spice

        def _sample_actor(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            probs = []
            for a in ["human", "robot"]:
                if self._pref_gen._current_recipe_name:
                    logp = self._pref_gen._hbm.log_prob_prefer(
                        self._pref_gen._human_id,
                        self._pref_gen._current_recipe_name,
                        current_spice,
                        a,
                    )
                    p = float(np.exp(logp))
                else:
                    p = 0.5
                probs.append(max(p, 1e-6))
            probs_arr = np.array(probs)
            probs_arr /= probs_arr.sum()
            return {actor: rng.choice(["human", "robot"], p=probs_arr)}

        return [FunctionalCSPSampler(_sample_actor, csp, {actor})]

    def _generate_policy(
        self, obs: SpiceState, csp_variables: Collection[CSPVariable]
    ) -> CSPPolicy:
        return _SpiceCSPPolicy(csp_variables, seed=self._seed)

    def observe_transition(
        self,
        obs: SpiceState,
        act: SpiceAction,
        next_obs: SpiceState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._disable_learning:
            self._pref_gen.learn_from_transition(obs, act, next_obs, done, info)

        # Fallback: populate mood keys when learning is disabled or the transition
        # was skipped (missing last_spice / last_actor), so info is always complete.
        if "mood_posterior" not in info:
            info["mood_posterior"] = self._pref_gen.get_mood_posterior_breakdown()
            info["expected_mood"] = self._pref_gen.get_expected_mood()

    def get_metrics(self) -> dict[str, float]:
        return self._pref_gen.get_metrics()
