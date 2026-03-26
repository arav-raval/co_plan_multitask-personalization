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
from multitask_personalization.envs.spices.config.spices_config import DEFAULT_CONFIG, SpicesConfig
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
        shared_hbm: HierarchicalPreferenceModel | None = None,
    ) -> None:
        super().__init__(seed)

        self.config = config if config is not None else DEFAULT_CONFIG
        self._human_id = human_id
        self._recipe_list: list[str] = recipe_list if recipe_list is not None else []
        self._neutral_confidence_threshold = neutral_confidence_threshold
        self._verbose = verbose

        if shared_hbm is not None:
            self._hbm = shared_hbm
            self._hbm.register_human(human_id)
            for r in self._recipe_list:
                self._hbm.register_recipe(human_id, r)
        else:
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
        shared_hbm: HierarchicalPreferenceModel | None = None,
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
            shared_hbm=shared_hbm,
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
        """
        Soft preference constraint using phi + running_psi as the effective logit.

        Stage 3: the old hard mood gate (force human/robot when mood posterior is
        confident) is replaced by the psi-adjusted log_prob_prefer. The HBM's
        log_prob_prefer now uses phi + running_psi, so mid-episode mood signals
        automatically shift the soft constraint without any explicit threshold check.
        A strong mood (large |psi|) pushes log_prob_prefer strongly in one direction,
        achieving the same effect as the old hard constraint but in a principled way.
        """
        return [self._pref_gen.generate(obs, variables, "user_preference")]

    def _generate_nonpersonal_constraints(
        self, obs: SpiceState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        return []

    def _generate_cost(
        self, obs: SpiceState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        """
        Stage 3: variance-weighted combined cost for max-entropy training mode.

        In max-entropy training mode, replaces the base class's variance-blind
        entropy cost with a combined exploit + explore cost:

            cost(actor) = exploit_cost(actor) + explore_cost(actor)

        where:
            exploit_cost(actor) = -log_prob_prefer(actor)
                                  (lower for the preferred actor — pure exploitation)
            explore_cost(actor) = 0                         if actor == preferred
                                = -get_phi_entropy(spice)   if actor != preferred
                                  (reward for trying the unexpected actor, scaled
                                   by H(B(sigmoid(phi_mean))) * phi_var)

        This naturally transitions from exploration to exploitation:
          - Large phi_var (few observations): explore_cost dominates → unexpected
            actor gets negative cost bonus → CSP explores.
          - Small phi_var (many observations): explore_cost ≈ 0 → exploit_cost
            dominates → CSP picks the preferred actor.

        No explicit annealing schedule is needed; the posterior variance provides
        the signal automatically. This implements the CBTL entropy criterion from
        the migration plan: H(Bernoulli(sigma(mean))) scaled by var.

        In eval mode or non-max-entropy methods, falls back to exploit_cost only.
        """
        if self._train_or_eval != "train" or self._explore_method != "max-entropy":
            return self._generate_exploit_cost(obs, variables)

        actor_var = variables[0]
        current = obs.current_spice
        hbm = self._pref_gen._hbm
        human_id = self._pref_gen._human_id

        def _combined_cost(actor_val: str) -> float:
            recipe = self._pref_gen._current_recipe_name
            if not recipe or not current:
                return 0.0

            # Exploit component: lower cost for the actor the model prefers
            log_p = hbm.log_prob_prefer(human_id, recipe, current, actor_val)
            exploit_cost = -log_p

            # Explore component: bonus for the unexpected actor, scaled by
            # variance-weighted entropy (large early, shrinks as phi converges)
            phi = hbm.get_phi(human_id, recipe, current)
            preferred = "human" if phi >= 0 else "robot"
            if actor_val != preferred:
                explore_val = hbm.get_phi_entropy(human_id, recipe, current)
                explore_cost = -explore_val
            else:
                explore_cost = 0.0

            return exploit_cost + explore_cost

        return CSPCost("variance_weighted_entropy", [actor_var], _combined_cost)

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
