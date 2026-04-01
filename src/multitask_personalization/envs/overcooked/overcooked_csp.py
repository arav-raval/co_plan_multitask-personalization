"""
overcooked_csp.py — CSP generator for subtask assignment in Overcooked.

Mirrors spices_csp.py with the following adaptations:
  - SpiceState / SpiceAction  →  OvercookedState / OvercookedAction
  - recipe_name / spice        →  layout_name / subtask
  - HierarchicalPreferenceModel  →  OvercookedPreferenceModel
  - Mood posterior monitoring removed (no discrete mood categories in Overcooked)

The max-entropy exploration cost is identical to spices Stage 3: variance-
weighted Bernoulli entropy transitions naturally from exploration to exploitation
as the phi posterior narrows.

The vector psi in OvercookedPreferenceModel is transparent to the CSP —
log_prob_prefer already includes the per-subtask running psi offset.
"""

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
from multitask_personalization.envs.overcooked.config.overcooked_config import (
    DEFAULT_CONFIG,
    OvercookedConfig,
)
from multitask_personalization.envs.overcooked.layouts import ALL_SUBTASKS
from multitask_personalization.envs.overcooked.overcooked_env import (
    OvercookedAction,
    OvercookedState,
)
from multitask_personalization.envs.overcooked.overcooked_hbm import (
    DEFAULT_HUMAN,
    OvercookedPreferenceModel,
)
from multitask_personalization.structs import (
    CSP,
    CSPConstraint,
    CSPCost,
    CSPPolicy,
    CSPSampler,
    CSPVariable,
    FunctionalCSPSampler,
    LogProbCSPConstraint,
)


# ---------------------------------------------------------------------------
# CSP policy
# ---------------------------------------------------------------------------

class _OvercookedCSPPolicy(CSPPolicy[OvercookedState, OvercookedAction]):
    """Converts a CSP solution (actor assignment) into an OvercookedAction."""

    def __init__(
        self, csp_variables: Collection[CSPVariable], seed: int = 0
    ) -> None:
        super().__init__(csp_variables, seed)
        self._actor: str | None = None
        self._done_emitted: bool = False

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._actor = self._get_value("actor")
        self._done_emitted = False

    def step(self, obs: OvercookedState) -> OvercookedAction:
        assert self._actor in ("human", "robot")
        return OvercookedAction(actor=self._actor)

    def check_termination(self, obs: OvercookedState) -> bool:
        return obs.current_subtask is None or self._done_emitted


# ---------------------------------------------------------------------------
# Constraint generator (wraps the HBM)
# ---------------------------------------------------------------------------

class _AssignPreferenceGenerator(
    CSPConstraintGenerator[OvercookedState, OvercookedAction]
):
    """Inner constraint generator: owns the HBM and generates soft constraints."""

    def __init__(
        self,
        subtask_list: list[str],
        human_id: str = DEFAULT_HUMAN,
        layout_list: list[str] | None = None,
        seed: int = 0,
        verbose: bool = False,
        config: OvercookedConfig | None = None,
        shared_hbm: OvercookedPreferenceModel | None = None,
    ) -> None:
        super().__init__(seed)

        self.config = config if config is not None else DEFAULT_CONFIG
        self._human_id = human_id
        self._layout_list: list[str] = layout_list if layout_list is not None else []
        self._verbose = verbose

        if shared_hbm is not None:
            self._hbm = shared_hbm
            self._hbm.register_human(human_id)
            for L in self._layout_list:
                self._hbm.register_layout(human_id, L)
        else:
            self._hbm = OvercookedPreferenceModel(
                subtasks=subtask_list,
                layouts=self._layout_list,
                mu0=self.config.hbm.mu0,
                sigma0=self.config.hbm.sigma0,
                sigma_h=self.config.hbm.sigma_h,
                sigma_r=self.config.hbm.sigma_r,
                sigma_obs=self.config.hbm.sigma_obs,
                config=self.config,
            )
        self._current_layout_name: str | None = None

    def save(self, model_dir: Path) -> None:
        if self._verbose:
            logging.info("[Save] Overcooked HBM (save not yet implemented)")

    def load(self, model_dir: Path) -> None:
        if self._verbose:
            logging.info("[Load] Overcooked HBM (load not yet implemented)")

    def generate(
        self,
        obs: OvercookedState,
        variables: list[CSPVariable],
        name: str,
    ) -> CSPConstraint:
        (actor_var,) = variables
        current = obs.current_subtask

        def _logprob(actor: str) -> float:
            if self._current_layout_name and current:
                return self._hbm.log_prob_prefer(
                    self._human_id, self._current_layout_name, current, actor
                )
            return np.log(0.5)

        return LogProbCSPConstraint(name, [actor_var], _logprob, threshold=np.log(0.3))

    def learn_from_transition(
        self,
        obs: OvercookedState,
        act: OvercookedAction,
        next_obs: OvercookedState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update HBM on each observed transition."""
        if info.get("last_subtask") is None or info.get("last_actor") is None:
            return

        layout_name = obs.layout_name
        if layout_name and layout_name not in self._layout_list:
            self._layout_list.append(layout_name)
        self._current_layout_name = layout_name

        actor = str(info["last_actor"])
        subtask = str(info["last_subtask"])
        task_score = float(info["task_score"])

        if layout_name:
            self._hbm.observe(self._human_id, layout_name, subtask, actor, task_score)

        if done:
            self._finalize_episode()

    def _finalize_episode(self) -> None:
        self._hbm.end_episode(self._human_id)
        if self._verbose:
            logging.info("[Episode] Overcooked HBM updated (theta, mu, psi)")

    def get_metrics(self) -> dict[str, float]:
        """Return psi diagnostics (replaces mood metrics from spices)."""
        psi_vec = self._hbm.get_psi_vec(self._human_id)
        return {
            f"psi_{i}": float(v) for i, v in enumerate(psi_vec)
        }


# ---------------------------------------------------------------------------
# Public CSP generator
# ---------------------------------------------------------------------------

class OvercookedAssignCSPGenerator(
    CSPGenerator[OvercookedState, OvercookedAction]
):
    """
    CSP generator for Overcooked task assignment.

    At each decision step the robot's CSP chooses which agent (human or robot)
    should execute the current subtask.  The HBM soft constraint provides the
    preference signal; variance-weighted entropy provides the exploration signal.
    """

    def __init__(
        self,
        subtask_list: list[str],
        layout_list: list[str] | None = None,
        human_id: str = DEFAULT_HUMAN,
        verbose: bool = False,
        config: OvercookedConfig | None = None,
        shared_hbm: OvercookedPreferenceModel | None = None,
        seed: int = 0,
        explore_method: str = "max-entropy",
        disable_learning: bool = False,
    ) -> None:
        super().__init__(seed=seed, explore_method=explore_method, disable_learning=disable_learning)
        self._subtasks = list(subtask_list)
        self._pref_gen = _AssignPreferenceGenerator(
            subtask_list=self._subtasks,
            human_id=human_id,
            layout_list=layout_list,
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
        """Return current P(prefer actor) per subtask from the HBM."""
        human_id = self._pref_gen._human_id
        layout = self._pref_gen._current_layout_name
        probs: dict[str, dict[str, float]] = {}
        for subtask in self._pref_gen._hbm.subtasks:
            subtask_probs: dict[str, float] = {}
            for actor in ["human", "robot"]:
                if layout:
                    logp = self._pref_gen._hbm.log_prob_prefer(
                        human_id, layout, subtask, actor
                    )
                    p = float(np.exp(logp))
                else:
                    p = 0.5
                subtask_probs[actor] = float(np.clip(p, 1e-6, 1.0 - 1e-6))
            total = sum(subtask_probs.values())
            probs[subtask] = {k: round(v / total, 3) for k, v in subtask_probs.items()}
        return probs

    # ------------------------------------------------------------------
    # CSPGenerator hooks
    # ------------------------------------------------------------------

    def _generate_variables(
        self, obs: OvercookedState
    ) -> tuple[list[CSPVariable], dict[CSPVariable, Any]]:
        actor = CSPVariable("actor", EnumSpace(["human", "robot"]))
        initialization = {actor: self._init_rng.choice(["human", "robot"])}
        return [actor], initialization

    def _generate_personal_constraints(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        """Soft preference constraint from HBM (phi + running_psi)."""
        return [self._pref_gen.generate(obs, variables, "user_preference")]

    def _generate_nonpersonal_constraints(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        return []

    def _generate_cost(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        """
        Variance-weighted combined explore + exploit cost (max-entropy training).

        Identical structure to spices Stage 3:
          cost(actor) = -log_prob_prefer(actor)           [exploit]
                      + (-phi_entropy if actor != preferred)  [explore]
        """
        if self._train_or_eval != "train" or self._explore_method != "max-entropy":
            return self._generate_exploit_cost(obs, variables)

        actor_var = variables[0]
        current = obs.current_subtask
        hbm = self._pref_gen._hbm
        human_id = self._pref_gen._human_id

        def _combined_cost(actor_val: str) -> float:
            layout = self._pref_gen._current_layout_name
            if not layout or not current:
                return 0.0
            log_p = hbm.log_prob_prefer(human_id, layout, current, actor_val)
            exploit_cost = -log_p

            phi = hbm.get_phi(human_id, layout, current)
            psi = hbm.get_running_psi(human_id, current)
            preferred = "human" if (phi + psi) >= 0 else "robot"
            if actor_val != preferred:
                explore_cost = -hbm.get_phi_entropy(human_id, layout, current)
            else:
                explore_cost = 0.0

            return exploit_cost + explore_cost

        return CSPCost("variance_weighted_entropy", [actor_var], _combined_cost)

    def _generate_exploit_cost(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        actor = variables[0]
        current = obs.current_subtask

        def _cost_fn(actor_val: str) -> float:
            layout = self._pref_gen._current_layout_name
            if layout and current:
                return -self._pref_gen._hbm.log_prob_prefer(
                    self._pref_gen._human_id, layout, current, actor_val
                )
            return 0.0

        return CSPCost("maximize_preference", [actor], _cost_fn)

    def _generate_samplers(
        self, obs: OvercookedState, csp: CSP
    ) -> list[CSPSampler]:
        actor = csp.variables[0]
        current = obs.current_subtask

        def _sample_actor(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            probs = []
            for a in ["human", "robot"]:
                layout = self._pref_gen._current_layout_name
                if layout and current:
                    logp = self._pref_gen._hbm.log_prob_prefer(
                        self._pref_gen._human_id, layout, current, a
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
        self, obs: OvercookedState, csp_variables: Collection[CSPVariable]
    ) -> CSPPolicy:
        return _OvercookedCSPPolicy(csp_variables, seed=self._seed)

    def observe_transition(
        self,
        obs: OvercookedState,
        act: OvercookedAction,
        next_obs: OvercookedState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._disable_learning:
            self._pref_gen.learn_from_transition(obs, act, next_obs, done, info)

    def get_metrics(self) -> dict[str, float]:
        return self._pref_gen.get_metrics()
