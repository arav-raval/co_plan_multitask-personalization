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
    """Converts a CSP solution (flag, optional count) into an OvercookedAction."""

    def __init__(
        self, csp_variables: Collection[CSPVariable], seed: int = 0
    ) -> None:
        super().__init__(csp_variables, seed)
        self._flag: int | None = None
        self._count: int | None = None
        self._has_count_var: bool = any(
            v.name == "ingredient_count" for v in csp_variables
        )
        self._done_emitted: bool = False

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._flag = self._get_value("flag")
        self._count = self._get_value("ingredient_count") if self._has_count_var else None
        self._last_subtask: str | None = None

    def step(self, obs: OvercookedState) -> OvercookedAction:
        assert self._flag in (0, 1)
        self._last_subtask = obs.current_subtask
        # Only attach count when the current subtask is load_pot (matches env).
        count = self._count if obs.current_subtask == "load_pot" else None
        return OvercookedAction(flag=self._flag, ingredient_count=count)

    def check_termination(self, obs: OvercookedState) -> bool:
        # Recompute when the subtask changes — each subtask needs a fresh
        # CSP decision. Without this, the same flag persists for all subtasks
        # in the episode, preventing per-subtask preference learning.
        if obs.current_subtask is None:
            return True
        if obs.current_subtask != self._last_subtask:
            return True
        return False


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
        preference_model: Any | None = None,
        scalar_psi: bool = False,
        continuous_pref_model: Any | None = None,
    ) -> None:
        super().__init__(seed)

        self.config = config if config is not None else DEFAULT_CONFIG
        self._human_id = human_id
        self._layout_list: list[str] = layout_list if layout_list is not None else []
        self._verbose = verbose

        if preference_model is not None:
            self._hbm = preference_model
            self._hbm.register_human(human_id)
            for L in self._layout_list:
                self._hbm.register_layout(human_id, L)
        elif shared_hbm is not None:
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
                scalar_psi=scalar_psi,
            )
        self._current_layout_name: str | None = None
        self._continuous_pref_model = continuous_pref_model

        # Conflict tracking: decreasing conflict_rate signals HBM convergence.
        self._episode_steps: int = 0
        self._episode_conflicts: int = 0
        self._conflict_rate: float = 0.0

    def save(self, model_dir: Path) -> None:
        if hasattr(self._hbm, "save"):
            self._hbm.save(model_dir)
        if self._continuous_pref_model is not None and hasattr(
            self._continuous_pref_model, "save"
        ):
            self._continuous_pref_model.save(model_dir)

    def load(self, model_dir: Path) -> None:
        if hasattr(self._hbm, "load"):
            self._hbm.load(model_dir)
        if self._continuous_pref_model is not None and hasattr(
            self._continuous_pref_model, "load"
        ):
            self._continuous_pref_model.load(model_dir)

    def generate(
        self,
        obs: OvercookedState,
        variables: list[CSPVariable],
        name: str,
    ) -> CSPConstraint:
        (flag_var,) = variables
        current = obs.current_subtask

        def _logprob(flag: int) -> float:
            if self._current_layout_name and current:
                # flag=1 → robot passes (predicts human will claim)
                # flag=0 → robot claims (predicts human won't claim)
                actor = "human" if flag == 1 else "robot"
                return self._hbm.log_prob_prefer(
                    self._human_id, self._current_layout_name, current, actor
                )
            return np.log(0.5)

        return LogProbCSPConstraint(name, [flag_var], _logprob, threshold=np.log(0.3))

    def learn_from_transition(
        self,
        obs: OvercookedState,
        act: OvercookedAction,
        next_obs: OvercookedState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update preference model on each observed transition.

        All models receive the same continuous satisfaction signal in [-1, +1].
        Conflict observations are NOT skipped — they carry valid behavioral
        information (the human acted) and the satisfaction signal includes the
        coordination penalty that reflects the misprediction. Only timeout
        steps (no subtask completion → satisfaction == 0) are skipped.
        """
        if info.get("last_subtask") is None or info.get("last_actor") is None:
            return

        # Track conflict rate as convergence diagnostic.
        self._episode_steps += 1
        if info.get("conflict", False):
            self._episode_conflicts += 1

        # Skip only on timeout steps (subtask not completed → satisfaction == 0).
        satisfaction = float(info.get("satisfaction", 0.0))
        if abs(satisfaction) < 1e-6:
            if done:
                self._conflict_rate = (
                    self._episode_conflicts / self._episode_steps
                    if self._episode_steps > 0 else 0.0
                )
                self._episode_steps = 0
                self._episode_conflicts = 0
                self._finalize_episode()
            return

        layout_name = obs.layout_name
        if layout_name and layout_name not in self._layout_list:
            self._layout_list.append(layout_name)
        self._current_layout_name = layout_name

        actor = str(info["last_actor"])
        subtask = str(info["last_subtask"])

        if layout_name:
            # All models receive the same continuous satisfaction signal.
            # This ensures fair comparison — all models get the same
            # information quality from the environment.
            sat = float(info.get("satisfaction", 0.0))
            self._hbm.observe(self._human_id, layout_name, subtask, actor, sat)

            # Continuous preference observation (phase 1: ingredient_count).
            # Only feed observations that carry a continuous-preference signal.
            cont_sat = info.get("continuous_satisfaction")
            count = info.get("ingredient_count")
            if (
                self._continuous_pref_model is not None
                and cont_sat is not None
                and count is not None
            ):
                self._continuous_pref_model.observe(
                    self._human_id, layout_name, int(count), float(cont_sat)
                )

        if done:
            self._conflict_rate = (
                self._episode_conflicts / self._episode_steps
                if self._episode_steps > 0 else 0.0
            )
            self._episode_steps = 0
            self._episode_conflicts = 0
            self._finalize_episode()

    def learn_from_transition_eval(
        self,
        obs: OvercookedState,
        act: OvercookedAction,
        next_obs: OvercookedState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Eval-time update: running psi only (no phi/theta/mu learning).

        Buffers observations and updates running psi so the CSP can adapt to
        session effects (fatigue/energy) mid-episode. At episode end, decays
        psi and resets episode buffers without updating phi/theta/mu.
        """
        if info.get("last_subtask") is None or info.get("last_actor") is None:
            return

        # Skip only on timeout steps; conflicts are valid observations.
        satisfaction = float(info.get("satisfaction", 0.0))
        if abs(satisfaction) < 1e-6:
            if done:
                self._hbm.end_episode_eval(self._human_id)
            return

        layout_name = obs.layout_name
        if layout_name and layout_name not in self._layout_list:
            self._layout_list.append(layout_name)
        self._current_layout_name = layout_name

        actor = str(info["last_actor"])
        subtask = str(info["last_subtask"])

        if layout_name:
            self._hbm.observe(self._human_id, layout_name, subtask, actor, satisfaction)

        if done:
            self._hbm.end_episode_eval(self._human_id)

    def _finalize_episode(self) -> None:
        self._hbm.end_episode(self._human_id)
        if self._continuous_pref_model is not None and hasattr(
            self._continuous_pref_model, "end_episode"
        ):
            self._continuous_pref_model.end_episode(self._human_id)
        if self._verbose:
            logging.info("[Episode] Overcooked HBM updated (theta, mu, psi)")

    def get_metrics(self) -> dict[str, float]:
        """Return psi diagnostics and conflict-rate convergence metric."""
        psi_vec = self._hbm.get_psi_vec(self._human_id)
        return {
            "conflict_rate": self._conflict_rate,
            **{f"psi_{i}": float(v) for i, v in enumerate(psi_vec)},
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
        preference_model: Any | None = None,
        feasibility: dict[str, dict[str, bool]] | None = None,
        seed: int = 0,
        explore_method: str = "max-entropy",
        disable_learning: bool = False,
        mood_learning_enabled: bool = True,
        scalar_psi: bool = False,
        continuous_pref_model: Any | None = None,
        ingredient_count_choices: list[int] | None = None,
    ) -> None:
        super().__init__(seed=seed, explore_method=explore_method, disable_learning=disable_learning)
        self._subtasks = list(subtask_list)
        self._feasibility = feasibility
        if preference_model is not None:
            self._baseline_model = preference_model
        else:
            self._baseline_model = None
        self._continuous_pref_model = continuous_pref_model
        self._ingredient_count_choices: list[int] = (
            list(ingredient_count_choices)
            if ingredient_count_choices is not None
            else [1, 2, 3]
        )
        self._pref_gen = _AssignPreferenceGenerator(
            subtask_list=self._subtasks,
            human_id=human_id,
            layout_list=layout_list,
            seed=self._seed,
            verbose=verbose,
            config=config,
            shared_hbm=shared_hbm if preference_model is None else None,
            preference_model=preference_model,
            scalar_psi=scalar_psi,
            continuous_pref_model=continuous_pref_model,
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
        # Ensure layout name is set from observation (critical for eval mode
        # where learn_from_transition doesn't run to set it).
        if obs.layout_name:
            self._pref_gen._current_layout_name = obs.layout_name
            if obs.layout_name not in self._pref_gen._layout_list:
                self._pref_gen._layout_list.append(obs.layout_name)
                self._pref_gen._hbm.register_layout(
                    self._pref_gen._human_id, obs.layout_name
                )
        # flag=0: robot claims the current subtask
        # flag=1: robot passes (predicts human will claim)
        flag = CSPVariable("flag", EnumSpace([0, 1]))
        initialization: dict[CSPVariable, Any] = {
            flag: int(self._init_rng.integers(0, 2))
        }
        variables: list[CSPVariable] = [flag]
        # Add ingredient_count variable only when applicable.
        if (
            self._continuous_pref_model is not None
            and obs.current_subtask == "load_pot"
        ):
            count_var = CSPVariable(
                "ingredient_count",
                EnumSpace(list(self._ingredient_count_choices)),
            )
            variables.append(count_var)
            initialization[count_var] = int(
                self._init_rng.choice(self._ingredient_count_choices)
            )
        return variables, initialization

    def _generate_personal_constraints(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        """Soft preference constraint from HBM (phi + running_psi).

        If an ingredient_count variable is present (i.e. load_pot subtask +
        continuous pref model), also add a soft constraint from the continuous
        preference model.
        """
        flag_var = next(v for v in variables if v.name == "flag")
        constraints: list[CSPConstraint] = [
            self._pref_gen.generate(obs, [flag_var], "user_preference")
        ]
        count_var = next(
            (v for v in variables if v.name == "ingredient_count"), None
        )
        if count_var is not None and self._continuous_pref_model is not None:
            layout = obs.layout_name or self._pref_gen._current_layout_name
            human_id = self._pref_gen._human_id
            cont_model = self._continuous_pref_model

            def _count_logprob(count: int) -> float:
                if layout is None:
                    return float(np.log(0.5))
                return float(cont_model.log_prob_accept(int(count), human_id, layout))

            constraints.append(
                LogProbCSPConstraint(
                    "continuous_pref",
                    [count_var],
                    _count_logprob,
                    threshold=np.log(0.05),
                )
            )
        return constraints

    def _generate_nonpersonal_constraints(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> list[CSPConstraint]:
        """Physical feasibility constraints from layout reachability.

        If the current subtask is infeasible for one agent, force the other.
        E.g. in forced_coordination, robot can't reach onion dispensers,
        so fetch_ingredient must go to human (flag=1).
        """
        if self._feasibility is None or obs.current_subtask is None:
            return []

        subtask = obs.current_subtask
        robot_ok = self._feasibility.get("robot", {}).get(subtask, True)
        human_ok = self._feasibility.get("human", {}).get(subtask, True)

        if robot_ok and human_ok:
            return []  # both can do it, no constraint needed

        flag_var = variables[0]

        if not robot_ok and human_ok:
            # Robot can't reach — force human (flag=1).
            def _must_be_human(flag: int) -> float:
                return 0.0 if flag == 1 else -100.0
            return [LogProbCSPConstraint(
                "feasibility_human_only", [flag_var], _must_be_human, threshold=-1.0
            )]

        if robot_ok and not human_ok:
            # Human can't reach — force robot (flag=0).
            def _must_be_robot(flag: int) -> float:
                return 0.0 if flag == 0 else -100.0
            return [LogProbCSPConstraint(
                "feasibility_robot_only", [flag_var], _must_be_robot, threshold=-1.0
            )]

        # Neither can reach — shouldn't happen, but don't constrain.
        return []

    def _generate_cost(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        """
        Variance-weighted combined explore + exploit cost (max-entropy training).

        Identical structure to spices Stage 3:
          cost(flag=1) = -log_prob_prefer("human")   [pass → predict human claims]
          cost(flag=0) = -log_prob_prefer("robot")   [claim → predict human won't]
          + (-phi_entropy if flag doesn't match predicted preference)  [explore]
        """
        if self._train_or_eval != "train" or self._explore_method != "max-entropy":
            return self._generate_exploit_cost(obs, variables)

        flag_var = next(v for v in variables if v.name == "flag")
        current = obs.current_subtask
        hbm = self._pref_gen._hbm
        human_id = self._pref_gen._human_id

        def _combined_cost(flag_val: int) -> float:
            layout = self._pref_gen._current_layout_name
            if not layout or not current:
                return 0.0
            actor_for_logprob = "human" if flag_val == 1 else "robot"
            log_p = hbm.log_prob_prefer(human_id, layout, current, actor_for_logprob)
            exploit_cost = -log_p

            phi = hbm.get_phi(human_id, layout, current)
            psi = hbm.get_running_psi(human_id, current)
            preferred_flag = 1 if (phi + psi) >= 0 else 0
            if flag_val != preferred_flag:
                explore_cost = -hbm.get_phi_entropy(human_id, layout, current)
            else:
                explore_cost = 0.0

            return exploit_cost + explore_cost

        return CSPCost("variance_weighted_entropy", [flag_var], _combined_cost)

    def _generate_exploit_cost(
        self, obs: OvercookedState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        flag = next(v for v in variables if v.name == "flag")
        current = obs.current_subtask

        def _cost_fn(flag_val: int) -> float:
            layout = self._pref_gen._current_layout_name
            if layout and current:
                actor_for_logprob = "human" if flag_val == 1 else "robot"
                return -self._pref_gen._hbm.log_prob_prefer(
                    self._pref_gen._human_id, layout, current, actor_for_logprob
                )
            return 0.0

        return CSPCost("maximize_preference", [flag], _cost_fn)

    def _generate_samplers(
        self, obs: OvercookedState, csp: CSP
    ) -> list[CSPSampler]:
        flag = next(v for v in csp.variables if v.name == "flag")
        count_var = next(
            (v for v in csp.variables if v.name == "ingredient_count"), None
        )
        current = obs.current_subtask

        def _sample_flag(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            layout = self._pref_gen._current_layout_name
            if layout and current:
                logp_human = self._pref_gen._hbm.log_prob_prefer(
                    self._pref_gen._human_id, layout, current, "human"
                )
                p_human = float(np.exp(logp_human))
            else:
                p_human = 0.5
            p_human = max(p_human, 1e-6)
            p_robot = max(1.0 - p_human, 1e-6)
            probs_arr = np.array([p_robot, p_human])  # [flag=0, flag=1]
            probs_arr /= probs_arr.sum()
            return {flag: int(rng.choice([0, 1], p=probs_arr))}

        samplers: list[CSPSampler] = [FunctionalCSPSampler(_sample_flag, csp, {flag})]

        if count_var is not None and self._continuous_pref_model is not None:
            choices = list(self._ingredient_count_choices)
            cont_model = self._continuous_pref_model
            human_id = self._pref_gen._human_id

            def _sample_count(
                sol: dict[CSPVariable, Any], rng: np.random.Generator
            ) -> dict[CSPVariable, Any]:
                layout = self._pref_gen._current_layout_name
                if layout is None:
                    return {count_var: int(rng.choice(choices))}
                probs = np.array(
                    [
                        max(cont_model.acceptance_prob(int(c), human_id, layout), 1e-6)
                        for c in choices
                    ],
                    dtype=float,
                )
                probs /= probs.sum()
                return {count_var: int(rng.choice(choices, p=probs))}

            samplers.append(FunctionalCSPSampler(_sample_count, csp, {count_var}))

        return samplers

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

    # NOTE: observe_transition_eval is intentionally omitted for now.
    # When present, csp_approach.py calls it during eval, which triggers
    # hbm.observe() + _update_running_psi() on every eval step. This causes
    # hangs in some seeds due to numerical issues in the psi optimizer after
    # loading saved models. The eval psi adaptation is only needed for
    # transfer/multi-human claims; for single-layout claims (1,2,3,5) it is
    # not required. Re-enable and debug for transfer experiments later.

    def get_metrics(self) -> dict[str, float]:
        return self._pref_gen.get_metrics()
