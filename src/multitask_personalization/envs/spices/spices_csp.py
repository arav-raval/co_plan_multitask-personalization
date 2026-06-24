"""CSP Elements for the spices environment."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Collection

import numpy as np
import torch
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
from multitask_personalization.envs.spices.spices_baselines import (
    FlatPreferenceModel,
    CBTLClassifierModel,
)

# Union type for any duck-typed preference model accepted by _AssignPreferenceGenerator.
_AnyPreferenceModel = (HierarchicalPreferenceModel | FlatPreferenceModel | CBTLClassifierModel)
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
        self._flag: int | None = None
        self._done_emitted = False
        self._action_taken = False

    def reset(self, solution: dict[CSPVariable, Any]) -> None:
        super().reset(solution)
        self._flag = self._get_value("flag")
        self._done_emitted = False
        self._action_taken = False

    def step(self, obs: SpiceState) -> SpiceAction:
        if (not obs.current_spice) or (
            len(obs.feasible_next) == 0 and len(obs.remaining_spices) == 0
        ):
            self._done_emitted = True
            return (1, None)

        assert self._flag in (0, 1)
        self._action_taken = True
        return (self._flag, None)

    def check_termination(self, obs: SpiceState) -> bool:
        # Re-solve the CSP for each new spice so per-spice phi is used.
        if self._action_taken:
            self._action_taken = False
            return True
        return self._done_emitted


class _AssignPreferenceGenerator(CSPConstraintGenerator[SpiceState, SpiceAction]):
    def __init__(
        self,
        spice_list: list[str],
        neutral_confidence_threshold: float,
        human_id: str = DEFAULT_HUMAN,
        recipe_list: list[str] | None = None,
        mood_learning_enabled: bool = True,
        seed: int = 0,
        verbose: bool = False,
        config: SpicesConfig | None = None,
        shared_hbm: HierarchicalPreferenceModel | None = None,
        preference_model: _AnyPreferenceModel | None = None,
    ) -> None:
        super().__init__(seed)

        self.config = config if config is not None else DEFAULT_CONFIG
        self._human_id = human_id
        self._recipe_list: list[str] = recipe_list if recipe_list is not None else []
        self._neutral_confidence_threshold = neutral_confidence_threshold
        self._verbose = verbose

        if preference_model is not None:
            # Use the provided alternative preference model (FlatPreferenceModel,
            # CBTLClassifierModel, etc.) instead of the default HBM.
            self._hbm = preference_model
            self._hbm.register_human(human_id)
            for r in self._recipe_list:
                self._hbm.register_recipe(human_id, r)
        elif shared_hbm is not None:
            self._hbm = shared_hbm
            self._hbm.register_human(human_id)
            for r in self._recipe_list:
                self._hbm.register_recipe(human_id, r)
        else:
            self._hbm = HierarchicalPreferenceModel(
                spices=spice_list,
                recipes=self._recipe_list,
                enable_mood_learning=mood_learning_enabled,
                mu0=self.config.hbm.mu0,
                sigma0=self.config.hbm.sigma0,
                sigma_h=self.config.hbm.sigma_h,
                sigma_r=self.config.hbm.sigma_r,
                sigma_obs=self.config.hbm.sigma_obs,
                config=self.config,
            )
        self._current_recipe_name: str | None = None

        # Sentinel spices for lightweight per-episode HBM diagnostics.
        # One from each theta-magnitude band (strong+, strong-, mid+, mid-).
        # Logged in get_metrics() so they appear in train_results.csv at each
        # episode boundary without flooding intra-episode steps.
        # Only populated when _hbm is HierarchicalPreferenceModel.
        self._SENTINELS: tuple[str, ...] = (
            "salt",       # theta=+2.0  (strong positive)
            "pepper",     # theta=-2.0  (strong negative)
            "coriander",  # theta=+0.8  (mid positive)
            "cardamom",   # theta=-0.8  (mid negative)
            "cinnamon",   # theta=+0.5  (nuanced positive)
        )
        # Holds the last-updated HBM diagnostics. Persists across steps so the CSV
        # always has a value (from most recent episode end) rather than NaN.
        self._hbm_metrics: dict[str, float] = {}

        # Conflict tracking: conflict rate decreases as HBM converges on human preferences.
        self._episode_steps: int = 0
        self._episode_conflicts: int = 0
        self._conflict_rate: float = 0.0  # from most recent completed episode

    def _get_mood_posterior(self) -> np.ndarray:
        """Return mood posterior array, defaulting to neutral for non-HBM models."""
        mp = self._hbm._mood_posterior.get(self._human_id)
        if mp is None:
            return np.array([0.0, 1.0, 0.0])
        return mp

    def get_expected_mood(self) -> float:
        """Return expected mood value: -1.0 (none_self) to +1.0 (all_self)."""
        mp = self._get_mood_posterior()
        mood_values = {"all_self": +1.0, "neutral": 0.0, "none_self": -1.0}
        return sum(mp[i] * mood_values[m] for i, m in enumerate(MOODS))

    def get_mood_posterior_breakdown(self) -> dict[str, float]:
        """Return mood posterior probabilities for each mood."""
        mp = self._get_mood_posterior()
        return {mood: float(mp[i]) for i, mood in enumerate(MOODS)}

    def get_most_likely_mood(self) -> tuple[str, float]:
        """Return the most likely mood and its probability."""
        mp = self._get_mood_posterior()
        idx = int(np.argmax(mp))
        return MOODS[idx], float(mp[idx])

    def save(self, model_dir: Path) -> None:
        if isinstance(self._hbm, HierarchicalPreferenceModel):
            hbm = self._hbm
            state = {
                "theta_m": {
                    h: {s: t.detach().clone() for s, t in sv.items()}
                    for h, sv in hbm._theta_m.items()
                },
                "theta_logv": {
                    h: {s: t.detach().clone() for s, t in sv.items()}
                    for h, sv in hbm._theta_logv.items()
                },
                "phi_m": {
                    h: {r: {s: t.detach().clone() for s, t in sv.items()} for r, sv in rv.items()}
                    for h, rv in hbm._phi_m.items()
                },
                "phi_logv": {
                    h: {r: {s: t.detach().clone() for s, t in sv.items()} for r, sv in rv.items()}
                    for h, rv in hbm._phi_logv.items()
                },
                "mu_mean": dict(hbm.mu_mean),
                "mu_var": dict(hbm.mu_var),
                "log_sigma_h": hbm.log_sigma_h.detach().clone(),
                "log_sigma_r": hbm.log_sigma_r.detach().clone(),
                "log_sigma_obs": hbm.log_sigma_obs.detach().clone(),
                "current_recipe_name": self._current_recipe_name,
            }
            with open(model_dir / "spice_hbm.pkl", "wb") as f:
                pickle.dump(state, f)
        else:
            # FlatPreferenceModel / CBTLClassifierModel delegate to their own save().
            self._hbm.save(model_dir)

    def load(self, model_dir: Path) -> None:
        if isinstance(self._hbm, HierarchicalPreferenceModel):
            path = model_dir / "spice_hbm.pkl"
            if not path.exists():
                return
            with open(path, "rb") as f:
                state = pickle.load(f)
            hbm = self._hbm
            for h, sv in state["theta_m"].items():
                if h not in hbm._theta_m:
                    hbm.register_human(h)
                for s, t in sv.items():
                    hbm._theta_m[h][s] = t.requires_grad_(True)
            for h, sv in state["theta_logv"].items():
                for s, t in sv.items():
                    hbm._theta_logv[h][s] = t.requires_grad_(True)
            for h, rv in state["phi_m"].items():
                for r, sv in rv.items():
                    if r not in hbm._phi_m.get(h, {}):
                        hbm.register_recipe(h, r)
                    for s, t in sv.items():
                        hbm._phi_m[h][r][s] = t.requires_grad_(True)
            for h, rv in state["phi_logv"].items():
                for r, sv in rv.items():
                    for s, t in sv.items():
                        hbm._phi_logv[h][r][s] = t.requires_grad_(True)
            hbm.mu_mean.update(state["mu_mean"])
            hbm.mu_var.update(state["mu_var"])
            with torch.no_grad():
                hbm.log_sigma_h.copy_(state["log_sigma_h"])
                hbm.log_sigma_r.copy_(state["log_sigma_r"])
                hbm.log_sigma_obs.copy_(state["log_sigma_obs"])
            self._current_recipe_name = state["current_recipe_name"]

            # Cross-recipe transfer: any recipe registered in this HBM but absent
            # from the loaded checkpoint needs phi re-initialized from the (now
            # loaded) theta.  This enables Claim 2: train on recipes A,B,C, load
            # into an eval approach that has recipe D registered — phi_D gets
            # warm-started from the learned theta.
            loaded_recipes = set(state.get("phi_m", {}).get("human", {}).keys())
            for h in list(hbm._phi_m.keys()):
                for r in list(hbm._phi_m[h].keys()):
                    if r not in loaded_recipes:
                        # Re-register: overwrites phi with current theta values
                        del hbm._phi_m[h][r]
                        del hbm._phi_logv[h][r]
                        hbm.register_recipe(h, r)
        else:
            self._hbm.load(model_dir)

    def generate(self, obs: SpiceState, variables: list[CSPVariable], name: str) -> CSPConstraint:
        (flag_var,) = variables
        current = obs.current_spice

        def _logprob(flag: int) -> float:
            if self._current_recipe_name and current:
                # flag=1 means robot predicts human will claim → prefer when P(human) is high
                # flag=0 means robot claims → prefer when P(robot) is high (P(human) is low)
                actor = "human" if flag == 1 else "robot"
                return self._hbm.log_prob_prefer(
                    self._human_id, self._current_recipe_name, current, actor
                )
            return np.log(0.5)

        return LogProbCSPConstraint(name, [flag_var], _logprob, threshold=np.log(0.3))

    def learn_from_transition(
        self,
        obs: SpiceState,
        act: SpiceAction,
        next_obs: SpiceState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update HBM on each observed transition.

        Uses task_score as the behavioral observation signal:
          +1  human claimed this spice (including conflicts where human won)
          -1  human did not claim this spice

        Conflict information is preserved in info["conflict"] for the
        conflict_rate metric, and the coordination penalty in the satisfaction
        signal captures coordination quality separately. We do NOT skip
        observations on conflict — every step provides a clean behavioral label.
        """
        if info.get("last_spice") is None or info.get("last_actor") is None:
            return

        # Track conflict rate as convergence diagnostic.
        self._episode_steps += 1
        if info.get("conflict", False):
            self._episode_conflicts += 1

        task_score = float(info.get("task_score", info.get("satisfaction", 0.0)))

        recipe_name = info.get("recipe_name") or self._current_recipe_name
        if recipe_name and recipe_name not in self._recipe_list:
            self._recipe_list.append(recipe_name)
            self._hbm.recipes = list(self._recipe_list)
        self._current_recipe_name = recipe_name

        actor = str(info["last_actor"])
        spice = str(info["last_spice"])

        if recipe_name:
            if isinstance(self._hbm, HierarchicalPreferenceModel):
                # HBM receives continuous satisfaction (Beta-sampled in [-1,+1]).
                # This encodes preference magnitude via true hidden phi, giving the
                # Gaussian likelihood term meaningful gradient signal beyond {±1} extremes.
                sat = float(info.get("satisfaction", task_score))
            else:
                # CBTL/FlatModel: keep binary task_score for faithful comparison.
                # Their soft-label weight (sat+1)/2 maps {+1,-1} → {1,0} cleanly.
                sat = task_score
            self._hbm.observe(self._human_id, recipe_name, spice, actor, sat)

        # Capture mood AFTER the observation update but BEFORE the episode reset
        # so callers always see the inferred mood for the current step.
        info["mood_posterior"] = self.get_mood_posterior_breakdown()
        info["expected_mood"] = self.get_expected_mood()

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
        obs: SpiceState,
        act: SpiceAction,
        next_obs: SpiceState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update running_psi only (no phi/theta learning) for mid-episode eval adaptation."""
        if info.get("last_spice") is None or info.get("last_actor") is None:
            return
        recipe_name = info.get("recipe_name") or self._current_recipe_name
        if not recipe_name:
            return
        self._current_recipe_name = recipe_name
        actor = str(info["last_actor"])
        spice = str(info["last_spice"])
        task_score = float(info.get("task_score", 0.0))
        # HBM eval psi adaptation uses continuous satisfaction; CBTL uses binary task_score.
        if isinstance(self._hbm, HierarchicalPreferenceModel):
            sat = float(info.get("satisfaction", task_score))
        else:
            sat = task_score
        self._hbm.observe_eval(
            self._human_id, recipe_name, spice, actor, sat, done
        )

    def _finalize_episode(self) -> None:
        """Update hierarchical HBM preferences and reset episode state.

        end_episode handles batch phi updates, theta/mu propagation, and mood
        posterior reset to prior (since each episode draws a fresh mood).
        """
        self._hbm.end_episode(self._human_id, neutral_threshold=self._neutral_confidence_threshold)
        if self._verbose:
            logging.info("[Episode] HBM updated hierarchical preferences (θ, μ)")
        self._update_hbm_metrics()

    def _update_hbm_metrics(self) -> None:
        """Snapshot sentinel-spice theta and phi values into _hbm_metrics.

        Only runs for HierarchicalPreferenceModel (no-op for baselines).
        Emits one theta value per sentinel (shared across recipes) and one phi
        value per sentinel per recipe — compact enough not to flood the CSV.
        """
        if not isinstance(self._hbm, HierarchicalPreferenceModel):
            return
        hbm = self._hbm
        h = self._human_id
        metrics: dict[str, float] = {}
        for spice in self._SENTINELS:
            if h in hbm._theta_m and spice in hbm._theta_m[h]:
                metrics[f"theta_{spice}"] = hbm.get_theta(h, spice)
                metrics[f"theta_var_{spice}"] = hbm.get_theta_var(h, spice)
            for recipe in self._recipe_list:
                if (
                    h in hbm._phi_m
                    and recipe in hbm._phi_m[h]
                    and spice in hbm._phi_m[h][recipe]
                ):
                    short = recipe[:8]  # truncate recipe name to keep key width sane
                    metrics[f"phi_{spice}_{short}"] = hbm.get_phi(h, recipe, spice)
                    metrics[f"phi_var_{spice}_{short}"] = hbm.get_phi_var(h, recipe, spice)
        self._hbm_metrics = metrics

    def get_metrics(self) -> dict[str, float]:
        """Return HBM diagnostics and conflict-rate convergence metric."""
        return {
            "conflict_rate": self._conflict_rate,
            **self._hbm_metrics,
        }


class SpicesAssignCSPGenerator(CSPGenerator[SpiceState, SpiceAction]):
    """CSP: choose the actor for the current spice; learn preferences via HBM."""

    def __init__(
        self,
        spice_list: list[str],
        recipe_list: list[str] | None = None,
        neutral_confidence_threshold: float = 0.75,
        human_id: str = DEFAULT_HUMAN,
        mood_learning_enabled: bool = True,
        verbose: bool = False,
        config: SpicesConfig | None = None,
        shared_hbm: HierarchicalPreferenceModel | None = None,
        preference_model: _AnyPreferenceModel | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._spices = list(spice_list)
        self._pref_gen = _AssignPreferenceGenerator(
            self._spices,
            neutral_confidence_threshold=neutral_confidence_threshold,
            human_id=human_id,
            recipe_list=recipe_list,
            mood_learning_enabled=mood_learning_enabled,
            seed=self._seed,
            verbose=verbose,
            config=config,
            shared_hbm=shared_hbm,
            preference_model=preference_model,
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
        # flag=0: robot claims the current spice
        # flag=1: robot passes (predicts human will claim)
        flag = CSPVariable("flag", EnumSpace([0, 1]))
        initialization = {flag: int(self._init_rng.integers(0, 2))}
        return [flag], initialization

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

            cost(flag) = exploit_cost(flag) + explore_cost(flag)

        where:
            exploit_cost(flag=1) = -log_prob_prefer("human")   [pass → predict human claims]
            exploit_cost(flag=0) = -log_prob_prefer("robot")   [claim → predict human won't]
            explore_cost(flag)   = 0                           if flag matches predicted preference
                                 = -get_phi_entropy(spice)     otherwise
                                   (reward for exploring uncertainty, scaled by
                                    H(B(sigmoid(phi_mean))) * phi_var)

        This naturally transitions from exploration to exploitation:
          - Large phi_var (few observations): explore_cost dominates → unexpected
            flag gets negative cost bonus → CSP explores.
          - Small phi_var (many observations): explore_cost ≈ 0 → exploit_cost
            dominates → CSP picks the flag that matches the model's preference.

        No explicit annealing schedule is needed; the posterior variance provides
        the signal automatically. This implements the CBTL entropy criterion from
        the migration plan: H(Bernoulli(sigma(mean))) scaled by var.

        In eval mode or non-max-entropy methods, falls back to exploit_cost only.
        """
        if self._train_or_eval != "train" or self._explore_method != "max-entropy":
            return self._generate_exploit_cost(obs, variables)

        flag_var = variables[0]
        current = obs.current_spice
        hbm = self._pref_gen._hbm
        human_id = self._pref_gen._human_id

        def _combined_cost(flag_val: int) -> float:
            recipe = self._pref_gen._current_recipe_name
            if not recipe or not current:
                return 0.0

            # flag=1 (pass) → robot predicts human will claim; flag=0 (claim) → robot predicts robot
            actor_for_logprob = "human" if flag_val == 1 else "robot"
            log_p = hbm.log_prob_prefer(human_id, recipe, current, actor_for_logprob)
            exploit_cost = -log_p

            # Explore component: bonus for the uncertain flag, scaled by variance-weighted entropy.
            phi = hbm.get_phi(human_id, recipe, current)
            preferred_flag = 1 if phi >= 0 else 0  # phi>0 → human prefers, robot should pass
            if flag_val != preferred_flag:
                explore_val = hbm.get_phi_entropy(human_id, recipe, current)
                explore_cost = -explore_val
            else:
                explore_cost = 0.0

            return exploit_cost + explore_cost

        return CSPCost("variance_weighted_entropy", [flag_var], _combined_cost)

    def _generate_exploit_cost(
        self, obs: SpiceState, variables: list[CSPVariable]
    ) -> CSPCost | None:
        """Minimize negative HBM log-probability for the predicted actor."""
        flag = variables[0]
        current = obs.current_spice

        def _cost_fn(flag_val: int) -> float:
            if self._pref_gen._current_recipe_name and current:
                actor_for_logprob = "human" if flag_val == 1 else "robot"
                return -self._pref_gen._hbm.log_prob_prefer(
                    self._pref_gen._human_id,
                    self._pref_gen._current_recipe_name,
                    current,
                    actor_for_logprob,
                )
            return 0.0

        return CSPCost("maximize_preference", [flag], _cost_fn)

    def _generate_samplers(self, obs: SpiceState, csp: CSP) -> list[CSPSampler]:
        flag = csp.variables[0]
        current_spice = obs.current_spice

        def _sample_flag(
            sol: dict[CSPVariable, Any], rng: np.random.Generator
        ) -> dict[CSPVariable, Any]:
            # P(flag=1) = P(human will claim) — sample proportional to HBM posterior.
            if self._pref_gen._current_recipe_name and current_spice:
                logp_human = self._pref_gen._hbm.log_prob_prefer(
                    self._pref_gen._human_id,
                    self._pref_gen._current_recipe_name,
                    current_spice,
                    "human",
                )
                p_human = float(np.exp(logp_human))
            else:
                p_human = 0.5
            p_human = max(p_human, 1e-6)
            p_robot = max(1.0 - p_human, 1e-6)
            probs_arr = np.array([p_robot, p_human])  # [flag=0, flag=1]
            probs_arr /= probs_arr.sum()
            return {flag: int(rng.choice([0, 1], p=probs_arr))}

        return [FunctionalCSPSampler(_sample_flag, csp, {flag})]

    def _generate_policy(
        self, obs: SpiceState, csp_variables: Collection[CSPVariable]
    ) -> CSPPolicy:
        return _SpiceCSPPolicy(csp_variables, seed=self._seed)

    def observe_transition_eval(
        self,
        obs: SpiceState,
        act: SpiceAction,
        next_obs: SpiceState,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        """Update running_psi mid-episode at eval time without any phi/theta learning."""
        self._pref_gen.learn_from_transition_eval(obs, act, next_obs, done, info)

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
