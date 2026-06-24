"""An approach that generates and solves a CSP to make decisions."""

import logging
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from pybullet_helpers.motion_planning import MotionPlanningHyperparameters
from tomsutils.llm import LargeLanguageModel

from multitask_personalization.csp_generation import CSPGenerator
from multitask_personalization.csp_solvers import CSPSolver
from multitask_personalization.envs.cooking.cooking_csp import CookingCSPGenerator
from multitask_personalization.envs.cooking.cooking_hidden_spec import (
    MealSpecMealPreferenceModel,
)
from multitask_personalization.envs.cooking.cooking_scene_spec import CookingSceneSpec
from multitask_personalization.envs.feeding.feeding_csp import FeedingCSPGenerator
from multitask_personalization.envs.feeding.feeding_env import FeedingEnv
from multitask_personalization.envs.feeding.feeding_scene_spec import FeedingSceneSpec
from multitask_personalization.envs.pybullet.pybullet_csp import PyBulletCSPGenerator
from multitask_personalization.envs.pybullet.pybullet_env import PyBulletEnv
from multitask_personalization.envs.pybullet.pybullet_scene_spec import (
    PyBulletSceneSpec,
)
from multitask_personalization.envs.overcooked.overcooked_baselines import (
    OvercookedCBTLClassifierModel,
    OvercookedFlatPreferenceModel,
)
from multitask_personalization.envs.overcooked.continuous_prefs import (
    ContinuousPreferenceCBTL,
    ContinuousPreferenceFlat,
    ContinuousPreferenceHBM,
)
from multitask_personalization.envs.overcooked.overcooked_csp import (
    OvercookedAssignCSPGenerator,
)
from multitask_personalization.envs.overcooked.overcooked_env import (
    OvercookedSceneSpec,
)
from multitask_personalization.envs.spices.recipes import get_recipe
from multitask_personalization.envs.spices.spices_baselines import (
    CBTLClassifierModel,
    FlatPreferenceModel,
)
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import SpiceSceneSpec
from multitask_personalization.envs.tiny.tiny_csp import TinyCSPGenerator
from multitask_personalization.envs.tiny.tiny_env import TinySceneSpec
from multitask_personalization.methods.approach import (
    ApproachFailure,
    BaseApproach,
    _ActType,
    _ObsType,
)
from multitask_personalization.rom.models import SphericalROMModel
from multitask_personalization.structs import CSPPolicy, CSPVariable, PublicSceneSpec
from multitask_personalization.utils import Threshold1DModel, visualize_csp_graph


class CSPApproach(BaseApproach[_ObsType, _ActType]):
    """An approach that generates and solves a CSP to make decisions."""

    def __init__(
        self,
        scene_spec: PublicSceneSpec,
        action_space: gym.spaces.Space[_ActType],
        csp_solver: CSPSolver,
        llm: LargeLanguageModel | None = None,
        motion_planning_quality: str = "normal",
        explore_method: str = "nothing-personal",
        disable_learning: bool = False,
        mood_learning_enabled: bool = True,
        preference_model_type: str = "hbm",
        psi_type: str = "vector",
        cbtl_pooled_across_humans: bool = True,
        csp_save_dir: str | None = None,
        seed: int = 0,
        lifelong_learning: dict | None = None,
    ):
        super().__init__(scene_spec, action_space, seed)
        self._llm = llm
        self._csp_solver = csp_solver
        self._current_policy: CSPPolicy | None = None
        self._current_sol: dict[CSPVariable, Any] | None = None
        self._explore_method = explore_method
        self._disable_learning = disable_learning
        self._mood_learning_enabled = mood_learning_enabled
        self._preference_model_type = preference_model_type
        self._psi_type = psi_type
        self._cbtl_pooled_across_humans = cbtl_pooled_across_humans
        self._motion_planning_quality = motion_planning_quality
        self._csp_save_dir = Path(csp_save_dir) if csp_save_dir else None
        self._lifelong_learning = lifelong_learning
        self._csp_generator = self._create_csp_generator()

    def reset(
        self,
        obs: _ObsType,
        info: dict[str, Any],
    ) -> None:
        super().reset(obs, info)
        self._sync_csp_generator_train_eval()
        self._recompute_policy(obs)

    def _recompute_policy(
        self, obs: _ObsType, force_exclude_personal_constraints: bool = False
    ) -> None:
        assert isinstance(self._csp_generator, CSPGenerator)
        csp, samplers, policy, initialization = self._csp_generator.generate(
            obs,
            force_exclude_personal_constraints=force_exclude_personal_constraints,
        )
        # Save the generated CSP.
        if self._csp_save_dir is not None:
            self._csp_save_dir.mkdir(exist_ok=True)
            while True:
                time_str = time.strftime("%Y%m%d-%H%M%S")
                viz_file = self._csp_save_dir / f"csp_{time_str}.png"
                if viz_file.exists():
                    time.sleep(1)
                else:
                    break
            visualize_csp_graph(csp, viz_file)
        self._current_sol = self._csp_solver.solve(csp, initialization, samplers)
        if self._current_sol is None:
            # Special case: if CSP is exploit-only, fall back to no personal
            # constraints in the (rare) case of failure.
            if self._explore_method in ("exploit-only", "epsilon-greedy"):
                return self._recompute_policy(
                    obs, force_exclude_personal_constraints=True
                )
            raise ApproachFailure("No solution found for generated CSP")
        self._current_policy = policy
        self._current_policy.reset(self._current_sol)

    def _get_action(self) -> _ActType:
        assert self._last_observation is not None
        assert self._last_info is not None
        if self._current_policy is None or self._current_policy.check_termination(
            self._last_observation
        ):
            logging.debug("Recomputing policy because of termination")
            self._recompute_policy(
                self._last_observation,
            )
        assert self._current_policy is not None
        return self._current_policy.step(self._last_observation)

    def update(
        self,
        obs: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        # At eval time, update running_psi so the CSP adapts mid-episode to mood
        # signals without modifying any learned phi/theta state.
        if self._train_or_eval == "eval" and hasattr(
            self._csp_generator, "observe_transition_eval"
        ):
            self._csp_generator.observe_transition_eval(
                self._last_observation, self._last_action, obs, done, info
            )
        super().update(obs, reward, done, info)

    def _learn_from_transition(
        self,
        obs: _ObsType,
        act: _ActType,
        next_obs: _ObsType,
        reward: float,
        done: bool,
        info: dict[str, Any],
    ) -> None:
        assert np.isclose(reward, 0.0), "Rewards not used in this project!"
        self._csp_generator.observe_transition(obs, act, next_obs, done, info)

    def get_step_metrics(self) -> dict[str, float]:
        step_metrics = super().get_step_metrics()
        csp_metrics = self._csp_generator.get_metrics()
        assert not set(csp_metrics) & set(step_metrics), "Metric name conflict"
        step_metrics.update(csp_metrics)
        return step_metrics

    def save(self, model_dir: Path) -> None:
        self._csp_generator.save(model_dir)

    def load(self, model_dir: Path) -> None:
        self._csp_generator.load(model_dir)

    def train(self) -> None:
        super().train()
        self._sync_csp_generator_train_eval()

    def eval(self) -> None:
        super().eval()
        self._sync_csp_generator_train_eval()

    def _create_csp_generator(self) -> CSPGenerator:
        if isinstance(self._scene_spec, TinySceneSpec):
            return TinyCSPGenerator(
                seed=self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
            )
        if isinstance(self._scene_spec, PyBulletSceneSpec):
            assert self._llm is not None
            pybullet_sim = PyBulletEnv(
                self._scene_spec, self._llm, seed=self._seed, use_gui=False
            )
            rom_model = SphericalROMModel(self._scene_spec.human_spec, self._seed)
            if self._motion_planning_quality == "normal":
                max_motion_planning_candidates = 1
                base_mp_hyperparameters = MotionPlanningHyperparameters()
            elif self._motion_planning_quality == "good":
                max_motion_planning_candidates = 25
                base_mp_hyperparameters = MotionPlanningHyperparameters(
                    birrt_extend_num_interp=50,
                    birrt_num_attempts=100,
                    birrt_num_iters=100,
                    birrt_smooth_amt=250,
                )
            else:
                raise ValueError(
                    f"Unknown motion planning quality: {self._motion_planning_quality}"
                )
            return PyBulletCSPGenerator(
                pybullet_sim,
                rom_model,
                self._llm,
                seed=self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
                max_motion_planning_candidates=max_motion_planning_candidates,
                base_mp_hyperparameters=base_mp_hyperparameters,
            )
        if isinstance(self._scene_spec, CookingSceneSpec):
            meal_model = MealSpecMealPreferenceModel(
                self._scene_spec.universal_meal_specs,
                self._scene_spec.preference_shift_spec,
                lifelong_learning=self._lifelong_learning,
            )
            return CookingCSPGenerator(
                self._scene_spec,
                meal_model,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
            )
        if isinstance(self._scene_spec, FeedingSceneSpec):
            occlusion_scale_model = Threshold1DModel(0.0, 1.0)
            feeding_sim = FeedingEnv(self._scene_spec)
            return FeedingCSPGenerator(
                feeding_sim,
                occlusion_scale_model,
                self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
            )
        if isinstance(self._scene_spec, OvercookedSceneSpec):
            spec = self._scene_spec
            subtask_list = list(spec.subtask_list)
            layout_list = [spec.layout_spec.name]
            preference_model = None
            if self._preference_model_type == "flat":
                preference_model = OvercookedFlatPreferenceModel(
                    subtasks=subtask_list, layouts=layout_list
                )
            elif self._preference_model_type == "cbtl":
                preference_model = OvercookedCBTLClassifierModel(
                    subtasks=subtask_list, layouts=layout_list,
                    pooled_across_humans=self._cbtl_pooled_across_humans,
                )
            # --- FUTURE WORK: continuous preferences model (phase 1) ---
            # Gated skeleton; only triggered by overcooked_continuous.yaml.
            # See envs/overcooked/continuous_prefs.py for status.
            continuous_pref_model: Any = None
            if getattr(spec, "continuous_prefs_enabled", False):
                choices = tuple(getattr(spec, "ingredient_count_choices", (1, 2, 3)))
                param_min = float(min(choices))
                param_max = float(max(choices))
                if self._preference_model_type == "flat":
                    continuous_pref_model = ContinuousPreferenceFlat(
                        param_min=param_min, param_max=param_max
                    )
                elif self._preference_model_type == "cbtl":
                    continuous_pref_model = ContinuousPreferenceCBTL(
                        param_min=param_min,
                        param_max=param_max,
                        pooled_across_humans=self._cbtl_pooled_across_humans,
                    )
                else:
                    continuous_pref_model = ContinuousPreferenceHBM(
                        param_min=param_min, param_max=param_max
                    )
            ingredient_count_choices = (
                list(getattr(spec, "ingredient_count_choices", (1, 2, 3)))
                if getattr(spec, "continuous_prefs_enabled", False)
                else None
            )
            return OvercookedAssignCSPGenerator(
                subtask_list=subtask_list,
                layout_list=layout_list,
                seed=self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
                mood_learning_enabled=self._mood_learning_enabled,
                preference_model=preference_model,
                feasibility=spec.feasibility,
                scalar_psi=(self._psi_type == "scalar"),
                continuous_pref_model=continuous_pref_model,
                ingredient_count_choices=ingredient_count_choices,
            )
        if isinstance(self._scene_spec, SpiceSceneSpec):
            spec = self._scene_spec
            if spec.train_recipe_names:
                train_names = list(spec.train_recipe_names)
                union: set[str] = set()
                for n in train_names:
                    union.update(get_recipe(n).spices)
                spice_list = sorted(union)
                recipe_list = train_names
            else:
                recipe = spec.recipe
                spice_list = list(recipe.spices)
                recipe_list = [recipe.name]
            preference_model = None
            if self._preference_model_type == "flat":
                preference_model = FlatPreferenceModel(
                    spices=spice_list, recipes=recipe_list
                )
            elif self._preference_model_type == "cbtl":
                preference_model = CBTLClassifierModel(
                    spices=spice_list, recipes=recipe_list,
                    pooled_across_humans=self._cbtl_pooled_across_humans,
                )
            return SpicesAssignCSPGenerator(
                spice_list=spice_list,
                recipe_list=recipe_list,
                seed=self._seed,
                explore_method=self._explore_method,
                disable_learning=self._disable_learning,
                mood_learning_enabled=self._mood_learning_enabled,
                preference_model=preference_model,
            )

        raise NotImplementedError()

    def _sync_csp_generator_train_eval(self) -> None:
        if self._train_or_eval == "train":
            self._csp_generator.train()
        else:
            assert self._train_or_eval == "eval"
            self._csp_generator.eval()
