"""Tests for spices_csp.py."""

import numpy as np

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec, RecipeSpec
)
from recipe import get_recipe

def _make_env(seed: int, name: str = "BreakingBread") -> SpiceEnv:
    recipe = get_recipe(name)
    scene_spec = SpiceSceneSpec(recipe=recipe)
    return SpiceEnv(scene_spec, hidden_spec=None, seed=seed, eval_mode=False, verbose=True)


def test_spices_csp():
    """Tests for spices_csp.py."""
    seed = 123
    env = _make_env(seed=seed, name="GrandmasSoup")

    obs, _ = env.reset()
    assert isinstance(obs, SpiceState)
    assert obs.current_spice in obs.feasible_next

    # Create CSP
    csp_generator = SpicesAssignCSPGenerator(
        spice_list=list(env.scene_spec.recipe.spices),
        seed=seed,
    )
    csp, samplers, policy, initialization = csp_generator.generate(obs)

    # Solve CSP
    solver = RandomWalkCSPSolver(seed, num_improvements=1, show_progress_bar=False)
    sol = solver.solve(
        csp,
        initialization,
        samplers,
    )

    assert sol is not None
    policy.reset(sol)

    terminated = False
    while not terminated:
        prev_obs = obs
        action = policy.step(obs)
        obs, reward, env_terminated, truncated, info = env.step(action)

        csp_generator.observe_transition(prev_obs, action, obs, env_terminated, info)
        terminated = env_terminated

    env.close()