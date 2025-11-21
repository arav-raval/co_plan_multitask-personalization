"""Debug script to trace spices CSP assignment."""

import logging
import numpy as np
import sys
sys.path.insert(0, 'src')

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec, RecipeSpec
)
from tests.envs.recipe import get_recipe

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_one_episode(env, generator, solver_seed=123):
    """Run a full episode of a recipe and update the generator (training step)."""
    obs, _ = env.reset()

    assert isinstance(obs, SpiceState)
    assert obs.current_spice in obs.feasible_next

    terminated = False
    step_count = 0

    while not terminated:
        step_count += 1
        prev_obs = obs
        print(f"\n{'='*60}")
        print(f"STEP {step_count} - Current spice: {obs.current_spice}")
        print(f"{'='*60}")

        # Generate CSP and solve for the current spice
        csp, samplers, policy, initialization = generator.generate(obs)
        print(f"[Test] Generated CSP with initialization: {initialization}")

        # Solve CSP
        solver = RandomWalkCSPSolver(solver_seed, show_progress_bar=False)
        sol = solver.solve(
            csp,
            initialization,
            samplers,
        )
        assert sol is not None
        print(f"[Test] Solver returned solution: {sol}")
        policy.reset(sol)

        # Step the policy
        action = policy.step(obs)
        print(f"[Test] Policy returned action: {action}")
        obs, reward, done, truncated, info = env.step(action)

        # Observe the transition
        generator.observe_transition(prev_obs, action, obs, done, info)
        terminated = done
    
    return info

if __name__ == "__main__":
    env_seed = 123
    csp_seed = 456
    
    # Make environment
    recipe = get_recipe("GrandmasSoup")
    scene_spec = SpiceSceneSpec(recipe=recipe)
    env = SpiceEnv(scene_spec, hidden_spec=None, seed=env_seed, eval_mode=False, verbose=True)
    
    # Make CSP generator
    csp_generator = SpicesAssignCSPGenerator(
        spice_list=list(env.scene_spec.recipe.spices),
        seed=csp_seed,
    )
    
    # Disable learning
    csp_generator._disable_learning = True
    
    # Run episode
    print("Starting episode with learning disabled...")
    info = run_one_episode(env, csp_generator, solver_seed=123)
    
    print(f"\n{'='*60}")
    print("EPISODE COMPLETE")
    print(f"{'='*60}")
    print(f"Action history: {info['action_history']}")
    print(f"Average satisfaction: {info['average_satisfaction']:.2f}")

