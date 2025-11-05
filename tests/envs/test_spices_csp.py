"""Tests for spices_csp.py."""

import numpy as np
import matplotlib.pyplot as plt

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec, RecipeSpec
)
from recipe import get_recipe
from collections import Counter

def _make_env(seed: int, name: str = "BreakingBread") -> SpiceEnv:
    recipe = get_recipe(name)
    scene_spec = SpiceSceneSpec(recipe=recipe)
    return SpiceEnv(scene_spec, hidden_spec=None, seed=seed, eval_mode=False, verbose=True)

def test_spices_csp_single_recipe(num_episodes: int = 5, recipe_name: str = "UltraComplexFeast"):
    env_seed = 123
    csp_seed = 456
    assert env_seed != csp_seed

    metrics = {
        "average_satisfactions": [],
        "satisfaction_variances": [],
        "actor_distributions": [],
    }

    # Make environment and CSP generator
    env = _make_env(seed=env_seed, name=recipe_name)
    
    csp_generator = SpicesAssignCSPGenerator(
        spice_list=list(env.scene_spec.recipe.spices),
        seed=csp_seed,
    )

    # Run episodes
    for i in range(num_episodes):
        print(f"\nEpisode {i+1}/{num_episodes}")
        obs, _  = env.reset()
        assert isinstance(obs, SpiceState)
        assert obs.current_spice in obs.feasible_next

        terminated = False
        while not terminated: 
            #import pdb; pdb.set_trace()
            prev_obs = obs

            # Generate CSP and solve for the current spice
            csp, samplers, policy, initialization = csp_generator.generate(obs)

            # import pdb; pdb.set_trace()
            solver = RandomWalkCSPSolver(csp_seed, show_progress_bar=False)
            sol = solver.solve(
                csp,
                initialization,
                samplers,
            )
            assert sol is not None
            policy.reset(sol)

            # Step the policy
            action = policy.step(obs)
            obs, reward, env_terminated, truncated, info = env.step(action)

            # Observe the transition
            csp_generator.observe_transition(prev_obs, action, obs, env_terminated, info)
            terminated = env_terminated
        
        metrics["average_satisfactions"].append(info["average_satisfaction"])
        metrics["satisfaction_variances"].append(info['satisfaction_variance'])

        filtered = [actor for _, actor in info['action_history'] if actor is not None]
        distribution = Counter(filtered)
        distribution = {k: round(v / len(filtered), 3) for k, v in distribution.items()}
        metrics["actor_distributions"].append(distribution)

    visualize(num_episodes, metrics)
    env.close()

def visualize(num_episodes, metrics):
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 3 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    for i, (metric_name, metric) in enumerate(metrics.items()):
        ax = axes[i]

        # TODO: Make this dynamic (temp just hardcoded for distribution metrics)
        if metric_name == "actor_distributions":
            humans = [d.get("human", 0) for d in metric]
            robots = [d.get("robot", 0) for d in metric]
            ax.bar(range(len(humans)), humans, label="Human", alpha=0.7)
            ax.bar(range(len(robots)), robots, bottom=humans, label="Robot", alpha=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Actor Fraction")
            ax.set_title("Actor Choice Distribution Over Time")
            ax.legend()
        else: 
            ax.plot(range(1, num_episodes + 1), metric, marker='o')
            ax.set_title(f"{metric_name} per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric_name)
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# def test_spices_csp():
#     """Tests for spices_csp.py."""
#     env_seed = 123
#     csp_seed = 456
#     env = _make_env(seed=env_seed, name="GrandmasSoup")

#     obs, _ = env.reset()
#     assert isinstance(obs, SpiceState)
#     assert obs.current_spice in obs.feasible_next

#     # Create CSP
#     csp_generator = SpicesAssignCSPGenerator(
#         spice_list=list(env.scene_spec.recipe.spices),
#         seed=csp_seed,
#     )
    
#     csp, samplers, policy, initialization = csp_generator.generate(obs)

#     # Solve CSP
#     solver = RandomWalkCSPSolver(csp_seed, num_improvements=1, show_progress_bar=False)
#     sol = solver.solve(
#         csp,
#         initialization,
#         samplers,
#     )

#     assert sol is not None
#     policy.reset(sol)

#     terminated = False
#     while not terminated:
#         prev_obs = obs
#         action = policy.step(obs)
#         obs, reward, env_terminated, truncated, info = env.step(action)

#         csp_generator.observe_transition(prev_obs, action, obs, env_terminated, info)
#         terminated = env_terminated

#     env.close()