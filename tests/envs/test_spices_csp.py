"""Tests for spices_csp.py."""

import logging
import numpy as np
import pytest
import matplotlib.pyplot as plt

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec, RecipeSpec
)
from recipe import get_recipe, get_profile
from collections import Counter

# ---------------- PARAMETER SETUP ----------------
PARAMETERS = {
    "num_episodes": 2,
    "profile": "ChefA",
    "recipe_name": "GrandmasSoup",
    "env_seed": 123,
    "csp_seed": 369,
    "logging_level": logging.INFO,
    "train_frac": 0.8,
    "verbose": True,
}
# --------------------------------------------------

# ---------------- HELPER FUNCTIONS ----------------
def _make_env(seed: int, name: str = PARAMETERS["recipe_name"]) -> SpiceEnv:
    recipe = get_recipe(name)
    scene_spec = SpiceSceneSpec(recipe=recipe)

    return SpiceEnv(scene_spec, hidden_spec=None, seed=seed, eval_mode=False, verbose=PARAMETERS["verbose"])

def run_one_episode(env, generator, solver_seed=123):
    """Run a full episode of a recipe and update the generator (training step)."""
    obs, _ = env.reset()

    assert isinstance(obs, SpiceState)
    assert obs.current_spice in obs.feasible_next

    terminated = False

    while not terminated:
        prev_obs = obs

        # Generate CSP and solve for the current spice
        csp, samplers, policy, initialization = generator.generate(obs)

        # Solve CSP
        solver = RandomWalkCSPSolver(solver_seed, show_progress_bar=False)
        sol = solver.solve(
            csp,
            initialization,
            samplers,
        )
        assert sol is not None
        policy.reset(sol)

        # Step the policy
        action = policy.step(obs)
        obs, reward, done, truncated, info = env.step(action)

        # Observe the transition
        generator.observe_transition(prev_obs, action, obs, done, info)
        terminated = done
    return info

def get_profile_vector(pref_gen):
    """Return flattened preference probabilities per spice/actor."""
    v = []
    for spice in pref_gen._spice_to_index.keys():
        for actor in ["human", "robot"]:
            if pref_gen._classifier is None:
                v.append(0.5)
            else:
                x = pref_gen._featurize(spice, actor)
                p = pref_gen._classifier.predict_proba([x])[0][1]
                v.append(float(np.clip(p, 1e-6, 1 - 1e-6)))
    return np.array(v, dtype=float)

def profile_similarity(v1, v2):
    """Cosine similarity + L1 distance between two profile vectors."""
    return {
        "cosine": 1 - cosine(v1, v2),
        "l1": float(np.mean(np.abs(v1 - v2)))
    }

def visualize(num_episodes, metrics):
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 3 * num_metrics))

    if num_metrics == 1:
        axes = [axes]

    print(f"Metrics: {metrics}")
    
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
        # elif metric_name == "pref_snapshots":
        #     for spice in metric[0].keys():
        #         human_probs = [p[spice]["human"] for p in metric]
        #         ax.plot(range(1, num_episodes + 1), human_probs, label=f"{spice} (human)")
        #         #ax.fill_between(range(1, num_episodes + 1), human_probs, [1 - p for p in human_probs], alpha=0.1)
        #         ax.set_xlabel("Episode")
        #         ax.set_ylabel("P(prefer = human)")
        #         ax.set_title(f"Preference Probability for {spice}")
        #         ax.legend()
        #         ax.grid(True)
        else: 
            ax.plot(range(1, num_episodes + 1), metric, marker='o')
            ax.set_title(f"{metric_name} per Episode")
            ax.set_xlabel("Episode")
            ax.set_ylabel(metric_name)
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# ---------------- TEST FUNCTIONS ----------------
@pytest.mark.single_recipe
def test_spices_csp_single_recipe(num_episodes: int = PARAMETERS["num_episodes"], recipe_name: str = PARAMETERS["recipe_name"]):
    env_seed = PARAMETERS["env_seed"]
    csp_seed = PARAMETERS["csp_seed"]
    assert env_seed != csp_seed

    metrics = {
        "average_satisfactions": [],
        "satisfaction_variances": [],
        "actor_distributions": [],
        #"pref_snapshots": [],
    }

    # Make environment and CSP generator
    env = _make_env(seed=env_seed, name=recipe_name)
    
    csp_generator = SpicesAssignCSPGenerator(
        spice_list=list(env.scene_spec.recipe.spices),
        seed=csp_seed,
    )
    
    # Run episodes
    for i in range(num_episodes):
        if PARAMETERS["verbose"]:
            logging.info(f"\nEpisode {i+1}/{num_episodes}")

        # Run a single episode
        info = run_one_episode(env, csp_generator)
        
        # Update metrics
        metrics["average_satisfactions"].append(info["average_satisfaction"])
        metrics["satisfaction_variances"].append(info['satisfaction_variance'])

        filtered = Counter([actor for _, actor in info['action_history'] if actor is not None])
        distribution = {k: round(v / len(filtered), 3) for k, v in filtered.items()}
        metrics["actor_distributions"].append(distribution)

        #metrics["pref_snapshots"].append(csp_generator.get_pref_snapshot()) TODO: Fix this implementation

    visualize(num_episodes, metrics)
    env.close()

@pytest.mark.multiple_recipes
def test_spices_csp_multiple_recipes(num_episodes: int = PARAMETERS["num_episodes"], profile: str = PARAMETERS["profile"]):
    env_seed = PARAMETERS["env_seed"]
    csp_seed = PARAMETERS["csp_seed"]
    train_frac = PARAMETERS["train_frac"]
    assert env_seed != csp_seed

    # Profile
    profile_spec = get_profile(profile)
    all_recipes = profile_spec.recipes
    
    # Split into train/test sets
    num_train = int(np.ceil(len(all_recipes) * train_frac))
    train_recipes = all_recipes[:num_train]
    test_recipes  = all_recipes[num_train:]

    logging.info(f"\n[Profile: {profile}]")
    logging.info(f"Train recipes: {train_recipes}")
    logging.info(f"Test recipes:  {test_recipes}")

    # Spice vocabulary
    all_spices = sorted({s for r in all_recipes for s in get_recipe(r).spices})
    generator = SpicesAssignCSPGenerator(spice_list=all_spices, seed=csp_seed)

    # ----------------- TRAINING -----------------
    train_results = {}
    for recipe_name in train_recipes:
        env = _make_env(seed=env_seed, name=recipe_name)
        logging.info(f"\n[Train] {recipe_name}")
        episode_sats = []
        for ep in range(num_episodes):
            info = run_one_episode(env, generator)
            episode_sats.append(info.get("average_satisfaction", np.nan))
            logging.info(f"\t Ep {ep+1}: mean sat={episode_sats[-1]:.3f}")
        train_results[recipe_name] = episode_sats
        env.close()

    # train_profile_vec = get_profile_vector(generator._pref_gen)

    # ----------------- TESTING -----------------
    test_results = {}
    #test_profiles = {}
    
    # disable learning during test
    generator._disable_learning = True
    for recipe_name in test_recipes:
        env = _make_env(seed=env_seed, name=recipe_name)
        episode_sats = []
        for ep in range(num_episodes):
            info = run_one_episode(env, generator)
            episode_sats.append(info.get("average_satisfaction", np.nan))
            logging.info(f"[Test] {recipe_name} | Ep {ep+1}: mean sat={episode_sats[-1]:.3f}")

        test_results[recipe_name] = episode_sats
        #test_profiles[recipe_name] = get_profile_vector(generator._pref_gen)
        env.close()
    
    # METRICS
    mean_train_sat = np.mean([np.mean(v) for v in train_results.values()])
    mean_test_sat  = np.mean([np.mean(v) for v in test_results.values()])
    logging.info(f"\nMean train satisfaction = {mean_train_sat:.3f}")
    logging.info(f"Mean test satisfaction  = {mean_test_sat:.3f}")

    # # CROSS-RECIPE SIMILARITY
    # profile_vectors = {**{r: train_profile_vec for r in train_recipes}, **test_profiles}
    # names = list(profile_vectors.keys())
    # n = len(names)
    # sim_matrix = np.zeros((n, n))
    # for i, a in enumerate(names):
    #     for j, b in enumerate(names):
    #         sim_matrix[i, j] = profile_similarity(profile_vectors[a], profile_vectors[b])["cosine"]

    # plt.figure(figsize=(6, 5))
    # sns.heatmap(sim_matrix, xticklabels=names, yticklabels=names, annot=True, cmap="viridis")
    # plt.title(f"Profile Similarity ({profile}) – Train vs Test")
    # plt.tight_layout()
    # plt.show()

    # 3. Satisfaction curves
    # plt.figure(figsize=(7, 4))
    # for r, vals in train_results.items():
    #     plt.plot(range(1, len(vals)+1), vals, "-o", label=f"{r} (train)")
    # for r, vals in test_results.items():
    #     plt.plot(range(1, len(vals)+1), vals, "--", label=f"{r} (test)")
    # plt.xlabel("Episode")
    # plt.ylabel("Mean Satisfaction")
    # plt.title(f"{profile}: Train/Test Satisfaction over Episodes")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    
    # for recipe in recipes:
    #     test_spices_csp_single_recipe(num_episodes, recipe.name)


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