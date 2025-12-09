"""Tests for spices_csp.py."""

import logging
import numpy as np
import pytest
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from multitask_personalization.csp_solvers import RandomWalkCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec, RecipeSpec
)
from recipe import get_recipe, get_profile
from collections import Counter

from multitask_personalization.envs.spices.spices_hbm import HierarchicalPreferenceModel

# ---------------- PARAMETER SETUP ----------------
PARAMETERS = {
    "num_episodes": 100,
    "num_epochs": 5, 
    "profile": "ChefA",
    "recipe_name": "UltraComplexFeast",
    "env_seed": 123,
    "csp_seed": 369,
    "logging_level": logging.INFO,
    "train_frac": 0.8,
    "verbose": True,
    "use_hbm": True,
}
# --------------------------------------------------

# ---------------- HELPER FUNCTIONS ----------------
def _make_env(seed: int, name: str = PARAMETERS["recipe_name"]) -> SpiceEnv:
    recipe = get_recipe(name)
    scene_spec = SpiceSceneSpec(recipe=recipe)

    return SpiceEnv(scene_spec, hidden_spec=None, seed=seed, eval_mode=False, verbose=PARAMETERS["verbose"])

def run_one_episode(env, generator, solver_seed=123, track_mood_evolution=False):
    """Run a full episode of a recipe and update the generator (training step).
    
    Args:
        track_mood_evolution: If True, returns step-by-step mood posterior evolution data
    """
    obs, _ = env.reset()

    assert isinstance(obs, SpiceState)
    assert obs.current_spice in obs.feasible_next

    terminated = False
    step_count = 0
    
    # Track mood evolution if requested
    mood_evolution = [] if track_mood_evolution else None

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

        # Track mood evolution
        if track_mood_evolution and info.get("last_spice") is not None:
            mood_posterior = info.get("mood_posterior", {})
            mood_evolution.append({
                "step": step_count,
                "spice": info.get("last_spice"),
                "actor": info.get("last_actor"),
                "preferred_actor": info.get("preferred_actor"),  # True preference
                "satisfaction": info.get("satisfaction", 0),
                "mood_posterior": mood_posterior.copy() if mood_posterior else {},
                "expected_mood": info.get("expected_mood", 0.0),
                "true_mood": info.get("mood"),  # True mood from environment
            })

        # Log mood estimates from info dict
        if PARAMETERS["verbose"] and info.get("last_spice") is not None:
            mood_posterior = info.get("mood_posterior", {})
            mood_str = ", ".join([
                f"{mood}={prob:.3f}" 
                for mood, prob in mood_posterior.items()
            ])
            
            logging.info(
                f"[Step {step_count}] Sat={info.get('satisfaction', 0):+d} | "
                f"Mood: ({mood_str}) | "
                f"Confident neutral: {info.get('is_confident_neutral', False)}\n"
            )
        
        step_count += 1
        terminated = done
    
    if track_mood_evolution:
        info["mood_evolution"] = mood_evolution
    return info

def update_metrics(metrics, info, hbm_info):
    metrics["average_satisfactions"].append(info["average_satisfaction"])
    metrics["satisfaction_variances"].append(info['satisfaction_variance'])

    filtered = Counter([actor for _, actor in info['action_history'] if actor is not None])
    distribution = {k: round(v / len(filtered), 3) for k, v in filtered.items()}
    metrics["actor_distributions"].append(distribution)
    
    # Track moods if available
    if "moods" not in metrics:
        metrics["moods"] = []
    if "mood" in info:
        metrics["moods"].append(info["mood"])

    # Track HBM evolution
    if PARAMETERS["use_hbm"]:
        generator, spices, recipe_name = hbm_info
        hbm = generator._pref_gen._hbm
        # Recipe-specific preferences (phi)
        phi_snapshot = {spice: hbm.get_phi(recipe_name, spice) for spice in spices}
        metrics["phi_history"].append(phi_snapshot)

        # Human-level preferences (theta)
        theta_snapshot = {spice: hbm.theta_mean[spice] for spice in spices}
        metrics["theta_history"].append(theta_snapshot)

        # Global preferences (mu)
        mu_snapshot = {spice: hbm.mu_mean[spice] for spice in spices}
        metrics["mu_history"].append(mu_snapshot)

        # Mood posterior
        metrics["mood_posterior_history"].append(hbm.mood_posterior.copy())

    return metrics
# --------------------------------------------------
# Visualization utilities
# --------------------------------------------------
def plot_phi_evolution(phi_history, recipe_name):
    plt.figure(figsize=(8,4))
    for spice in phi_history[0].keys():
        vals = [ph[spice] for ph in phi_history]
        plt.plot(vals, "-o", label=spice, alpha=0.7)

    plt.title(f"φ evolution for recipe={recipe_name}")
    plt.xlabel("Episode")
    plt.ylabel("φ_{r,s}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize(num_episodes, metrics):
    """
    Visualize standard metrics (satisfactions, actor distributions, etc.).
    
    Args:
        num_episodes: Number of episodes
        metrics: Dictionary of standard metrics (satisfactions, moods, etc.)
    """
    # Filter out HBM-related metrics and empty metrics for standard visualization
    standard_metrics = {
        k: v for k, v in metrics.items() 
        if k not in ["phi_history", "theta_history", "mu_history", "mood_posterior_history"]
        and v is not None and len(v) > 0  # Only include non-empty metrics
    }
    
    num_standard_plots = len(standard_metrics)
    if num_standard_plots == 0:
        logging.warning("No standard metrics to visualize")
        return
    
    fig, axes = plt.subplots(num_standard_plots, 1, figsize=(8, 3 * num_standard_plots))
    if num_standard_plots == 1:
        axes = [axes]
    
    _plot_standard_metrics(axes, num_episodes, standard_metrics)
    plt.suptitle("Episode Metrics", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_hbm_evolution(metrics, recipe_name=None, spices=None):
    """
    Visualize HBM evolution (phi, theta, mu).
    
    Args:
        metrics: Dictionary containing HBM data (phi_history, theta_history, etc.)
        recipe_name: Optional recipe name for HBM plots
        spices: Optional list of spices for HBM plots
    """
    # Extract HBM metrics
    hbm_metrics = {
        "phi_history": metrics.get("phi_history", []),
        "theta_history": metrics.get("theta_history", []),
        "mu_history": metrics.get("mu_history", []),
    }
    
    # Check if we have HBM data
    if not hbm_metrics["phi_history"]:
        logging.warning("No HBM data to visualize")
        return
    
    num_episodes = len(hbm_metrics["phi_history"])
    _plot_hbm_evolution(hbm_metrics, recipe_name, spices, num_episodes)


def visualize_all(num_episodes, metrics, hbm_metrics=None, recipe_name=None, spices=None):
    """
    Consolidated visualization function for both standard metrics and HBM evolution.
    
    Args:
        num_episodes: Number of episodes
        metrics: Dictionary of standard metrics (satisfactions, moods, etc.)
        hbm_metrics: Optional dictionary with HBM data (phi_history, theta_history, etc.)
        recipe_name: Optional recipe name for HBM plots
        spices: Optional list of spices for HBM plots
    """
    # Determine if we have HBM data
    has_hbm = (hbm_metrics is not None and 
               "phi_history" in hbm_metrics and 
               len(hbm_metrics["phi_history"]) > 0)
    
    # Calculate number of plots needed
    num_standard_plots = len(metrics)
    num_hbm_plots = 4 if has_hbm else 0
    
    # Create figure with appropriate layout
    if has_hbm:
        # Two separate figures: one for standard metrics, one for HBM
        # Figure 1: Standard metrics
        if num_standard_plots > 0:
            fig1, axes1 = plt.subplots(num_standard_plots, 1, figsize=(8, 3 * num_standard_plots))
            if num_standard_plots == 1:
                axes1 = [axes1]
            
            _plot_standard_metrics(axes1, num_episodes, metrics)
            plt.suptitle("Episode Metrics", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        # Figure 2: HBM evolution
        _plot_hbm_evolution(hbm_metrics, recipe_name, spices, num_episodes)
        
    else:
        # Only standard metrics
        fig, axes = plt.subplots(num_standard_plots, 1, figsize=(8, 3 * num_standard_plots))
        if num_standard_plots == 1:
            axes = [axes]

        _plot_standard_metrics(axes, num_episodes, metrics)
        plt.suptitle("Episode Metrics", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


def _plot_standard_metrics(axes, num_episodes, metrics):
    """Helper function to plot standard metrics."""
    moods = metrics.get("moods", None)

    mood_symbols = {
        "all_self": "A",
        "neutral": "N",
        "none_self": "X",
    }

    mood_colors = {
        "all_self": "green",
        "neutral": "gray",
        "none_self": "red",
    }

    for i, (metric_name, metric) in enumerate(metrics.items()):
        ax = axes[i]

        # Skip empty metrics (shouldn't happen due to filtering, but safety check)
        if not metric or len(metric) == 0:
            ax.text(0.5, 0.5, f"No data for {metric_name}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{metric_name} per Episode (No Data)")
            continue

        # Special case: actor distributions
        if metric_name == "actor_distributions":
            humans = [d.get("human", 0) for d in metric]
            robots = [d.get("robot", 0) for d in metric]
            ax.bar(range(len(humans)), humans, label="Human", alpha=0.7)
            ax.bar(range(len(robots)), robots, bottom=humans, label="Robot", alpha=0.7)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Actor Fraction")
            ax.set_title("Actor Choice Distribution Over Time")
            ax.legend()
            continue

        # Normal plot (including satisfaction)
        # Use actual length of metric data, not num_episodes
        actual_episodes = len(metric)
        x_vals = np.arange(1, actual_episodes + 1)
        ax.plot(x_vals, metric, marker='o')

        # Annotate mood on satisfaction graph
        if metric_name == "average_satisfactions" and moods is not None and len(moods) == len(metric):
            for ep, (x, y, mood) in enumerate(zip(x_vals, metric, moods)):
                symbol = mood_symbols.get(mood, "?")
                color = mood_colors.get(mood, "black")
                ax.text(
                    x,
                    y + 0.05,
                    symbol,
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                )

            # Add legend for mood symbols
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           label='All-Self (A)', markerfacecolor='green', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w',
                           label='Neutral (N)', markerfacecolor='gray', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w',
                           label='None-Self (X)', markerfacecolor='red', markersize=10),
            ]
            ax.legend(handles=legend_handles, loc="upper left")

        ax.set_title(f"{metric_name} per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel(metric_name)
        ax.grid(True)
    

def _plot_hbm_evolution(hbm_metrics, recipe_name, spices, num_episodes):
    """Helper function to plot HBM evolution."""
    phi_history = hbm_metrics["phi_history"]
    theta_history = hbm_metrics["theta_history"]
    mu_history = hbm_metrics["mu_history"]
    
    # Use actual length of history data
    actual_episodes = len(phi_history) if phi_history else 0
    if actual_episodes == 0:
        logging.warning("No HBM history data to plot")
        return
    
    episodes = range(actual_episodes)
    
    # Select a subset of spices to plot (to avoid clutter)
    num_spices_to_plot = min(8, len(spices)) if spices else 8
    spices_to_plot = spices[:num_spices_to_plot] if spices else []
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Recipe-specific preferences (phi)
    ax1 = plt.subplot(2, 2, 1)
    for spice in spices_to_plot:
        if spice in phi_history[0] if phi_history else False:
            phi_vals = [ph.get(spice, 0.0) for ph in phi_history]
            ax1.plot(episodes, phi_vals, "-o", label=spice, alpha=0.7, markersize=3)
    ax1.set_title(f"Recipe-Specific Preferences (φ) - {recipe_name or 'Recipe'}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("φ_{r,s}")
    ax1.legend(ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Human-level preferences (theta)
    ax2 = plt.subplot(2, 2, 2)
    for spice in spices_to_plot:
        if spice in theta_history[0] if theta_history else False:
            theta_vals = [th.get(spice, 0.0) for th in theta_history]
            ax2.plot(episodes, theta_vals, "-s", label=spice, alpha=0.7, markersize=3)
    ax2.set_title("Human-Level Preferences (θ)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("θ_s")
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Global preferences (mu)
    ax3 = plt.subplot(2, 2, 3)
    for spice in spices_to_plot:
        if spice in mu_history[0] if mu_history else False:
            mu_vals = [mu.get(spice, 0.0) for mu in mu_history]
            ax3.plot(episodes, mu_vals, "-^", label=spice, alpha=0.7, markersize=3)
    ax3.set_title("Global Preferences (μ)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("μ_s")
    ax3.legend(ncol=2, fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: Preference convergence (show variance reduction)
    ax4 = plt.subplot(2, 2, 4)
    if phi_history and spices and spices[0] in phi_history[0]:
        # Compute variance of preferences over a rolling window
        window_size = min(10, actual_episodes // 5)
        if window_size > 1:
            variances = []
            for i in range(actual_episodes):
                start = max(0, i - window_size // 2)
                end = min(actual_episodes, i + window_size // 2)
                window_phis = [ph.get(spices[0], 0.0) for ph in phi_history[start:end]]
                variances.append(np.var(window_phis))
            ax4.plot(episodes, variances, "-", label="Preference Variance", color="purple")
            ax4.set_title("Preference Convergence (Variance)")
            ax4.set_xlabel("Episode")
            ax4.set_ylabel("Var(φ)")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Not enough episodes\nfor variance analysis", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Preference Convergence")
    else:
        ax4.text(0.5, 0.5, "No HBM data available", 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Preference Convergence")
    
    plt.suptitle(f"HBM Evolution Over {actual_episodes} Episodes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    if phi_history and theta_history and mu_history and spices:
        logging.info(f"\n{'='*60}")
        logging.info(f"HBM Summary (Final Episode)")
        logging.info(f"{'='*60}")
        
        final_phi = phi_history[-1]
        final_theta = theta_history[-1]
        final_mu = mu_history[-1]
        
        logging.info(f"\nRecipe-Specific (φ) - {recipe_name or 'Recipe'}:")
        for spice in spices_to_plot:
            logging.info(f"  {spice:15s}: {final_phi[spice]:+7.3f}")
        
        logging.info(f"\nHuman-Level (θ):")
        for spice in spices_to_plot:
            logging.info(f"  {spice:15s}: {final_theta[spice]:+7.3f}")
        
        logging.info(f"\nGlobal (μ):")
        for spice in spices_to_plot:
            logging.info(f"  {spice:15s}: {final_mu[spice]:+7.3f}")
        
        # Show convergence (variance reduction)
        if len(phi_history) > 10 and spices and spices[0] in phi_history[0]:
            early_phi = np.mean([abs(ph.get(spices[0], 0.0)) for ph in phi_history[:5]])
            late_phi = np.mean([abs(ph.get(spices[0], 0.0)) for ph in phi_history[-5:]])
            logging.info(f"\nConvergence indicator (|φ| for {spices[0]}):")
            logging.info(f"  Early episodes (avg): {early_phi:.3f}")
            logging.info(f"  Late episodes (avg): {late_phi:.3f}")
            logging.info(f"  Change: {late_phi - early_phi:+.3f}")

def visualize_mood_posterior_evolution(mood_evolution, recipe_name=None):
    """Visualize step-by-step mood posterior evolution with simplified annotations."""
    if not mood_evolution:
        return
    
    steps = [d["step"] for d in mood_evolution]
    all_self_probs = [d["mood_posterior"].get("all_self", 0.0) for d in mood_evolution]
    neutral_probs = [d["mood_posterior"].get("neutral", 0.0) for d in mood_evolution]
    none_self_probs = [d["mood_posterior"].get("none_self", 0.0) for d in mood_evolution]
    
    # Get true mood (should be same for all steps in episode)
    true_mood = mood_evolution[0].get("true_mood", "unknown")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot mood posterior probabilities
    ax.plot(steps, all_self_probs, "-o", label="all_self", color="green", linewidth=2, markersize=6)
    ax.plot(steps, neutral_probs, "-s", label="neutral", color="gray", linewidth=2, markersize=6)
    ax.plot(steps, none_self_probs, "-^", label="none_self", color="red", linewidth=2, markersize=6)
    
    # Set x-axis ticks and labels directly on the axis line
    ax.set_xticks(steps)
    x_labels = []
    for data in mood_evolution:
        spice = data.get("spice", "?")
        preferred = data.get("preferred_actor", "?")
        satisfaction = data.get("satisfaction", 0)
        
        # Convert preference to ±1: +1 if human, -1 if robot
        pref_value = +1.0 if preferred == "human" else -1.0
        
        # Create label: spice name with satisfaction and preference
        label = f"{spice}\ny={satisfaction:+d}\npref={pref_value:+.0f}"
        x_labels.append(label)
    
    ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=7)
    
    # Add horizontal line at confidence threshold
    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Confidence Threshold')
    
    # Formatting
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("P(mood)", fontsize=12)
    ax.set_title(f"Mood Posterior Evolution - {recipe_name or 'Recipe'}\n(True Mood: {true_mood})", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)  # Standard y-limit
    
    # Add text box with legend in bottom right corner
    legend_text = (
        "Labels:\n"
        "• Spice name\n"
        "• y = satisfaction value\n"
        "• pref = true preference\n"
        "  (+1 = human, -1 = robot)"
    )
    ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray', linewidth=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    logging.info(f"\n{'='*60}")
    logging.info(f"Mood Posterior Evolution Summary - {recipe_name or 'Recipe'}")
    logging.info(f"{'='*60}")
    logging.info(f"True Mood: {true_mood}")
    logging.info(f"Total Steps: {len(mood_evolution)}")
    
    final_posterior = mood_evolution[-1]["mood_posterior"]
    logging.info(f"\nFinal Posterior:")
    for mood, prob in final_posterior.items():
        logging.info(f"  {mood:12s}: {prob:.3f}")
    
    # Show when confidence was reached
    for i, data in enumerate(mood_evolution):
        posterior = data["mood_posterior"]
        max_mood = max(posterior.items(), key=lambda x: x[1])
        if max_mood[1] >= 0.5:
            logging.info(f"\nConfidence reached at step {i}: {max_mood[0]} (P={max_mood[1]:.3f})")
            break


# ---------------- TEST FUNCTIONS ----------------
@pytest.mark.single_recipe
def test_spices_csp_single_recipe(num_episodes: int = PARAMETERS["num_episodes"], recipe_name: str = PARAMETERS["recipe_name"]):
    env_seed = PARAMETERS["env_seed"]
    csp_seed = PARAMETERS["csp_seed"]
    assert env_seed != csp_seed

    metrics = {
        "average_satisfactions": [],
        "percent_satisfied": [],
        "satisfaction_variances": [],
        "actor_distributions": [],
        "moods": [],
        "phi_history": [],
        "theta_history": [],
        "mu_history": [],
        "mood_posterior_history": [],
    }

    # Make environment and CSP generator
    env = _make_env(seed=env_seed, name=recipe_name)
    
    csp_generator = SpicesAssignCSPGenerator(
        spice_list=list(env.scene_spec.recipe.spices),
        recipe_list=[recipe_name], 
        seed=csp_seed,
        use_hbm=PARAMETERS["use_hbm"],
    )
    
    spices = list(env.scene_spec.recipe.spices)
    
    # Track mood evolution from last episode for visualization
    last_episode_mood_evolution = None
    
    # Run episodes
    for i in range(num_episodes):
        if PARAMETERS["verbose"]:
            logging.info(f"\n{'=' * 30} Episode {i+1}/{num_episodes} {'=' * 30}")

        # Run a single episode (track mood evolution for last episode only)
        track_evolution = (i == num_episodes - 1)  # Only track last episode
        info = run_one_episode(env, csp_generator, track_mood_evolution=track_evolution)
        
        if track_evolution and "mood_evolution" in info:
            last_episode_mood_evolution = info["mood_evolution"]
        
        # Update metrics
        hbm_info = (csp_generator, spices, recipe_name)
        metrics = update_metrics(metrics, info, hbm_info)

        if PARAMETERS["verbose"]:
            logging.info(
                f"[Episode {i+1} Summary] "
                f"Average satisfaction: {info['average_satisfaction']:.3f} | "
                f"True mood: {info['mood']} | "
                f"Neutral confidence: {info.get('neutral_confidence', 0):.3f}"
            )
            # Log HBM preference for first spice as example
            if PARAMETERS["use_hbm"] and spices:
                first_spice = spices[0]
                phi = csp_generator._pref_gen._hbm.get_phi(recipe_name, first_spice)
                theta = csp_generator._pref_gen._hbm.theta_mean[first_spice]
                mu = csp_generator._pref_gen._hbm.mu_mean[first_spice]
                logging.info(
                    f"  HBM [{first_spice}]: φ={phi:.3f}, θ={theta:.3f}, μ={mu:.3f}"
                )

    visualize(num_episodes, metrics)
    
    #Visualize HBM evolution if available
    if metrics["phi_history"]:
        visualize_hbm_evolution(metrics, recipe_name, spices)
    
    # Visualize mood posterior evolution from last episode
    if last_episode_mood_evolution:
        visualize_mood_posterior_evolution(last_episode_mood_evolution, recipe_name)
    
    env.close()

@pytest.mark.multiple_recipes
def test_spices_csp_multiple_recipes(num_episodes: int = PARAMETERS["num_episodes"], profile: str = PARAMETERS["profile"]):
    env_seed = PARAMETERS["env_seed"]
    csp_seed = PARAMETERS["csp_seed"]
    train_frac = PARAMETERS["train_frac"]
    assert env_seed != csp_seed

    # Profile
    profile_spec = get_profile(profile)
    all_recipes = list(profile_spec.recipes)
    
    # Split into train/test sets
    num_train = int(np.ceil(len(all_recipes) * train_frac))
    train_recipes = all_recipes[:num_train]
    test_recipes  = all_recipes[num_train:]

    logging.info(f"\n{'='*60}")
    logging.info(f"[Profile: {profile}]")
    logging.info(f"Train recipes: {train_recipes}")
    logging.info(f"Test recipes:  {test_recipes}")

    # Spice vocabulary - collect from all recipes
    train_spices = sorted({s for r in train_recipes for s in get_recipe(r).spices})
    test_spices = sorted({s for r in test_recipes for s in get_recipe(r).spices})
    all_spices = sorted(set(train_spices) | set(test_spices))
    
    # # Validate: all test spices must be in training spices
    # missing_spices = set(test_spices) - set(train_spices)
    # if missing_spices:
    #     raise ValueError(
    #         f"Test recipes contain spices not seen in training: {missing_spices}. "
    #         f"Training spices: {train_spices}, Test spices: {test_spices}"
    #    )
    
    logging.info(f"\n[Spice Vocabulary]")
    logging.info(f"Training spices ({len(train_spices)}): {train_spices}")
    logging.info(f"Test spices ({len(test_spices)}): {test_spices}")
    logging.info(f"All spices ({len(all_spices)}): {all_spices}")
    
    # Create generator with all spices (needed for test recipes)
    generator = SpicesAssignCSPGenerator(
        spice_list=all_spices,
        recipe_list=all_recipes,
        seed=csp_seed,
        use_hbm=PARAMETERS["use_hbm"],
        verbose=PARAMETERS["verbose"]
    )

    # ----------------- TRAINING -----------------
    logging.info(f"\n{'='*60}")
    logging.info(f"[TRAINING PHASE] - {PARAMETERS['num_epochs']} epochs")
    logging.info(f"{'='*60}")
    
    train_results = {
        "satisfactions": {},
        "moods": {},
        "expected_moods": {},
        "neutral_confidences": {},
        "neutral_episodes": {},  # Track which episodes were used for learning
    }
    
    total_neutral_episodes = 0
    total_episodes = 0
    
    # Train for multiple epochs
    for epoch in range(PARAMETERS["num_epochs"]):
        logging.info(f"\n{'='*60}")
        logging.info(f"[Training Epoch {epoch+1}/{PARAMETERS['num_epochs']}]")
        logging.info(f"{'='*60}")
        
        for recipe_name in train_recipes:
            env = _make_env(seed=env_seed, name=recipe_name)
            logging.info(f"\n[Train Epoch {epoch+1}] {recipe_name}")
            
            # Initialize recipe tracking on first epoch
            if epoch == 0:
                recipe_sats = []
                recipe_moods = []
                recipe_expected_moods = []
                recipe_neutral_confs = []
                recipe_neutral_episodes = []
            else: 
                # Append to existing lists for subsequent epochs
                recipe_sats = train_results["satisfactions"][recipe_name]
                recipe_moods = train_results["moods"][recipe_name]
                recipe_expected_moods = train_results["expected_moods"][recipe_name]
                recipe_neutral_confs = train_results["neutral_confidences"][recipe_name]
                recipe_neutral_episodes = train_results["neutral_episodes"][recipe_name]
            
            for ep in range(num_episodes):
                total_episodes += 1
                neutral_examples_before_episode = len(generator._pref_gen._neutral_training_inputs)
                
                info = run_one_episode(env, generator)
                
                recipe_sats.append(info.get("average_satisfaction", np.nan))
                recipe_moods.append(info.get("mood"))
                recipe_expected_moods.append(info.get("expected_mood", 0.0))
                recipe_neutral_confs.append(info.get("neutral_confidence", 0.0))
                
                # Check if this episode was used for learning (neutral-confident)
                neutral_examples_after_episode = len(generator._pref_gen._neutral_training_inputs)
                was_used_for_learning = neutral_examples_after_episode > neutral_examples_before_episode
                
                recipe_neutral_episodes.append(was_used_for_learning)
                if was_used_for_learning:
                    total_neutral_episodes += 1
                
                if PARAMETERS["verbose"]:
                    logging.info(
                        f"\t Epoch {epoch+1} Ep {ep+1}: sat={recipe_sats[-1]:.3f} | "
                        f"mood={recipe_moods[-1]} | "
                        f"E[mood]={recipe_expected_moods[-1]:+.3f} | "
                        f"P(neutral)={recipe_neutral_confs[-1]:.3f} | "
                        f"used_for_learning={recipe_neutral_episodes[-1]}"
                    )
            
            # Update results (create on first epoch, extend on later epochs)
            if epoch == 0:
                train_results["satisfactions"][recipe_name] = recipe_sats
                train_results["moods"][recipe_name] = recipe_moods
                train_results["expected_moods"][recipe_name] = recipe_expected_moods
                train_results["neutral_confidences"][recipe_name] = recipe_neutral_confs
                train_results["neutral_episodes"][recipe_name] = recipe_neutral_episodes
            else:
                # Already updated lists in place above
                pass
            
            env.close()

    # Training summary
    logging.info(f"\n[Training Summary]")
    logging.info(f"Total epochs: {PARAMETERS['num_epochs']}")
    logging.info(f"Total episodes: {total_episodes} ({len(train_recipes)} recipes × {num_episodes} episodes × {PARAMETERS['num_epochs']} epochs)")
    logging.info(f"Neutral-confident episodes used for learning: {total_neutral_episodes} ({100*total_neutral_episodes/total_episodes:.1f}%)")
    logging.info(f"Neutral examples collected: {len(generator._pref_gen._neutral_training_inputs)}")
    
    # Classifier status
    mood_metrics = generator.get_metrics()
    if mood_metrics.get("classifier_trained", 0) > 0:
        logging.info(f"Classifier trained: Yes")
        logging.info(f"Classifier training accuracy: {mood_metrics.get('classifier_train_accuracy', 0):.3f}")
    else:
        logging.info(f"Classifier trained: No (insufficient neutral episodes)")
    
    # Compute mean satisfaction across all epochs
    mean_train_sat = np.mean([np.mean(v) for v in train_results["satisfactions"].values()])
    logging.info(f"Mean train satisfaction (all epochs): {mean_train_sat:.3f}")
    
    # Show satisfaction by epoch (if multiple epochs)
    if PARAMETERS["num_epochs"] > 1:
        episodes_per_epoch = len(train_recipes) * num_episodes
        for epoch in range(PARAMETERS["num_epochs"]):
            epoch_start = epoch * episodes_per_epoch
            epoch_end = (epoch + 1) * episodes_per_epoch
            epoch_sats = []
            for recipe_sats in train_results["satisfactions"].values():
                epoch_sats.extend(recipe_sats[epoch_start:epoch_end])
            epoch_mean = np.mean(epoch_sats) if epoch_sats else np.nan
            logging.info(f"  Epoch {epoch+1} mean satisfaction: {epoch_mean:.3f}")

    # ----------------- TESTING -----------------
    logging.info(f"\n{'='*60}")
    logging.info(f"[TESTING PHASE] (Learning Disabled)")
    logging.info(f"{'='*60}")
    
    # Disable learning during test
    generator._disable_learning = True
    
    test_results = {
        "satisfactions": {},
        "moods": {},
        "expected_moods": {},
        "neutral_confidences": {},
        "mood_accuracy": {},  # Track if inferred mood matches true mood
    }
    
    for recipe_name in test_recipes:
        env = _make_env(seed=env_seed, name=recipe_name)
        logging.info(f"\n[Test] {recipe_name}")
        
        recipe_sats = []
        recipe_moods = []
        recipe_expected_moods = []
        recipe_neutral_confs = []
        recipe_mood_accuracy = []
        
        for ep in range(num_episodes):
            info = run_one_episode(env, generator)
            
            recipe_sats.append(info.get("average_satisfaction", np.nan))
            true_mood = info.get("mood")
            expected_mood = info.get("expected_mood", 0.0)
            neutral_conf = info.get("neutral_confidence", 0.0)
            
            recipe_moods.append(true_mood)
            recipe_expected_moods.append(expected_mood)
            recipe_neutral_confs.append(neutral_conf)
            
            # Determine inferred mood from expected_mood value
            if expected_mood > 0.25:
                inferred_mood = "all_self"
            elif expected_mood < -0.25:
                inferred_mood = "none_self"
            else:
                inferred_mood = "neutral"
            
            mood_correct = (inferred_mood == true_mood)
            recipe_mood_accuracy.append(mood_correct)
            
            if PARAMETERS["verbose"]:
                logging.info(
                    f"\t Ep {ep+1}: sat={recipe_sats[-1]:.3f} | "
                    f"true_mood={true_mood} | "
                    f"inferred_mood={inferred_mood} (E[m]={expected_mood:+.3f}) | "
                    f"P(neutral)={neutral_conf:.3f} | "
                    f"correct={mood_correct}"
                )
        
        test_results["satisfactions"][recipe_name] = recipe_sats
        test_results["moods"][recipe_name] = recipe_moods
        test_results["expected_moods"][recipe_name] = recipe_expected_moods
        test_results["neutral_confidences"][recipe_name] = recipe_neutral_confs
        test_results["mood_accuracy"][recipe_name] = recipe_mood_accuracy
        env.close()
    
    # Test summary
    logging.info(f"\n{'='*60}")
    logging.info(f"[Test Summary]")
    logging.info(f"{'='*60}")
    
    mean_test_sat = np.mean([np.mean(v) for v in test_results["satisfactions"].values()])
    mean_mood_accuracy = np.mean([np.mean(v) for v in test_results["mood_accuracy"].values()])
    
    logging.info(f"Mean test satisfaction: {mean_test_sat:.3f}")
    logging.info(f"Mean mood inference accuracy: {mean_mood_accuracy:.3f} ({100*mean_mood_accuracy:.1f}%)")
    
    # Per-recipe breakdown
    logging.info(f"\n[Per-Recipe Test Results]")
    for recipe_name in test_recipes:
        recipe_sat = np.mean(test_results["satisfactions"][recipe_name])
        recipe_acc = np.mean(test_results["mood_accuracy"][recipe_name])
        logging.info(
            f"  {recipe_name}: "
            f"sat={recipe_sat:.3f}, "
            f"mood_acc={recipe_acc:.3f}"
        )
    
    logging.info(f"\n{'='*60}")
    logging.info(f"[Cross-Recipe Transfer Summary]")
    logging.info(f"Train satisfaction: {mean_train_sat:.3f}")
    logging.info(f"Test satisfaction:  {mean_test_sat:.3f}")
    logging.info(f"Transfer ratio:     {mean_test_sat/mean_train_sat:.3f}" if mean_train_sat > 0 else "N/A")
    logging.info(f"Mood inference accuracy: {mean_mood_accuracy:.3f}")
    logging.info(f"{'='*60}")

