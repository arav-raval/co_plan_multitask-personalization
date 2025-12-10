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
    "num_episodes": 5,
    "num_epochs": 5, 
    "profile": "ChefAExpanded",
    "recipe_name": "UltraComplexFeast",
    "env_seed": 123,
    "csp_seed": 369,
    "logging_level": logging.INFO,
    "train_frac": 0.7,  # 70% train, 30% test for better split
    "verbose": True,
    "use_hbm": True,
}
# --------------------------------------------------

# ---------------- HELPER FUNCTIONS ----------------
def _make_env(seed: int, name: str = PARAMETERS["recipe_name"]) -> SpiceEnv:
    recipe = get_recipe(name)
    scene_spec = SpiceSceneSpec(recipe=recipe)

    return SpiceEnv(scene_spec, hidden_spec=None, seed=seed, eval_mode=False, verbose=PARAMETERS["verbose"])

def run_one_episode(env, generator, solver_seed=123, track_mood_evolution=True):
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

        # Track mood evolution (only during episode, not after it ends)
        if track_mood_evolution and not done and info.get("last_spice") is not None:
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
    
    # Separate average satisfaction into its own figure
    if "average_satisfactions" in standard_metrics:
        _plot_average_satisfaction(standard_metrics["average_satisfactions"], 
                                  metrics.get("moods", None))
        # Remove satisfaction from other metrics
        other_metrics = {k: v for k, v in standard_metrics.items() 
                        if k != "average_satisfactions"}
    else:
        other_metrics = standard_metrics
    
    # Plot other metrics if any
    if len(other_metrics) > 0:
        num_other_plots = len(other_metrics)
        fig, axes = plt.subplots(num_other_plots, 1, figsize=(10, 4 * num_other_plots))
        fig.patch.set_facecolor('white')
        if num_other_plots == 1:
            axes = [axes]
        
        _plot_standard_metrics(axes, num_episodes, other_metrics)
        plt.suptitle("Episode Metrics", fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
    elif len(standard_metrics) == 0:
        logging.warning("No standard metrics to visualize")


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


def _plot_average_satisfaction(satisfactions, moods=None):
    """Plot average satisfaction as a separate figure."""
    if not satisfactions or len(satisfactions) == 0:
        return
    
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
    
    # Create separate figure for satisfaction
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Use actual length of metric data
    actual_episodes = len(satisfactions)
    x_vals = np.arange(1, actual_episodes + 1)
    
    # Professional styling for satisfaction plot
    ax.plot(x_vals, satisfactions, marker='o', linewidth=2, markersize=6, 
           markerfacecolor='#3498db', markeredgecolor='white', markeredgewidth=1.5,
           color='#3498db', alpha=0.85)
    
    # Annotate mood on satisfaction graph with better styling
    if moods is not None and len(moods) == len(satisfactions):
        for ep, (x, y, mood) in enumerate(zip(x_vals, satisfactions, moods)):
            symbol = mood_symbols.get(mood, "?")
            color = mood_colors.get(mood, "black")
            ax.text(
                x,
                y + 0.03,
                symbol,
                color=color,
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                         edgecolor=color, linewidth=1.5, alpha=0.8)
            )

        # Add legend for mood symbols - positioned at bottom right
        legend_handles = [
            plt.Line2D([0], [0], marker='s', color='w', markersize=8,
                       label='All-Self (A)', markerfacecolor='green', 
                       markeredgecolor='white', markeredgewidth=1),
            plt.Line2D([0], [0], marker='s', color='w', markersize=8,
                       label='Neutral (N)', markerfacecolor='gray',
                       markeredgecolor='white', markeredgewidth=1),
            plt.Line2D([0], [0], marker='s', color='w', markersize=8,
                       label='None-Self (X)', markerfacecolor='red',
                       markeredgecolor='white', markeredgewidth=1),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9,
                 frameon=True, fancybox=True, shadow=True, framealpha=0.9)
    
    # Clean title and labels
    ax.set_title("Average Satisfaction per Episode", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Episode", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Satisfaction", fontsize=12, fontweight='bold')
    
    # Professional grid and styling
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax.set_ylim(-0.05, 1.05)  # Ensure full range is visible
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    plt.tight_layout()
    plt.show()


def _plot_standard_metrics(axes, num_episodes, metrics):
    """Helper function to plot standard metrics (excluding average satisfaction)."""
    moods = None  # Moods are handled separately in satisfaction plot

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

        # Normal plot (satisfaction is handled separately)
        # Use actual length of metric data, not num_episodes
        actual_episodes = len(metric)
        x_vals = np.arange(1, actual_episodes + 1)
        
        # Standard styling for other metrics
        ax.plot(x_vals, metric, marker='o', linewidth=2, markersize=5, alpha=0.7)
        ax.set_title(f"{metric_name.replace('_', ' ').title()} per Episode", 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Episode", fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
    

def _plot_hbm_evolution(hbm_metrics, recipe_name, spices, num_episodes, training_stats=None):
    """Helper function to plot HBM evolution with improved styling.
    
    Args:
        training_stats: Optional dict with keys 'num_episodes', 'num_epochs', 'training_set_size'
                        Used to add subtitle to global preferences plot.
    """
    phi_history = hbm_metrics["phi_history"]
    theta_history = hbm_metrics["theta_history"]
    mu_history = hbm_metrics["mu_history"]
    
    # Use actual length of history data
    phi_data_points = len(phi_history) if phi_history else 0
    theta_data_points = len(theta_history) if theta_history else 0
    mu_data_points = len(mu_history) if mu_history else 0
    
    if phi_data_points == 0 and theta_data_points == 0 and mu_data_points == 0:
        logging.warning("No HBM history data to plot")
        return
    
    # Determine the maximum data points (for cases where theta/mu are tracked globally)
    max_data_points = max(phi_data_points, theta_data_points, mu_data_points)
    
    # Calculate actual episodes (excluding initial state if present)
    actual_episodes = num_episodes  # Use the provided num_episodes for title
    
    # Create episode ranges for each history type
    phi_episodes = range(phi_data_points) if phi_data_points > 0 else []
    theta_episodes = range(theta_data_points) if theta_data_points > 0 else []
    mu_episodes = range(mu_data_points) if mu_data_points > 0 else []
    
    # Select a subset of spices to plot (to avoid clutter)
    num_spices_to_plot = min(8, len(spices)) if spices else 8
    spices_to_plot = spices[:num_spices_to_plot] if spices else []
    
    # Enhanced color palette with better contrast
    colors = plt.cm.Set2(np.linspace(0, 1, len(spices_to_plot)))
    
    # Clean up recipe name (remove parenthetical text)
    recipe_display = recipe_name.replace("_", " ").title() if recipe_name else "Recipe"
    if "(" in recipe_display:
        recipe_display = recipe_display.split("(")[0].strip()
    
    # Create three separate figures for easy downloading
    
    # --- Figure 1: Global Preferences (μ) ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor('white')
    for i, spice in enumerate(spices_to_plot):
        if mu_history and spice in mu_history[0]:
            mu_vals = [mu.get(spice, 0.0) for mu in mu_history]
            spice_display = spice.replace("_", " ").title()
            ax1.plot(mu_episodes, mu_vals, "-", label=spice_display, 
                    color=colors[i], linewidth=2, markersize=4, alpha=0.85, marker='o', markevery=max(1, len(mu_episodes)//20))
    ax1.set_title("Global Preferences (μ)", fontsize=14, fontweight='bold', pad=15)
    
    # Add subtitle with training statistics if provided
    if training_stats:
        subtitle = (f"Episodes: {training_stats.get('num_episodes', 'N/A')} per recipe | "
                   f"Epochs: {training_stats.get('num_epochs', 'N/A')} | "
                   f"Training Set: {training_stats.get('training_set_size', 'N/A')} recipes")
        ax1.text(0.5, -0.15, subtitle, transform=ax1.transAxes, 
                fontsize=9, ha='center', style='italic', color='dimgray',
                bbox=dict(boxstyle="round,pad=0.6", fc="#f0f0f0", ec="#666666", lw=1.5, alpha=0.9))
    
    ax1.set_xlabel("Episode", fontsize=12, fontweight='bold')
    ax1.set_ylabel("μ (Global Preference)", fontsize=12, fontweight='bold')
    ax1.legend(ncol=1, fontsize=7, frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, loc='lower right', handlelength=1.5, handletextpad=0.5)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1.2, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    if training_stats:
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    else:
        plt.tight_layout()
    plt.show()
    
    # --- Figure 2: Human-Level Preferences (θ) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('white')
    for i, spice in enumerate(spices_to_plot):
        if theta_history and spice in theta_history[0]:
            theta_vals = [th.get(spice, 0.0) for th in theta_history]
            spice_display = spice.replace("_", " ").title()
            ax2.plot(theta_episodes, theta_vals, "-", label=spice_display, 
                    color=colors[i], linewidth=2, markersize=4, alpha=0.85, marker='s', markevery=max(1, len(theta_episodes)//20))
    ax2.set_title("Human-Level Preferences (θ)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Episode", fontsize=12, fontweight='bold')
    ax2.set_ylabel("θ (Human-Level Preference)", fontsize=12, fontweight='bold')
    ax2.legend(ncol=1, fontsize=7, frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, loc='lower right', handlelength=1.5, handletextpad=0.5)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.2, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    plt.tight_layout()
    plt.show()
    
    # --- Figure 3: Recipe-Specific Preferences (φ) ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.patch.set_facecolor('white')
    for i, spice in enumerate(spices_to_plot):
        if phi_history and spice in phi_history[0]:
            phi_vals = [ph.get(spice, 0.0) for ph in phi_history]
            spice_display = spice.replace("_", " ").title()
            ax3.plot(phi_episodes, phi_vals, "-", label=spice_display, 
                    color=colors[i], linewidth=2, markersize=4, alpha=0.85, marker='^', markevery=max(1, len(phi_episodes)//20))
    ax3.set_title(f"Recipe-Specific Preferences (φ) - {recipe_display}", fontsize=14, fontweight='bold', pad=15)
    ax3.set_xlabel("Episode", fontsize=12, fontweight='bold')
    ax3.set_ylabel("φ (Recipe-Specific Preference)", fontsize=12, fontweight='bold')
    ax3.legend(ncol=1, fontsize=7, frameon=True, fancybox=True, shadow=True, 
              framealpha=0.9, loc='lower right', handlelength=1.5, handletextpad=0.5)
    ax3.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1.2, alpha=0.3)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(1.2)
    ax3.spines['bottom'].set_linewidth(1.2)
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

def _plot_multirecipe_hbm_evolution_unused(hbm_metrics_by_recipe, train_recipes, num_episodes, num_epochs, all_spices):
    """
    Visualize HBM evolution across all training recipes showing generalization.
    Shows how theta/mu generalize across recipes while phi stays recipe-specific.
    """
    if not hbm_metrics_by_recipe:
        return
    
    # Find common spices across recipes
    common_spices = []
    for recipe_name in train_recipes:
        if recipe_name in hbm_metrics_by_recipe:
            recipe_spices = hbm_metrics_by_recipe[recipe_name]["spices"]
            if not common_spices:
                common_spices = list(recipe_spices)
            else:
                common_spices = [s for s in common_spices if s in recipe_spices]
    
    common_spices = sorted(common_spices)[:6]  # Limit to 6 for clarity
    
    # Aggregate data across all recipes
    # For theta and mu: use first recipe's history (they're shared)
    # For phi: collect from all recipes
    first_recipe = train_recipes[0]
    if first_recipe not in hbm_metrics_by_recipe:
        return
    
    theta_history = hbm_metrics_by_recipe[first_recipe]["theta_history"]
    mu_history = hbm_metrics_by_recipe[first_recipe]["mu_history"]
    total_episodes = len(theta_history)
    
    # Calculate recipe boundaries (episodes per recipe per epoch)
    episodes_per_recipe_per_epoch = num_episodes
    recipe_boundaries = []
    for epoch in range(num_epochs):
        for i, recipe_name in enumerate(train_recipes):
            episode_start = epoch * len(train_recipes) * num_episodes + i * num_episodes
            episode_end = episode_start + num_episodes
            recipe_boundaries.append((episode_start, episode_end, recipe_name, epoch))
    
    # Professional figure styling
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('white')
    
    # Color palette for spices
    spice_colors = plt.cm.tab10(np.linspace(0, 1, len(common_spices)))
    
    # Plot 1: Recipe-specific preferences (phi) - show divergence per recipe
    ax1 = plt.subplot(2, 2, 1)
    
    for recipe_idx, recipe_name in enumerate(train_recipes):
        if recipe_name not in hbm_metrics_by_recipe:
            continue
        metrics = hbm_metrics_by_recipe[recipe_name]
        phi_history = metrics["phi_history"]
        
        # Find where this recipe appears in the timeline
        recipe_episodes = []
        recipe_phi_vals = {spice: [] for spice in common_spices}
        
        for start, end, r_name, epoch in recipe_boundaries:
            if r_name == recipe_name:
                for ep_idx in range(start, min(end, total_episodes)):
                    if ep_idx < len(phi_history):
                        recipe_episodes.append(ep_idx)
                        for spice in common_spices:
                            if spice in phi_history[ep_idx]:
                                recipe_phi_vals[spice].append(phi_history[ep_idx][spice])
                            else:
                                recipe_phi_vals[spice].append(0.0)
        
        # Plot phi for each spice for this recipe
        for spice_idx, spice in enumerate(common_spices):
            if recipe_episodes and len(recipe_phi_vals[spice]) == len(recipe_episodes):
                spice_display = spice.replace("_", " ").title()
                label = f"{spice_display} ({recipe_name.replace('_', ' ').title()})" if recipe_idx == 0 else None
                ax1.plot(recipe_episodes, recipe_phi_vals[spice], "-o", 
                        label=label, color=spice_colors[spice_idx], 
                        linewidth=1.5, markersize=3, alpha=0.6,
                        linestyle='-' if recipe_idx == 0 else '--')
    
    # Add recipe transition markers
    for start, end, recipe_name, epoch in recipe_boundaries:
        if start < total_episodes:
            ax1.axvline(x=start, color='gray', linestyle=':', linewidth=1, alpha=0.3)
            if epoch == 0:  # Only label first epoch
                recipe_display = recipe_name.replace("_", " ").title()[:15]
                ax1.text(start + num_episodes/2, ax1.get_ylim()[1] * 0.9, recipe_display,
                        ha='center', fontsize=7, rotation=90, alpha=0.6)
    
    # Add epoch markers (align with episode numbers: 10, 20, 30, etc.)
    for epoch in range(1, num_epochs + 1):
        epoch_pos = epoch * len(train_recipes) * num_episodes
        if epoch_pos <= total_episodes:
            ax1.axvline(x=epoch_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax1.text(epoch_pos, ax1.get_ylim()[1] * 0.98, f'E{epoch}', 
                    ha='center', fontsize=9, fontweight='bold', alpha=0.8)
    
    ax1.set_title("Recipe-Specific Preferences (φ) - Diverges Per Recipe", 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel("Training Episode (Across All Recipes)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("φ (Recipe-Specific Preference)", fontsize=11, fontweight='bold')
    ax1.legend(ncol=2, fontsize=8, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    ax1.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Human-level preferences (theta) - shows generalization
    ax2 = plt.subplot(2, 2, 2)
    episodes = range(total_episodes)
    for spice_idx, spice in enumerate(common_spices):
        if spice in theta_history[0]:
            theta_vals = [th.get(spice, 0.0) for th in theta_history]
            spice_display = spice.replace("_", " ").title()
            ax2.plot(episodes, theta_vals, "-s", label=spice_display, 
                    color=spice_colors[spice_idx], linewidth=1.5, markersize=4,
                    markerfacecolor=spice_colors[spice_idx], markeredgecolor='white', 
                    markeredgewidth=1, alpha=0.7)
    
    # Add epoch markers (align with episode numbers)
    for epoch in range(1, num_epochs + 1):
        epoch_pos = epoch * len(train_recipes) * num_episodes
        if epoch_pos <= total_episodes:
            ax2.axvline(x=epoch_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax2.text(epoch_pos, ax2.get_ylim()[1] * 0.98, f'E{epoch}', 
                    ha='center', fontsize=9, fontweight='bold', alpha=0.8)
    
    ax2.set_title("Human-Level Preferences (θ) - Generalizes Across Recipes", 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel("Training Episode (Across All Recipes)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("θ (Human-Level Preference)", fontsize=11, fontweight='bold')
    ax2.legend(ncol=2, fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    ax2.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Global preferences (mu) - shows generalization
    ax3 = plt.subplot(2, 2, 3)
    for spice_idx, spice in enumerate(common_spices):
        if spice in mu_history[0]:
            mu_vals = [mu.get(spice, 0.0) for mu in mu_history]
            spice_display = spice.replace("_", " ").title()
            ax3.plot(episodes, mu_vals, "-^", label=spice_display, 
                    color=spice_colors[spice_idx], linewidth=1.5, markersize=4,
                    markerfacecolor=spice_colors[spice_idx], markeredgecolor='white', 
                    markeredgewidth=1, alpha=0.7)
    
    # Add epoch markers (align with episode numbers)
    for epoch in range(1, num_epochs + 1):
        epoch_pos = epoch * len(train_recipes) * num_episodes
        if epoch_pos <= total_episodes:
            ax3.axvline(x=epoch_pos, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
            ax3.text(epoch_pos, ax3.get_ylim()[1] * 0.98, f'E{epoch}', 
                    ha='center', fontsize=9, fontweight='bold', alpha=0.8)
    
    ax3.set_title("Global Preferences (μ) - Generalizes Across Recipes", 
                 fontsize=12, fontweight='bold')
    ax3.set_xlabel("Training Episode (Across All Recipes)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("μ (Global Preference)", fontsize=11, fontweight='bold')
    ax3.legend(ncol=2, fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    ax3.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.4)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Comparison showing generalization vs specialization
    ax4 = plt.subplot(2, 2, 4)
    # Show final values: phi varies by recipe, theta/mu are shared
    final_phi_by_recipe = {}
    for recipe_name in train_recipes[:3]:  # Show first 3 recipes
        if recipe_name in hbm_metrics_by_recipe:
            phi_history = hbm_metrics_by_recipe[recipe_name]["phi_history"]
            if phi_history:
                final_phi_by_recipe[recipe_name] = phi_history[-1]
    
    x_pos = np.arange(len(common_spices))
    width = 0.25
    for i, recipe_name in enumerate(list(final_phi_by_recipe.keys())[:3]):
        recipe_display = recipe_name.replace("_", " ").title()[:10]
        phi_vals = [final_phi_by_recipe[recipe_name].get(spice, 0.0) for spice in common_spices]
        ax4.bar(x_pos + i * width, phi_vals, width, label=f'φ ({recipe_display})', 
               alpha=0.7)
    
    # Add theta and mu for comparison
    if theta_history and mu_history:
        final_theta = [theta_history[-1].get(spice, 0.0) for spice in common_spices]
        final_mu = [mu_history[-1].get(spice, 0.0) for spice in common_spices]
        ax4.bar(x_pos + 3 * width, final_theta, width, label='θ (Shared)', 
               color='green', alpha=0.7)
        ax4.bar(x_pos + 4 * width, final_mu, width, label='μ (Shared)', 
               color='blue', alpha=0.7)
    
    ax4.set_title("Final Preferences: Recipe-Specific vs Shared", 
                 fontsize=12, fontweight='bold')
    ax4.set_xlabel("Spice", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Preference Value", fontsize=11, fontweight='bold')
    ax4.set_xticks(x_pos + 2 * width)
    ax4.set_xticklabels([s.replace("_", " ").title() for s in common_spices], 
                       rotation=45, ha='right')
    ax4.legend(fontsize=8, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    ax4.grid(True, alpha=0.25, axis='y', linestyle='-', linewidth=0.5, color='gray')
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.4)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.suptitle(f"HBM Multi-Recipe Training: Generalization vs Specialization\n" +
                f"{len(train_recipes)} Recipes × {num_episodes} Episodes × {num_epochs} Epochs = {total_episodes} Total Episodes", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_multirecipe_hbm(hbm_history, train_recipes, test_recipes, generator, all_spices, profile_name=None):
    """
    Visualize HBM preferences across multiple recipes showing transfer and shifts.
    
    Args:
        hbm_history: Dictionary with phi_by_recipe, theta_history, mu_history, recipe_episode_map
        train_recipes: List of training recipe names
        test_recipes: List of test recipe names
        generator: CSP generator with HBM
        all_spices: List of all spices across recipes
        profile_name: Optional profile name for title
    """
    if not PARAMETERS["use_hbm"] or generator._pref_gen._hbm is None:
        logging.warning("HBM not enabled or not available")
        return
    
    hbm = generator._pref_gen._hbm
    
    # Get final preferences for all recipes (train + test)
    all_recipes = train_recipes + test_recipes
    final_phi_by_recipe = {}
    for recipe_name in all_recipes:
        recipe_spices = [s for s in all_spices if s in get_recipe(recipe_name).spices]
        final_phi_by_recipe[recipe_name] = {
            spice: hbm.get_phi(recipe_name, spice) for spice in recipe_spices
        }
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Recipe-specific preferences (phi) comparison across recipes
    ax1 = plt.subplot(2, 3, 1)
    # Select a subset of spices that appear in multiple recipes
    spice_counts = Counter()
    for recipe_name in all_recipes:
        recipe_spices = list(get_recipe(recipe_name).spices)
        spice_counts.update(recipe_spices)
    
    # Get spices that appear in at least 2 recipes
    common_spices = [s for s, count in spice_counts.items() if count >= 2]
    if not common_spices:
        common_spices = list(spice_counts.keys())[:8]  # Fallback to first 8 spices
    
    common_spices = sorted(common_spices)[:8]  # Limit to 8 for clarity
    
    if not common_spices:
        logging.warning("No common spices found for visualization")
        return
    
    x_pos = np.arange(len(common_spices))
    width = 0.8 / max(len(all_recipes), 1)  # Avoid division by zero
    
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_recipes), 1)))
    for i, recipe_name in enumerate(all_recipes):
        recipe_display = recipe_name.replace("_", " ").title()
        phi_vals = [final_phi_by_recipe[recipe_name].get(spice, 0.0) for spice in common_spices]
        offset = (i - len(all_recipes)/2 + 0.5) * width
        ax1.bar(x_pos + offset, phi_vals, width, label=recipe_display, 
                color=colors[i], alpha=0.8)
    
    ax1.set_xlabel("Spice", fontsize=11, fontweight='bold')
    ax1.set_ylabel("φ (Recipe-Specific Preference)", fontsize=11, fontweight='bold')
    ax1.set_title("Recipe-Specific Preferences (φ) Across Recipes", fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.replace("_", " ").title() for s in common_spices], rotation=45, ha='right')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Human-level preferences (theta) evolution
    ax2 = plt.subplot(2, 3, 2)
    if hbm_history["theta_history"]:
        theta_history = hbm_history["theta_history"]
        episodes = range(len(theta_history))
        for spice in common_spices[:6]:  # Limit to 6 spices
            theta_vals = [th.get(spice, 0.0) for th in theta_history]
            ax2.plot(episodes, theta_vals, "-o", label=spice.replace("_", " ").title(), 
                    alpha=0.7, markersize=3, linewidth=1.5)
    ax2.set_xlabel("Training Episode", fontsize=11, fontweight='bold')
    ax2.set_ylabel("θ (Human-Level Preference)", fontsize=11, fontweight='bold')
    ax2.set_title("Human-Level Preferences (θ) Evolution", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 3: Global preferences (mu) evolution
    ax3 = plt.subplot(2, 3, 3)
    if hbm_history["mu_history"]:
        mu_history = hbm_history["mu_history"]
        episodes = range(len(mu_history))
        for spice in common_spices[:6]:  # Limit to 6 spices
            mu_vals = [mu.get(spice, 0.0) for mu in mu_history]
            ax3.plot(episodes, mu_vals, "-s", label=spice.replace("_", " ").title(), 
                    alpha=0.7, markersize=3, linewidth=1.5)
    ax3.set_xlabel("Training Episode", fontsize=11, fontweight='bold')
    ax3.set_ylabel("μ (Global Preference)", fontsize=11, fontweight='bold')
    ax3.set_title("Global Preferences (μ) Evolution", fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 4: Preference transfer heatmap (train vs test)
    ax4 = plt.subplot(2, 3, 4)
    if test_recipes:
        # Create heatmap showing phi values for test recipes
        test_recipe_names = [r.replace("_", " ").title() for r in test_recipes]
        heatmap_data = []
        for recipe_name in test_recipes:
            recipe_spices = [s for s in common_spices if s in get_recipe(recipe_name).spices]
            row = [final_phi_by_recipe[recipe_name].get(spice, 0.0) for spice in common_spices]
            heatmap_data.append(row)
        
        if heatmap_data:
            im = ax4.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
            ax4.set_yticks(range(len(test_recipes)))
            ax4.set_yticklabels(test_recipe_names, fontsize=9)
            ax4.set_xticks(range(len(common_spices)))
            ax4.set_xticklabels([s.replace("_", " ").title() for s in common_spices], 
                               rotation=45, ha='right', fontsize=8)
            ax4.set_title("Test Recipe Preferences (φ)", fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax4, label='φ value')
    
    # Plot 5: Recipe-specific phi evolution for selected recipes
    ax5 = plt.subplot(2, 3, 5)
    selected_recipes = train_recipes[:3]  # Show first 3 training recipes
    for recipe_name in selected_recipes:
        if recipe_name in hbm_history["phi_by_recipe"]:
            phi_history = hbm_history["phi_by_recipe"][recipe_name]
            if phi_history:
                episodes = range(len(phi_history))
                # Plot average phi magnitude for this recipe
                avg_phi = [np.mean([abs(ph.get(spice, 0.0)) for spice in common_spices if spice in ph]) 
                                  for ph in phi_history]
                recipe_display = recipe_name.replace("_", " ").title()
                ax5.plot(episodes, avg_phi, "-o", label=recipe_display, alpha=0.7, markersize=4)
    ax5.set_xlabel("Episode", fontsize=11, fontweight='bold')
    ax5.set_ylabel("Average |φ|", fontsize=11, fontweight='bold')
    ax5.set_title("Recipe-Specific Preference Learning", fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Hierarchy comparison (mu, theta, phi) for selected spices
    ax6 = plt.subplot(2, 3, 6)
    selected_spices = common_spices[:4]  # Top 4 common spices
    x_pos_hier = np.arange(len(selected_spices))
    width_hier = 0.25
    
    mu_vals = [hbm.mu_mean.get(spice, 0.0) for spice in selected_spices]
    theta_vals = [hbm.theta_mean.get(spice, 0.0) for spice in selected_spices]
    # Average phi across all recipes for each spice
    phi_avg_vals = [np.mean([final_phi_by_recipe[r].get(spice, 0.0) 
                            for r in all_recipes if spice in final_phi_by_recipe[r]]) 
                   for spice in selected_spices]
    
    ax6.bar(x_pos_hier - width_hier, mu_vals, width_hier, label='μ (Global)', 
           color='#3498db', alpha=0.8)
    ax6.bar(x_pos_hier, theta_vals, width_hier, label='θ (Human)', 
           color='#2ecc71', alpha=0.8)
    ax6.bar(x_pos_hier + width_hier, phi_avg_vals, width_hier, label='φ (Recipe)', 
           color='#e74c3c', alpha=0.8)
    
    ax6.set_xlabel("Spice", fontsize=11, fontweight='bold')
    ax6.set_ylabel("Preference Value", fontsize=11, fontweight='bold')
    ax6.set_title("Hierarchical Preference Comparison", fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos_hier)
    ax6.set_xticklabels([s.replace("_", " ").title() for s in selected_spices], 
                        rotation=45, ha='right')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    profile_display = profile_name.replace("_", " ").title() if profile_name else "Multi-Recipe"
    plt.suptitle(f"HBM Preference Learning: {profile_display}\n" +
                f"Train: {len(train_recipes)} recipes | Test: {len(test_recipes)} recipes", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    logging.info(f"\n{'='*60}")
    logging.info(f"HBM Multi-Recipe Summary")
    logging.info(f"{'='*60}")
    logging.info(f"\nFinal Human-Level Preferences (θ):")
    for spice in common_spices[:10]:
        logging.info(f"  {spice:20s}: {hbm.theta_mean.get(spice, 0.0):+7.3f}")
    
    logging.info(f"\nFinal Global Preferences (μ):")
    for spice in common_spices[:10]:
        logging.info(f"  {spice:20s}: {hbm.mu_mean.get(spice, 0.0):+7.3f}")
    
    logging.info(f"\nRecipe-Specific Preferences (φ) - Final Values:")
    for recipe_name in all_recipes[:5]:  # Show first 5 recipes
        recipe_display = recipe_name.replace("_", " ").title()
        logging.info(f"\n  {recipe_display}:")
        recipe_spices = [s for s in common_spices if s in get_recipe(recipe_name).spices]
        for spice in recipe_spices[:5]:
            phi_val = final_phi_by_recipe[recipe_name].get(spice, 0.0)
            logging.info(f"    {spice:20s}: {phi_val:+7.3f}")

def visualize_mood_posterior_evolution(mood_evolution, recipe_name=None):
    """Visualize step-by-step mood posterior evolution with professional styling."""
    if not mood_evolution:
        return
    
    # Filter out any invalid entries (safety check)
    mood_evolution = [d for d in mood_evolution if d.get("mood_posterior") is not None]
    if not mood_evolution:
        logging.warning("No valid mood evolution data to visualize")
        return
    
    steps = [d["step"] for d in mood_evolution]
    all_self_probs = [d["mood_posterior"].get("all_self", 0.0) for d in mood_evolution]
    neutral_probs = [d["mood_posterior"].get("neutral", 0.0) for d in mood_evolution]
    none_self_probs = [d["mood_posterior"].get("none_self", 0.0) for d in mood_evolution]
    
    # Get true mood (should be same for all steps in episode)
    true_mood = mood_evolution[0].get("true_mood", "unknown")
    
    # Professional figure styling
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    
    # Plot mood posterior probabilities with professional styling
    ax.plot(steps, all_self_probs, "-o", label="All-Self", color="#2ecc71", 
            linewidth=2.5, markersize=8, markerfacecolor="#2ecc71", markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(steps, neutral_probs, "-s", label="Neutral", color="#7f8c8d", 
            linewidth=2.5, markersize=8, markerfacecolor="#7f8c8d", markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(steps, none_self_probs, "-^", label="None-Self", color="#e74c3c", 
            linewidth=2.5, markersize=8, markerfacecolor="#e74c3c", markeredgecolor='white', markeredgewidth=1.5)
    
    # Set x-axis ticks and labels
    ax.set_xticks(steps)
    x_labels = []
    for data in mood_evolution:
        spice = data.get("spice", "?")
        preferred = data.get("preferred_actor", "?")
        satisfaction = data.get("satisfaction", 0)
        
        # Convert preference to ±1: +1 if human, -1 if robot
        pref_value = +1.0 if preferred == "human" else -1.0
        
        # Clean up spice name for display
        spice_display = spice.replace("_", " ").title()
        
        # Create cleaner label
        label = f"{spice_display}\n(Sat={satisfaction:+d})"
        x_labels.append(label)
    
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    
    # Add horizontal line at confidence threshold
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.4, 
               label='Confidence Threshold (0.5)', zorder=0)
    
    # Professional formatting
    ax.set_xlabel("Recipe Step", fontsize=13, fontweight='bold')
    ax.set_ylabel("Posterior Probability P(mood)", fontsize=13, fontweight='bold')
    
    # Clean title
    true_mood_display = true_mood.replace("_", " ").title() if isinstance(true_mood, str) else str(true_mood)
    recipe_display = recipe_name.replace("_", " ").title() if recipe_name else "Recipe"
    ax.set_title(f"Bayesian Mood Filtering: {recipe_display}\n(True Mood: {true_mood_display})", 
                 fontsize=15, fontweight='bold', pad=15)
    
    # Professional legend
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, 
              shadow=True, framealpha=0.95, edgecolor='gray')
    
    # Grid styling
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add subtle background color
    ax.axhspan(0, 0.5, alpha=0.05, color='red', zorder=0)
    ax.axhspan(0.5, 1.0, alpha=0.05, color='green', zorder=0)
    
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
    
    # Capture initial state before first episode (ensure preferences start at 0)
    if PARAMETERS["use_hbm"] and csp_generator._pref_gen._hbm is not None:
        hbm = csp_generator._pref_gen._hbm
        initial_phi = {spice: 0.0 for spice in spices}  # Phi starts at 0
        initial_theta = {spice: hbm.theta_mean[spice] for spice in spices}
        initial_mu = {spice: hbm.mu_mean[spice] for spice in spices}
        metrics["phi_history"].append(initial_phi)
        metrics["theta_history"].append(initial_theta)
        metrics["mu_history"].append(initial_mu)
    
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
    
    # Track HBM evolution per recipe (similar to single recipe format)
    hbm_metrics_by_recipe = {}
    for recipe_name in train_recipes:
        recipe_spices = list(get_recipe(recipe_name).spices)
        hbm_metrics_by_recipe[recipe_name] = {
            "phi_history": [],
            "theta_history": [],
            "mu_history": [],
            "spices": recipe_spices,
        }
    
    # Track GLOBAL theta and mu evolution across ALL recipes (shared across recipes)
    global_hbm_metrics = {
        "theta_history": [],
        "mu_history": [],
        "spices": all_spices,  # All spices seen across all recipes
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
                
                # Track HBM evolution if using HBM (same format as single recipe)
                if PARAMETERS["use_hbm"] and generator._pref_gen._hbm is not None:
                    hbm = generator._pref_gen._hbm
                    recipe_spices = list(get_recipe(recipe_name).spices)
                    
                    # Capture initial state before first episode of this recipe in first epoch
                    if epoch == 0 and ep == 0 and len(hbm_metrics_by_recipe[recipe_name]["phi_history"]) == 0:
                        # Initial state - phi should be 0, theta/mu may have values from previous recipes
                        initial_phi = {spice: 0.0 for spice in recipe_spices}
                        initial_theta = {spice: hbm.theta_mean[spice] for spice in recipe_spices}
                        initial_mu = {spice: hbm.mu_mean[spice] for spice in recipe_spices}
                        hbm_metrics_by_recipe[recipe_name]["phi_history"].append(initial_phi)
                        hbm_metrics_by_recipe[recipe_name]["theta_history"].append(initial_theta)
                        hbm_metrics_by_recipe[recipe_name]["mu_history"].append(initial_mu)
                    
                    # Recipe-specific phi snapshot (after episode)
                    phi_snapshot = {spice: hbm.get_phi(recipe_name, spice) for spice in recipe_spices}
                    hbm_metrics_by_recipe[recipe_name]["phi_history"].append(phi_snapshot)
                    
                    # Shared theta and mu snapshots (same for all recipes, evolve across recipes)
                    theta_snapshot = {spice: hbm.theta_mean[spice] for spice in recipe_spices}
                    mu_snapshot = {spice: hbm.mu_mean[spice] for spice in recipe_spices}
                    hbm_metrics_by_recipe[recipe_name]["theta_history"].append(theta_snapshot)
                    hbm_metrics_by_recipe[recipe_name]["mu_history"].append(mu_snapshot)
                    
                    # Track global theta and mu evolution (across all recipes)
                    global_theta_snapshot = {spice: hbm.theta_mean[spice] for spice in all_spices if spice in hbm.theta_mean}
                    global_mu_snapshot = {spice: hbm.mu_mean[spice] for spice in all_spices if spice in hbm.mu_mean}
                    global_hbm_metrics["theta_history"].append(global_theta_snapshot)
                    global_hbm_metrics["mu_history"].append(global_mu_snapshot)
                
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
    
    # Visualize HBM evolution for each training recipe (using existing visualization)
    if PARAMETERS["use_hbm"] and generator._pref_gen._hbm is not None:
        logging.info(f"\n{'='*60}")
        logging.info(f"[HBM Visualization]")
        logging.info(f"{'='*60}")
        
        # Show GLOBAL theta and mu evolution across ALL training episodes (from all recipes)
        if global_hbm_metrics["theta_history"]:
            logging.info(f"\nVisualizing GLOBAL HBM evolution (θ and μ across all training recipes)")
            total_training_episodes = len(global_hbm_metrics["theta_history"])
            
            # For global visualization, show phi from first recipe as example, 
            # but theta and mu show evolution across ALL recipes
            first_recipe = train_recipes[0]
            if first_recipe in hbm_metrics_by_recipe and hbm_metrics_by_recipe[first_recipe]["phi_history"]:
                # Create visualization showing:
                # - phi from first recipe (as example of recipe-specific evolution)
                # - theta and mu from global tracking (showing evolution across ALL recipes)
                global_hbm_for_viz = {
                    "phi_history": hbm_metrics_by_recipe[first_recipe]["phi_history"],
                    "theta_history": global_hbm_metrics["theta_history"],
                    "mu_history": global_hbm_metrics["mu_history"],
                }
                # Use common spices across recipes
                common_spices = sorted(list(set(hbm_metrics_by_recipe[first_recipe]["spices"]) & set(global_hbm_metrics["spices"])))[:8]
                # Pass training statistics for subtitle
                training_stats = {
                    'num_episodes': num_episodes,
                    'num_epochs': PARAMETERS["num_epochs"],
                    'training_set_size': len(train_recipes)
                }
                _plot_hbm_evolution(
                    global_hbm_for_viz,
                    recipe_name=first_recipe,
                    spices=common_spices,
                    num_episodes=total_training_episodes,
                    training_stats=training_stats
                )
        
        # Then visualize each training recipe individually (showing recipe-specific phi with final global theta/mu)
        for recipe_name in train_recipes:
            if recipe_name in hbm_metrics_by_recipe:
                metrics = hbm_metrics_by_recipe[recipe_name]
                if metrics["phi_history"]:
                    logging.info(f"\nVisualizing HBM evolution for training recipe: {recipe_name}")
                    # Get final global theta/mu (after all training)
                    final_global_theta = global_hbm_metrics["theta_history"][-1] if global_hbm_metrics["theta_history"] else {}
                    final_global_mu = global_hbm_metrics["mu_history"][-1] if global_hbm_metrics["mu_history"] else {}
                    
                    # Show recipe-specific phi evolution, but use final global theta/mu (constant line)
                    hbm_metrics_for_viz = {
                        "phi_history": metrics["phi_history"],
                        # Show final global values (constant) to indicate these are shared after all training
                        "theta_history": [final_global_theta] * len(metrics["phi_history"]),
                        "mu_history": [final_global_mu] * len(metrics["phi_history"]),
                    }
                    total_episodes_per_recipe = num_episodes * PARAMETERS["num_epochs"]
                    _plot_hbm_evolution(
                        hbm_metrics_for_viz,
                        recipe_name=recipe_name,
                        spices=metrics["spices"],
                        num_episodes=total_episodes_per_recipe
                    )
        
        # Also visualize for test recipes (final state only - no learning)
        if test_recipes:
            logging.info(f"\n[Test Recipe Final Preferences]")
            hbm = generator._pref_gen._hbm
            for recipe_name in test_recipes:
                recipe_spices = list(get_recipe(recipe_name).spices)
                # Create a single-episode snapshot for visualization (test set - no evolution)
                test_metrics = {
                    "phi_history": [{spice: hbm.get_phi(recipe_name, spice) for spice in recipe_spices}],
                    "theta_history": [{spice: hbm.theta_mean[spice] for spice in recipe_spices}],
                    "mu_history": [{spice: hbm.mu_mean[spice] for spice in recipe_spices}],
                }
                logging.info(f"\nFinal preferences for test recipe: {recipe_name}")
                # Test recipes show final state only (no evolution, learning disabled)
                _plot_hbm_evolution(
                    test_metrics,
                    recipe_name=recipe_name,
                    spices=recipe_spices,
                    num_episodes=1
                )

