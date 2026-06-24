from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Config
# ---------------------------------------------------------------------------
RUNS_DIR = Path("logs/runs")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent styling
COLORS = {
    "DBTL (ours)": "#2563EB",       # blue
    "DBTL w/o ψ": "#7C3AED",        # purple
    "Flat HBM": "#DC2626",           # red
    "CBTL (pooled)": "#059669",      # green
    "Exploit-only": "#D97706",       # amber
    "No learning": "#9CA3AF",        # gray
    "Scalar ψ": "#EA580C",           # orange
}

MARKERS = {
    "DBTL (ours)": "o",
    "DBTL w/o ψ": "s",
    "Flat HBM": "^",
    "CBTL (pooled)": "D",
    "Exploit-only": "v",
    "No learning": "x",
    "Scalar ψ": "p",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


# Helpers
# ---------------------------------------------------------------------------
def load_eval_csvs(run_dirs: List[Path]) -> pd.DataFrame:
    """Load and concat eval_results.csv from multiple seed directories."""
    frames = []
    for d in sorted(run_dirs):
        csv = d / "eval_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed_dir"] = str(d)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_approach(prefix: str) -> pd.DataFrame:
    """Load all seeds for a run prefix like 'spices_c14_ours'."""
    base = RUNS_DIR / prefix
    if not base.exists():
        return pd.DataFrame()
    seed_dirs = sorted([base / d for d in os.listdir(base)
                       if (base / d / "eval_results.csv").exists()])
    return load_eval_csvs(seed_dirs)


def agg_metric(df: pd.DataFrame, metric: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean, se) aggregated across seeds."""
    if df.empty or metric not in df.columns:
        return np.array([]), np.array([]), np.array([])
    grouped = df.groupby("training_step")[metric]
    mean = grouped.mean()
    se = grouped.std() / np.sqrt(grouped.count())
    steps = mean.index.values
    return steps, mean.values, se.values

# Figure 1: SpiceEnv Multi-Recipe Joint Training
# ---------------------------------------------------------------------------
def plot_spice_multi_recipe():
    """Learning curves for 4-recipe joint training."""
    approaches = {
        "DBTL (ours)": "spices_c14_ours",
        "DBTL w/o ψ": "spices_c14_nomood",
        "Flat HBM": "spices_c14_flat",
        "CBTL (pooled)": "spices_c14_cbtl",
    }

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for metric, ax, title in [
        ("neutral_eval_mean_user_satisfaction_per_step", axes[0], "Neutral Evaluation (ψ = 0)"),
        ("natural_eval_mean_user_satisfaction_per_step", axes[1], "Natural Evaluation (ψ sampled)"),
    ]:
        for label, prefix in approaches.items():
            df = load_approach(prefix)
            steps, mean, se = agg_metric(df, metric)
            if len(steps) == 0:
                continue
            ax.plot(steps, mean, label=label, color=COLORS[label],
                    marker=MARKERS[label], markevery=4, markersize=5, linewidth=1.8)
            ax.fill_between(steps, mean - se, mean + se,
                           alpha=0.15, color=COLORS[label])

        ax.set_ylabel("Mean Per-Step Satisfaction")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9)

    axes[1].set_xlabel("Training Steps")

    fig.suptitle("SpiceEnv: Multi-Recipe Joint Training (4 recipes, 10 seeds)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spice_multi_recipe.pdf")
    fig.savefig(OUT_DIR / "spice_multi_recipe.png")
    plt.close(fig)
    print(f"  Saved spice_multi_recipe")


# Figure 2: SpiceEnv LOO Transfer — ME Tight Cluster
# ---------------------------------------------------------------------------
def plot_spice_me_loo():
    """Learning curves for Middle-Eastern tight-cluster LOO."""
    me_dir = RUNS_DIR / "spices_me_loo"
    if not me_dir.exists():
        print("  SKIP spice_me_loo (directory not found)")
        return

    frames = []
    for seed_dir in sorted(me_dir.iterdir()):
        csv = seed_dir / "eval_results.csv"
        cfg = seed_dir / "config.yaml"
        if not csv.exists() or not cfg.exists():
            continue
        df = pd.read_csv(csv)

        cfg_text = cfg.read_text()
        approach = "unknown"
        for line in cfg_text.splitlines():
            if "approach_name:" in line:
                approach = line.split(":")[-1].strip()
                break
        df["approach"] = approach
        df["seed_dir"] = str(seed_dir)
        frames.append(df)

    if not frames:
        print("  SKIP spice_me_loo (no data)")
        return

    all_df = pd.concat(frames, ignore_index=True)

    approach_map = {
        "ours": "DBTL (ours)",
        "cbtl_adapted": "CBTL (pooled)",
        "flat_model": "Flat HBM",
        "without_mood_learning": "DBTL w/o ψ",
    }

    metric = "neutral_eval_mean_user_satisfaction_per_step"

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    for code, label in approach_map.items():
        sub = all_df[all_df["approach"] == code]
        if sub.empty or metric not in sub.columns:
            continue
        grouped = sub.groupby("training_step")[metric]
        mean = grouped.mean()
        se = grouped.std() / np.sqrt(grouped.count())
        steps = mean.index.values
        ax.plot(steps, mean.values, label=label, color=COLORS.get(label, "gray"),
                marker=MARKERS.get(label, "."), markevery=4, markersize=5, linewidth=1.8)
        ax.fill_between(steps, (mean - se).values, (mean + se).values,
                       alpha=0.15, color=COLORS.get(label, "gray"))

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Per-Step Satisfaction")
    ax.set_title("SpiceEnv: Middle-Eastern LOO Transfer (4 splits × 10 seeds)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "spice_me_loo.pdf")
    fig.savefig(OUT_DIR / "spice_me_loo.png")
    plt.close(fig)
    print(f"  Saved spice_me_loo")


# Figure 3: SpiceEnv Diverse-Pool LOO
# ---------------------------------------------------------------------------
def plot_spice_diverse_loo():
    """Per-split learning curves for diverse-pool LOO."""
    splits = {
        "leave-asian": ("spices_loo_asian_ours", "spices_loo_asian_cbtl", "spices_loo_asian_flat"),
        "leave-indian": ("spices_loo_indian_ours", "spices_loo_indian_cbtl", "spices_loo_indian_flat"),
        "leave-ultra": ("spices_loo_ultra_ours", "spices_loo_ultra_cbtl", "spices_loo_ultra_flat"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    metric = "neutral_eval_mean_user_satisfaction_per_step"

    for idx, (split_name, (ours_prefix, cbtl_prefix, flat_prefix)) in enumerate(splits.items()):
        ax = axes[idx]
        for label, prefix in [("DBTL (ours)", ours_prefix), ("CBTL (pooled)", cbtl_prefix), ("Flat HBM", flat_prefix)]:
            df = load_approach(prefix)
            steps, mean, se = agg_metric(df, metric)
            if len(steps) == 0:
                continue
            ax.plot(steps, mean, label=label, color=COLORS[label],
                    marker=MARKERS[label], markevery=4, markersize=5, linewidth=1.8)
            ax.fill_between(steps, mean - se, mean + se, alpha=0.15, color=COLORS[label])

        ax.set_xlabel("Training Steps")
        ax.set_title(split_name)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Per-Step Satisfaction")
    axes[0].legend(loc="lower right", framealpha=0.9)

    fig.suptitle("SpiceEnv: Diverse-Pool LOO Transfer (10 seeds per split)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spice_diverse_loo.pdf")
    fig.savefig(OUT_DIR / "spice_diverse_loo.png")
    plt.close(fig)
    print(f"  Saved spice_diverse_loo")

# Figure 4: Overcooked Single-Context
# ---------------------------------------------------------------------------
def plot_overcooked_single_context():
    """Learning curves for single-context Overcooked (CrampedRoom, RealisticCook)."""
    approaches = {
        "DBTL (ours)": "oc_main_ours",
        "DBTL w/o ψ": "oc_main_nomood",
        "Flat HBM": "oc_main_flat",
        "CBTL (pooled)": "oc_main_cbtl",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for metric, ax, title in [
        ("neutral_eval_mean_user_satisfaction_per_step", axes[0], "Neutral (ψ = 0)"),
        ("natural_eval_mean_user_satisfaction_per_step", axes[1], "Natural (ψ sampled)"),
    ]:
        for label, prefix in approaches.items():
            df = load_approach(prefix)
            steps, mean, se = agg_metric(df, metric)
            if len(steps) == 0:
                continue
            ax.plot(steps, mean, label=label, color=COLORS[label],
                    marker=MARKERS[label], markevery=4, markersize=5, linewidth=1.8)
            ax.fill_between(steps, mean - se, mean + se,
                           alpha=0.15, color=COLORS[label])

        ax.set_xlabel("Training Steps")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Per-Step Satisfaction")
    axes[0].legend(loc="lower right", framealpha=0.9)

    fig.suptitle("Overcooked: Single-Context Saturation (CrampedRoom, 10 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overcooked_single_context.pdf")
    fig.savefig(OUT_DIR / "overcooked_single_context.png")
    plt.close(fig)
    print(f"  Saved overcooked_single_context")

# Figure 5: Exploration Ablation (SpiceEnv)
# ---------------------------------------------------------------------------
def plot_exploration_ablation():
    """DBTL (ours) vs exploit-only on SpiceEnv multi-recipe."""
    approaches = {
        "DBTL (ours)": "spices_c14_ours",
        "Exploit-only": None,  # check spices_comparison
    }

    # Exploit-only data
    exploit_dirs = []
    comp_dir = Path("experiments/logs/spices_comparison")
    if comp_dir.exists():
        for seed_name in os.listdir(comp_dir):
            cfg_path = comp_dir / seed_name / "config.yaml"
            if cfg_path.exists():
                text = cfg_path.read_text()
                if "approach_name: exploit_only" in text:
                    exploit_dirs.append(comp_dir / seed_name)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    metric = "neutral_eval_mean_user_satisfaction_per_step"

    # DBTL (ours)
    df_ours = load_approach("spices_c14_ours")
    steps, mean, se = agg_metric(df_ours, metric)
    if len(steps) > 0:
        ax.plot(steps, mean, label="DBTL (ours)", color=COLORS["DBTL (ours)"],
                marker=MARKERS["DBTL (ours)"], markevery=4, markersize=5, linewidth=1.8)
        ax.fill_between(steps, mean - se, mean + se, alpha=0.15, color=COLORS["DBTL (ours)"])

    # Exploit-only
    if exploit_dirs:
        df_exploit = load_eval_csvs(exploit_dirs)
        steps, mean, se = agg_metric(df_exploit, metric)
        if len(steps) > 0:
            ax.plot(steps, mean, label="Exploit-only", color=COLORS["Exploit-only"],
                    marker=MARKERS["Exploit-only"], markevery=4, markersize=5, linewidth=1.8)
            ax.fill_between(steps, mean - se, mean + se, alpha=0.15, color=COLORS["Exploit-only"])

    # CBTL
    df_cbtl = load_approach("spices_c14_cbtl")
    steps, mean, se = agg_metric(df_cbtl, metric)
    if len(steps) > 0:
        ax.plot(steps, mean, label="CBTL (pooled)", color=COLORS["CBTL (pooled)"],
                marker=MARKERS["CBTL (pooled)"], markevery=4, markersize=5, linewidth=1.8,
                linestyle="--", alpha=0.7)
        ax.fill_between(steps, mean - se, mean + se, alpha=0.1, color=COLORS["CBTL (pooled)"])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Per-Step Satisfaction")
    ax.set_title("Exploration Ablation: DBTL vs. Exploit-Only (SpiceEnv, neutral eval)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "exploration_ablation.pdf")
    fig.savefig(OUT_DIR / "exploration_ablation.png")
    plt.close(fig)
    print(f"  Saved exploration_ablation")


# Figure 6: Final satisfaction summary across all experiments
# ---------------------------------------------------------------------------
def plot_summary_bar_chart():
    """Summary bar chart of final neutral satisfaction across key experiments."""
    # Load final-checkpoint values for spice multi-recipe
    experiments = {
        "SpiceEnv\nMulti-Recipe": {
            "DBTL (ours)": "spices_c14_ours",
            "DBTL w/o ψ": "spices_c14_nomood",
            "Flat HBM": "spices_c14_flat",
            "CBTL (pooled)": "spices_c14_cbtl",
        },
        "Overcooked\nSingle-Context": {
            "DBTL (ours)": "oc_main_ours",
            "DBTL w/o ψ": "oc_main_nomood",
            "Flat HBM": "oc_main_flat",
            "CBTL (pooled)": "oc_main_cbtl",
        },
    }

    metric = "neutral_eval_mean_user_satisfaction_per_step"
    method_order = ["DBTL (ours)", "DBTL w/o ψ", "CBTL (pooled)", "Flat HBM"]

    fig, axes = plt.subplots(1, len(experiments), figsize=(10, 4.5), sharey=True)
    if len(experiments) == 1:
        axes = [axes]

    bar_width = 0.18

    for ax_idx, (exp_name, approaches) in enumerate(experiments.items()):
        ax = axes[ax_idx]
        vals = []
        errs = []
        labels = []

        for method in method_order:
            prefix = approaches.get(method)
            if prefix is None:
                vals.append(0)
                errs.append(0)
                labels.append(method)
                continue
            df = load_approach(prefix)
            if df.empty or metric not in df.columns:
                vals.append(0)
                errs.append(0)
            else:
                # Get final checkpoint
                final_step = df["training_step"].max()
                final = df[df["training_step"] == final_step][metric]
                vals.append(final.mean())
                errs.append(final.std() / np.sqrt(len(final)))
            labels.append(method)

        x = np.arange(len(method_order))
        bars = ax.bar(x, vals, bar_width * 3, yerr=errs, capsize=3,
                     color=[COLORS[m] for m in method_order], alpha=0.85,
                     edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(method_order, rotation=25, ha="right", fontsize=8)
        ax.set_title(exp_name, fontsize=11)
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Final Neutral Satisfaction")

    fig.suptitle("Final Per-Step Satisfaction at Convergence (10 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_bar_chart.pdf")
    fig.savefig(OUT_DIR / "summary_bar_chart.png")
    plt.close(fig)
    print(f"  Saved summary_bar_chart")

def main():
    print("Generating thesis figures...")
    print(f"  Output directory: {OUT_DIR}")
    print()

    plot_spice_multi_recipe()
    plot_spice_me_loo()
    plot_spice_diverse_loo()
    plot_overcooked_single_context()
    plot_exploration_ablation()
    plot_summary_bar_chart()

    print()
    print("Done! Check", OUT_DIR)


if __name__ == "__main__":
    main()
