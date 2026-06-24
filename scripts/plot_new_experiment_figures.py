"""Generate publication-quality figures for expanded LOO and nonstationary experiments."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Config
RUNS_DIR = Path("logs/runs")
OUT_DIR = Path("docs/figures/thesis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling — matches plot_thesis_figures.py
COLORS = {
    "DBTL (ours)": "#2563EB",
    "DBTL w/o ψ": "#7C3AED",
    "Flat HBM": "#DC2626",
    "CBTL (pooled)": "#059669",
}

MARKERS = {
    "DBTL (ours)": "o",
    "DBTL w/o ψ": "s",
    "Flat HBM": "^",
    "CBTL (pooled)": "D",
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

METRIC = "neutral_eval_mean_user_satisfaction_per_step"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_seeds(base_dir: Path) -> pd.DataFrame:
    """Load eval_results.csv from all seed subdirectories."""
    frames = []
    if not base_dir.exists():
        return pd.DataFrame()
    for d in sorted(base_dir.iterdir()):
        csv = d / "eval_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed_dir"] = str(d)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def agg_metric(df: pd.DataFrame, metric: str = METRIC):
    """Return (steps, mean, se) aggregated across seeds."""
    if df.empty or metric not in df.columns:
        return np.array([]), np.array([]), np.array([])
    grouped = df.groupby("training_step")[metric]
    mean = grouped.mean()
    se = grouped.std() / np.sqrt(grouped.count())
    return mean.index.values, mean.values, se.values


APPROACH_MAP = {
    "DBTL (ours)": "ours",
    "CBTL (pooled)": "cbtl_adapted",
    "Flat HBM": "flat_model",
    "DBTL w/o ψ": "without_mood_learning",
}


# ---------------------------------------------------------------------------
# Figure 1: Expanded LOO — Bar chart of warm-start (t0) satisfaction
# ---------------------------------------------------------------------------
def plot_expanded_loo_warmstart():
    """Bar chart: t0 satisfaction per split, grouped by method."""
    splits = ["ultra", "asian", "indian", "mediterranean",
              "moroccan", "ethiopian", "thai", "spanish"]
    split_labels = ["Ultra", "Asian", "Indian", "Medit.",
                    "Moroccan", "Ethiopian", "Thai", "Spanish"]

    data = {}  # {method_label: [t0_per_split]}
    errs = {}

    for label, dirname in APPROACH_MAP.items():
        t0_means, t0_ses = [], []
        for split in splits:
            base = RUNS_DIR / f"expanded_loo_{split}_{dirname}"
            df = load_seeds(base)
            if df.empty:
                t0_means.append(np.nan)
                t0_ses.append(0)
                continue
            # t0 = first eval step
            t0_vals = df[df["training_step"] == 0][METRIC].values
            t0_means.append(np.nanmean(t0_vals))
            t0_ses.append(np.nanstd(t0_vals) / np.sqrt(len(t0_vals)))
        data[label] = t0_means
        errs[label] = t0_ses

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(splits))
    width = 0.2
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, (label, vals) in enumerate(data.items()):
        bars = ax.bar(x + offsets[i], vals, width, label=label,
                      color=COLORS[label], yerr=errs[label],
                      capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Held-Out Recipe")
    ax.set_ylabel("Step-0 Satisfaction (warm-start)")
    ax.set_title("Expanded 8-Recipe LOO: Warm-Start Transfer Quality (10 seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_ylim(0.25, 0.75)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "expanded_loo_warmstart.pdf")
    fig.savefig(OUT_DIR / "expanded_loo_warmstart.png")
    plt.close(fig)
    print("  Saved expanded_loo_warmstart")


# ---------------------------------------------------------------------------
# Figure 2: Expanded LOO — Learning curves (aggregate across all 8 splits)
# ---------------------------------------------------------------------------
def plot_expanded_loo_curves():
    """Aggregate learning curves across all 8 LOO splits."""
    splits = ["ultra", "asian", "indian", "mediterranean",
              "moroccan", "ethiopian", "thai", "spanish"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, dirname in APPROACH_MAP.items():
        all_frames = []
        for split in splits:
            base = RUNS_DIR / f"expanded_loo_{split}_{dirname}"
            df = load_seeds(base)
            if not df.empty:
                df["split"] = split
                all_frames.append(df)

        if not all_frames:
            continue
        combined = pd.concat(all_frames, ignore_index=True)
        steps, mean, se = agg_metric(combined)
        if len(steps) == 0:
            continue
        ax.plot(steps, mean, label=label, color=COLORS[label],
                marker=MARKERS[label], markevery=4, markersize=5, linewidth=1.8)
        ax.fill_between(steps, mean - se, mean + se,
                        alpha=0.15, color=COLORS[label])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Per-Step Satisfaction")
    ax.set_title("Expanded 8-Recipe LOO: Aggregate Learning Curves\n"
                 "(8 held-out splits × 10 seeds = 80 runs per method)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "expanded_loo_curves.pdf")
    fig.savefig(OUT_DIR / "expanded_loo_curves.png")
    plt.close(fig)
    print("  Saved expanded_loo_curves")


# ---------------------------------------------------------------------------
# Figure 3: Expanded LOO — Per-split learning curves (2x4 grid)
# ---------------------------------------------------------------------------
def plot_expanded_loo_per_split():
    """2x4 grid of learning curves, one panel per held-out recipe."""
    splits = ["ultra", "asian", "indian", "mediterranean",
              "moroccan", "ethiopian", "thai", "spanish"]
    split_titles = ["Hold out: Ultra", "Hold out: Asian",
                    "Hold out: Indian", "Hold out: Mediterranean",
                    "Hold out: Moroccan", "Hold out: Ethiopian",
                    "Hold out: Thai", "Hold out: Spanish"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True, sharex=True)
    axes_flat = axes.flatten()

    for idx, (split, title) in enumerate(zip(splits, split_titles)):
        ax = axes_flat[idx]
        for label, dirname in APPROACH_MAP.items():
            base = RUNS_DIR / f"expanded_loo_{split}_{dirname}"
            df = load_seeds(base)
            steps, mean, se = agg_metric(df)
            if len(steps) == 0:
                continue
            ax.plot(steps, mean, label=label, color=COLORS[label],
                    marker=MARKERS[label], markevery=4, markersize=3, linewidth=1.4)
            ax.fill_between(steps, mean - se, mean + se,
                            alpha=0.12, color=COLORS[label])
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2)
        if idx >= 4:
            ax.set_xlabel("Training Steps")
        if idx % 4 == 0:
            ax.set_ylabel("Satisfaction")

    axes_flat[0].legend(loc="lower right", fontsize=7, framealpha=0.9)
    fig.suptitle("Expanded 8-Recipe LOO: Per-Split Learning Curves (10 seeds)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "expanded_loo_per_split.pdf")
    fig.savefig(OUT_DIR / "expanded_loo_per_split.png")
    plt.close(fig)
    print("  Saved expanded_loo_per_split")


# ---------------------------------------------------------------------------
# Figure 4: Nonstationary — Recovery curves with shift annotation
# ---------------------------------------------------------------------------
def plot_nonstationary_recovery():
    """Learning/recovery curves with vertical line at preference shift."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    shift_step = 2500

    for label, dirname in APPROACH_MAP.items():
        base = RUNS_DIR / f"nonstationary_{dirname}"
        df = load_seeds(base)
        steps, mean, se = agg_metric(df)
        if len(steps) == 0:
            continue
        ax.plot(steps, mean, label=label, color=COLORS[label],
                marker=MARKERS[label], markevery=2, markersize=5, linewidth=1.8)
        ax.fill_between(steps, mean - se, mean + se,
                        alpha=0.15, color=COLORS[label])

    # Shift annotation
    ax.axvline(x=shift_step, color="#6B7280", linestyle="--", linewidth=1.5,
               alpha=0.8, zorder=0)
    ax.annotate("Preference Shift",
                xy=(shift_step, ax.get_ylim()[1] * 0.98),
                xytext=(shift_step + 200, ax.get_ylim()[1] * 0.98),
                fontsize=10, color="#6B7280", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2),
                va="top")

    # Add shaded regions
    ylim = ax.get_ylim()
    ax.axvspan(0, shift_step, alpha=0.03, color="#2563EB", zorder=0)
    ax.axvspan(shift_step, 5000, alpha=0.03, color="#DC2626", zorder=0)
    ax.text(shift_step / 2, ylim[0] + 0.01, "Pre-shift", ha="center",
            fontsize=9, color="#6B7280", style="italic")
    ax.text(shift_step + (5000 - shift_step) / 2, ylim[0] + 0.01, "Post-shift",
            ha="center", fontsize=9, color="#6B7280", style="italic")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Per-Step Satisfaction")
    ax.set_title("Non-Stationarity: Preference Shift Recovery\n"
                 "(φ shifts at step 2500; 4 recipes, 10 seeds)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "nonstationary_recovery.pdf")
    fig.savefig(OUT_DIR / "nonstationary_recovery.png")
    plt.close(fig)
    print("  Saved nonstationary_recovery")


# ---------------------------------------------------------------------------
# Figure 5: Nonstationary — Recovery detail (zoomed post-shift)
# ---------------------------------------------------------------------------
def plot_nonstationary_zoom():
    """Zoomed view of post-shift recovery (steps 2250-5000)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    shift_step = 2500

    for label, dirname in APPROACH_MAP.items():
        base = RUNS_DIR / f"nonstationary_{dirname}"
        df = load_seeds(base)
        steps, mean, se = agg_metric(df)
        if len(steps) == 0:
            continue
        # Filter to post-shift region (include one pre-shift point for context)
        mask = steps >= 2250
        ax.plot(steps[mask], mean[mask], label=label, color=COLORS[label],
                marker=MARKERS[label], markevery=1, markersize=6, linewidth=2.0)
        ax.fill_between(steps[mask], (mean - se)[mask], (mean + se)[mask],
                        alpha=0.15, color=COLORS[label])

    ax.axvline(x=shift_step, color="#6B7280", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate("Shift", xy=(shift_step, ax.get_ylim()[1]),
                fontsize=10, color="#6B7280", fontweight="bold",
                ha="center", va="bottom")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Per-Step Satisfaction")
    ax.set_title("Post-Shift Recovery Detail (steps 2250–5000, 10 seeds)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "nonstationary_zoom.pdf")
    fig.savefig(OUT_DIR / "nonstationary_zoom.png")
    plt.close(fig)
    print("  Saved nonstationary_zoom")


# ---------------------------------------------------------------------------
# Figure 6: Nonstationary — Drop & recovery bar chart
# ---------------------------------------------------------------------------
def plot_nonstationary_bars():
    """Bar chart of drop magnitude and recovery amount per method."""
    shift_step = 2500
    methods_order = ["DBTL (ours)", "DBTL w/o ψ", "CBTL (pooled)", "Flat HBM"]

    drops, recoveries, drop_ses, rec_ses = {}, {}, {}, {}

    for label, dirname in APPROACH_MAP.items():
        base = RUNS_DIR / f"nonstationary_{dirname}"
        df = load_seeds(base)
        if df.empty:
            continue

        seed_drops, seed_recs = [], []
        for sd in df["seed_dir"].unique():
            sdf = df[df["seed_dir"] == sd].sort_values("training_step")
            pre = sdf[sdf["training_step"] == 2250][METRIC].values
            at = sdf[sdf["training_step"] == shift_step][METRIC].values
            final = sdf[METRIC].iloc[-1]
            if len(pre) > 0 and len(at) > 0:
                seed_drops.append(pre[0] - at[0])
                seed_recs.append(final - at[0])

        drops[label] = np.mean(seed_drops)
        recoveries[label] = np.mean(seed_recs)
        drop_ses[label] = np.std(seed_drops) / np.sqrt(len(seed_drops))
        rec_ses[label] = np.std(seed_recs) / np.sqrt(len(seed_recs))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Drop magnitude
    ax = axes[0]
    labels_present = [m for m in methods_order if m in drops]
    x = np.arange(len(labels_present))
    vals = [drops[m] for m in labels_present]
    errs = [drop_ses[m] for m in labels_present]
    colors = [COLORS[m] for m in labels_present]
    ax.bar(x, vals, color=colors, yerr=errs, capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (ours)", "\n(ours)").replace(" (pooled)", "\n(pooled)")
                        for m in labels_present], fontsize=9)
    ax.set_ylabel("Satisfaction Drop")
    ax.set_title("Drop at Shift Point")
    ax.grid(True, alpha=0.2, axis="y")

    # Recovery
    ax = axes[1]
    vals = [recoveries[m] for m in labels_present]
    errs = [rec_ses[m] for m in labels_present]
    ax.bar(x, vals, color=colors, yerr=errs, capsize=4, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (ours)", "\n(ours)").replace(" (pooled)", "\n(pooled)")
                        for m in labels_present], fontsize=9)
    ax.set_ylabel("Satisfaction Recovery")
    ax.set_title("Recovery (final − shift)")
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Non-Stationarity: Drop & Recovery (10 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "nonstationary_bars.pdf")
    fig.savefig(OUT_DIR / "nonstationary_bars.png")
    plt.close(fig)
    print("  Saved nonstationary_bars")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating new experiment figures...")
    plot_expanded_loo_warmstart()
    plot_expanded_loo_curves()
    plot_expanded_loo_per_split()
    plot_nonstationary_recovery()
    plot_nonstationary_zoom()
    plot_nonstationary_bars()
    print("Done!")
