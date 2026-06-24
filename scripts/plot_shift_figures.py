"""Generate publication-quality figures for the 4-variant shift experiments."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS_DIR = Path("logs/runs")
OUT_DIR = Path("docs/figures/thesis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC = "neutral_eval_mean_user_satisfaction_per_step"

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
APPROACH_MAP = {
    "DBTL (ours)": "ours",
    "CBTL (pooled)": "cbtl_adapted",
    "Flat HBM": "flat_model",
    "DBTL w/o ψ": "without_mood_learning",
}
SHIFT_VARIANTS = {
    "Soft (16 spices)": "spices_shift_soft",
    "Medium (14 spices)": "spices_shift_medium",
    "Strong (7 spices)": "spices_shift_strong",
    "Random (~14/seed)": "spices_shift_random",
}
SHIFT_COLORS = {
    "Soft (16 spices)": "#60A5FA",
    "Medium (14 spices)": "#F59E0B",
    "Strong (7 spices)": "#EF4444",
    "Random (~14/seed)": "#8B5CF6",
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


def load_seeds(base_dir: Path) -> pd.DataFrame:
    frames = []
    if not base_dir.exists():
        return pd.DataFrame()
    for d in sorted(base_dir.iterdir()):
        csv = d / "eval_results.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df["seed_dir"] = str(d)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def agg(df, metric=METRIC):
    if df.empty or metric not in df.columns:
        return np.array([]), np.array([]), np.array([])
    g = df.groupby("training_step")[metric]
    m = g.mean()
    se = g.std() / np.sqrt(g.count())
    return m.index.values, m.values, se.values


# ---------------------------------------------------------------------------
# Figure 1: 2x2 grid — recovery curves for each shift variant
# ---------------------------------------------------------------------------
def plot_shift_grid():
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), sharey=True)

    for idx, (sname, senv) in enumerate(SHIFT_VARIANTS.items()):
        ax = axes[idx]
        for label, dirname in APPROACH_MAP.items():
            df = load_seeds(RUNS_DIR / f"{senv}_{dirname}")
            steps, mean, se = agg(df)
            if len(steps) == 0:
                continue
            ax.plot(steps, mean, label=label, color=COLORS[label],
                    marker=MARKERS[label], markevery=2, markersize=5, linewidth=1.8)
            ax.fill_between(steps, mean - se, mean + se, alpha=0.15, color=COLORS[label])

        ax.axvline(x=2500, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"Shift: {sname}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Training Steps")
        if idx == 0:
            ax.set_ylabel("Mean Per-Step Satisfaction")
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle("Non-Stationarity: Recovery by Shift Magnitude\n"
                 "(sign-flip at step 2500, magnitude preserved, 10 seeds)", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shift_grid.pdf")
    fig.savefig(OUT_DIR / "shift_grid.png")
    plt.close(fig)
    print("  Saved shift_grid")


# ---------------------------------------------------------------------------
# Figure 2: DBTL recovery comparison across variants (overlay)
# ---------------------------------------------------------------------------
def plot_shift_dbtl_overlay():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for sname, senv in SHIFT_VARIANTS.items():
        df = load_seeds(RUNS_DIR / f"{senv}_ours")
        steps, mean, se = agg(df)
        if len(steps) == 0:
            continue
        ax.plot(steps, mean, label=sname, color=SHIFT_COLORS[sname],
                marker="o", markevery=2, markersize=5, linewidth=2.0)
        ax.fill_between(steps, mean - se, mean + se, alpha=0.12, color=SHIFT_COLORS[sname])

    ax.axvline(x=2500, color="#6B7280", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.annotate("Preference Shift", xy=(2500, 0.66), fontsize=10, color="#6B7280",
                fontweight="bold", ha="right", va="bottom")

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Per-Step Satisfaction")
    ax.set_title("DBTL Recovery: Soft vs Medium vs Strong vs Random Shifts\n"
                 "(sign-flip preserves magnitude, 10 seeds)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "shift_dbtl_overlay.pdf")
    fig.savefig(OUT_DIR / "shift_dbtl_overlay.png")
    plt.close(fig)
    print("  Saved shift_dbtl_overlay")


# ---------------------------------------------------------------------------
# Figure 3: Drop & Recovery bar chart (grouped by variant)
# ---------------------------------------------------------------------------
def plot_shift_drop_recovery_bars():
    methods_order = ["DBTL (ours)", "CBTL (pooled)", "Flat HBM", "DBTL w/o ψ"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Drop by variant ---
    ax = axes[0]
    x = np.arange(len(SHIFT_VARIANTS))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, label in enumerate(methods_order):
        dirname = APPROACH_MAP[label]
        vals, errs = [], []
        for sname, senv in SHIFT_VARIANTS.items():
            seed_drops = []
            for seed in range(10, 20):
                try:
                    df = pd.read_csv(f"logs/runs/{senv}_{dirname}/{seed}/eval_results.csv")
                    pre = df[df["training_step"] == 2250][METRIC].iloc[0]
                    at = df[df["training_step"] == 2500][METRIC].iloc[0]
                    seed_drops.append(pre - at)
                except:
                    pass
            vals.append(np.mean(seed_drops) if seed_drops else 0)
            errs.append(np.std(seed_drops) / np.sqrt(len(seed_drops)) if seed_drops else 0)
        ax.bar(x + offsets[i], vals, width, label=label, color=COLORS[label],
               yerr=errs, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.split(" (")[0] for s in SHIFT_VARIANTS], fontsize=10)
    ax.set_ylabel("Satisfaction Drop")
    ax.set_title("Drop at Shift Point")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y")

    # --- Panel B: Recovery by variant ---
    ax = axes[1]
    for i, label in enumerate(methods_order):
        dirname = APPROACH_MAP[label]
        vals, errs = [], []
        for sname, senv in SHIFT_VARIANTS.items():
            seed_recs = []
            for seed in range(10, 20):
                try:
                    df = pd.read_csv(f"logs/runs/{senv}_{dirname}/{seed}/eval_results.csv")
                    at = df[df["training_step"] == 2500][METRIC].iloc[0]
                    final = df[METRIC].iloc[-1]
                    seed_recs.append(final - at)
                except:
                    pass
            vals.append(np.mean(seed_recs) if seed_recs else 0)
            errs.append(np.std(seed_recs) / np.sqrt(len(seed_recs)) if seed_recs else 0)
        ax.bar(x + offsets[i], vals, width, label=label, color=COLORS[label],
               yerr=errs, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.split(" (")[0] for s in SHIFT_VARIANTS], fontsize=10)
    ax.set_ylabel("Satisfaction Recovery (final − shift)")
    ax.set_title("Total Recovery")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Non-Stationarity: Drop & Recovery by Shift Band (10 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shift_drop_recovery.pdf")
    fig.savefig(OUT_DIR / "shift_drop_recovery.png")
    plt.close(fig)
    print("  Saved shift_drop_recovery")


# ---------------------------------------------------------------------------
# Figure 4: Recovery speed (first 250 steps post-shift)
# ---------------------------------------------------------------------------
def plot_shift_early_recovery():
    """Bar chart: satisfaction gain in first 250 steps after shift (speed metric)."""
    methods_order = ["DBTL (ours)", "CBTL (pooled)", "Flat HBM", "DBTL w/o ψ"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SHIFT_VARIANTS))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, label in enumerate(methods_order):
        dirname = APPROACH_MAP[label]
        vals, errs = [], []
        for sname, senv in SHIFT_VARIANTS.items():
            seed_speed = []
            for seed in range(10, 20):
                try:
                    df = pd.read_csv(f"logs/runs/{senv}_{dirname}/{seed}/eval_results.csv")
                    at = df[df["training_step"] == 2500][METRIC].iloc[0]
                    at_2750 = df[df["training_step"] == 2750][METRIC].iloc[0]
                    seed_speed.append(at_2750 - at)
                except:
                    pass
            vals.append(np.mean(seed_speed) if seed_speed else 0)
            errs.append(np.std(seed_speed) / np.sqrt(len(seed_speed)) if seed_speed else 0)
        ax.bar(x + offsets[i], vals, width, label=label, color=COLORS[label],
               yerr=errs, capsize=3, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([s.split(" (")[0] for s in SHIFT_VARIANTS], fontsize=10)
    ax.set_ylabel("Satisfaction Gain (first 250 steps)")
    ax.set_title("Early Recovery Speed: Satisfaction Gain in First 250 Steps Post-Shift\n(10 seeds)")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.2, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "shift_early_recovery.pdf")
    fig.savefig(OUT_DIR / "shift_early_recovery.png")
    plt.close(fig)
    print("  Saved shift_early_recovery")


# ---------------------------------------------------------------------------
# Figure 5: Zoomed post-shift recovery (2x2 grid)
# ---------------------------------------------------------------------------
def plot_shift_zoom_grid():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (sname, senv) in enumerate(SHIFT_VARIANTS.items()):
        ax = axes_flat[idx]
        for label, dirname in APPROACH_MAP.items():
            df = load_seeds(RUNS_DIR / f"{senv}_{dirname}")
            steps, mean, se = agg(df)
            if len(steps) == 0:
                continue
            mask = steps >= 2250
            ax.plot(steps[mask], mean[mask], label=label, color=COLORS[label],
                    marker=MARKERS[label], markevery=1, markersize=5, linewidth=2.0)
            ax.fill_between(steps[mask], (mean - se)[mask], (mean + se)[mask],
                            alpha=0.15, color=COLORS[label])

        ax.axvline(x=2500, color="#6B7280", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_title(f"Shift: {sname}", fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Training Steps")
        if idx % 2 == 0:
            ax.set_ylabel("Mean Per-Step Satisfaction")

    axes_flat[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Post-Shift Recovery Detail (steps 2250–5000, 10 seeds)", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "shift_zoom_grid.pdf")
    fig.savefig(OUT_DIR / "shift_zoom_grid.png")
    plt.close(fig)
    print("  Saved shift_zoom_grid")


if __name__ == "__main__":
    print("Generating shift experiment figures...")
    plot_shift_grid()
    plot_shift_dbtl_overlay()
    plot_shift_drop_recovery_bars()
    plot_shift_early_recovery()
    plot_shift_zoom_grid()
    print("Done!")
