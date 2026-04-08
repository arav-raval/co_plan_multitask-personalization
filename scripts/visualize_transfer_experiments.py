"""Visualize results for Claim 2 (cross-recipe generalization) and Claim 3 (cross-human transfer).

Claim 2: Uses the same Hydra multirun format as the main experiment. The key difference
  is that eval is on a held-out recipe — the plots show how well theta transfers.

Claim 3: Uses the run_transfer_experiment.py output format with warm/cold comparison.
  Plots convergence curves showing warm-start advantage from population mu.

Usage:
  # Claim 2 (cross-recipe): same format as main visualizer
  python scripts/visualize_transfer_experiments.py \
    --claim 2 --log-dir logs/2026-04-06/22-45-58

  # Claim 3 (cross-human transfer)
  python scripts/visualize_transfer_experiments.py \
    --claim 3 --log-dir logs/transfer/2026-04-06
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paper-quality style (shared with main visualizer)
# ---------------------------------------------------------------------------

_METHOD_LABELS: dict[str, str] = {
    "ours":                  "HBM (Ours)",
    "without_mood_learning": "HBM (no psi)",
    "flat_model":            "Flat Bayesian",
    "cbtl_classifier":       "CBTL",
}
_METHOD_COLORS: dict[str, str] = {
    "ours":                  "#0072B2",
    "without_mood_learning": "#E69F00",
    "flat_model":            "#009E73",
    "cbtl_classifier":       "#CC79A7",
}
_METHOD_LINESTYLES: dict[str, str] = {
    "ours":                  "-",
    "without_mood_learning": "--",
    "flat_model":            "-.",
    "cbtl_classifier":       ":",
}
_METHOD_MARKERS: dict[str, str] = {
    "ours":                  "o",
    "without_mood_learning": "s",
    "flat_model":            "^",
    "cbtl_classifier":       "D",
}

# Transfer-specific: warm vs cold within a method
_TRANSFER_STYLES = {
    "warm": {"linestyle": "-",  "marker": "o", "label_suffix": " (warm-start)"},
    "cold": {"linestyle": "--", "marker": "x", "label_suffix": " (cold-start)"},
}


def _setup_rcparams() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _mean_se(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0


def _plot_timeseries(
    aggregate: dict[str, dict[int, tuple[float, float, int]]],
    title: str,
    y_label: str,
    out_path: Path,
    x_label: str = "Training Steps",
) -> None:
    _setup_rcparams()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    all_steps: list[int] = []
    for step_map in aggregate.values():
        all_steps.extend(step_map.keys())
    n_unique = len(set(all_steps))
    markevery = max(1, n_unique // 8)

    for method in sorted(aggregate.keys()):
        step_map = aggregate[method]
        steps = sorted(step_map.keys())
        means = [step_map[s][0] for s in steps]
        ses = [step_map[s][1] for s in steps]

        label = _METHOD_LABELS.get(method, method)
        color = _METHOD_COLORS.get(method, None)
        ls = _METHOD_LINESTYLES.get(method, "-")
        marker = _METHOD_MARKERS.get(method, "o")

        ax.plot(steps, means, label=label, color=color, linestyle=ls,
                marker=marker, markevery=markevery, markersize=5, linewidth=2)
        ax.fill_between(steps,
                        [m - se for m, se in zip(means, ses)],
                        [m + se for m, se in zip(means, ses)],
                        alpha=0.15, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def _write_csv(
    aggregate: dict[str, dict[int, tuple[float, float, int]]],
    out_path: Path,
    metric_name: str,
) -> None:
    rows: list[dict[str, object]] = []
    for method, step_map in sorted(aggregate.items()):
        for step in sorted(step_map):
            mean, se, n = step_map[step]
            rows.append({
                "method": method,
                "training_step": step,
                "metric": metric_name,
                "mean": mean,
                "se": se,
                "n": n,
            })
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Claim 2: Cross-recipe generalization
# ---------------------------------------------------------------------------

def _visualize_claim2(log_dir: Path) -> None:
    """Visualize cross-recipe generalization results (Hydra multirun format).

    Same format as main experiment but eval is on held-out MediterraneanComplex.
    Key metric: prediction accuracy on the held-out recipe over training steps.
    """
    import yaml

    out_dir = log_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect: method -> {step -> [values across seeds]}
    neutral_pred: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    natural_pred: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    neutral_sat: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    natural_sat: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    run_dirs = sorted([p for p in log_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        eval_csv = run_dir / "eval_results.csv"
        if not cfg_path.exists() or not eval_csv.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        method = str(cfg.get("approach_name", "unknown"))

        with open(eval_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(float(row.get("training_step", 0)))
                for key, store in [
                    ("neutral_eval_mean_prediction_accuracy", neutral_pred),
                    ("natural_eval_mean_prediction_accuracy", natural_pred),
                    ("neutral_eval_mean_user_satisfaction", neutral_sat),
                    ("natural_eval_mean_user_satisfaction", natural_sat),
                    # Fallback for older runs without neutral/natural prefix
                    ("eval_mean_prediction_accuracy", neutral_pred),
                    ("eval_mean_user_satisfaction_per_step", neutral_sat),
                ]:
                    val = row.get(key)
                    if val is not None and val != "":
                        try:
                            store[method][step].append(float(val))
                        except ValueError:
                            pass

    def _to_agg(raw: dict[str, dict[int, list[float]]]) -> dict[str, dict[int, tuple[float, float, int]]]:
        out: dict[str, dict[int, tuple[float, float, int]]] = {}
        for method, step_map in raw.items():
            out[method] = {}
            for step, vals in step_map.items():
                m, se = _mean_se(vals)
                out[method][step] = (m, se, len(vals))
        return out

    if neutral_pred:
        agg = _to_agg(neutral_pred)
        _plot_timeseries(agg,
            title="Cross-Recipe: Prediction Accuracy on Held-Out Recipe\n(Neutral Mood, mean ± SE)",
            y_label="Prediction Accuracy",
            out_path=out_dir / "cross_recipe_neutral_prediction_accuracy.png")
        _write_csv(agg, out_dir / "cross_recipe_neutral_prediction_accuracy.csv",
                   "neutral_prediction_accuracy")

    if natural_pred:
        agg = _to_agg(natural_pred)
        _plot_timeseries(agg,
            title="Cross-Recipe: Prediction Accuracy on Held-Out Recipe\n(Natural Mood, mean ± SE)",
            y_label="Prediction Accuracy",
            out_path=out_dir / "cross_recipe_natural_prediction_accuracy.png")
        _write_csv(agg, out_dir / "cross_recipe_natural_prediction_accuracy.csv",
                   "natural_prediction_accuracy")

    if neutral_sat:
        agg = _to_agg(neutral_sat)
        _plot_timeseries(agg,
            title="Cross-Recipe: Satisfaction on Held-Out Recipe\n(Neutral Mood, mean ± SE)",
            y_label="Episode Average Satisfaction",
            out_path=out_dir / "cross_recipe_neutral_satisfaction.png")
        _write_csv(agg, out_dir / "cross_recipe_neutral_satisfaction.csv",
                   "neutral_satisfaction")

    if natural_sat:
        agg = _to_agg(natural_sat)
        _plot_timeseries(agg,
            title="Cross-Recipe: Satisfaction on Held-Out Recipe\n(Natural Mood, mean ± SE)",
            y_label="Episode Average Satisfaction",
            out_path=out_dir / "cross_recipe_natural_satisfaction.png")
        _write_csv(agg, out_dir / "cross_recipe_natural_satisfaction.csv",
                   "natural_satisfaction")

    print(f"\nClaim 2 analysis written to {out_dir}")


# ---------------------------------------------------------------------------
# Claim 3: Cross-human transfer
# ---------------------------------------------------------------------------

def _visualize_claim3(log_dir: Path) -> None:
    """Visualize cross-human transfer results.

    Input: directory containing {approach}_seed{N}/ subdirectories, each with
    phase2_eval_results.csv from run_transfer_experiment.py.

    Key plots:
    1. Warm vs Cold convergence per approach (4 subplots showing transfer advantage)
    2. Warm-start comparison across approaches (which method benefits most from transfer)
    3. Transfer gap (warm - cold) over training steps
    """
    out_dir = log_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect: approach -> temp_type -> {step -> [values across seeds]}
    data: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for run_dir in sorted(log_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        p2 = run_dir / "phase2_eval_results.csv"
        if not p2.exists():
            continue
        # Parse approach from dir name: e.g. "ours_seed0"
        name = run_dir.name
        parts = name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        approach = parts[0]

        with open(p2) as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(float(row["training_step"]))
                for prefix in ["warm_neutral_", "cold_neutral_", "warm_natural_", "cold_natural_"]:
                    for metric_suffix in ["eval_mean_prediction_accuracy", "eval_mean_user_satisfaction"]:
                        key = f"{prefix}{metric_suffix}"
                        val = row.get(key)
                        if val is not None and val != "":
                            try:
                                data[approach][key][step].append(float(val))
                            except ValueError:
                                pass

    if not data:
        print("No transfer results found.")
        return

    approaches = sorted(data.keys())

    # --- Plot 1: Warm vs Cold prediction accuracy per approach (2x2 grid) ---
    _setup_rcparams()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, approach in enumerate(approaches[:4]):
        ax = axes_flat[idx]
        label = _METHOD_LABELS.get(approach, approach)

        for temp, style in [("warm_neutral_", {"color": "#0072B2", "ls": "-", "marker": "o", "label": "Warm-start (transfer)"}),
                            ("cold_neutral_", {"color": "#CC79A7", "ls": "--", "marker": "x", "label": "Cold-start (no transfer)"})]:
            key = f"{temp}eval_mean_prediction_accuracy"
            step_vals = data[approach].get(key, {})
            if not step_vals:
                continue
            steps = sorted(step_vals.keys())
            means = [np.mean(step_vals[s]) for s in steps]
            ses = [np.std(step_vals[s], ddof=1) / math.sqrt(len(step_vals[s])) if len(step_vals[s]) > 1 else 0 for s in steps]

            ax.plot(steps, means, color=style["color"], linestyle=style["ls"],
                    marker=style["marker"], markersize=5, linewidth=2, label=style["label"],
                    markevery=max(1, len(steps) // 8))
            ax.fill_between(steps,
                            [m - se for m, se in zip(means, ses)],
                            [m + se for m, se in zip(means, ses)],
                            alpha=0.15, color=style["color"])

        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if idx >= 2:
            ax.set_xlabel("Training Steps (on new human)")
        if idx % 2 == 0:
            ax.set_ylabel("Prediction Accuracy")
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Cross-Human Transfer: Warm-Start vs Cold-Start\n(Neutral Eval, mean ± SE)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "transfer_warm_vs_cold_per_approach.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'transfer_warm_vs_cold_per_approach.png'}")

    # --- Plot 2: Warm-start comparison across approaches ---
    warm_agg: dict[str, dict[int, tuple[float, float, int]]] = {}
    for approach in approaches:
        key = "warm_neutral_eval_mean_prediction_accuracy"
        step_vals = data[approach].get(key, {})
        if step_vals:
            warm_agg[approach] = {}
            for step in sorted(step_vals.keys()):
                m, se = _mean_se(step_vals[step])
                warm_agg[approach][step] = (m, se, len(step_vals[step]))

    if warm_agg:
        _plot_timeseries(warm_agg,
            title="Cross-Human Transfer: Warm-Start Prediction Accuracy\n(Neutral Eval, mean ± SE)",
            y_label="Prediction Accuracy",
            out_path=out_dir / "transfer_warm_comparison.png",
            x_label="Training Steps (on new human)")

    # --- Plot 3: Transfer gap (warm - cold) over time ---
    _setup_rcparams()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for approach in approaches:
        warm_key = "warm_neutral_eval_mean_prediction_accuracy"
        cold_key = "cold_neutral_eval_mean_prediction_accuracy"
        warm_vals = data[approach].get(warm_key, {})
        cold_vals = data[approach].get(cold_key, {})
        if not warm_vals or not cold_vals:
            continue

        steps = sorted(set(warm_vals.keys()) & set(cold_vals.keys()))
        gaps = []
        gap_ses = []
        for s in steps:
            w = np.array(warm_vals[s])
            c = np.array(cold_vals[s])
            # Paired difference
            n = min(len(w), len(c))
            diff = w[:n] - c[:n]
            gaps.append(float(np.mean(diff)))
            gap_ses.append(float(np.std(diff, ddof=1) / math.sqrt(n)) if n > 1 else 0.0)

        label = _METHOD_LABELS.get(approach, approach)
        color = _METHOD_COLORS.get(approach, None)
        ls = _METHOD_LINESTYLES.get(approach, "-")
        marker = _METHOD_MARKERS.get(approach, "o")

        ax.plot(steps, gaps, label=label, color=color, linestyle=ls,
                marker=marker, markersize=5, linewidth=2,
                markevery=max(1, len(steps) // 8))
        ax.fill_between(steps,
                        [g - se for g, se in zip(gaps, gap_ses)],
                        [g + se for g, se in zip(gaps, gap_ses)],
                        alpha=0.15, color=color)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Training Steps (on new human)")
    ax.set_ylabel("Transfer Gap (warm − cold)")
    ax.set_title("Transfer Advantage over Training\n(Prediction Accuracy, mean ± SE)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer_gap_over_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'transfer_gap_over_time.png'}")

    # --- CSV export ---
    rows: list[dict[str, object]] = []
    for approach in approaches:
        for key_prefix, metric_label in [
            ("warm_neutral_eval_mean_prediction_accuracy", "warm_neutral_pred_acc"),
            ("cold_neutral_eval_mean_prediction_accuracy", "cold_neutral_pred_acc"),
            ("warm_natural_eval_mean_prediction_accuracy", "warm_natural_pred_acc"),
            ("cold_natural_eval_mean_prediction_accuracy", "cold_natural_pred_acc"),
            ("warm_neutral_eval_mean_user_satisfaction", "warm_neutral_sat"),
            ("cold_neutral_eval_mean_user_satisfaction", "cold_neutral_sat"),
        ]:
            step_vals = data[approach].get(key_prefix, {})
            for step in sorted(step_vals.keys()):
                m, se = _mean_se(step_vals[step])
                rows.append({
                    "approach": approach,
                    "training_step": step,
                    "metric": metric_label,
                    "mean": m,
                    "se": se,
                    "n": len(step_vals[step]),
                })

    if rows:
        csv_path = out_dir / "transfer_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {csv_path}")

    print(f"\nClaim 3 analysis written to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize transfer experiment results")
    parser.add_argument("--claim", type=int, required=True, choices=[2, 3],
                        help="Which claim to visualize: 2=cross-recipe, 3=cross-human")
    parser.add_argument("--log-dir", required=True, help="Log directory")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    if args.claim == 2:
        _visualize_claim2(log_dir)
    elif args.claim == 3:
        _visualize_claim3(log_dir)


if __name__ == "__main__":
    main()
