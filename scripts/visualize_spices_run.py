"""Generate SPICES run visualizations for a given run folder.

Usage:
  python scripts/visualize_spices_run.py --run-dir logs/spices_test_reports/runs/<run_id>
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _read_metric_csv(path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = str(row["metric"])
            rows[metric] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "se": float(row["se"]),
                "n": float(row["n"]),
            }
    return rows


def _read_episode_csv(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                {
                    "seed": int(row["seed"]),
                    "episode": int(row["episode"]),
                    "mood": str(row["mood"]),
                    "average_satisfaction": float(row["average_satisfaction"]),
                }
            )
    return out


def _find_file(run_dir: Path, relative_candidates: List[str]) -> Path | None:
    for rel in relative_candidates:
        p = run_dir / rel
        if p.exists():
            return p
    return None


def _plot_metric_bars(metrics: Dict[str, Dict[str, float]], title: str, out_path: Path) -> None:
    keys = list(metrics.keys())
    means = [metrics[k]["mean"] for k in keys]
    ses = [metrics[k]["se"] for k in keys]
    x = list(range(len(keys)))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, means, yerr=ses, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel("Mean (error bars = SE)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_bars_subset(
    metrics: Dict[str, Dict[str, float]],
    keys: List[str],
    title: str,
    out_path: Path,
) -> None:
    existing = [k for k in keys if k in metrics]
    if not existing:
        return
    subset = {k: metrics[k] for k in existing}
    _plot_metric_bars(subset, title, out_path)


def _plot_single_episode_progression(episodes: List[dict], out_path: Path) -> None:
    by_mood_by_ep: Dict[str, Dict[int, List[float]]] = {}
    for row in episodes:
        mood = row["mood"]
        ep = row["episode"]
        sat = row["average_satisfaction"]
        by_mood_by_ep.setdefault(mood, {}).setdefault(ep, []).append(sat)

    fig, ax = plt.subplots(figsize=(12, 5))
    for mood, by_ep in sorted(by_mood_by_ep.items()):
        eps = sorted(by_ep.keys())
        means = [sum(by_ep[e]) / len(by_ep[e]) for e in eps]
        ses = []
        for e in eps:
            vals = by_ep[e]
            if len(vals) <= 1:
                ses.append(0.0)
            else:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                ses.append((var ** 0.5) / (len(vals) ** 0.5))
        ax.plot(eps, means, label=mood)
        upper = [m + s for m, s in zip(means, ses)]
        lower = [m - s for m, s in zip(means, ses)]
        ax.fill_between(eps, lower, upper, alpha=0.15)

    ax.set_title("Single Recipe: Episode Satisfaction Progression")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Satisfaction")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_sigma_comparison(
    single: Dict[str, Dict[str, float]],
    cross: Dict[str, Dict[str, float]],
    multi: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    tests = ["single_recipe", "cross_transfer", "multi_human"]
    sigma_names = ["sigma_h", "sigma_r", "sigma_obs"]
    data = [single, cross, multi]

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.22
    x = list(range(len(tests)))
    offsets = [-width, 0.0, width]
    for i, sigma in enumerate(sigma_names):
        means = [d[sigma]["mean"] for d in data]
        ses = [d[sigma]["se"] for d in data]
        ax.bar([v + offsets[i] for v in x], means, width=width, yerr=ses, capsize=4, label=sigma)

    ax.set_xticks(x)
    ax.set_xticklabels(tests)
    ax.set_ylabel("Learned Sigma (mean +/- SE)")
    ax.set_title("Sigma Comparison Across SPICES Tests")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to run folder under logs/spices_test_reports/runs/")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    single_csv = _find_file(
        run_dir,
        [
            "single_recipe/single_recipe_metrics.csv",
            "single_recipe_metrics.csv",
        ],
    )
    single_ep_csv = _find_file(
        run_dir,
        [
            "single_recipe/single_recipe_metrics_episode_satisfaction.csv",
            "single_recipe_metrics_episode_satisfaction.csv",
        ],
    )
    cross_csv = _find_file(
        run_dir,
        [
            "cross_transfer/cross_transfer_metrics.csv",
            "cross_transfer_metrics.csv",
        ],
    )
    multi_csv = _find_file(
        run_dir,
        [
            "multi_human/multi_human_metrics.csv",
            "multi_human_metrics.csv",
        ],
    )

    if not single_csv or not single_ep_csv or not cross_csv or not multi_csv:
        missing = [
            ("single_recipe_metrics.csv", single_csv),
            ("single_recipe_metrics_episode_satisfaction.csv", single_ep_csv),
            ("cross_transfer_metrics.csv", cross_csv),
            ("multi_human_metrics.csv", multi_csv),
        ]
        missing_names = [name for name, path in missing if path is None]
        raise FileNotFoundError(f"Missing required metric files: {missing_names}")

    single_metrics = _read_metric_csv(single_csv)
    cross_metrics = _read_metric_csv(cross_csv)
    multi_metrics = _read_metric_csv(multi_csv)
    single_eps = _read_episode_csv(single_ep_csv)

    # Single recipe: split by your requested grouping.
    _plot_metric_bars_subset(
        single_metrics,
        ["avg_sat_neutral", "avg_sat_all_self", "avg_sat_none_self", "avg_sat_overall"],
        "Single Recipe: Average Satisfaction by Mood (mean +/- SE)",
        plots_dir / "single_recipe_avg_satisfaction.png",
    )
    _plot_metric_bars_subset(
        single_metrics,
        ["psi_align_non_neutral", "phi_sign_acc", "neutral_match_auc"],
        "Single Recipe: Alignment/Accuracy Summary (mean +/- SE)",
        plots_dir / "single_recipe_alignment_accuracy.png",
    )
    _plot_single_episode_progression(
        single_eps,
        plots_dir / "single_recipe_episode_progression.png",
    )

    # Cross transfer: split by your requested grouping and drop sigmas.
    _plot_metric_bars_subset(
        cross_metrics,
        ["delta_neutral", "delta_non_neutral", "delta_all"],
        "Cross Transfer: Satisfaction Deltas (mean +/- SE)",
        plots_dir / "cross_transfer_deltas.png",
    )
    _plot_metric_bars_subset(
        cross_metrics,
        ["theta_sign_acc", "phi_sign_acc", "win_rate_recipes"],
        "Cross Transfer: Accuracy/Win Summary (mean +/- SE)",
        plots_dir / "cross_transfer_accuracy_winrate.png",
    )

    # Multi human: split into theta accuracy and delta satisfaction, no sigmas.
    _plot_metric_bars_subset(
        multi_metrics,
        ["theta_acc_h1", "theta_acc_h2", "theta_acc_new_human", "mu_sign_acc"],
        "Multi Human: Theta/Mu Accuracy (mean +/- SE)",
        plots_dir / "multi_human_theta_accuracy.png",
    )
    _plot_metric_bars_subset(
        multi_metrics,
        ["delta_sat_h1", "delta_sat_h2", "delta_sat_h3"],
        "Multi Human: Delta Satisfaction by Human (mean +/- SE)",
        plots_dir / "multi_human_delta_satisfaction.png",
    )

    # Keep sigma comparison as shared plot across all tests.
    _plot_sigma_comparison(
        single_metrics,
        cross_metrics,
        multi_metrics,
        plots_dir / "sigma_comparison.png",
    )

    summary_path = plots_dir / "README.txt"
    summary_path.write_text(
        "\n".join(
            [
                "SPICES visualization artifacts",
                f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
                f"Run dir: {run_dir}",
                "",
                "Generated files:",
                "- single_recipe_avg_satisfaction.png",
                "- single_recipe_alignment_accuracy.png",
                "- single_recipe_episode_progression.png",
                "- cross_transfer_deltas.png",
                "- cross_transfer_accuracy_winrate.png",
                "- multi_human_theta_accuracy.png",
                "- multi_human_delta_satisfaction.png",
                "- sigma_comparison.png",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote plots to: {plots_dir}")


if __name__ == "__main__":
    main()
