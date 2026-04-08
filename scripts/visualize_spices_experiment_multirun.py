"""Visualize SPICES multirun logs from run_single_experiment.py.

This script compares methods (e.g., ours/no_learning/nothing_personal) averaged
across seeds from a Hydra multirun folder.

It produces:
  1) User satisfaction over training steps (mean +/- SE by method)
  2) Phi learning progress over training steps (phi sign accuracy vs. hidden theta)

Usage:
  python scripts/visualize_spices_experiment_multirun.py \
    --log-dir logs/2026-04-02/10-55-47
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multitask_personalization.envs.spices.config import (
    create_theta_params_from_config,
    get_hidden_hbm_config,
)


DEFAULT_HUMAN = "human"


# Theta magnitude bands for stratified sign-accuracy analysis.
# Boundaries chosen to align with _SPICE_SPECIFIC_THETA clusters:
#   nuanced  (0.3, 0.8]: turmeric, basil, coriander, paprika, ghee, …
#   mid      (0.8, 1.5]: garlic, ginger, saffron, harissa, fenugreek, …
#   strong   (1.5, ∞):   salt (+2.0), pepper (−2.0), chili/gochujang (−1.5)
THETA_BANDS: list[tuple[str, float, float]] = [
    ("nuanced\n(0.3–0.8]",   0.3, 0.8),
    ("mid\n(0.8–1.5]",       0.8, 1.5),
    ("strong\n(>1.5)",        1.5, float("inf")),
]
# Short labels used as dict keys (no newlines).
_BAND_KEYS: list[str] = ["nuanced", "mid", "strong"]


@dataclass
class SeedSeries:
    method: str
    seed: int
    # step -> value
    neutral_satisfaction: dict[int, float]
    natural_satisfaction: dict[int, float]
    phi_mae: dict[int, float]
    phi_rmse: dict[int, float]
    # tanh_phi_mae: mean over signal spices of |tanh(phi_learned) - tanh(phi_true)|
    # Bounded [0, 2]; not subject to tanh saturation artifacts.  Primary metric.
    tanh_phi_mae: dict[int, float]
    # sign_accuracy: fraction of signal spices where sign(phi_learned) == sign(phi_true)
    sign_accuracy: dict[int, float]
    # sign_accuracy_by_band: band_key -> {step -> accuracy}
    sign_accuracy_by_band: dict[str, dict[int, float]]
    # smoothed training satisfaction: step -> rolling mean
    train_satisfaction: dict[int, float]
    # prediction accuracy: fraction of steps robot flag matched human behavior [0, 1]
    neutral_pred_accuracy: dict[int, float]
    natural_pred_accuracy: dict[int, float]


def _mean_se(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / math.sqrt(arr.size)) if arr.size > 0 else float("nan")
    return mean, se


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_train_satisfaction_series(
    train_csv_path: Path,
    window: int = 50,
) -> dict[int, float]:
    """Return rolling-mean training satisfaction sampled every `window` steps."""
    if not train_csv_path.exists():
        return {}
    steps: list[int] = []
    sats: list[float] = []
    with train_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s = int(float(row.get("step", row.get("", 0))))
                v = float(row.get("user_satisfaction", "nan"))
                if not math.isnan(v):
                    steps.append(s)
                    sats.append(v)
            except (ValueError, KeyError):
                continue
    if not steps:
        return {}
    # Compute rolling mean in windows of `window` steps
    out: dict[int, float] = {}
    arr = np.array(sats, dtype=float)
    for i in range(0, len(arr), window):
        chunk = arr[i : i + window]
        if chunk.size > 0:
            out[steps[i]] = float(np.mean(chunk))
    return out


def _load_eval_satisfaction_series(
    eval_csv_path: Path,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return neutral and natural satisfaction series (fallback to legacy single metric)."""
    neutral: dict[int, float] = {}
    natural: dict[int, float] = {}
    with eval_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["training_step"]))
            if (
                "neutral_eval_mean_episode_average_satisfaction" in row
                and row["neutral_eval_mean_episode_average_satisfaction"] != ""
            ):
                neutral[step] = float(row["neutral_eval_mean_episode_average_satisfaction"])
            elif "eval_mean_episode_average_satisfaction" in row and row["eval_mean_episode_average_satisfaction"] != "":
                neutral[step] = float(row["eval_mean_episode_average_satisfaction"])
            else:
                neutral[step] = float(row["eval_mean_user_satisfaction_per_step"])

            if (
                "natural_eval_mean_episode_average_satisfaction" in row
                and row["natural_eval_mean_episode_average_satisfaction"] != ""
            ):
                natural[step] = float(row["natural_eval_mean_episode_average_satisfaction"])
    return neutral, natural


def _load_eval_prediction_accuracy_series(
    eval_csv_path: Path,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return neutral and natural prediction accuracy series from eval CSV."""
    neutral: dict[int, float] = {}
    natural: dict[int, float] = {}
    with eval_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(float(row["training_step"]))
            if "neutral_eval_mean_prediction_accuracy" in row and row["neutral_eval_mean_prediction_accuracy"] != "":
                try:
                    neutral[step] = float(row["neutral_eval_mean_prediction_accuracy"])
                except ValueError:
                    pass
            if "natural_eval_mean_prediction_accuracy" in row and row["natural_eval_mean_prediction_accuracy"] != "":
                try:
                    natural[step] = float(row["natural_eval_mean_prediction_accuracy"])
                except ValueError:
                    pass
    return neutral, natural


def _extract_true_theta_from_config(
    run_cfg: dict[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (global_theta, per_recipe_theta).

    global_theta: spice -> global theta (used for recipes with no overrides)
    per_recipe_theta: recipe_name -> spice -> true theta (applying any overrides)
    """
    hidden = (
        run_cfg.get("env", {})
        .get("env", {})
        .get("hidden_spec", {})
        .get("hidden_hbm", {})
    )
    config_name = hidden.get("hidden_hbm_config_name", "SpiceSpecificHuman")
    cfg = get_hidden_hbm_config(str(config_name))

    from multitask_personalization.envs.spices.spices_experiment import (
        MULTI_RECIPE_EXPERIMENT_POOL,
    )
    from multitask_personalization.envs.spices.recipes import get_recipe

    all_spices: set[str] = set()
    recipe_spices: dict[str, list[str]] = {}
    for recipe_name in MULTI_RECIPE_EXPERIMENT_POOL:
        spices = list(get_recipe(recipe_name).spices)
        recipe_spices[recipe_name] = spices
        all_spices.update(spices)
    spice_list = sorted(all_spices)

    global_theta = create_theta_params_from_config(cfg, spice_list)
    per_recipe_theta: dict[str, dict[str, float]] = {}
    for recipe_name, spices in recipe_spices.items():
        per_recipe_theta[recipe_name] = cfg.generate_theta_for_recipe(spices, recipe_name)
    return global_theta, per_recipe_theta


def _load_phi_from_checkpoint(ckpt: Path) -> dict[str, dict[str, float]] | None:
    """Load phi as {recipe: {spice: float}} from any supported checkpoint format.

    Supports:
    - spice_hbm.pkl  (HierarchicalPreferenceModel):  phi_m[human][recipe][spice] tensor
    - flat_model.pkl (FlatPreferenceModel):           alpha/beta -> logit(mean)
    - cbtl_classifier.pkl (CBTLClassifierModel):      w[human][spice] shared across recipes

    Returns None if no recognised file is found.
    """
    # HBM
    hbm_path = ckpt / "spice_hbm.pkl"
    if hbm_path.exists():
        with hbm_path.open("rb") as f:
            state = pickle.load(f)
        phi_m = state.get("phi_m", {})
        human_phi = phi_m.get(DEFAULT_HUMAN, {})
        # Values may be tensors; normalise to plain floats.
        return {
            recipe: {
                spice: float(v.item() if hasattr(v, "item") else v)
                for spice, v in spice_dict.items()
            }
            for recipe, spice_dict in human_phi.items()
        }

    # FlatPreferenceModel: phi = logit(alpha / (alpha + beta))
    flat_path = ckpt / "flat_model.pkl"
    if flat_path.exists():
        with flat_path.open("rb") as f:
            state = pickle.load(f)
        alpha = state.get("alpha", {}).get(DEFAULT_HUMAN, {})
        beta  = state.get("beta",  {}).get(DEFAULT_HUMAN, {})
        out: dict[str, dict[str, float]] = {}
        for recipe, spice_alpha in alpha.items():
            spice_beta = beta.get(recipe, {})
            out[recipe] = {}
            for spice, a in spice_alpha.items():
                b = spice_beta.get(spice, 1.0)
                n = a + b
                p = max(min(a / n, 1.0 - 1e-9), 1e-9)
                out[recipe][spice] = math.log(p / (1.0 - p))
        return out

    # CBTLClassifierModel: w[human][spice], shared across recipes.
    # Replicate the shared weight into every recipe so the per-recipe error
    # computation compares CBTL's single w against each recipe's true phi
    # (including recipe-specific overrides).  This is fair: the HBM is also
    # evaluated per-recipe, and CBTL's structural limitation (no recipe-level
    # split) should be visible as higher error on conflict spices.
    cbtl_path = ckpt / "cbtl_classifier.pkl"
    if cbtl_path.exists():
        with cbtl_path.open("rb") as f:
            state = pickle.load(f)
        w = state.get("w", {}).get(DEFAULT_HUMAN, {})
        spice_vals = {spice: float(w_val) for spice, w_val in w.items()}
        # Replicate into all known recipes.  We pull the recipe list from the
        # experiment pool so every recipe gets a row, just like HBM checkpoints.
        from multitask_personalization.envs.spices.spices_experiment import (
            MULTI_RECIPE_EXPERIMENT_POOL,
        )
        return {recipe: dict(spice_vals) for recipe in MULTI_RECIPE_EXPERIMENT_POOL}

    return None


def _load_phi_error_series(
    model_dir: Path,
    global_theta: dict[str, float],
    per_recipe_theta: dict[str, dict[str, float]],
    signal_threshold: float,
) -> tuple[
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[str, dict[int, float]],
]:
    """Compute phi MAE/RMSE/tanh_phi_mae/sign_accuracy at each checkpoint.

    Supports HierarchicalPreferenceModel (spice_hbm.pkl), FlatPreferenceModel
    (flat_model.pkl), and CBTLClassifierModel (cbtl_classifier.pkl).

    Metrics compare learned phi against the *intended phi_true* for each (recipe, spice):
      - For stable spices: phi_true == theta (same across recipes).
      - For recipe-conflicted spices: phi_true is the per-recipe override value, which
        differs from theta. This correctly penalises CBTL (whose shared w averages toward
        zero on conflicted spices) and rewards the HBM (whose per-recipe phi tracks each
        recipe's true direction independently).

    Averaging: each metric is computed per-recipe first (mean over signal spices in that
    recipe), then averaged across recipes. This gives equal weight to each recipe
    regardless of how many spices it contains, so a recipe with 18 spices does not
    dominate one with 15.

    Returns (mae, rmse, tanh_phi_mae, sign_accuracy, sign_accuracy_by_band).
    sign_accuracy_by_band maps band_key -> {step -> accuracy}.
    Step 0 is excluded: all phi_m are zero at init, so metrics reflect prior geometry
    rather than anything the model has learned.
    """
    out_mae: dict[int, float] = {}
    out_rmse: dict[int, float] = {}
    out_tanh_mae: dict[int, float] = {}
    out_sign_acc: dict[int, float] = {}
    out_sign_by_band: dict[str, dict[int, float]] = {k: {} for k in _BAND_KEYS}
    if not model_dir.exists():
        return out_mae, out_rmse, out_tanh_mae, out_sign_acc, out_sign_by_band

    ckpt_dirs = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    )
    for ckpt in ckpt_dirs:
        step = int(ckpt.name)
        # Skip step 0: phi_m is all-zeros at init, so metrics reflect prior geometry
        # rather than learning (positive-theta spices appear "correct" because 0>=0).
        if step == 0:
            continue
        human_phi = _load_phi_from_checkpoint(ckpt)
        if human_phi is None:
            continue

        # Collect per-recipe means, then average across recipes.
        recipe_mae: list[float] = []
        recipe_rmse: list[float] = []
        recipe_tanh_mae: list[float] = []
        recipe_sign_acc: list[float] = []
        # band_key -> list of per-recipe accuracies
        band_recipe_acc: dict[str, list[float]] = {k: [] for k in _BAND_KEYS}

        for recipe_name, spice_dict in human_phi.items():
            # phi_true for this recipe: per-recipe override if available (conflict
            # configs), otherwise global theta. This is the *intended* phi value —
            # what the env uses to determine the preferred actor deterministically.
            recipe_true = per_recipe_theta.get(recipe_name, global_theta)

            raw_errors: list[float] = []
            tanh_errors: list[float] = []
            sign_correct: list[float] = []
            # band_key -> correct flags for this recipe
            band_correct: dict[str, list[float]] = {k: [] for k in _BAND_KEYS}

            for spice, tensor_val in spice_dict.items():
                phi_true = float(recipe_true.get(spice, global_theta.get(spice, 0.0)))
                if abs(phi_true) <= signal_threshold:
                    continue
                learned = float(tensor_val)
                raw_errors.append(abs(learned - phi_true))
                tanh_errors.append(abs(math.tanh(learned) - math.tanh(phi_true)))
                correct = 1.0 if (learned >= 0) == (phi_true >= 0) else 0.0
                sign_correct.append(correct)
                # Assign to the matching theta band.
                abs_true = abs(phi_true)
                for (_, lo, hi), bkey in zip(THETA_BANDS, _BAND_KEYS):
                    if lo < abs_true <= hi:
                        band_correct[bkey].append(correct)
                        break

            if raw_errors:
                arr = np.array(raw_errors, dtype=float)
                recipe_mae.append(float(np.mean(arr)))
                recipe_rmse.append(float(np.sqrt(np.mean(arr ** 2))))
            if tanh_errors:
                recipe_tanh_mae.append(float(np.mean(tanh_errors)))
            if sign_correct:
                recipe_sign_acc.append(float(np.mean(sign_correct)))
            for bkey in _BAND_KEYS:
                if band_correct[bkey]:
                    band_recipe_acc[bkey].append(float(np.mean(band_correct[bkey])))

        if recipe_mae:
            out_mae[step] = float(np.mean(recipe_mae))
            out_rmse[step] = float(np.mean(recipe_rmse))
        if recipe_tanh_mae:
            out_tanh_mae[step] = float(np.mean(recipe_tanh_mae))
        if recipe_sign_acc:
            out_sign_acc[step] = float(np.mean(recipe_sign_acc))
        for bkey in _BAND_KEYS:
            if band_recipe_acc[bkey]:
                out_sign_by_band[bkey][step] = float(np.mean(band_recipe_acc[bkey]))

    return out_mae, out_rmse, out_tanh_mae, out_sign_acc, out_sign_by_band


def _collect_seed_series(log_dir: Path, signal_threshold: float) -> list[SeedSeries]:
    all_series: list[SeedSeries] = []
    run_dirs = sorted([p for p in log_dir.iterdir() if p.is_dir() and p.name.isdigit()])
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        eval_csv = run_dir / "eval_results.csv"
        train_csv = run_dir / "train_results.csv"
        model_dir = run_dir / "models"
        if not cfg_path.exists() or not eval_csv.exists():
            continue

        cfg = _read_yaml(cfg_path)
        method = str(cfg.get("approach_name", "unknown"))
        seed = int(cfg.get("seed", -1))

        neutral_satisfaction, natural_satisfaction = _load_eval_satisfaction_series(eval_csv)
        neutral_pred_accuracy, natural_pred_accuracy = _load_eval_prediction_accuracy_series(eval_csv)
        train_satisfaction = _load_train_satisfaction_series(train_csv)
        global_theta, per_recipe_theta = _extract_true_theta_from_config(cfg)
        phi_mae, phi_rmse, tanh_phi_mae, sign_accuracy, sign_accuracy_by_band = (
            _load_phi_error_series(
                model_dir=model_dir,
                global_theta=global_theta,
                per_recipe_theta=per_recipe_theta,
                signal_threshold=signal_threshold,
            )
        )
        all_series.append(
            SeedSeries(
                method=method,
                seed=seed,
                neutral_satisfaction=neutral_satisfaction,
                natural_satisfaction=natural_satisfaction,
                phi_mae=phi_mae,
                phi_rmse=phi_rmse,
                tanh_phi_mae=tanh_phi_mae,
                sign_accuracy=sign_accuracy,
                sign_accuracy_by_band=sign_accuracy_by_band,
                train_satisfaction=train_satisfaction,
                neutral_pred_accuracy=neutral_pred_accuracy,
                natural_pred_accuracy=natural_pred_accuracy,
            )
        )
    return all_series


def _aggregate_by_method(
    all_series: list[SeedSeries],
    attr: str,
) -> dict[str, dict[int, tuple[float, float, int]]]:
    """Return method -> step -> (mean, se, n)."""
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in all_series:
        series = getattr(s, attr)
        for step, val in series.items():
            if np.isnan(val):
                continue
            grouped[s.method][step].append(float(val))

    out: dict[str, dict[int, tuple[float, float, int]]] = {}
    for method, step_vals in grouped.items():
        out[method] = {}
        for step, vals in step_vals.items():
            mean, se = _mean_se(vals)
            out[method][step] = (mean, se, len(vals))
    return out


# ---------------------------------------------------------------------------
# Paper-quality plot configuration
# ---------------------------------------------------------------------------

# Display names for methods (raw key -> formatted label)
_METHOD_LABELS: dict[str, str] = {
    "ours":                  "HBM (Ours)",
    "with_mood_learning":    "HBM (with psi)",
    "without_mood_learning": "HBM (Ours; no psi)",
    "flat_model":            "Flat Bayesian",
    "cbtl_classifier":       "CBTL",
    "exploit_only":          "Exploit Only",
    "no_learning":           "No Learning",
}

# Consistent color palette (colorblind-safe: Okabe-Ito)
_METHOD_COLORS: dict[str, str] = {
    "ours":                  "#0072B2",  # blue
    "with_mood_learning":    "#0072B2",  # blue
    "without_mood_learning": "#E69F00",  # amber
    "flat_model":            "#009E73",  # green
    "cbtl_classifier":       "#CC79A7",  # pink
    "exploit_only":          "#D55E00",  # vermillion
    "no_learning":           "#999999",  # grey
}

# Line styles to aid black-and-white printing
_METHOD_LINESTYLES: dict[str, str] = {
    "ours":                  "-",
    "with_mood_learning":    "-",
    "without_mood_learning": "--",
    "flat_model":            "-.",
    "cbtl_classifier":       ":",
    "exploit_only":          (0, (3, 1, 1, 1)),
    "no_learning":           (0, (5, 5)),
}

_METHOD_MARKERS: dict[str, str] = {
    "ours":                  "o",
    "with_mood_learning":    "o",
    "without_mood_learning": "s",
    "flat_model":            "^",
    "cbtl_classifier":       "D",
    "exploit_only":          "v",
    "no_learning":           "x",
}

def _plot_timeseries(
    aggregate: dict[str, dict[int, tuple[float, float, int]]],
    title: str,
    y_label: str,
    out_path: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Determine marker step: place ~8 markers per line regardless of resolution
    all_steps: list[int] = []
    for step_map in aggregate.values():
        all_steps.extend(step_map.keys())
    if all_steps:
        min_s, max_s = min(all_steps), max(all_steps)
        n_unique = len(set(all_steps))
        markevery = max(1, n_unique // 8)
    else:
        markevery = 1

    for method in sorted(aggregate.keys()):
        step_map = aggregate[method]
        steps = sorted(step_map.keys())
        means = [step_map[s][0] for s in steps]
        ses = [step_map[s][1] for s in steps]
        upper = [m + se for m, se in zip(means, ses)]
        lower = [m - se for m, se in zip(means, ses)]

        label = _METHOD_LABELS.get(method, method)
        color = _METHOD_COLORS.get(method, None)
        ls = _METHOD_LINESTYLES.get(method, "-")
        marker = _METHOD_MARKERS.get(method, "o")

        line, = ax.plot(
            steps, means,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.8,
            marker=marker,
            markersize=5,
            markevery=markevery,
        )
        ax.fill_between(steps, lower, upper, alpha=0.15, color=line.get_color())

    ax.set_title(title, pad=10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(framealpha=0.9, edgecolor="0.8", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()


def _write_summary_csv(
    aggregate: dict[str, dict[int, tuple[float, float, int]]],
    out_path: Path,
    metric_name: str,
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "training_step", "metric", "mean", "se", "n"],
        )
        writer.writeheader()
        for method in sorted(aggregate.keys()):
            for step in sorted(aggregate[method].keys()):
                mean, se, n = aggregate[method][step]
                writer.writerow(
                    {
                        "method": method,
                        "training_step": step,
                        "metric": metric_name,
                        "mean": mean,
                        "se": se,
                        "n": n,
                    }
                )


def _aggregate_band_accuracy_at_final(
    all_series: list[SeedSeries],
) -> dict[str, dict[str, tuple[float, float, int]]]:
    """Return method -> band_key -> (mean, se, n) using each run's final checkpoint."""
    # method -> band_key -> list of final-step accuracies across seeds
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in all_series:
        for bkey, step_vals in s.sign_accuracy_by_band.items():
            if not step_vals:
                continue
            final_step = max(step_vals.keys())
            val = step_vals[final_step]
            if not math.isnan(val):
                grouped[s.method][bkey].append(val)

    out: dict[str, dict[str, tuple[float, float, int]]] = {}
    for method, band_vals in grouped.items():
        out[method] = {}
        for bkey, vals in band_vals.items():
            mean, se = _mean_se(vals)
            out[method][bkey] = (mean, se, len(vals))
    return out


def _plot_sign_accuracy_by_band(
    band_agg: dict[str, dict[str, tuple[float, float, int]]],
    out_path: Path,
) -> None:
    """Grouped bar chart: sign accuracy at final checkpoint, stratified by |theta| band.

    Layout: one group of bars per theta band (x-axis), one bar per method (color).
    Error bars show ±1 SE across seeds.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Determine method order: prefer a canonical order, then alphabetical fallback.
    preferred_order = [
        "ours",
        "with_mood_learning",
        "without_mood_learning",
        "flat_model",
        "cbtl_classifier",
        "exploit_only",
        "no_learning",
    ]
    methods = [m for m in preferred_order if m in band_agg]
    methods += sorted(m for m in band_agg if m not in preferred_order)

    n_methods = len(methods)
    n_bands = len(_BAND_KEYS)
    bar_width = 0.72 / n_methods  # bars fill ~72% of each group slot
    x = np.arange(n_bands)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, method in enumerate(methods):
        offsets = x + (i - (n_methods - 1) / 2.0) * bar_width
        means = []
        ses = []
        for bkey in _BAND_KEYS:
            entry = band_agg[method].get(bkey)
            if entry is not None:
                means.append(entry[0])
                ses.append(entry[1])
            else:
                means.append(float("nan"))
                ses.append(0.0)

        color = _METHOD_COLORS.get(method, None)
        label = _METHOD_LABELS.get(method, method)
        bars = ax.bar(
            offsets,
            means,
            width=bar_width * 0.88,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        # Error bars
        ax.errorbar(
            offsets,
            means,
            yerr=ses,
            fmt="none",
            ecolor="black",
            elinewidth=1.1,
            capsize=3,
            capthick=1.1,
        )
        # Annotate bars with the numeric value (skip NaN)
        for bar, m in zip(bars, means):
            if not math.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.012,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="0.25",
                )

    band_labels = [label for label, _, _ in THETA_BANDS]
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=11)
    ax.set_xlabel(r"Preference strength band  $|\theta^*|$", labelpad=6)
    ax.set_ylabel(r"Sign accuracy  $\mathrm{sign}(\hat{\phi}) = \mathrm{sign}(\phi^*)$")
    ax.set_title("Sign Accuracy by Preference Strength (final checkpoint, mean ± SE)", pad=10)
    ax.set_ylim(0.0, 1.12)
    ax.axhline(1.0, color="0.6", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.7)
    ax.legend(framealpha=0.9, edgecolor="0.8", loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()


def _write_band_accuracy_csv(
    band_agg: dict[str, dict[str, tuple[float, float, int]]],
    out_path: Path,
) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "theta_band", "mean", "se", "n"],
        )
        writer.writeheader()
        for method in sorted(band_agg.keys()):
            for bkey in _BAND_KEYS:
                entry = band_agg[method].get(bkey)
                if entry is None:
                    continue
                mean, se, n = entry
                writer.writerow(
                    {"method": method, "theta_band": bkey, "mean": mean, "se": se, "n": n}
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Hydra multirun folder, e.g. logs/2026-04-02/10-55-47",
    )
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.3,
        help="Signal spice threshold on |true theta| (default: 0.3).",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    out_dir = log_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_series = _collect_seed_series(log_dir, signal_threshold=args.signal_threshold)
    if not all_series:
        raise RuntimeError("No valid run subdirectories found (expected numeric dirs with config/eval/models).")

    neutral_sat_agg = _aggregate_by_method(all_series, "neutral_satisfaction")
    natural_sat_agg = _aggregate_by_method(all_series, "natural_satisfaction")
    train_sat_agg = _aggregate_by_method(all_series, "train_satisfaction")
    neutral_pred_acc_agg = _aggregate_by_method(all_series, "neutral_pred_accuracy")
    natural_pred_acc_agg = _aggregate_by_method(all_series, "natural_pred_accuracy")
    phi_mae_agg = _aggregate_by_method(all_series, "phi_mae")
    phi_rmse_agg = _aggregate_by_method(all_series, "phi_rmse")
    tanh_phi_mae_agg = _aggregate_by_method(all_series, "tanh_phi_mae")
    sign_accuracy_agg = _aggregate_by_method(all_series, "sign_accuracy")

    _plot_timeseries(
        neutral_sat_agg,
        title="User Satisfaction under Neutral Mood (mean ± SE)",
        y_label="Episode Average Satisfaction",
        out_path=out_dir / "neutral_satisfaction_over_time.png",
    )
    if natural_sat_agg:
        _plot_timeseries(
            natural_sat_agg,
            title="User Satisfaction under Natural Mood Variation (mean ± SE)",
            y_label="Episode Average Satisfaction",
            out_path=out_dir / "natural_satisfaction_over_time.png",
        )
    else:
        # Backward compatibility for older logs that only have one eval mode.
        _plot_timeseries(
            neutral_sat_agg,
            title="User Satisfaction over Training (mean ± SE)",
            y_label="Episode Average Satisfaction",
            out_path=out_dir / "satisfaction_over_time.png",
        )

    if train_sat_agg:
        _plot_timeseries(
            train_sat_agg,
            title="Training Satisfaction over Time (rolling mean ± SE)",
            y_label="User Satisfaction (rolling mean)",
            out_path=out_dir / "train_satisfaction_over_time.png",
        )

    _plot_timeseries(
        phi_mae_agg,
        title="Preference Parameter Error over Training (mean ± SE)",
        y_label=r"Mean $|\hat{\phi} - \phi^*|$",
        out_path=out_dir / "phi_mae_over_time.png",
    )
    _plot_timeseries(
        phi_rmse_agg,
        title="Preference Parameter RMSE over Training (mean ± SE)",
        y_label=r"RMSE$(\hat{\phi},\, \phi^*)$",
        out_path=out_dir / "phi_rmse_over_time.png",
    )
    if tanh_phi_mae_agg:
        _plot_timeseries(
            tanh_phi_mae_agg,
            title="Preference Prediction Error over Training (mean ± SE)",
            y_label=r"Mean $|\tanh(\hat{\phi}) - \tanh(\phi^*)|$",
            out_path=out_dir / "tanh_phi_mae_over_time.png",
        )
    if sign_accuracy_agg:
        _plot_timeseries(
            sign_accuracy_agg,
            title="Preference Sign Accuracy over Training (mean ± SE)",
            y_label=r"Fraction correct: $\mathrm{sign}(\hat{\phi}) = \mathrm{sign}(\phi^*)$",
            out_path=out_dir / "sign_accuracy_over_time.png",
        )

    if neutral_pred_acc_agg:
        _plot_timeseries(
            neutral_pred_acc_agg,
            title="Prediction Accuracy under Neutral Mood (mean ± SE)",
            y_label="Fraction of Steps Correctly Predicted",
            out_path=out_dir / "neutral_prediction_accuracy_over_time.png",
        )
    if natural_pred_acc_agg:
        _plot_timeseries(
            natural_pred_acc_agg,
            title="Prediction Accuracy under Natural Mood Variation (mean ± SE)",
            y_label="Fraction of Steps Correctly Predicted",
            out_path=out_dir / "natural_prediction_accuracy_over_time.png",
        )

    band_agg = _aggregate_band_accuracy_at_final(all_series)
    if band_agg:
        _plot_sign_accuracy_by_band(
            band_agg,
            out_path=out_dir / "sign_accuracy_by_theta_band.png",
        )
        _write_band_accuracy_csv(
            band_agg,
            out_path=out_dir / "sign_accuracy_by_theta_band.csv",
        )

    _write_summary_csv(
        neutral_sat_agg,
        out_dir / "neutral_satisfaction_over_time.csv",
        metric_name="neutral_eval_mean_episode_average_satisfaction",
    )
    if natural_sat_agg:
        _write_summary_csv(
            natural_sat_agg,
            out_dir / "natural_satisfaction_over_time.csv",
            metric_name="natural_eval_mean_episode_average_satisfaction",
        )
    else:
        _write_summary_csv(
            neutral_sat_agg,
            out_dir / "satisfaction_over_time.csv",
            metric_name="eval_mean_episode_average_satisfaction",
        )
    
    if train_sat_agg:
        _write_summary_csv(
            train_sat_agg,
            out_dir / "train_satisfaction_over_time.csv",
            metric_name="train_satisfaction_rolling_mean",
        )

    _write_summary_csv(
        phi_mae_agg,
        out_dir / "phi_mae_over_time.csv",
        metric_name="phi_mae",
    )
    _write_summary_csv(
        phi_rmse_agg,
        out_dir / "phi_rmse_over_time.csv",
        metric_name="phi_rmse",
    )
    if tanh_phi_mae_agg:
        _write_summary_csv(
            tanh_phi_mae_agg,
            out_dir / "tanh_phi_mae_over_time.csv",
            metric_name="tanh_phi_mae",
        )
    if sign_accuracy_agg:
        _write_summary_csv(
            sign_accuracy_agg,
            out_dir / "sign_accuracy_over_time.csv",
            metric_name="sign_accuracy",
        )
    if neutral_pred_acc_agg:
        _write_summary_csv(
            neutral_pred_acc_agg,
            out_dir / "neutral_prediction_accuracy_over_time.csv",
            metric_name="neutral_eval_mean_prediction_accuracy",
        )
    if natural_pred_acc_agg:
        _write_summary_csv(
            natural_pred_acc_agg,
            out_dir / "natural_prediction_accuracy_over_time.csv",
            metric_name="natural_eval_mean_prediction_accuracy",
        )

    print(f"Wrote analysis outputs to: {out_dir}")
    print(f"- {out_dir / 'neutral_satisfaction_over_time.png'}")
    if natural_sat_agg:
        print(f"- {out_dir / 'natural_satisfaction_over_time.png'}")
    if train_sat_agg:
        print(f"- {out_dir / 'train_satisfaction_over_time.png'}")
    print(f"- {out_dir / 'phi_mae_over_time.png'}")
    print(f"- {out_dir / 'phi_rmse_over_time.png'}")
    if tanh_phi_mae_agg:
        print(f"- {out_dir / 'tanh_phi_mae_over_time.png'}")
    if sign_accuracy_agg:
        print(f"- {out_dir / 'sign_accuracy_over_time.png'}")
    print(f"- {out_dir / 'neutral_satisfaction_over_time.csv'}")
    if natural_sat_agg:
        print(f"- {out_dir / 'natural_satisfaction_over_time.csv'}")
    print(f"- {out_dir / 'phi_mae_over_time.csv'}")
    print(f"- {out_dir / 'phi_rmse_over_time.csv'}")
    if tanh_phi_mae_agg:
        print(f"- {out_dir / 'tanh_phi_mae_over_time.csv'}")
    if sign_accuracy_agg:
        print(f"- {out_dir / 'sign_accuracy_over_time.csv'}")
    if neutral_pred_acc_agg:
        print(f"- {out_dir / 'neutral_prediction_accuracy_over_time.png'}")
        print(f"- {out_dir / 'neutral_prediction_accuracy_over_time.csv'}")
    if natural_pred_acc_agg:
        print(f"- {out_dir / 'natural_prediction_accuracy_over_time.png'}")
        print(f"- {out_dir / 'natural_prediction_accuracy_over_time.csv'}")
    if band_agg:
        print(f"- {out_dir / 'sign_accuracy_by_theta_band.png'}")
        print(f"- {out_dir / 'sign_accuracy_by_theta_band.csv'}")


if __name__ == "__main__":
    main()
