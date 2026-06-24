"""Tests for spices_csp.py."""

import csv
import json
import logging
from pathlib import Path
import numpy as np

from multitask_personalization.csp_solvers import EnumerationCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec,
)
from multitask_personalization.envs.spices.recipes import get_recipe, get_profile
from multitask_personalization.envs.spices.spices_hbm import (
    DEFAULT_HUMAN,
    HierarchicalPreferenceModel,
)
from multitask_personalization.envs.spices.config import (
    get_hidden_hbm_config,
    create_theta_params_from_config,
    PARAMETERS,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _create_hidden_hbm(spices: list, recipes: list, config_name: str) -> HierarchicalPreferenceModel:
    cfg = get_hidden_hbm_config(config_name)
    theta_mean = create_theta_params_from_config(cfg, spices)
    hbm = HierarchicalPreferenceModel(
        spices=spices,
        recipes=recipes,
        sigma_r=cfg.sigma_r,
        sigma_h=cfg.sigma_h,
        sigma0=cfg.sigma0,
        sigma_obs=cfg.sigma_obs,
    )
    hbm.set_theta(DEFAULT_HUMAN, theta_mean, sigma_h=cfg.sigma_h)
    return hbm


def _make_env(env_seed: int, recipe_name: str, hidden_hbm: HierarchicalPreferenceModel,
              training: bool = False) -> SpiceEnv:
    recipe = get_recipe(recipe_name)
    scene_spec = SpiceSceneSpec(recipe=recipe)
    hidden_spec = SpiceHiddenSpec(
        preferred_actor={},
        hidden_hbm=hidden_hbm,
        force_neutral_mood=training,
    )
    return SpiceEnv(scene_spec, hidden_spec, seed=env_seed)


def _signal_spices(hidden_hbm: HierarchicalPreferenceModel, spices: list,
                   threshold: float = 0.3) -> list[str]:
    return sorted(
        [s for s in spices if abs(hidden_hbm.theta_mean.get(s, 0.0)) > threshold],
        key=lambda s: abs(hidden_hbm.theta_mean.get(s, 0.0)),
        reverse=True,
    )


def _theta_sign_accuracy(hbm: HierarchicalPreferenceModel,
                         hidden_hbm: HierarchicalPreferenceModel,
                         signal_spices: list,
                         human_id: str = DEFAULT_HUMAN) -> tuple[int, int]:
    correct = sum(
        np.sign(hbm.get_theta(human_id, s)) == np.sign(hidden_hbm.theta_mean.get(s, 0.0))
        for s in signal_spices
    )
    return correct, len(signal_spices)


def _phi_sign_accuracy(hbm: HierarchicalPreferenceModel,
                       hidden_hbm: HierarchicalPreferenceModel,
                       signal_spices: list,
                       recipe_name: str,
                       human_id: str = DEFAULT_HUMAN) -> tuple[int, int]:
    correct = sum(
        np.sign(hbm.phi_mean.get(recipe_name, {}).get(s, 0.0))
        == np.sign(hidden_hbm.theta_mean.get(s, 0.0))
        for s in signal_spices
    )
    return correct, len(signal_spices)


def _psi_alignment(batch_psi_m: float, mood: str) -> bool:
    """True if batch_psi_m sign matches the mood's expected direction."""
    if mood == "neutral":
        return True  # neutral is always aligned — psi near 0 is correct
    if mood == "all_self":
        return batch_psi_m > 0.0
    if mood == "none_self":
        return batch_psi_m < 0.0
    return True


def _get_hidden_hbm_config_names() -> list[str]:
    """
    Return configured hidden HBM names with backward-compatible fallback.
    """
    names = PARAMETERS.get("hidden_hbm_config_names")
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    legacy = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")
    return [str(legacy)]


def _get_primary_hidden_hbm_config_name() -> str:
    return _get_hidden_hbm_config_names()[0]


def _mean_std_se(vals: list[float]) -> tuple[float, float, float]:
    """Return (mean, std, standard_error) using sample std when possible."""
    if not vals:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / np.sqrt(arr.size)) if arr.size > 0 else float("nan")
    return mean, std, se


def _export_metrics_report(
    test_name: str,
    summary_rows: list[dict[str, float | str]],
    details_payload: dict,
) -> None:
    """
    Write metrics to both JSON and CSV for paper reproducibility.

    Output dir: logs/spices_test_reports/
    """
    out_dir = Path("logs/spices_test_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{test_name}.json"
    csv_path = out_dir / f"{test_name}.csv"

    def _json_default(obj: object) -> object:
        if isinstance(obj, np.generic):
            return obj.item()
        return str(obj)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(details_payload, f, indent=2, default=_json_default)

    fieldnames = ["metric", "mean", "std", "se", "n", "notes"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({
                "metric": row.get("metric", ""),
                "mean": row.get("mean", ""),
                "std": row.get("std", ""),
                "se": row.get("se", ""),
                "n": row.get("n", ""),
                "notes": row.get("notes", ""),
            })

    logging.info(f"\n  [Report] wrote JSON: {json_path}")
    logging.info(f"  [Report] wrote CSV : {csv_path}")


def _export_episode_satisfaction_csv(
    test_name: str,
    rows: list[dict[str, float | str]],
) -> None:
    """Write per-episode satisfaction rows for figure recreation."""
    out_dir = Path("logs/spices_test_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{test_name}_episode_satisfaction.csv"
    fieldnames = ["seed", "episode", "mood", "average_satisfaction", "inferred_mood", "psi_m"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "seed": row.get("seed", ""),
                "episode": row.get("episode", ""),
                "mood": row.get("mood", ""),
                "average_satisfaction": row.get("average_satisfaction", ""),
                "inferred_mood": row.get("inferred_mood", ""),
                "psi_m": row.get("psi_m", ""),
            })


def test_spices_csp_single_recipe():
    """Single Recipe | Multi-Episode | Multi-Seed.

    Metrics logged:
        - Running_psi trajectory for mood episodes (last seed, first 10 eps)
        - Average satisfaction by mood
        - Phi sign accuracy and learned vs. true theta (top spices, final seed)
        - Learned hyperparameters (sigma_h, sigma_r, sigma_obs)
    """
    num_seeds    = PARAMETERS["num_seeds"]
    num_episodes = PARAMETERS["num_episodes"]
    recipe_name  = PARAMETERS["recipe_name"]
    recipe       = get_recipe(recipe_name)
    spices       = list(recipe.spices)
    config_name  = _get_primary_hidden_hbm_config_name()
    hidden_hbm   = _create_hidden_hbm(spices, [recipe_name], config_name=config_name)
    sig_spices   = _signal_spices(hidden_hbm, spices)

    # (seed, ep, mood, avg_sat, inferred_mood, batch_psi_m)
    all_records: list[tuple] = []
    neutral_match_rates: list[float] = []
    neutral_oracle_gaps: list[float] = []
    seed_metrics: list[dict[str, float]] = []
    final_seed_spice_rows: list[dict[str, float | str]] = []
    final_hbm: HierarchicalPreferenceModel | None = None

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] + i
        assert env_seed != csp_seed

        env     = _make_env(env_seed, recipe_name, hidden_hbm)
        csp_gen = SpicesAssignCSPGenerator(spice_list=spices, recipe_list=[recipe_name], seed=csp_seed)
        solver  = EnumerationCSPSolver(csp_seed)
        hbm     = csp_gen._pref_gen._hbm
        max_steps = len(spices) + 1
        log_this_seed = (i == num_seeds - 1)
        seed_mood_sats: dict[str, list[float]] = {"neutral": [], "all_self": [], "none_self": []}
        seed_neutral_match_rates: list[float] = []
        seed_neutral_oracle_gaps: list[float] = []
        seed_non_neutral_align: list[float] = []

        if log_this_seed:
            logging.info(f"\n{'='*60}")
            logging.info(f"[SingleRecipe] recipe={recipe_name}  config={config_name}  seed={i}")
            logging.info(f"  Signal spices: {len(sig_spices)} / {len(spices)}")

        for ep in range(num_episodes):
            obs, _ = env.reset()
            assert isinstance(obs, SpiceState)

            step_match_flags: list[bool] = []
            step_oracle_minus_actual: list[float] = []
            for _ in range(max_steps):
                prev_obs = obs
                csp, samplers, policy, init = csp_gen.generate(obs)
                sol = solver.solve(csp, init, samplers)
                assert sol is not None
                policy.reset(sol)
                act = policy.step(obs)
                obs, reward, done, _, info = env.step(act)
                assert np.isclose(reward, 0.0)
                csp_gen.observe_transition(prev_obs, act, obs, done, info)
                # Diagnostics for neutral episodes: robot prediction accuracy.
                # Under autonomous-human semantics, task_score=+1 means human claimed,
                # task_score=-1 means human didn't claim. Conflicts also produce
                # task_score=+1 (the human won) but are excluded from prediction
                # accuracy since they reflect coordination, not preference recovery.
                task_score = float(info.get("task_score", 0.0))
                conflict = bool(info.get("conflict", False))
                robot_flag = act[0]  # 0=claim, 1=pass
                if not conflict and task_score != 0.0:
                    # Match: robot passed (flag=1) when human claimed (+1), or
                    #        robot claimed (flag=0) when human didn't claim (-1).
                    human_claimed = task_score > 0
                    robot_passed = (robot_flag == 1)
                    step_match_flags.append(human_claimed == robot_passed)
                    # Oracle gap: oracle always matches human behavior (max task_score)
                    # actual_score is abs(task_score) for a correct match, 0 otherwise.
                    correct = (human_claimed == robot_passed)
                    oracle_exp = 1.0  # oracle always gets task_score=+1
                    actual_exp = 1.0 if correct else -1.0
                    step_oracle_minus_actual.append(oracle_exp - actual_exp)
                if done:
                    break

            mood       = info["mood"]
            psi_m      = hbm.get_psi_m(DEFAULT_HUMAN)
            inferred   = max(info["mood_posterior"], key=info["mood_posterior"].get)
            all_records.append((i, ep, mood, info["average_satisfaction"], inferred, psi_m))
            if mood in seed_mood_sats:
                seed_mood_sats[mood].append(float(info["average_satisfaction"]))
            if mood == "neutral" and step_match_flags:
                neutral_match_rates.append(float(np.mean(step_match_flags)))
                seed_neutral_match_rates.append(float(np.mean(step_match_flags)))
            if mood == "neutral" and step_oracle_minus_actual:
                neutral_oracle_gaps.append(float(np.mean(step_oracle_minus_actual)))
                seed_neutral_oracle_gaps.append(float(np.mean(step_oracle_minus_actual)))
            if mood != "neutral":
                seed_non_neutral_align.append(1.0 if _psi_alignment(psi_m, mood) else 0.0)

        env.close()
        if log_this_seed:
            final_hbm = hbm
        c, tot = _phi_sign_accuracy(hbm, hidden_hbm, sig_spices, recipe_name)
        sigmas = hbm.get_learned_sigmas()
        seed_metrics.append({
            "seed": float(i),
            "avg_sat_neutral": float(np.mean(seed_mood_sats["neutral"])) if seed_mood_sats["neutral"] else float("nan"),
            "avg_sat_all_self": float(np.mean(seed_mood_sats["all_self"])) if seed_mood_sats["all_self"] else float("nan"),
            "avg_sat_none_self": float(np.mean(seed_mood_sats["none_self"])) if seed_mood_sats["none_self"] else float("nan"),
            "avg_sat_overall": float(np.mean(
                seed_mood_sats["neutral"] + seed_mood_sats["all_self"] + seed_mood_sats["none_self"]
            )) if (seed_mood_sats["neutral"] or seed_mood_sats["all_self"] or seed_mood_sats["none_self"]) else float("nan"),
            "phi_sign_acc": (c / tot) if tot > 0 else float("nan"),
            "sigma_h": float(sigmas["sigma_h"]),
            "sigma_r": float(sigmas["sigma_r"]),
            "sigma_obs": float(sigmas["sigma_obs"]),
            "psi_align_non_neutral": float(np.mean(seed_non_neutral_align)) if seed_non_neutral_align else float("nan"),
            "neutral_match_auc": float(np.mean(seed_neutral_match_rates)) if seed_neutral_match_rates else float("nan"),
            "neutral_oracle_gap_auc": float(np.mean(seed_neutral_oracle_gaps)) if seed_neutral_oracle_gaps else float("nan"),
        })

    # ── Average satisfaction by mood ────────────────────────────────────────
    logging.info(f"\n  --- Average Satisfaction by Mood ---")
    for mood in ["neutral", "all_self", "none_self"]:
        sats = [sat for (_, _, m, sat, _, _) in all_records if m == mood]
        if sats:
            logging.info(f"  {mood:10}: avg={np.mean(sats):+.3f}  (n={len(sats)})")
    overall = np.mean([r[3] for r in all_records])
    logging.info(f"  {'overall':10}: avg={overall:+.3f}  (n={len(all_records)})")

    # ── Neutral-only progression: average every 10 neutral episodes ─────────
    neutral_series = [sat for (_, _, m, sat, _, _) in all_records if m == "neutral"]
    if neutral_series:
        window = 10
        logging.info(
            f"\n  --- Neutral Satisfaction Progression (avg per {window} neutral episodes) ---"
        )
        for i in range(0, len(neutral_series), window):
            chunk = neutral_series[i : i + window]
            lo = i + 1
            hi = i + len(chunk)
            logging.info(
                f"  neutral_eps[{lo:>3}-{hi:>3}] : avg={np.mean(chunk):+.3f}  (n={len(chunk)})"
            )
    if neutral_match_rates:
        window = 10
        logging.info(
            f"\n  --- Neutral Actor-Match Rate Progression (avg per {window} neutral episodes) ---"
        )
        for i in range(0, len(neutral_match_rates), window):
            chunk = neutral_match_rates[i : i + window]
            lo = i + 1
            hi = i + len(chunk)
            logging.info(
                f"  neutral_eps[{lo:>3}-{hi:>3}] : match_rate={np.mean(chunk):.1%}  (n={len(chunk)})"
            )
    # ── Phi sign accuracy + learned vs. true (final seed) ───────────────────
    if final_hbm is not None and sig_spices:
        c, tot = _phi_sign_accuracy(final_hbm, hidden_hbm, sig_spices, recipe_name)
        sigmas = final_hbm.get_learned_sigmas()
        logging.info(f"\n  --- Learned Preferences (final seed) ---")
        logging.info(f"  Phi sign accuracy: {c}/{tot} = {c/tot:.0%}")
        logging.info(f"  Learned sigmas: sigma_h={sigmas['sigma_h']:.3f}  "
                     f"sigma_r={sigmas['sigma_r']:.3f}  sigma_obs={sigmas['sigma_obs']:.3f}")
        logging.info(f"\n  {'spice':<20} {'true_theta':>10} {'learned_phi':>12} {'phi_var':>8}")
        logging.info(f"  {'-'*54}")
        for s in sig_spices:
            true_t  = hidden_hbm.theta_mean.get(s, 0.0)
            learned = final_hbm.phi_mean.get(recipe_name, {}).get(s, 0.0)
            phi_var = final_hbm.get_phi_var(DEFAULT_HUMAN, recipe_name, s)
            correct = "✓" if np.sign(learned) == np.sign(true_t) else "✗"
            logging.info(f"  {s:<20} {true_t:+10.3f} {learned:+12.3f} {phi_var:8.4f} {correct}")
            final_seed_spice_rows.append({
                "spice": s,
                "true_theta": float(true_t),
                "learned_phi": float(learned),
                "phi_var": float(phi_var),
                "correct_sign": bool(np.sign(learned) == np.sign(true_t)),
            })

    # ── Seed-level summary stats (mean ± std, SE) ───────────────────────────
    summary_rows: list[dict[str, float | str]] = []
    if seed_metrics:
        metric_keys = [
            "avg_sat_neutral",
            "avg_sat_all_self",
            "avg_sat_none_self",
            "avg_sat_overall",
            "psi_align_non_neutral",
            "phi_sign_acc",
            "sigma_h",
            "sigma_r",
            "sigma_obs",
            "neutral_match_auc",
            "neutral_oracle_gap_auc",
        ]
        logging.info(f"\n  --- Seed Summary (mean ± std, SE over {len(seed_metrics)} seeds) ---")
        for key in metric_keys:
            vals = [float(m[key]) for m in seed_metrics if not np.isnan(float(m[key]))]
            mean, std, se = _mean_std_se(vals)
            logging.info(f"  {key:<24} {mean:+.4f} ± {std:.4f}  (SE={se:.4f}, n={len(vals)})")
            summary_rows.append({
                "metric": key,
                "mean": mean,
                "std": std,
                "se": se,
                "n": float(len(vals)),
                "notes": "single_recipe",
            })

    _export_metrics_report(
        "single_recipe_metrics",
        summary_rows,
        {
            "config": dict(PARAMETERS),
            "seed_metrics": seed_metrics,
            "all_records": all_records,
            "neutral_match_rates": neutral_match_rates,
            "neutral_oracle_gaps": neutral_oracle_gaps,
            "final_seed_spice_rows": final_seed_spice_rows,
        },
    )
    _export_episode_satisfaction_csv(
        "single_recipe_metrics",
        [
            {
                "seed": int(seed),
                "episode": int(ep),
                "mood": str(mood),
                "average_satisfaction": float(avg_sat),
                "inferred_mood": str(inferred),
                "psi_m": float(psi_m),
            }
            for (seed, ep, mood, avg_sat, inferred, psi_m) in all_records
        ],
    )

def test_spices_csp_cross_transfer():
    """Single Human | Train on K recipes | Evaluate on M held-out recipes vs. cold-start.

    Training: forced-neutral mood.
    Evaluation: real mood; report neutral, non-neutral, and all-episode satisfaction.

    Metrics logged:
        - Theta sign accuracy (post-training, signal spices)
        - Per-test-recipe satisfaction: trained vs. baseline + delta
          (neutral, non-neutral, all episodes)
        - Aggregate satisfaction delta for all three buckets
        - Learned hyperparameters
    """
    num_seeds     = PARAMETERS["num_seeds"]
    num_train_eps = PARAMETERS["num_episodes"]
    num_test_eps  = PARAMETERS["num_test_episodes"]
    train_frac    = PARAMETERS["train_frac"]
    config_name   = _get_primary_hidden_hbm_config_name()

    profile_spec  = get_profile(PARAMETERS["profile"])
    all_recipes   = list(profile_spec.recipes)
    n_train       = int(np.ceil(len(all_recipes) * train_frac))
    train_recipes = all_recipes[:n_train]
    test_recipes  = all_recipes[n_train:]
    all_spices    = sorted({s for r in all_recipes for s in get_recipe(r).spices})

    hidden_hbm  = _create_hidden_hbm(all_spices, all_recipes, config_name=config_name)
    sig_spices  = _signal_spices(hidden_hbm, all_spices)
    max_steps   = max(len(get_recipe(r).spices) for r in all_recipes) + 5

    logging.info(f"\n{'='*60}")
    logging.info(f"[CrossTransfer] profile={profile_spec.name}  config={config_name}")
    logging.info(f"  Train: {len(train_recipes)} recipes  |  Test: {len(test_recipes)} recipes  "
                 f"|  Vocab: {len(all_spices)} spices  |  Signal: {len(sig_spices)} spices")

    trained_sats_neutral = {r: [] for r in test_recipes}
    trained_sats_non_neutral = {r: [] for r in test_recipes}
    trained_sats_all = {r: [] for r in test_recipes}
    baseline_sats_neutral = {r: [] for r in test_recipes}
    baseline_sats_non_neutral = {r: [] for r in test_recipes}
    baseline_sats_all = {r: [] for r in test_recipes}
    post_train_theta_acc: tuple[int, int] | None = None
    seed_metrics: list[dict[str, float]] = []

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] + i
        solver   = EnumerationCSPSolver(csp_seed)

        trained_gen = SpicesAssignCSPGenerator(
            spice_list=all_spices, recipe_list=train_recipes, seed=csp_seed
        )
        seed_trained_neutral = {r: [] for r in test_recipes}
        seed_trained_non_neutral = {r: [] for r in test_recipes}
        seed_trained_all = {r: [] for r in test_recipes}
        seed_baseline_neutral = {r: [] for r in test_recipes}
        seed_baseline_non_neutral = {r: [] for r in test_recipes}
        seed_baseline_all = {r: [] for r in test_recipes}

        logging.info(f"Seed {i} of {num_seeds}")
        for recipe_name in train_recipes:
            logging.info(f"    Recipe: {recipe_name}")
            env = _make_env(env_seed, recipe_name, hidden_hbm, training=True)
            for _ in range(num_train_eps):
                obs, _ = env.reset()
                for _ in range(max_steps):
                    prev_obs = obs
                    csp, samplers, policy, init = trained_gen.generate(obs)
                    sol = solver.solve(csp, init, samplers)
                    assert sol is not None
                    policy.reset(sol)
                    act = policy.step(obs)
                    obs, _, done, _, info = env.step(act)
                    trained_gen.observe_transition(prev_obs, act, obs, done, info)
                    if done:
                        break
            env.close()

        post_train_hbm = trained_gen._pref_gen._hbm
        post_train_hbm.flush_theta_mu()
        post_train_theta_acc = _theta_sign_accuracy(post_train_hbm, hidden_hbm, sig_spices)

        for recipe_name in test_recipes:
            env = _make_env(env_seed, recipe_name, hidden_hbm, training=False)
            for _ in range(num_test_eps):
                obs, _ = env.reset()
                for _ in range(max_steps):
                    prev_obs = obs
                    csp, samplers, policy, init = trained_gen.generate(obs)
                    sol = solver.solve(csp, init, samplers)
                    assert sol is not None
                    policy.reset(sol)
                    act = policy.step(obs)
                    obs, _, done, _, info = env.step(act)
                    trained_gen.observe_transition(prev_obs, act, obs, done, info)
                    if done:
                        break
                avg_sat = info["average_satisfaction"]
                trained_sats_all[recipe_name].append(avg_sat)
                seed_trained_all[recipe_name].append(avg_sat)
                if info.get("mood") == "neutral":
                    trained_sats_neutral[recipe_name].append(avg_sat)
                    seed_trained_neutral[recipe_name].append(avg_sat)
                else:
                    trained_sats_non_neutral[recipe_name].append(avg_sat)
                    seed_trained_non_neutral[recipe_name].append(avg_sat)
            env.close()

        baseline_solver = EnumerationCSPSolver(csp_seed)
        for recipe_name in test_recipes:
            test_spices = sorted(get_recipe(recipe_name).spices)
            baseline_gen = SpicesAssignCSPGenerator(
                spice_list=test_spices, recipe_list=[recipe_name], seed=csp_seed
            )
            env = _make_env(env_seed, recipe_name, hidden_hbm, training=False)
            for _ in range(num_test_eps):
                obs, _ = env.reset()
                for _ in range(max_steps):
                    prev_obs = obs
                    csp, samplers, policy, init = baseline_gen.generate(obs)
                    sol = baseline_solver.solve(csp, init, samplers)
                    assert sol is not None
                    policy.reset(sol)
                    act = policy.step(obs)
                    obs, _, done, _, info = env.step(act)
                    baseline_gen.observe_transition(prev_obs, act, obs, done, info)
                    if done:
                        break
                avg_sat = info["average_satisfaction"]
                baseline_sats_all[recipe_name].append(avg_sat)
                seed_baseline_all[recipe_name].append(avg_sat)
                if info.get("mood") == "neutral":
                    baseline_sats_neutral[recipe_name].append(avg_sat)
                    seed_baseline_neutral[recipe_name].append(avg_sat)
                else:
                    baseline_sats_non_neutral[recipe_name].append(avg_sat)
                    seed_baseline_non_neutral[recipe_name].append(avg_sat)
            env.close()

        # Per-seed aggregates for SE reporting
        c_theta, tot_theta = _theta_sign_accuracy(post_train_hbm, hidden_hbm, sig_spices)
        phi_c = 0
        phi_tot = 0
        for recipe_name in test_recipes:
            c_phi, tot_phi = _phi_sign_accuracy(post_train_hbm, hidden_hbm, sig_spices, recipe_name)
            phi_c += c_phi
            phi_tot += tot_phi
        sigmas = post_train_hbm.get_learned_sigmas()

        def _agg(d: dict[str, list[float]]) -> float:
            vals = [x for r in test_recipes for x in d[r]]
            return float(np.mean(vals)) if vals else float("nan")

        agg_tr_neu = _agg(seed_trained_neutral)
        agg_bl_neu = _agg(seed_baseline_neutral)
        agg_tr_non = _agg(seed_trained_non_neutral)
        agg_bl_non = _agg(seed_baseline_non_neutral)
        agg_tr_all = _agg(seed_trained_all)
        agg_bl_all = _agg(seed_baseline_all)

        wins = 0
        win_total = 0
        for recipe_name in test_recipes:
            t_vals = seed_trained_all[recipe_name]
            b_vals = seed_baseline_all[recipe_name]
            if t_vals and b_vals:
                win_total += 1
                if float(np.mean(t_vals)) > float(np.mean(b_vals)):
                    wins += 1
        win_rate = float(wins / win_total) if win_total > 0 else float("nan")

        seed_metrics.append({
            "seed": float(i),
            "theta_sign_acc": float(c_theta / tot_theta) if tot_theta > 0 else float("nan"),
            "phi_sign_acc": float(phi_c / phi_tot) if phi_tot > 0 else float("nan"),
            "delta_neutral": agg_tr_neu - agg_bl_neu if not np.isnan(agg_tr_neu) and not np.isnan(agg_bl_neu) else float("nan"),
            "delta_non_neutral": agg_tr_non - agg_bl_non if not np.isnan(agg_tr_non) and not np.isnan(agg_bl_non) else float("nan"),
            "delta_all": agg_tr_all - agg_bl_all if not np.isnan(agg_tr_all) and not np.isnan(agg_bl_all) else float("nan"),
            "win_rate_recipes": win_rate,
            "sigma_h": float(sigmas["sigma_h"]),
            "sigma_r": float(sigmas["sigma_r"]),
            "sigma_obs": float(sigmas["sigma_obs"]),
        })

    # ── Theta sign accuracy ──────────────────────────────────────────────────
    if post_train_theta_acc is not None:
        c, tot = post_train_theta_acc
        logging.info(f"\n  --- Theta Sign Accuracy (post-training, {tot} signal spices) ---")
        logging.info(f"  {c}/{tot} = {c/tot:.0%}")

    # ── Satisfaction: trained vs. baseline by episode bucket ────────────────
    def _log_satisfaction_table(
        title: str,
        trained_dict: dict[str, list[float]],
        baseline_dict: dict[str, list[float]],
    ) -> None:
        logging.info(f"\n  --- Satisfaction: Trained vs. Baseline ({title}) ---")
        logging.info(f"  {'recipe':<28} {'trained':>8} {'baseline':>9} {'delta':>7}")
        logging.info(f"  {'-'*56}")
        trained_all_vals: list[float] = []
        baseline_all_vals: list[float] = []
        for recipe_name in test_recipes:
            t = trained_dict[recipe_name]
            b = baseline_dict[recipe_name]
            trained_all_vals.extend(t)
            baseline_all_vals.extend(b)
            t_mean = np.mean(t) if t else float("nan")
            b_mean = np.mean(b) if b else float("nan")
            delta = t_mean - b_mean if t and b else float("nan")
            logging.info(f"  {recipe_name:<28} {t_mean:+8.3f} {b_mean:+9.3f} {delta:+7.3f}")
        if trained_all_vals and baseline_all_vals:
            agg_t = np.mean(trained_all_vals)
            agg_b = np.mean(baseline_all_vals)
            logging.info(f"  {'-'*56}")
            logging.info(f"  {'AGGREGATE':<28} {agg_t:+8.3f} {agg_b:+9.3f} {agg_t-agg_b:+7.3f}")

    _log_satisfaction_table("neutral episodes", trained_sats_neutral, baseline_sats_neutral)
    _log_satisfaction_table(
        "non-neutral episodes", trained_sats_non_neutral, baseline_sats_non_neutral
    )
    _log_satisfaction_table("all episodes", trained_sats_all, baseline_sats_all)

    # ── Learned hyperparameters ──────────────────────────────────────────────
    hbm = trained_gen._pref_gen._hbm
    sigmas = hbm.get_learned_sigmas()
    logging.info(f"\n  Learned sigmas: sigma_h={sigmas['sigma_h']:.3f}  "
                 f"sigma_r={sigmas['sigma_r']:.3f}  sigma_obs={sigmas['sigma_obs']:.3f}")

    # ── Seed-level summary (effect sizes + CI proxy) ────────────────────────
    summary_rows: list[dict[str, float | str]] = []
    if seed_metrics:
        metric_keys = [
            "theta_sign_acc",
            "phi_sign_acc",
            "delta_neutral",
            "delta_non_neutral",
            "delta_all",
            "win_rate_recipes",
            "sigma_h",
            "sigma_r",
            "sigma_obs",
        ]
        logging.info(f"\n  --- Seed Summary (mean ± std, SE over {len(seed_metrics)} seeds) ---")
        for key in metric_keys:
            vals = [float(m[key]) for m in seed_metrics if not np.isnan(float(m[key]))]
            mean, std, se = _mean_std_se(vals)
            ci95 = 1.96 * se if not np.isnan(se) else float("nan")
            logging.info(
                f"  {key:<20} {mean:+.4f} ± {std:.4f}  (SE={se:.4f}, 95%CI~±{ci95:.4f}, n={len(vals)})"
            )
            summary_rows.append({
                "metric": key,
                "mean": mean,
                "std": std,
                "se": se,
                "n": float(len(vals)),
                "notes": "cross_transfer",
            })

    _export_metrics_report(
        "cross_transfer_metrics",
        summary_rows,
        {
            "config": dict(PARAMETERS),
            "seed_metrics": seed_metrics,
            "trained_sats_neutral": trained_sats_neutral,
            "trained_sats_non_neutral": trained_sats_non_neutral,
            "trained_sats_all": trained_sats_all,
            "baseline_sats_neutral": baseline_sats_neutral,
            "baseline_sats_non_neutral": baseline_sats_non_neutral,
            "baseline_sats_all": baseline_sats_all,
        },
    )

def test_spices_csp_multi_human():
    """Multi-Human | Shared HBM | Population pooling + new-human cold-start.

    Metrics logged:
        - Per-trained-human theta sign accuracy
        - New human (μ warm-start) theta accuracy at zero episodes
        - Population μ sign accuracy
        - Satisfaction: trained humans + new human vs. cold baseline
        - Learned μ and θ for top signal spices
    """
    num_seeds   = PARAMETERS["num_seeds"]
    config_names = _get_hidden_hbm_config_names()

    profile_spec  = get_profile(PARAMETERS["profile"])
    all_recipes   = list(profile_spec.recipes)
    n_train       = int(np.ceil(len(all_recipes) * PARAMETERS["train_frac"]))
    train_recipes = all_recipes[:n_train]
    test_recipes  = all_recipes[n_train:]
    all_spices    = sorted({s for r in all_recipes for s in get_recipe(r).spices})

    num_humans      = PARAMETERS["num_humans"]
    train_human_ids = [f"h{i + 1}" for i in range(num_humans - 1)]
    new_human_id    = f"h{num_humans}"
    all_human_ids   = train_human_ids + [new_human_id]

    num_train_eps = max(1, PARAMETERS["num_episodes"] // len(train_human_ids))
    num_test_eps  = max(1, PARAMETERS["num_test_episodes"] // len(all_human_ids))
    max_steps     = max(len(get_recipe(r).spices) for r in all_recipes) + 5

    hidden_hbms = {}
    for idx, h in enumerate(all_human_ids):
        # Use first num_humans configured names; if fewer are provided, reuse the last.
        cfg_name = config_names[min(idx, len(config_names) - 1)]
        hidden_hbms[h] = _create_hidden_hbm(all_spices, all_recipes, config_name=cfg_name)
    sig_spices = _signal_spices(hidden_hbms["h1"], all_spices)

    logging.info(f"\n{'='*60}")
    logging.info(
        f"[MultiHuman] profile={profile_spec.name}  "
        f"configs={config_names[:len(all_human_ids)]}"
    )
    logging.info(f"  Train: {len(train_recipes)} recipes  |  Test: {len(test_recipes)} recipes  "
                 f"|  Humans: {train_human_ids} + new={new_human_id}  "
                 f"|  Signal spices: {len(sig_spices)}")

    trained_sats  = {h: {r: [] for r in test_recipes} for h in all_human_ids}
    baseline_sats = {h: {r: [] for r in test_recipes} for h in all_human_ids}
    post_train_theta_acc: dict[str, tuple[int, int]] = {}
    new_human_theta_acc: tuple[int, int] | None = None
    shared_hbm: HierarchicalPreferenceModel | None = None
    seed_metrics: list[dict[str, float]] = []

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] + i
        solver   = EnumerationCSPSolver(csp_seed)

        shared_hbm = HierarchicalPreferenceModel(spices=all_spices)
        gens = {
            h: SpicesAssignCSPGenerator(
                spice_list=all_spices, recipe_list=train_recipes,
                seed=csp_seed, human_id=h, shared_hbm=shared_hbm,
            )
            for h in train_human_ids
        }
        seed_trained_sats = {h: {r: [] for r in test_recipes} for h in all_human_ids}
        seed_baseline_sats = {h: {r: [] for r in test_recipes} for h in all_human_ids}

        for recipe_name in train_recipes:
            for h in train_human_ids:
                env = _make_env(env_seed, recipe_name, hidden_hbms[h], training=True)
                for _ in range(num_train_eps):
                    obs, _ = env.reset()
                    for _ in range(max_steps):
                        prev_obs = obs
                        csp, samplers, policy, init = gens[h].generate(obs)
                        sol = solver.solve(csp, init, samplers)
                        assert sol is not None
                        policy.reset(sol)
                        act = policy.step(obs)
                        obs, _, done, _, info = env.step(act)
                        gens[h].observe_transition(prev_obs, act, obs, done, info)
                        if done:
                            break
                env.close()

        shared_hbm.flush_theta_mu()

        for h in train_human_ids:
            post_train_theta_acc[h] = _theta_sign_accuracy(shared_hbm, hidden_hbms[h], sig_spices, h)

        shared_hbm.register_human(new_human_id)
        gens[new_human_id] = SpicesAssignCSPGenerator(
            spice_list=all_spices, recipe_list=train_recipes,
            seed=csp_seed, human_id=new_human_id, shared_hbm=shared_hbm,
        )
        new_human_theta_acc = _theta_sign_accuracy(shared_hbm, hidden_hbms[new_human_id], sig_spices, new_human_id)

        for h in all_human_ids:
            for recipe_name in test_recipes:
                env = _make_env(env_seed, recipe_name, hidden_hbms[h], training=False)
                for _ in range(num_test_eps):
                    obs, _ = env.reset()
                    for _ in range(max_steps):
                        prev_obs = obs
                        csp, samplers, policy, init = gens[h].generate(obs)
                        sol = solver.solve(csp, init, samplers)
                        assert sol is not None
                        policy.reset(sol)
                        act = policy.step(obs)
                        obs, _, done, _, info = env.step(act)
                        gens[h].observe_transition(prev_obs, act, obs, done, info)
                        if done:
                            break
                    if info.get("mood") == "neutral":
                        trained_sats[h][recipe_name].append(info["average_satisfaction"])
                        seed_trained_sats[h][recipe_name].append(info["average_satisfaction"])
                env.close()

        baseline_solver = EnumerationCSPSolver(csp_seed)
        for h in all_human_ids:
            for recipe_name in test_recipes:
                test_spices  = sorted(get_recipe(recipe_name).spices)
                baseline_gen = SpicesAssignCSPGenerator(
                    spice_list=test_spices, recipe_list=[recipe_name], seed=csp_seed
                )
                env = _make_env(env_seed, recipe_name, hidden_hbms[h], training=False)
                for _ in range(num_test_eps):
                    obs, _ = env.reset()
                    for _ in range(max_steps):
                        prev_obs = obs
                        csp, samplers, policy, init = baseline_gen.generate(obs)
                        sol = baseline_solver.solve(csp, init, samplers)
                        assert sol is not None
                        policy.reset(sol)
                        act = policy.step(obs)
                        obs, _, done, _, info = env.step(act)
                        baseline_gen.observe_transition(prev_obs, act, obs, done, info)
                        if done:
                            break
                    if info.get("mood") == "neutral":
                        baseline_sats[h][recipe_name].append(info["average_satisfaction"])
                        seed_baseline_sats[h][recipe_name].append(info["average_satisfaction"])
                env.close()

        # Per-seed metrics for SE reporting
        seed_row: dict[str, float] = {"seed": float(i)}
        for h in train_human_ids:
            c, tot = _theta_sign_accuracy(shared_hbm, hidden_hbms[h], sig_spices, h)
            seed_row[f"theta_acc_{h}"] = float(c / tot) if tot > 0 else float("nan")
        c_new, tot_new = _theta_sign_accuracy(shared_hbm, hidden_hbms[new_human_id], sig_spices, new_human_id)
        seed_row["theta_acc_new_human"] = float(c_new / tot_new) if tot_new > 0 else float("nan")
        mu_correct = sum(
            np.sign(shared_hbm.get_mu(s)) == np.sign(hidden_hbms["h1"].theta_mean.get(s, 0.0))
            for s in sig_spices
        )
        seed_row["mu_sign_acc"] = float(mu_correct / len(sig_spices)) if sig_spices else float("nan")
        for h in all_human_ids:
            t_all = [s for r in test_recipes for s in seed_trained_sats[h][r]]
            b_all = [s for r in test_recipes for s in seed_baseline_sats[h][r]]
            t_mean = float(np.mean(t_all)) if t_all else float("nan")
            b_mean = float(np.mean(b_all)) if b_all else float("nan")
            delta = t_mean - b_mean if t_all and b_all else float("nan")
            seed_row[f"delta_sat_{h}"] = delta
        sigmas_seed = shared_hbm.get_learned_sigmas()
        seed_row["sigma_h"] = float(sigmas_seed["sigma_h"])
        seed_row["sigma_r"] = float(sigmas_seed["sigma_r"])
        seed_row["sigma_obs"] = float(sigmas_seed["sigma_obs"])
        seed_metrics.append(seed_row)

    # ── Theta sign accuracy ──────────────────────────────────────────────────
    logging.info(f"\n  --- Theta Sign Accuracy (post-training, {len(sig_spices)} signal spices) ---")
    for h in train_human_ids:
        if h in post_train_theta_acc:
            c, tot = post_train_theta_acc[h]
            logging.info(f"  {h} (trained):     {c}/{tot} = {c/tot:.0%}")
    if new_human_theta_acc is not None:
        c, tot = new_human_theta_acc
        logging.info(f"  {new_human_id} (μ warm-start): {c}/{tot} = {c/tot:.0%}  (zero episodes)")

    if shared_hbm is not None:
        mu_correct = sum(
            np.sign(shared_hbm.get_mu(s)) == np.sign(hidden_hbms["h1"].theta_mean.get(s, 0.0))
            for s in sig_spices
        )
        logging.info(f"  μ population sign accuracy: {mu_correct}/{len(sig_spices)} = {mu_correct/len(sig_spices):.0%}")

    # ── Satisfaction ─────────────────────────────────────────────────────────
    logging.info(f"\n  --- Satisfaction vs. Baseline (neutral episodes only) ---")
    logging.info(f"  {'human':<6} {'type':<14} {'trained':>8} {'baseline':>9} {'delta':>7}")
    logging.info(f"  {'-'*48}")
    for h in all_human_ids:
        label = "μ-warm (new)" if h == new_human_id else "trained"
        t_all = [s for r in test_recipes for s in trained_sats[h][r]]
        b_all = [s for r in test_recipes for s in baseline_sats[h][r]]
        t_mean = np.mean(t_all) if t_all else float("nan")
        b_mean = np.mean(b_all) if b_all else float("nan")
        delta  = t_mean - b_mean if t_all and b_all else float("nan")
        logging.info(f"  {h:<6} {label:<14} {t_mean:+8.3f} {b_mean:+9.3f} {delta:+7.3f}")

    # ── Learned μ and θ (top 10 signal spices) ──────────────────────────────
    if shared_hbm is not None:
        top_sig = sig_spices[:10]
        logging.info(f"\n  --- Learned μ and θ (top {len(top_sig)} signal spices) ---")
        header = f"  {'spice':<20} {'true':>6} {'μ':>6}" + "".join(f" {'  '+h:>7}" for h in all_human_ids)
        logging.info(header)
        logging.info(f"  {'-'*(len(header)-2)}")
        for s in top_sig:
            true_val = hidden_hbms["h1"].theta_mean.get(s, 0.0)
            mu_val   = shared_hbm.get_mu(s)
            theta_str = "".join(f" {shared_hbm.get_theta(h, s):+7.2f}" for h in all_human_ids)
            logging.info(f"  {s:<20} {true_val:+6.2f} {mu_val:+6.2f}{theta_str}")

        sigmas = shared_hbm.get_learned_sigmas()
        logging.info(f"\n  Learned sigmas: sigma_h={sigmas['sigma_h']:.3f}  "
                     f"sigma_r={sigmas['sigma_r']:.3f}  sigma_obs={sigmas['sigma_obs']:.3f}")

    # ── Seed-level summary + exports ─────────────────────────────────────────
    summary_rows: list[dict[str, float | str]] = []
    if seed_metrics:
        metric_keys = []
        for h in train_human_ids:
            metric_keys.append(f"theta_acc_{h}")
        metric_keys.extend([
            "theta_acc_new_human",
            "mu_sign_acc",
        ])
        for h in all_human_ids:
            metric_keys.append(f"delta_sat_{h}")
        metric_keys.extend(["sigma_h", "sigma_r", "sigma_obs"])

        logging.info(f"\n  --- Seed Summary (mean ± std, SE over {len(seed_metrics)} seeds) ---")
        for key in metric_keys:
            vals = [float(m[key]) for m in seed_metrics if key in m and not np.isnan(float(m[key]))]
            mean, std, se = _mean_std_se(vals)
            logging.info(f"  {key:<20} {mean:+.4f} ± {std:.4f}  (SE={se:.4f}, n={len(vals)})")
            summary_rows.append({
                "metric": key,
                "mean": mean,
                "std": std,
                "se": se,
                "n": float(len(vals)),
                "notes": "multi_human",
            })

    _export_metrics_report(
        "multi_human_metrics",
        summary_rows,
        {
            "config": dict(PARAMETERS),
            "seed_metrics": seed_metrics,
            "trained_sats": trained_sats,
            "baseline_sats": baseline_sats,
            "train_human_ids": train_human_ids,
            "new_human_id": new_human_id,
            "test_recipes": test_recipes,
        },
    )
