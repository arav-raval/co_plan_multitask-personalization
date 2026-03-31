"""Tests for spices_csp.py."""

import logging
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


# --- TEMP DEBUG: set True locally, or delete this block + dbg_* uses in test_spices_csp_single_recipe ---
_SINGLE_RECIPE_DEBUG = True


def _psi_running_convergence_step(psi_trace: list[float], eps: float = 0.02) -> str:
    """
    First 1-based timestep where running_psi changes by less than eps for two steps in a row.
    Returns '--' if not reached (or trace too short).
    """
    if len(psi_trace) < 3:
        return "--"
    for k in range(2, len(psi_trace)):
        if abs(psi_trace[k] - psi_trace[k - 1]) < eps and abs(
            psi_trace[k - 1] - psi_trace[k - 2]
        ) < eps:
            return str(k + 1)
    return "--"


# --- end TEMP DEBUG ---


# ---------------------------------------------------------------------------
# test_spices_csp_single_recipe
# ---------------------------------------------------------------------------

def test_spices_csp_single_recipe():
    """Single Recipe | Multi-Episode | Multi-Seed.

    Metrics logged:
        - Psi/mood alignment rate by episode quartile (all seeds)
        - Running_psi trajectory for mood episodes (last seed, first 10 eps)
        - Average satisfaction by mood
        - Phi sign accuracy and learned vs. true theta (top spices, final seed)
        - Learned hyperparameters (sigma_h, sigma_r, sigma_obs)

    Why average_satisfaction is often small (e.g. 0.1–0.2):
        Per step, the env maps logit -> tanh(logit) in [-1,1], then samples a Beta
        around p=(tanh+1)/2 with concentration kappa, then maps to [-1,1]. That
        adds noise and shrinks extremes toward 0. Episode average_satisfaction
        is the mean over steps; mixing neutral and non-neutral moods, partial
        episodes, and policies that do not always match the latent optimum all
        pull the aggregate toward a modest positive or near-zero mean.
    """
    num_seeds    = PARAMETERS["num_seeds"]
    num_episodes = PARAMETERS["num_episodes"]
    recipe_name  = PARAMETERS["recipe_name"]
    recipe       = get_recipe(recipe_name)
    spices       = list(recipe.spices)
    config_name  = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")
    hidden_hbm   = _create_hidden_hbm(spices, [recipe_name], config_name=config_name)
    sig_spices   = _signal_spices(hidden_hbm, spices)

    # (seed, ep, mood, avg_sat, inferred_mood, batch_psi_m)
    all_records: list[tuple] = []
    neutral_match_rates: list[float] = []
    neutral_oracle_gaps: list[float] = []
    final_hbm: HierarchicalPreferenceModel | None = None

    num_psi_log_eps = min(num_episodes, 10)

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

        if log_this_seed:
            logging.info(f"\n{'='*60}")
            logging.info(f"[SingleRecipe] recipe={recipe_name}  config={config_name}  seed={i}")
            logging.info(f"  Signal spices: {len(sig_spices)} / {len(spices)}")
            logging.info(f"\n  --- Psi Trajectory (first {num_psi_log_eps} eps) ---")
            logging.info(f"  {'Ep':>4}  {'mood':10}  {'exp':>4}  {'aligned':>7}  "
                         f"{'psi_m':>8}  {'peak':>8}  trajectory")

        # Last occurrence of each mood in this seed (for TEMP DEBUG dump after training).
        dbg_last_episode_by_mood: dict[str, tuple[int, list[float], list[float]]] = {}
        if log_this_seed and _SINGLE_RECIPE_DEBUG:
            logging.info(
                "\n  --- TEMP DEBUG (last seed): will log *last* episode per mood after run | "
                "psi_conv_step = first step where |Δrunning_psi| < 0.02 twice ---"
            )

        for ep in range(num_episodes):
            obs, _ = env.reset()
            assert isinstance(obs, SpiceState)

            step_psi: list[float] = []
            step_sats: list[float] = []
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
                step_psi.append(hbm.get_running_psi(DEFAULT_HUMAN))
                step_sats.append(float(info.get("satisfaction", 0.0)))
                # TEMP diagnostics for neutral episodes: action match rate and
                # expected satisfaction gap vs. oracle actor choice.
                preferred_actor = str(info.get("preferred_actor"))
                last_actor = str(info.get("last_actor"))
                if preferred_actor in ("human", "robot") and last_actor in ("human", "robot"):
                    step_match_flags.append(last_actor == preferred_actor)
                    pref_sign = 1.0 if preferred_actor == "human" else -1.0
                    base = float(env._base_satisfaction_bias)
                    psi_true = float(env._current_psi_true)
                    phi_latent = pref_sign * base
                    logit_h = 1.0 * (phi_latent + psi_true)
                    logit_r = -1.0 * (phi_latent + psi_true)
                    exp_h = float(np.tanh(logit_h))
                    exp_r = float(np.tanh(logit_r))
                    actual_exp = exp_h if last_actor == "human" else exp_r
                    oracle_exp = max(exp_h, exp_r)
                    step_oracle_minus_actual.append(oracle_exp - actual_exp)
                if done:
                    break

            mood       = info["mood"]
            psi_m      = hbm.get_psi_m(DEFAULT_HUMAN)
            inferred   = max(info["mood_posterior"], key=info["mood_posterior"].get)
            all_records.append((i, ep, mood, info["average_satisfaction"], inferred, psi_m))
            if mood == "neutral" and step_match_flags:
                neutral_match_rates.append(float(np.mean(step_match_flags)))
            if mood == "neutral" and step_oracle_minus_actual:
                neutral_oracle_gaps.append(float(np.mean(step_oracle_minus_actual)))

            if log_this_seed and ep < num_psi_log_eps:
                in_ep  = [v for v in step_psi if v != 0.0] or step_psi[:1]
                peak   = max(in_ep, key=abs)
                exp    = "+" if mood == "all_self" else ("-" if mood == "none_self" else "~")
                ok     = "✓" if _psi_alignment(psi_m, mood) else "✗"
                traj   = "  ".join(f"{v:+.2f}" for v in step_psi)
                logging.info(f"  {ep+1:4d}  {mood:10}  {exp:>4}  {ok:>7}  "
                             f"{psi_m:+8.4f}  {peak:+8.4f}  [{traj}]")

            if log_this_seed and _SINGLE_RECIPE_DEBUG:
                dbg_last_episode_by_mood[mood] = (
                    ep + 1,
                    list(step_psi),
                    list(step_sats),
                )

        if log_this_seed and _SINGLE_RECIPE_DEBUG:
            logging.info("\n  --- TEMP DEBUG: last episode per mood (same last seed) ---")
            for m in ("neutral", "all_self", "none_self"):
                if m not in dbg_last_episode_by_mood:
                    logging.info(f"  [TEMP DEBUG] no '{m}' episode in this run")
                    continue
                ep_num, step_psi, step_sats = dbg_last_episode_by_mood[m]
                conv = _psi_running_convergence_step(step_psi)
                sat_str = "[" + ", ".join(f"{s:+.3f}" for s in step_sats) + "]"
                psi_str = "[" + ", ".join(f"{p:+.3f}" for p in step_psi) + "]"
                logging.info(
                    f"  [TEMP DEBUG] last '{m}' ep={ep_num}  "
                    f"psi_conv_step(approx)={conv}  n_steps={len(step_sats)}"
                )
                logging.info(f"  [TEMP DEBUG]   satisfaction (per step): {sat_str}")
                logging.info(f"  [TEMP DEBUG]   running_psi (after each observe): {psi_str}")

        env.close()
        if log_this_seed:
            final_hbm = hbm

    # ── Psi/mood alignment by episode quartile ──────────────────────────────
    q = num_episodes // 4
    quartile_labels = [f"Q1(1-{q})", f"Q2({q+1}-{2*q})", f"Q3({2*q+1}-{3*q})", f"Q4({3*q+1}-{num_episodes})"]
    quartile_ranges = [(0, q), (q, 2*q), (2*q, 3*q), (3*q, num_episodes)]

    mood_records = [(ep, mood, psi_m) for (_, ep, mood, _, _, psi_m) in all_records if mood != "neutral"]
    logging.info(f"\n  --- Psi/Mood Alignment by Quartile (non-neutral eps, all {num_seeds} seeds) ---")
    for label, (lo, hi) in zip(quartile_labels, quartile_ranges):
        subset = [(mood, psi_m) for (ep, mood, psi_m) in mood_records if lo <= ep < hi]
        if not subset:
            logging.info(f"  {label}: no mood episodes")
            continue
        aligned = sum(_psi_alignment(psi_m, mood) for mood, psi_m in subset)
        logging.info(f"  {label}: {aligned}/{len(subset)} = {aligned/len(subset):.0%} aligned")

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
    if neutral_oracle_gaps:
        window = 10
        logging.info(
            f"\n  --- Neutral Oracle Gap Progression (avg per {window} neutral episodes) ---"
        )
        for i in range(0, len(neutral_oracle_gaps), window):
            chunk = neutral_oracle_gaps[i : i + window]
            lo = i + 1
            hi = i + len(chunk)
            logging.info(
                f"  neutral_eps[{lo:>3}-{hi:>3}] : oracle_minus_actual={np.mean(chunk):+.3f}  (n={len(chunk)})"
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
        for s in sig_spices[:10]:
            true_t  = hidden_hbm.theta_mean.get(s, 0.0)
            learned = final_hbm.phi_mean.get(recipe_name, {}).get(s, 0.0)
            phi_var = final_hbm.get_phi_var(DEFAULT_HUMAN, recipe_name, s)
            correct = "✓" if np.sign(learned) == np.sign(true_t) else "✗"
            logging.info(f"  {s:<20} {true_t:+10.3f} {learned:+12.3f} {phi_var:8.4f} {correct}")


# ---------------------------------------------------------------------------
# test_spices_csp_cross_transfer
# ---------------------------------------------------------------------------

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
    config_name   = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")

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

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] - i
        solver   = EnumerationCSPSolver(csp_seed)

        trained_gen = SpicesAssignCSPGenerator(
            spice_list=all_spices, recipe_list=train_recipes, seed=csp_seed
        )

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
                if info.get("mood") == "neutral":
                    trained_sats_neutral[recipe_name].append(avg_sat)
                else:
                    trained_sats_non_neutral[recipe_name].append(avg_sat)
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
                if info.get("mood") == "neutral":
                    baseline_sats_neutral[recipe_name].append(avg_sat)
                else:
                    baseline_sats_non_neutral[recipe_name].append(avg_sat)
            env.close()

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


# ---------------------------------------------------------------------------
# test_spices_csp_multi_human
# ---------------------------------------------------------------------------

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
    config_name = PARAMETERS["hidden_hbm_config_name"]

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

    hidden_hbms = {
        h: _create_hidden_hbm(all_spices, all_recipes, config_name=config_name)
        for h in all_human_ids
    }
    sig_spices = _signal_spices(hidden_hbms["h1"], all_spices)

    logging.info(f"\n{'='*60}")
    logging.info(f"[MultiHuman] profile={profile_spec.name}  config={config_name}")
    logging.info(f"  Train: {len(train_recipes)} recipes  |  Test: {len(test_recipes)} recipes  "
                 f"|  Humans: {train_human_ids} + new={new_human_id}  "
                 f"|  Signal spices: {len(sig_spices)}")

    trained_sats  = {h: {r: [] for r in test_recipes} for h in all_human_ids}
    baseline_sats = {h: {r: [] for r in test_recipes} for h in all_human_ids}
    post_train_theta_acc: dict[str, tuple[int, int]] = {}
    new_human_theta_acc: tuple[int, int] | None = None
    shared_hbm: HierarchicalPreferenceModel | None = None

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] - i
        solver   = EnumerationCSPSolver(csp_seed)

        shared_hbm = HierarchicalPreferenceModel(spices=all_spices)
        gens = {
            h: SpicesAssignCSPGenerator(
                spice_list=all_spices, recipe_list=train_recipes,
                seed=csp_seed, human_id=h, shared_hbm=shared_hbm,
            )
            for h in train_human_ids
        }

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
                env.close()

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


# ---------------------------------------------------------------------------
# test_mood_inference
# ---------------------------------------------------------------------------

def test_mood_inference():
    """Unit test: verify correct mood inference by episode end for each mood type."""
    MOODS = ("all_self", "neutral", "none_self")
    recipe_name = PARAMETERS["recipe_name"]
    recipe      = get_recipe(recipe_name)
    spices      = list(recipe.spices)
    config_name = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")
    hidden_hbm  = _create_hidden_hbm(spices, [recipe_name], config_name=config_name)

    env_seed = PARAMETERS["env_seed"]
    csp_seed = PARAMETERS["csp_seed"]
    assert env_seed != csp_seed

    mood_outcomes: dict[str, str] = {}
    offset = 0

    while set(mood_outcomes.keys()) != set(MOODS):
        env         = _make_env(env_seed + offset, recipe_name, hidden_hbm)
        csp_gen     = SpicesAssignCSPGenerator(spice_list=spices, recipe_list=[recipe_name], seed=csp_seed)
        solver      = EnumerationCSPSolver(csp_seed)
        max_steps   = len(spices) + 5

        obs, _ = env.reset()
        assert isinstance(obs, SpiceState)

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
            if done:
                break
        env.close()

        true_mood     = info["mood"]
        inferred_mood = max(info["mood_posterior"], key=info["mood_posterior"].get)
        if true_mood not in mood_outcomes:
            mood_outcomes[true_mood] = inferred_mood
        offset += 1

    logging.info(f"\n  --- Mood Inference Results ---")
    for mood in MOODS:
        inferred = mood_outcomes.get(mood, "N/A")
        ok = "✓" if inferred == mood else "✗"
        logging.info(f"  {mood:10}: inferred={inferred:10}  {ok}")
