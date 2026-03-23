"""Tests for spices_csp.py."""

import logging
import numpy as np
import pytest
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import cosine

from multitask_personalization.csp_solvers import EnumerationCSPSolver
from multitask_personalization.envs.spices.spices_csp import SpicesAssignCSPGenerator
from multitask_personalization.envs.spices.spices_env import (
    SpiceEnv, SpiceState, SpiceSceneSpec, SpiceHiddenSpec, RecipeSpec
)
from multitask_personalization.envs.spices.recipes import get_recipe, get_profile
from collections import Counter

from multitask_personalization.envs.spices.spices_hbm import (
    DEFAULT_HUMAN,
    HierarchicalPreferenceModel,
)
from multitask_personalization.envs.spices.config import (
    get_hidden_hbm_config,
    create_theta_params_from_config,
    list_hidden_hbm_configs,
    PARAMETERS,
)

# ---- Helper functions ----
def _create_hidden_hbm(spices: list, recipes: list, config_name: str):
    """Build an HBM from a hidden config for testing."""
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
    hbm._theta_mean[DEFAULT_HUMAN] = theta_mean
    hbm._theta_var[DEFAULT_HUMAN] = {s: cfg.sigma_h**2 for s in spices}
    return hbm

def _make_env(env_seed: int, recipe_name: str, hidden_hbm, training: bool = False):
    """Build a SpiceEnv for testing."""
    recipe = get_recipe(recipe_name)
    scene_spec = SpiceSceneSpec(recipe=recipe)
    hidden_spec = SpiceHiddenSpec(
        preferred_actor={},
        hidden_hbm=hidden_hbm,
        force_neutral_mood=training,
    )
    return SpiceEnv(scene_spec, hidden_spec, seed=env_seed)

# ---- Test Functions
def test_spices_csp_single_recipe():
    """Single Recipe | Multi-Episode | Multi-Seed learning curves

    Tracks:
        - Split satisfaction by mood (neutral / all_self / none_self).
        - Mood inference accuracy
        - Theta-sign accuracy and phi convergence (final seed, signal spices)
    """
    num_seeds    = PARAMETERS["num_seeds"]
    num_episodes = PARAMETERS["num_episodes"]
    recipe_name  = PARAMETERS["recipe_name"]
    recipe       = get_recipe(recipe_name)
    spices       = list(recipe.spices)

    config_name = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")
    hidden_hbm  = _create_hidden_hbm(spices, [recipe_name], config_name=config_name)

    # (seed_idx, ep_idx, true_mood, avg_satisfaction, inferred_mood)
    all_records: list[tuple] = []

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] + i
        assert env_seed != csp_seed

        env           = _make_env(env_seed, recipe_name, hidden_hbm)
        csp_generator = SpicesAssignCSPGenerator(spice_list=spices, recipe_list=[recipe_name], seed=csp_seed)
        solver        = EnumerationCSPSolver(csp_seed)
        max_steps     = len(spices) + 1

        for ep in range(num_episodes):
            obs, _ = env.reset()
            assert isinstance(obs, SpiceState)
            assert obs.current_spice in obs.feasible_next

            for _ in range(max_steps):
                prev_obs = obs
                csp, samplers, policy, initialization = csp_generator.generate(obs)
                sol = solver.solve(csp, initialization, samplers)
                assert sol is not None
                policy.reset(sol)
                act = policy.step(obs)
                obs, reward, done, truncated, info = env.step(act)
                assert np.isclose(reward, 0.0)
                csp_generator.observe_transition(prev_obs, act, obs, done, info)
                if done:
                    break

            inferred_mood = max(info["mood_posterior"], key=info["mood_posterior"].get)
            all_records.append((i, ep, info["mood"], info["average_satisfaction"], inferred_mood))

        env.close()

    # Satisfaction separated by mood
    for mood in ["neutral", "all_self", "none_self"]:
        ep_sats: dict[int, list[float]] = {}
        for (_, ep, m, sat, _) in all_records:
            if m == mood:
                ep_sats.setdefault(ep, []).append(sat)
        if not ep_sats:
            continue
        logging.info(f"\n---- Learning Curve ({mood}) | {recipe_name} ----")
        for ep in sorted(ep_sats):
            sats = ep_sats[ep]
            logging.info(f"  Ep {ep+1:3d} (n={len(sats)}): avg_sat={np.mean(sats):+.3f}")

    # Average Satisfaction
    average_satisfaction = np.mean([r[3] for r in all_records]) 
    logging.info(f"\n---- Average Satisfaction (All Moods): {average_satisfaction:+.3f}")

    # Mood inference accuracy
    mood_correct = sum(1 for (_, _, mood, _, inferred) in all_records if mood == inferred)
    logging.info(
        f"\n---- Mood Inference: {mood_correct}/{len(all_records)} "
        f"= {mood_correct / len(all_records):.1%} ----"
    )

    # Theta and Phi Convergence
    if hidden_hbm is not None:
        hbm           = csp_generator._pref_gen._hbm
        signal_spices = [s for s in spices if abs(hidden_hbm.theta_mean.get(s, 0.0)) > 0.3]
        if signal_spices:
            correct = sum(
                np.sign(hbm.phi_mean[recipe_name].get(s, 0.0)) == np.sign(hidden_hbm.theta_mean.get(s, 0.0))
                for s in signal_spices
            )
            logging.info(
                f"\n---- Theta-Sign Accuracy (final seed): "
                f"{correct}/{len(signal_spices)} = {correct / len(signal_spices):.1%} ----"
            )
            logging.info("\n---- Phi Convergence (final seed, first 5 signal spices) ----")
            for s in signal_spices[:5]:
                learned = hbm.phi_mean[recipe_name].get(s, 0.0)
                true    = hidden_hbm.theta_mean.get(s, 0.0)
                logging.info(f"  {s:20s}: learned={learned:+.3f}  true_theta={true:+.3f}")

def test_spices_csp_cross_transfer():
    """Single Human | Train on K recipes | Evaluate on M held-out recipes vs. cold-start baseline.

    Training uses forced-neutral mood (100% learning efficiency).
    Evaluation uses real mood; satisfaction is recorded only for neutral episodes so the
    comparison is uncontaminated by mood overrides.

    Metrics:
        - Per-test-recipe and aggregate average satisfaction: trained vs. baseline
        - Theta-sign accuracy on test recipes (final seed)
    """
    num_seeds       = PARAMETERS["num_seeds"]
    num_train_eps   = PARAMETERS["num_episodes"] 
    num_test_eps    = PARAMETERS["num_test_episodes"]
    train_frac      = PARAMETERS["train_frac"]

    profile_spec  = get_profile(PARAMETERS["profile"])
    all_recipes   = list(profile_spec.recipes)
    n_train       = int(np.ceil(len(all_recipes) * train_frac))
    train_recipes = all_recipes[:n_train]
    test_recipes  = all_recipes[n_train:]

    all_spices = sorted({s for r in all_recipes for s in get_recipe(r).spices})

    logging.info(f"\n{'='*60}")
    logging.info(f"[Cross-Transfer] profile={profile_spec.name}")
    logging.info(f"  Train ({len(train_recipes)}): {train_recipes}")
    logging.info(f"  Test  ({len(test_recipes)}):  {test_recipes}")
    logging.info(f"  Vocabulary: {len(all_spices)} spices")

    config_name = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")
    hidden_hbm  = _create_hidden_hbm(all_spices, all_recipes, config_name=config_name)
    logging.info(f"  Preference Configuration: {config_name}")

    trained_sats         = {r: [] for r in test_recipes}
    baseline_sats        = {r: [] for r in test_recipes}
    post_train_theta_acc = None  # (correct, total) snapshotted after training, before eval

    max_steps = max(len(get_recipe(r).spices) for r in all_recipes) + 5

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] - i
        solver   = EnumerationCSPSolver(csp_seed)

        trained_gen = SpicesAssignCSPGenerator(
            spice_list=all_spices, recipe_list=train_recipes, seed=csp_seed
        )

        # Training (mood = False)
        for recipe_name in train_recipes:
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

        # Post-training theta accuracy (spices seen in training)
        if hidden_hbm is not None:
            post_train_hbm = trained_gen._pref_gen._hbm
            post_train_hbm.flush_theta_mu()
            train_vocab     = {s for r in train_recipes for s in get_recipe(r).spices}
            train_signal    = [
                s for s in train_vocab
                if abs(hidden_hbm.theta_mean.get(s, 0.0)) > 0.3
            ]
            post_train_theta_acc = (
                sum(
                    np.sign(post_train_hbm.theta_mean.get(s, 0.0))
                    == np.sign(hidden_hbm.theta_mean.get(s, 0.0))
                    for s in train_signal
                ),
                len(train_signal),
            )

        # Evaluation (all moods); only neutral episodes reported
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
                if info.get("mood") == "neutral":
                    trained_sats[recipe_name].append(info["average_satisfaction"])
            env.close()

        # Baseline: fresh generator + fresh solver, each test recipe in isolation.
        # Fresh solver avoids RNG contamination from the training + eval steps above.
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
                if info.get("mood") == "neutral":
                    baseline_sats[recipe_name].append(info["average_satisfaction"])
            env.close()

    # Log metrics
    logging.info(f"\n{'='*60}")
    logging.info(f"[Results] trained vs. baseline (neutral episodes only)")
    trained_all, baseline_all = [], []
    for recipe_name in test_recipes:
        t = trained_sats[recipe_name]
        b = baseline_sats[recipe_name]
        trained_all.extend(t)
        baseline_all.extend(b)
        logging.info(
            f"  {recipe_name:28s}: "
            f"trained={np.mean(t) if t else float('nan'):+.3f} (n={len(t)})  "
            f"baseline={np.mean(b) if b else float('nan'):+.3f} (n={len(b)})  "
            f"delta={np.mean(t) - np.mean(b) if t and b else float('nan'):+.3f}"
        )
    if trained_all and baseline_all:
        logging.info(
            f"\n  AGGREGATE: trained={np.mean(trained_all):+.3f}  "
            f"baseline={np.mean(baseline_all):+.3f}  "
            f"delta={np.mean(trained_all) - np.mean(baseline_all):+.3f}"
        )

    if hidden_hbm is not None:
        hbm = trained_gen._pref_gen._hbm

        # ── Post-training theta accuracy (per-human preference) ---
        if post_train_theta_acc is not None:
            correct_t, total_t = post_train_theta_acc
            logging.info(
                f"\n  Post-Training Theta Accuracy (final seed, {total_t} signal spices): "
                f"{correct_t}/{total_t} = {correct_t / total_t:.1%}"
            )

        # ── Post-eval accuracy per test recipe  ---
        logging.info(f"\n  Post-Eval Phi Accuracy per test recipe (final seed):")
        for recipe_name in test_recipes:
            r_spices      = list(get_recipe(recipe_name).spices)
            signal_spices = [s for s in r_spices if abs(hidden_hbm.theta_mean.get(s, 0.0)) > 0.3]
            if signal_spices:
                correct = sum(
                    np.sign(hbm.phi_mean.get(recipe_name, {}).get(s, 0.0))
                    == np.sign(hidden_hbm.theta_mean.get(s, 0.0))
                    for s in signal_spices
                )
                logging.info(
                    f"    {recipe_name:28s}: "
                    f"{correct}/{len(signal_spices)} = {correct / len(signal_spices):.1%}"
                )

def test_spices_csp_multi_human():
    """Multi-Human | Shared HBM | Population pooling.

    Two humans (h1, h2) train on the same K recipes using a single shared
    HierarchicalPreferenceModel.  Their observations jointly update a shared
    population mean μ.  After training, a third human (h3) is registered from
    the learned mu

    Metrics:
        - Per-trained-human theta accuracy after training (h1, h2).
        - Population sign accuracy vs. true preferences.
        - h3 initial theta accuracy
        - Satisfaction on test recipes: h1/h2 trained vs. cold baseline.
        - Satisfaction on test recipes: h3 vs. cold baseline.
    """
    num_seeds     = PARAMETERS["num_seeds"]

    profile_spec  = get_profile(PARAMETERS["profile"])
    all_recipes   = list(profile_spec.recipes)
    n_train       = int(np.ceil(len(all_recipes) * PARAMETERS["train_frac"]))
    train_recipes = all_recipes[:n_train]
    test_recipes  = all_recipes[n_train:]
    all_spices    = sorted({s for r in all_recipes for s in get_recipe(r).spices})

    num_humans = PARAMETERS["num_humans"]
    train_human_ids = [f"h{i + 1}" for i in range(num_humans - 1)]
    new_human_id    = f"h{num_humans}"
    all_human_ids   = train_human_ids + [new_human_id]

    # Scale to number of humans 
    num_train_eps = max(1, PARAMETERS["num_episodes"] // len(train_human_ids))
    num_test_eps  = max(1, PARAMETERS["num_test_episodes"] // len(all_human_ids))

    config_name = PARAMETERS["hidden_hbm_config_name"]
    hidden_hbms = {
        h: _create_hidden_hbm(all_spices, all_recipes, config_name=config_name)
        for h in all_human_ids
    }

    logging.info(f"\n{'='*60}")
    logging.info(f"[Multi-Human] profile={profile_spec.name}  config={config_name}")
    logging.info(f"  Train ({len(train_recipes)}): {train_recipes}")
    logging.info(f"  Test  ({len(test_recipes)}):  {test_recipes}")
    logging.info(f"  Trained humans: {train_human_ids}  New human: {new_human_id}")
    logging.info(f"  Vocabulary: {len(all_spices)} spices")

    trained_sats  = {h: {r: [] for r in test_recipes} for h in all_human_ids}
    baseline_sats = {h: {r: [] for r in test_recipes} for h in all_human_ids}
    post_train_theta_acc = {h: None for h in train_human_ids}
    new_human_theta_acc  = None  

    max_steps = max(len(get_recipe(r).spices) for r in all_recipes) + 5

    for i in range(num_seeds):
        env_seed = PARAMETERS["env_seed"] + i
        csp_seed = PARAMETERS["csp_seed"] - i
        solver   = EnumerationCSPSolver(csp_seed)

        shared_hbm = HierarchicalPreferenceModel(spices=all_spices)

        gens = {
            h: SpicesAssignCSPGenerator(
                spice_list=all_spices,
                recipe_list=train_recipes,
                seed=csp_seed,
                human_id=h,
                shared_hbm=shared_hbm,
            )
            for h in train_human_ids
        }

        # Training
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

        train_vocab  = {s for r in train_recipes for s in get_recipe(r).spices}
        train_signal = [
            s for s in train_vocab
            if abs(hidden_hbms["h1"].theta_mean.get(s, 0.0)) > 0.3
        ]
        for h in train_human_ids:
            post_train_theta_acc[h] = (
                sum(
                    np.sign(shared_hbm._theta_mean[h].get(s, 0.0))
                    == np.sign(hidden_hbms[h].theta_mean.get(s, 0.0))
                    for s in train_signal
                ),
                len(train_signal),
            )

        # Register h3
        shared_hbm.register_human(new_human_id)
        gens[new_human_id] = SpicesAssignCSPGenerator(
            spice_list=all_spices,
            recipe_list=train_recipes,
            seed=csp_seed,
            human_id=new_human_id,
            shared_hbm=shared_hbm,
        )

        new_human_theta_acc = (
            sum(
                np.sign(shared_hbm._theta_mean[new_human_id].get(s, 0.0))
                == np.sign(hidden_hbms[new_human_id].theta_mean.get(s, 0.0))
                for s in train_signal
            ),
            len(train_signal),
        )

        # Evaluation
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

        # Baseline
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

    # Metrics Summary
    logging.info(f"\n{'='*60}")
    logging.info(f"[Results] trained vs. baseline (neutral episodes only)")

    for h in all_human_ids:
        label = "μ-warm (new)" if h == new_human_id else "trained"
        t_all = [s for r in test_recipes for s in trained_sats[h][r]]
        b_all = [s for r in test_recipes for s in baseline_sats[h][r]]
        logging.info(
            f"  {h} ({label:14s}): "
            f"trained={np.mean(t_all) if t_all else float('nan'):+.3f} (n={len(t_all)})  "
            f"baseline={np.mean(b_all) if b_all else float('nan'):+.3f} (n={len(b_all)})  "
            f"delta={np.mean(t_all) - np.mean(b_all) if t_all and b_all else float('nan'):+.3f}"
        )

    logging.info(f"\n  Post-Training Theta Accuracy (final seed, {len(train_signal)} signal spices):")
    for h in train_human_ids:
        if post_train_theta_acc[h] is not None:
            c, tot = post_train_theta_acc[h]
            logging.info(f"    {h}: {c}/{tot} = {c/tot:.1%}")

    if new_human_theta_acc is not None:
        c, tot = new_human_theta_acc
        logging.info(
            f"\n  h{num_humans} Initial Theta Accuracy (μ warm-start, zero episodes): "
            f"{c}/{tot} = {c/tot:.1%}"
        )

    mu_correct = sum(
        np.sign(shared_hbm.get_mu(s)) == np.sign(hidden_hbms[train_human_ids[0]].theta_mean.get(s, 0.0))
        for s in train_signal
    )
    logging.info(
        f"\n  Population μ Sign Accuracy (final seed): "
        f"{mu_correct}/{len(train_signal)} = {mu_correct/len(train_signal):.1%}"
    )

    # Report global μ and θ for each human (top 12 signal spices by |true θ|).
    top_signal = sorted(
        train_signal,
        key=lambda s: abs(hidden_hbms["h1"].theta_mean.get(s, 0.0)),
        reverse=True,
    )[:12]
    logging.info(f"\n  Learned μ and θ (final seed, top 12 signal spices):")
    logging.info(f"  {'spice':<18} | {'true':>6} | {'μ':>6} | " + " | ".join(f"{h:>6}" for h in all_human_ids))
    for s in top_signal:
        true_val = hidden_hbms["h1"].theta_mean.get(s, 0.0)
        mu_val   = shared_hbm.get_mu(s)
        theta_vals = [shared_hbm._theta_mean[h].get(s, 0.0) for h in all_human_ids]
        theta_str = " | ".join(f"{v:+.2f}" for v in theta_vals)
        logging.info(f"  {s:<18} | {true_val:+.2f} | {mu_val:+.2f} | {theta_str}")

def test_mood_inference():
    """
    Basic unit test for correct mood inference by end of episode for each mood type
    """
    MOODS = ("all_self", "neutral", "none_self")
    mood_outcomes = {}

    # Seeds
    env_seed = PARAMETERS["env_seed"]
    csp_seed = PARAMETERS["csp_seed"]
    offset = 0
    assert env_seed != csp_seed

    # Recipe
    recipe_name = PARAMETERS["recipe_name"]
    recipe = get_recipe(recipe_name)
    spices = list(recipe.spices)

    # Hidden HBM Model
    config_name = PARAMETERS.get("hidden_hbm_config_name", "AlternatingHuman")
    hidden_hbm = _create_hidden_hbm(spices, [recipe_name], config_name=config_name)

    while set(mood_outcomes.keys()) != set(MOODS):
        # Environment (generates SpiceSceneSpec, SpiceHiddenSpec, SpiceEnv)
        env = _make_env(env_seed + offset, recipe_name, hidden_hbm)
        obs, _ = env.reset()
        assert isinstance(obs, SpiceState)
        assert obs.current_spice in obs.feasible_next

        # Create the CSP
        csp_generator = SpicesAssignCSPGenerator(
            spice_list=spices,
            recipe_list=[recipe_name],
            seed=csp_seed,
        )

        # Create the solver
        solver = EnumerationCSPSolver(csp_seed)

        max_steps = len(spices) + 5

        # Iterate through recipe
        for _ in range(max_steps):
            prev_obs = obs

            # Regenerate spice-specific CSP
            csp, samplers, policy, initialization = csp_generator.generate(obs)    
            sol = solver.solve(
                csp, 
                initialization,
                samplers
            )
            assert sol is not None
            policy.reset(sol)

            act = policy.step(obs)
            obs, reward, done, truncated, info = env.step(act)

            assert isinstance(obs, SpiceState)
            assert np.isclose(reward, 0.0)

            # Update mood posteriors
            csp_generator.observe_transition(prev_obs, act, obs, done, info)

            if done: 
                break
        env.close()

        inferred_mood = max(info['mood_posterior'], key=info['mood_posterior'].get)
        true_mood = info['mood']
        
        if true_mood not in mood_outcomes.keys():
            mood_outcomes[true_mood] = inferred_mood

        offset += 1

    logging.info("\n --- Testing Mood Inference ---")
    logging.info(mood_outcomes)