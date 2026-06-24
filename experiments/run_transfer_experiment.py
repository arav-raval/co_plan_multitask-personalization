"""Cross-human transfer experiment
Phase 1: Population Training 
Phase 2: Transfer evaluation (introduce a new human)

Usage:
  python experiments/run_transfer_experiment.py \
    --approach ours --seed 0 \
    --phase1-steps 3000 --phase2-steps 2000 \
    --eval-frequency 250 --num-eval-trials 50

Outputs are written to logs/<date>/<time>/ with:
  - phase1_train_results.csv   (population training metrics)
  - phase2_train_results.csv   (new-human training metrics)
  - phase2_eval_results.csv    (new-human eval metrics — the main comparison)
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from multitask_personalization.csp_solvers import EnumerationCSPSolver
from multitask_personalization.envs.spices.config.spices_config import DEFAULT_CONFIG
from multitask_personalization.envs.spices.spices_env import SpiceEnv, SpiceHiddenSpec
from multitask_personalization.envs.spices.spices_experiment import (
    build_spice_experiment_hidden_hbm,
    build_spice_scene_spec_multi,
    MULTI_RECIPE_EXPERIMENT_POOL,
    POPULATION_HUMAN_CONFIGS,
    NEW_HUMAN_CONFIG,
)
from multitask_personalization.envs.spices.spices_hbm import (
    HierarchicalPreferenceModel,
)
from multitask_personalization.methods.csp_approach import CSPApproach

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-human transfer experiment")
    p.add_argument("--approach", default="ours", choices=["ours", "without_mood_learning", "cbtl_adapted", "flat_model"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase1-steps", type=int, default=3000, help="Training steps per population human")
    p.add_argument("--phase2-steps", type=int, default=2000, help="Training steps on new human")
    p.add_argument("--eval-frequency", type=int, default=250)
    p.add_argument("--num-eval-trials", type=int, default=50)
    p.add_argument("--max-eval-episode-length", type=int, default=40)
    p.add_argument("--hidden-hbm-config", default="SpiceSpecificHumanRecipeConflict")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    return p.parse_args()


def _build_approach(approach_name: str, scene_spec: Any, action_space: Any, seed: int) -> Any:
    """Build a CSPApproach with the right preference model type."""

    preference_model_map = {
        "ours": ("hbm", "max-entropy", True),
        "without_mood_learning": ("hbm", "max-entropy", False),
        "cbtl_adapted": ("cbtl", "exploit-only", True),
        "flat_model": ("flat", "exploit-only", True),
    }
    pref_type, explore, mood_enabled = preference_model_map[approach_name]

    return CSPApproach(
        scene_spec=scene_spec,
        action_space=action_space,
        csp_solver=EnumerationCSPSolver(seed=seed),
        explore_method=explore,
        disable_learning=False,
        mood_learning_enabled=mood_enabled,
        preference_model_type=pref_type,
        seed=seed,
    )


def _run_training_phase(
    env: Any,
    approach: Any,
    n_steps: int,
    sync_scene_spec: bool = True,
    human_id: str = "human",
) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []

    # Register this human with the preference model
    try:
        pref_gen = approach._csp_generator._pref_gen
        pref_gen._human_id = human_id
        hbm = pref_gen._hbm
        if hasattr(hbm, "register_human"):
            hbm.register_human(human_id)
        if hasattr(hbm, "register_recipe"):
            for recipe in list(pref_gen._recipe_list):
                hbm.register_recipe(human_id, recipe)
    except AttributeError:
        pass

    obs, info = env.reset()
    approach.reset(obs, info)
    if sync_scene_spec:
        approach._scene_spec = env.unwrapped.scene_spec

    for t in range(n_steps):
        act = approach.step()
        obs, rew, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)
        approach.update(obs, float(rew), done, info)

        metrics.append({
            "step": t,
            "human_id": human_id,
            "user_satisfaction": info.get("user_satisfaction", float("nan")),
            "prediction_accuracy": info.get("prediction_accuracy", float("nan")),
        })

        if done:
            obs, info = env.reset()
            approach.reset(obs, info)
            if sync_scene_spec:
                approach._scene_spec = env.unwrapped.scene_spec

    return metrics


def _run_eval(
    eval_approach: Any,
    eval_env: Any,
    num_trials: int,
    max_episode_length: int,
    training_step: int,
    force_neutral_mood: bool,
    metric_prefix: str = "",
) -> dict[str, float]:
    """Run evaluation trials and return aggregated metrics."""
    satisfactions: list[float] = []
    pred_accuracies: list[float] = []

    for trial in range(num_trials):
        seed = 100000 + trial
        obs, info = eval_env.reset(seed=seed)
        eval_approach.reset(obs, info)
        eval_approach._scene_spec = eval_env.unwrapped.scene_spec

        # Set mood mode
        original_fnm = eval_env.unwrapped._hidden_spec.force_neutral_mood
        eval_env.unwrapped._hidden_spec.force_neutral_mood = force_neutral_mood

        episode_sat = 0.0
        n_steps = 0
        for _ in range(max_episode_length):
            act = eval_approach.step()
            obs, rew, terminated, truncated, info = eval_env.step(act)
            eval_approach.update(obs, float(rew), bool(terminated or truncated), info)
            episode_sat += info.get("user_satisfaction", 0.0)
            n_steps += 1
            if terminated or truncated:
                break

        eval_env.unwrapped._hidden_spec.force_neutral_mood = original_fnm

        avg_sat = episode_sat / max(n_steps, 1)
        satisfactions.append(avg_sat)
        pred_acc = info.get("prediction_accuracy", float("nan"))
        if not math.isnan(pred_acc):
            pred_accuracies.append(pred_acc)

    result = {
        "training_step": training_step,
        f"{metric_prefix}eval_mean_user_satisfaction": float(np.mean(satisfactions)),
    }
    if pred_accuracies:
        result[f"{metric_prefix}eval_mean_prediction_accuracy"] = float(np.mean(pred_accuracies))
    return result


def main() -> None:
    args = _parse_args()
    recipe_names = list(MULTI_RECIPE_EXPERIMENT_POOL)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("logs") / "runs" / f"transfer_{args.env}_{args.approach}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)

    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Approach: {args.approach}, Seed: {args.seed}")

    # Save config
    config = vars(args)
    config["population_humans"] = [h for h, _ in POPULATION_HUMAN_CONFIGS]
    config["new_human"] = NEW_HUMAN_CONFIG[0]
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # PHASE 1: Population training
    # =====================================================================
    logging.info("=" * 60)
    logging.info("PHASE 1: Population training")
    logging.info("=" * 60)

    scene_spec = build_spice_scene_spec_multi(recipe_names)

    dummy_hidden = SpiceHiddenSpec(
        preferred_actor={},
        hidden_hbm=build_spice_experiment_hidden_hbm(recipe_names, "SpiceSpecificHumanStrong"),
    )
    dummy_env = SpiceEnv(hidden_spec=dummy_hidden, scene_spec=scene_spec, seed=args.seed)

    train_approach = _build_approach(args.approach, scene_spec, dummy_env.action_space, args.seed)
    train_approach.train()
    dummy_env.close()

    phase1_metrics: list[dict[str, float]] = []

    for human_label, config_name in POPULATION_HUMAN_CONFIGS:
        logging.info(f"--- Training on {human_label} ({config_name}) ---")

        hidden_hbm = build_spice_experiment_hidden_hbm(recipe_names, config_name)
        hidden_spec = SpiceHiddenSpec(preferred_actor={}, hidden_hbm=hidden_hbm)
        train_env = SpiceEnv(hidden_spec=hidden_spec, scene_spec=scene_spec, seed=args.seed)

        human_metrics = _run_training_phase(
            train_env, train_approach, args.phase1_steps,
            human_id=human_label,
        )
        phase1_metrics.extend(human_metrics)
        train_env.close()

    # Save Phase 1 model
    phase1_model_dir = model_dir / "phase1_final"
    phase1_model_dir.mkdir(exist_ok=True)
    train_approach.save(phase1_model_dir)
    logging.info(f"Saved Phase 1 model to {phase1_model_dir}")

    # Save Phase 1 metrics
    with open(out_dir / "phase1_train_results.csv", "w", newline="") as f:
        if phase1_metrics:
            writer = csv.DictWriter(f, fieldnames=phase1_metrics[0].keys())
            writer.writeheader()
            writer.writerows(phase1_metrics)

    # PHASE 2: New human with warm-started mu
    # =====================================================================
    logging.info("=" * 60)
    logging.info("PHASE 2: Transfer to new human")
    logging.info("=" * 60)

    new_human_label, new_human_config = NEW_HUMAN_CONFIG

    def _make_new_human_env(seed: int, eval_mode: bool = False) -> SpiceEnv:
        hbm = build_spice_experiment_hidden_hbm(recipe_names, new_human_config)
        spec = SpiceHiddenSpec(preferred_actor={}, hidden_hbm=hbm)
        return SpiceEnv(hidden_spec=spec, scene_spec=scene_spec, seed=seed, eval_mode=eval_mode)

    warm_train_env = _make_new_human_env(args.seed + 1000)
    cold_train_env = _make_new_human_env(args.seed + 1000)  # same seed for identical episodes
    new_eval_env = _make_new_human_env(args.seed + 2000, eval_mode=True)

    warm_approach = _build_approach(args.approach, scene_spec, warm_train_env.action_space, args.seed + 500)
    warm_approach.train()
    warm_approach.load(phase1_model_dir)

    def _register_new_human(approach_obj, new_id: str) -> None:
        try:
            pref_gen = approach_obj._csp_generator._pref_gen
            pref_gen._human_id = new_id
            hbm = pref_gen._hbm
            if hasattr(hbm, "register_human"):
                hbm.register_human(new_id)
            if hasattr(hbm, "register_recipe"):
                for recipe in list(pref_gen._recipe_list):
                    hbm.register_recipe(new_id, recipe)
        except AttributeError:
            pass

    _register_new_human(warm_approach, new_human_label)

    # Reset learned variance hyperparameters to defaults. 
    try:
        hbm = warm_approach._csp_generator._pref_gen._hbm
        if isinstance(hbm, HierarchicalPreferenceModel):
            default_sigma_h = DEFAULT_CONFIG.hbm.sigma_h
            default_sigma_r = DEFAULT_CONFIG.hbm.sigma_r
            default_sigma_obs = DEFAULT_CONFIG.hbm.sigma_obs
            with torch.no_grad():
                hbm.log_sigma_h.fill_(math.log(default_sigma_h))
                hbm.log_sigma_r.fill_(math.log(default_sigma_r))
                hbm.log_sigma_obs.fill_(math.log(default_sigma_obs))
            logging.info(
                f"Reset sigmas to defaults: sigma_h={default_sigma_h}, "
                f"sigma_r={default_sigma_r}, sigma_obs={default_sigma_obs}"
            )
    except AttributeError:
        pass 

    logging.info("Loaded Phase 1 model into warm-started approach (mu transfer)")

    cold_approach = _build_approach(args.approach, scene_spec, cold_train_env.action_space, args.seed + 600)
    cold_approach.train()
    _register_new_human(cold_approach, new_human_label)
    logging.info("Created cold-start approach (no transfer)")

    warm_eval = _build_approach(args.approach, scene_spec, new_eval_env.action_space, args.seed + 700)
    warm_eval.eval()
    _register_new_human(warm_eval, new_human_label)
    cold_eval = _build_approach(args.approach, scene_spec, new_eval_env.action_space, args.seed + 800)
    cold_eval.eval()
    _register_new_human(cold_eval, new_human_label)

    phase2_train_metrics: list[dict[str, float]] = []
    phase2_eval_metrics: list[dict[str, float]] = []

    warm_obs, warm_info = warm_train_env.reset()
    cold_obs, cold_info = cold_train_env.reset()
    warm_approach.reset(warm_obs, warm_info)
    warm_approach._scene_spec = warm_train_env.unwrapped.scene_spec
    cold_approach.reset(cold_obs, cold_info)
    cold_approach._scene_spec = cold_train_env.unwrapped.scene_spec

    for t in range(args.phase2_steps + 1):
        # Eval checkpoint
        if t % args.eval_frequency == 0:
            logging.info(f"Phase 2 eval at step {t}")

            # Save and load for eval
            warm_ckpt = model_dir / f"warm_{t}"
            warm_ckpt.mkdir(exist_ok=True)
            warm_approach.save(warm_ckpt)
            warm_eval.load(warm_ckpt)

            cold_ckpt = model_dir / f"cold_{t}"
            cold_ckpt.mkdir(exist_ok=True)
            cold_approach.save(cold_ckpt)
            cold_eval.load(cold_ckpt)

            # Neutral eval for both
            warm_metrics = _run_eval(
                warm_eval, new_eval_env, args.num_eval_trials,
                args.max_eval_episode_length, t, True, "warm_neutral_",
            )
            cold_metrics = _run_eval(
                cold_eval, new_eval_env, args.num_eval_trials,
                args.max_eval_episode_length, t, True, "cold_neutral_",
            )
            # Natural eval for both
            warm_natural = _run_eval(
                warm_eval, new_eval_env, args.num_eval_trials,
                args.max_eval_episode_length, t, False, "warm_natural_",
            )
            cold_natural = _run_eval(
                cold_eval, new_eval_env, args.num_eval_trials,
                args.max_eval_episode_length, t, False, "cold_natural_",
            )

            step_eval = {**warm_metrics, **cold_metrics, **warm_natural, **cold_natural}
            phase2_eval_metrics.append(step_eval)

        if t >= args.phase2_steps:
            break

        warm_act = warm_approach.step()
        warm_obs, warm_rew, warm_term, warm_trunc, warm_info = warm_train_env.step(warm_act)
        warm_done = bool(warm_term or warm_trunc)
        warm_approach.update(warm_obs, float(warm_rew), warm_done, warm_info)

        cold_act = cold_approach.step()
        cold_obs, cold_rew, cold_term, cold_trunc, cold_info = cold_train_env.step(cold_act)
        cold_done = bool(cold_term or cold_trunc)
        cold_approach.update(cold_obs, float(cold_rew), cold_done, cold_info)

        phase2_train_metrics.append({
            "step": t,
            "warm_satisfaction": warm_info.get("user_satisfaction", float("nan")),
            "warm_prediction_accuracy": warm_info.get("prediction_accuracy", float("nan")),
            "cold_satisfaction": cold_info.get("user_satisfaction", float("nan")),
            "cold_prediction_accuracy": cold_info.get("prediction_accuracy", float("nan")),
        })

        if warm_done:
            warm_obs, warm_info = warm_train_env.reset()
            warm_approach.reset(warm_obs, warm_info)
            warm_approach._scene_spec = warm_train_env.unwrapped.scene_spec
        if cold_done:
            cold_obs, cold_info = cold_train_env.reset()
            cold_approach.reset(cold_obs, cold_info)
            cold_approach._scene_spec = cold_train_env.unwrapped.scene_spec

    warm_train_env.close()
    cold_train_env.close()
    new_eval_env.close()

    # Save Phase 2 results
    if phase2_train_metrics:
        pd.DataFrame(phase2_train_metrics).to_csv(out_dir / "phase2_train_results.csv", index=False)
    if phase2_eval_metrics:
        pd.DataFrame(phase2_eval_metrics).to_csv(out_dir / "phase2_eval_results.csv", index=False)

    logging.info(f"Results written to {out_dir}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
