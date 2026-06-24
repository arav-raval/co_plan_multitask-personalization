"""Overcooked cross-layout leave-one-out experiment.
Trains on multiple layouts and evaluates on a held-out layout.

Phase 1: Training on layouts
Phase 2: Evaluate on held-out layout

Usage:
  python experiments/run_overcooked_layout_loo.py \\
    --approach ours --seed 0 \\
    --train-layouts CrampedRoom,AsymmetricAdvantages \\
    --held-out CoordinationRing \\
    --phase1-steps 2000 --phase2-steps 2000 \\
    --eval-frequency 250 --num-eval-trials 15

Outputs:
  - phase1_train_results.csv
  - phase2_train_results.csv
  - phase2_eval_results.csv   (main warm-vs-cold comparison)
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
from multitask_personalization.envs.overcooked.config.overcooked_config import (
    DEFAULT_CONFIG,
)
from multitask_personalization.envs.overcooked.layouts import get_layout
from multitask_personalization.envs.overcooked.overcooked_env import OvercookedEnv
from multitask_personalization.envs.overcooked.overcooked_experiment import (
    build_overcooked_hidden_spec,
)
from multitask_personalization.envs.overcooked.overcooked_hbm import (
    OvercookedPreferenceModel,
)
from multitask_personalization.methods.csp_approach import CSPApproach

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overcooked cross-layout LOO")
    p.add_argument("--approach", default="ours",
                   choices=["ours", "without_mood_learning", "cbtl_adapted", "flat_model"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase1-steps", type=int, default=2000,
                   help="Training steps per layout in Phase 1")
    p.add_argument("--phase2-steps", type=int, default=2000,
                   help="Training steps on the held-out layout in Phase 2")
    p.add_argument("--eval-frequency", type=int, default=250)
    p.add_argument("--num-eval-trials", type=int, default=15)
    p.add_argument("--max-eval-episode-length", type=int, default=200)
    p.add_argument("--train-layouts", default="CrampedRoom,AsymmetricAdvantages",
                   help="Comma-separated list of training layouts")
    p.add_argument("--held-out", default="CoordinationRing",
                   help="Held-out layout for Phase 2 evaluation")
    p.add_argument("--human-config", default="RealisticCook")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _build_approach(approach_name: str, scene_spec: Any, action_space: Any, seed: int) -> Any:
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
    layout_name: str,
    human_id: str = "human",
) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []

    # Ensure the HBM knows about this (human, layout) pair
    try:
        pref_gen = approach._csp_generator._pref_gen
        pref_gen._human_id = human_id
        hbm = pref_gen._hbm
        if hasattr(hbm, "register_human"):
            hbm.register_human(human_id)
        if hasattr(hbm, "register_layout"):
            hbm.register_layout(human_id, layout_name)
    except AttributeError:
        pass

    obs, info = env.reset()
    approach.reset(obs, info)
    approach._scene_spec = env.unwrapped.scene_spec

    for t in range(n_steps):
        act = approach.step()
        obs, rew, terminated, truncated, info = env.step(act)
        done = bool(terminated or truncated)
        approach.update(obs, float(rew), done, info)

        metrics.append({
            "step": t,
            "layout": layout_name,
            "user_satisfaction": info.get("user_satisfaction", float("nan")),
            "prediction_accuracy": info.get("prediction_accuracy", float("nan")),
        })

        if done:
            obs, info = env.reset()
            approach.reset(obs, info)
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
    satisfactions: list[float] = []
    pred_accuracies: list[float] = []

    for trial in range(num_trials):
        seed = 100000 + trial
        obs, info = eval_env.reset(seed=seed)
        eval_approach.reset(obs, info)
        eval_approach._scene_spec = eval_env.unwrapped.scene_spec

        original_fnm = eval_env.unwrapped._hidden_spec.force_neutral_session
        eval_env.unwrapped._hidden_spec.force_neutral_session = force_neutral_mood

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

        eval_env.unwrapped._hidden_spec.force_neutral_session = original_fnm

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
    train_layout_names = [s.strip() for s in args.train_layouts.split(",")]
    held_out_name = args.held_out.strip()
    human_id = "human"

    assert held_out_name not in train_layout_names, (
        f"Held-out layout {held_out_name} must not be in training set {train_layout_names}"
    )

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("logs") / "runs" / "oc_layout_loo" / f"{args.approach}_{held_out_name}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)

    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Approach: {args.approach}, Seed: {args.seed}")
    logging.info(f"Train layouts: {train_layout_names}")
    logging.info(f"Held-out: {held_out_name}")

    # Save config
    config = vars(args).copy()
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # PHASE 1: Train on layouts sequentially
    # =====================================================================
    logging.info("=" * 60)
    logging.info(f"PHASE 1: Training on {len(train_layout_names)} layouts")
    logging.info("=" * 60)

    # Build learner with a dummy env for action space
    first_layout_spec = get_layout(train_layout_names[0])
    dummy_hidden = build_overcooked_hidden_spec(train_layout_names[0], args.human_config)
    dummy_env = OvercookedEnv(
        layout_spec=first_layout_spec,
        hidden_spec=dummy_hidden,
        seed=args.seed,
    )
    train_approach = _build_approach(
        args.approach, dummy_env.unwrapped.scene_spec,
        dummy_env.action_space, args.seed,
    )
    train_approach.train()
    dummy_env.close()

    phase1_metrics: list[dict[str, float]] = []

    for layout_name in train_layout_names:
        logging.info(f"--- Training on {layout_name} ---")

        layout_spec = get_layout(layout_name)
        hidden_spec = build_overcooked_hidden_spec(layout_name, args.human_config)
        train_env = OvercookedEnv(
            layout_spec=layout_spec,
            hidden_spec=hidden_spec,
            seed=args.seed,
        )

        layout_metrics = _run_training_phase(
            train_env, train_approach, args.phase1_steps,
            layout_name=layout_name, human_id=human_id,
        )
        phase1_metrics.extend(layout_metrics)
        train_env.close()

    # Save Phase 1 model
    phase1_model_dir = model_dir / "phase1_final"
    phase1_model_dir.mkdir(exist_ok=True)
    train_approach.save(phase1_model_dir)
    logging.info(f"Saved Phase 1 model to {phase1_model_dir}")

    if phase1_metrics:
        with open(out_dir / "phase1_train_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=phase1_metrics[0].keys())
            writer.writeheader()
            writer.writerows(phase1_metrics)

    # PHASE 2: Evaluate on held-out layout
    # =====================================================================
    logging.info("=" * 60)
    logging.info(f"PHASE 2: Transfer to held-out layout: {held_out_name}")
    logging.info("=" * 60)

    held_out_layout_spec = get_layout(held_out_name)

    def _make_heldout_env(seed: int) -> Any:
        hidden = build_overcooked_hidden_spec(held_out_name, args.human_config)
        return OvercookedEnv(
            layout_spec=held_out_layout_spec,
            hidden_spec=hidden,
            seed=seed,
        )

    warm_train_env = _make_heldout_env(args.seed + 1000)
    cold_train_env = _make_heldout_env(args.seed + 1000)
    new_eval_env = _make_heldout_env(args.seed + 2000)

    # Warm start approach
    warm_approach = _build_approach(
        args.approach, warm_train_env.unwrapped.scene_spec,
        warm_train_env.action_space, args.seed + 500,
    )
    warm_approach.train()
    warm_approach.load(phase1_model_dir)

    try:
        pref_gen = warm_approach._csp_generator._pref_gen
        pref_gen._human_id = human_id
        if hasattr(pref_gen._hbm, "register_layout"):
            pref_gen._hbm.register_layout(human_id, held_out_name)
    except AttributeError:
        pass

    # Reset learned variance hyperparameters to defaults
    try:
        hbm = warm_approach._csp_generator._pref_gen._hbm
        if isinstance(hbm, OvercookedPreferenceModel):
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
    except (AttributeError, ImportError):
        pass

    logging.info("Loaded Phase 1 model into warm-started approach (theta transfer)")

    # Cold start approach
    cold_approach = _build_approach(
        args.approach, cold_train_env.unwrapped.scene_spec,
        cold_train_env.action_space, args.seed + 600,
    )
    cold_approach.train()
    try:
        pref_gen = cold_approach._csp_generator._pref_gen
        pref_gen._human_id = human_id
        if hasattr(pref_gen._hbm, "register_layout"):
            pref_gen._hbm.register_layout(human_id, held_out_name)
    except AttributeError:
        pass
    logging.info("Created cold-start approach (no transfer)")

    warm_eval = _build_approach(
        args.approach, new_eval_env.unwrapped.scene_spec,
        new_eval_env.action_space, args.seed + 700,
    )
    warm_eval.eval()
    cold_eval = _build_approach(
        args.approach, new_eval_env.unwrapped.scene_spec,
        new_eval_env.action_space, args.seed + 800,
    )
    cold_eval.eval()

    phase2_train_metrics: list[dict[str, float]] = []
    phase2_eval_metrics: list[dict[str, float]] = []

    warm_obs, warm_info = warm_train_env.reset()
    cold_obs, cold_info = cold_train_env.reset()
    warm_approach.reset(warm_obs, warm_info)
    warm_approach._scene_spec = warm_train_env.unwrapped.scene_spec
    cold_approach.reset(cold_obs, cold_info)
    cold_approach._scene_spec = cold_train_env.unwrapped.scene_spec

    for t in range(args.phase2_steps + 1):
        if t % args.eval_frequency == 0:
            logging.info(f"Phase 2 eval at step {t}")

            warm_ckpt = model_dir / f"warm_{t}"
            warm_ckpt.mkdir(exist_ok=True)
            warm_approach.save(warm_ckpt)
            warm_eval.load(warm_ckpt)

            cold_ckpt = model_dir / f"cold_{t}"
            cold_ckpt.mkdir(exist_ok=True)
            cold_approach.save(cold_ckpt)
            cold_eval.load(cold_ckpt)

            warm_metrics = _run_eval(
                warm_eval, new_eval_env, args.num_eval_trials,
                args.max_eval_episode_length, t, True, "warm_neutral_",
            )
            cold_metrics = _run_eval(
                cold_eval, new_eval_env, args.num_eval_trials,
                args.max_eval_episode_length, t, True, "cold_neutral_",
            )
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

    if phase2_train_metrics:
        pd.DataFrame(phase2_train_metrics).to_csv(
            out_dir / "phase2_train_results.csv", index=False,
        )
    if phase2_eval_metrics:
        pd.DataFrame(phase2_eval_metrics).to_csv(
            out_dir / "phase2_eval_results.csv", index=False,
        )

    logging.info(f"Results written to {out_dir}")
    logging.info("Done!")


if __name__ == "__main__":
    main()
